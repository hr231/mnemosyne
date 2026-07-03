from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from mnemosyne.context.token_budget import DEFAULT_ENCODING, TokenBudget
from mnemosyne.db.models.memory import ScoredMemory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.monitoring.metrics import global_registry
from mnemosyne.providers.base import MemoryProvider


@dataclass
class Section:
    name: str
    content: str
    token_count: int


class ContextBlock(BaseModel):
    text: str
    token_count: int
    sections: list[Any] | None = None


async def assemble_context(
    provider: MemoryProvider,
    user_id: UUID,
    query_embedding: list[float],
    embedder: EmbeddingClient,
    token_budget: int = 2000,
    entity_store: Any = None,
    query_text: str = "",
    encoding_name: str = DEFAULT_ENCODING,
) -> ContextBlock:
    reg = global_registry()
    started = time.perf_counter()
    try:
        budget = TokenBudget(max_tokens=token_budget, encoding_name=encoding_name)

        # Entity names discovered by entity-aware search, reused by the entity
        # section so NER runs at most once per assembly (off the event loop).
        entity_names: list[str] = []

        # Fetch candidates — use entity-aware search when entity_store is provided
        if entity_store is not None and query_text:
            from mnemosyne.retrieval.entity_search import entity_aware_search
            hits = await entity_aware_search(
                provider,
                entity_store,
                query_text,
                query_embedding,
                user_id,
                embedder,
                limit=20,
                out_entity_names=entity_names,
            )
        else:
            hits = await provider.search(query_embedding, user_id=user_id, limit=20)

        # Memory ids already placed in a higher-priority section. Each memory is
        # emitted at most once: profile > relevant > recent.
        placed: set[UUID] = set()

        # Section 1: Profile — high-importance memories (importance > 0.7)
        profile_mems = [sm for sm in hits if sm.memory.importance > 0.7]
        profile_mems.sort(key=lambda sm: sm.memory.importance, reverse=True)
        profile_section = _build_section("profile", profile_mems[:5], budget, placed)

        # Section 2: Query-relevant — all search results, scored by relevance
        relevant_section = _build_section("relevant", hits[:10], budget, placed)

        # Section 3: Recent — sorted by created_at descending
        recent = sorted(hits, key=lambda sm: sm.memory.created_at, reverse=True)
        recent_section = _build_section("recent", recent[:5], budget, placed)

        # Section 4: Entities — names surfaced from the query (entity_store path only)
        entity_section = _build_entity_section(entity_names, budget)

        sections = [
            s
            for s in [profile_section, relevant_section, recent_section, entity_section]
            if s.content
        ]
        text = "".join(s.content for s in sections)

        used = budget.used
        cap = max(1, token_budget)
        utilization = used / cap
        reg.set_context_token_utilization(utilization)
        if utilization >= 0.95:
            reg.record_context_truncate()
        else:
            reg.record_context_inject()

        return ContextBlock(
            text=text,
            token_count=used,
            sections=sections,
        )
    finally:
        reg.record_retrieval_latency_ms((time.perf_counter() - started) * 1000.0)


def _build_section(
    name: str,
    scored_memories: list[ScoredMemory],
    budget: TokenBudget,
    placed: set[UUID] | None = None,
) -> Section:
    lines: list[str] = []
    section_tokens = 0
    for sm in scored_memories:
        mid = sm.memory.memory_id
        if placed is not None and mid in placed:
            continue
        line = f"- {sm.memory.content}\n"
        fitted, used = budget.consume(line)
        if used == 0:
            break
        lines.append(fitted)
        section_tokens += used
        if placed is not None:
            placed.add(mid)
    return Section(name=name, content="".join(lines), token_count=section_tokens)


def _build_entity_section(entity_names: list[str], budget: TokenBudget) -> Section:
    """Build Section 4 from entity names already extracted upstream.

    The names come from ``entity_aware_search`` (which runs NER off the event
    loop), so this function performs no I/O and no extraction of its own.
    """
    if not entity_names:
        return Section(name="entities", content="", token_count=0)

    lines: list[str] = []
    section_tokens = 0
    for name in entity_names:
        line = f"- {name}\n"
        fitted, used = budget.consume(line)
        if used == 0:
            break
        lines.append(fitted)
        section_tokens += used

    return Section(name="entities", content="".join(lines), token_count=section_tokens)
