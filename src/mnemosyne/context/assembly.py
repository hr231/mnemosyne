from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import tiktoken
from pydantic import BaseModel

from mnemosyne.context.token_budget import TokenBudget
from mnemosyne.db.models.memory import ScoredMemory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider

_ENC = tiktoken.encoding_for_model("gpt-4")


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
) -> ContextBlock:
    budget = TokenBudget(max_tokens=token_budget, encoding=_ENC)

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
        )
    else:
        hits = await provider.search(query_embedding, user_id=user_id, limit=20)

    # Section 1: Profile — high-importance memories (importance > 0.7)
    profile_mems = [sm for sm in hits if sm.memory.importance > 0.7]
    profile_mems.sort(key=lambda sm: sm.memory.importance, reverse=True)
    profile_section = _build_section("profile", profile_mems[:5], budget)

    # Section 2: Query-relevant — all search results, scored by relevance
    relevant_section = _build_section("relevant", hits[:10], budget)

    # Section 3: Recent — sorted by created_at descending
    recent = sorted(hits, key=lambda sm: sm.memory.created_at, reverse=True)
    recent_section = _build_section("recent", recent[:5], budget)

    # Section 4: Entities — names of entities found in the query (entity_store path only)
    entity_section = _build_entity_section(
        user_id, query_text, entity_store, budget
    )

    sections = [
        s
        for s in [profile_section, relevant_section, recent_section, entity_section]
        if s.content
    ]
    text = "".join(s.content for s in sections)
    return ContextBlock(
        text=text,
        token_count=budget.used,
        sections=sections,
    )


def _build_section(
    name: str,
    scored_memories: list[ScoredMemory],
    budget: TokenBudget,
) -> Section:
    lines: list[str] = []
    section_tokens = 0
    for sm in scored_memories:
        line = f"- {sm.memory.content}\n"
        fitted, used = budget.consume(line)
        if used == 0:
            break
        lines.append(fitted)
        section_tokens += used
    return Section(name=name, content="".join(lines), token_count=section_tokens)


def _build_entity_section(
    user_id: UUID,
    query_text: str,
    entity_store: Any,
    budget: TokenBudget,
) -> Section:
    """Build Section 4: entity names found in the query.

    This is a best-effort synchronous summary derived from the entity_store.
    When entity_store is None or query_text is empty, returns an empty section.
    The actual entity lookup is done inline using NER (no I/O — entity data
    already fetched during entity_aware_search).
    """
    if entity_store is None or not query_text:
        return Section(name="entities", content="", token_count=0)

    # Collect entity names via the same lightweight NER path used in entity search
    entity_names: list[str] = []
    seen_names: set[str] = set()

    try:
        from mnemosyne.pipeline.ner.spacy_extractor import extract_entities_spacy
        for raw in extract_entities_spacy(query_text):
            key = raw.name.strip().lower()
            if key not in seen_names:
                entity_names.append(raw.name)
                seen_names.add(key)
    except Exception:
        pass

    try:
        from mnemosyne.pipeline.ner.gliner_extractor import extract_entities_gliner
        for raw in extract_entities_gliner(query_text):
            key = raw.name.strip().lower()
            if key not in seen_names:
                entity_names.append(raw.name)
                seen_names.add(key)
    except Exception:
        pass

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
