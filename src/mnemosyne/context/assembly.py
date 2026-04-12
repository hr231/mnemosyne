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
) -> ContextBlock:
    budget = TokenBudget(max_tokens=token_budget, encoding=_ENC)

    # Fetch candidates
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

    sections = [s for s in [profile_section, relevant_section, recent_section] if s.content]
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
