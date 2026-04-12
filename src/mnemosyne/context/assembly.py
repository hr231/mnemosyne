from __future__ import annotations

from typing import Any
from uuid import UUID

import tiktoken
from pydantic import BaseModel

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider

_ENC = tiktoken.encoding_for_model("gpt-4")


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
    hits = await provider.search(query_embedding, user_id=user_id, limit=10)

    lines: list[str] = []
    total_tokens = 0
    for scored_mem in hits:
        line = f"- {scored_mem.memory.content}\n"
        line_tokens = len(_ENC.encode(line))
        if total_tokens + line_tokens > token_budget:
            remaining = token_budget - total_tokens
            if remaining > 0:
                encoded = _ENC.encode(line)[:remaining]
                lines.append(_ENC.decode(encoded))
                total_tokens += remaining
            break
        lines.append(line)
        total_tokens += line_tokens

    text = "".join(lines)
    return ContextBlock(text=text, token_count=total_tokens)
