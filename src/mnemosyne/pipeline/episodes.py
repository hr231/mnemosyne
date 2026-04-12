from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.episode import Episode
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.llm.base import LLMClient
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)


async def create_episode(
    provider: MemoryProvider,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    memory_ids: list[uuid.UUID],
    summary: str | None = None,
    llm_client: LLMClient | None = None,
    embedder: EmbeddingClient | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> Episode:
    """Create an Episode record for a completed session.

    Summary generation priority:
    1. Use *summary* if provided directly (e.g. from lakebase).
    2. If *llm_client* is given, generate a memory-specific summary
       from the content of the linked memories.
    3. Fall back to concatenating memory contents with ``"; "``.
    4. If no memories, use ``"Empty session"``.

    The episode summary embedding is generated when *embedder* is provided.

    This function does NOT persist the episode — the caller is responsible
    for storing the returned ``Episode`` object via the appropriate
    repository or provider.
    """
    # Resolve summary text
    if not summary:
        if memory_ids:
            contents: list[str] = []
            for mid in memory_ids:
                mem = await provider.get_by_id(mid)
                if mem:
                    contents.append(mem.content)

            if contents and llm_client is not None:
                prompt = _build_episode_summary_prompt(contents)
                try:
                    summary = await llm_client.complete(prompt)
                except Exception:
                    logger.exception(
                        "LLM summary generation failed for session %s — using fallback",
                        session_id,
                    )
                    summary = "; ".join(contents)
            elif contents:
                summary = "; ".join(contents)
            else:
                summary = "Empty session"
        else:
            summary = "Empty session"

    # Generate summary embedding when an embedder is available
    summary_embedding: list[float] | None = None
    if embedder is not None and summary:
        try:
            summary_embedding = await embedder.embed(summary)
        except Exception:
            logger.exception(
                "Failed to embed episode summary for session %s", session_id
            )

    episode = Episode(
        user_id=user_id,
        session_id=session_id,
        summary=summary,
        summary_embedding=summary_embedding,
        memory_ids=list(memory_ids),
        started_at=started_at or datetime.now(timezone.utc),
        ended_at=ended_at,
    )

    logger.debug(
        "create_episode: session=%s memories=%d summary_len=%d",
        session_id,
        len(memory_ids),
        len(summary),
    )
    return episode


def _build_episode_summary_prompt(memory_contents: list[str]) -> str:
    """Build the LLM prompt for generating a memory-specific episode summary."""
    items = "\n".join(f"- {c}" for c in memory_contents)
    return (
        "You are summarising a user session for a memory system.\n"
        "Given the following memories extracted from the session, write a single "
        "concise sentence (max 50 words) that captures the key information.\n\n"
        f"Memories:\n{items}\n\n"
        "Summary:"
    )
