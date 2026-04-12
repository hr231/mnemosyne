from __future__ import annotations

import logging
import uuid

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)


async def embed_pending_memories(
    provider: MemoryProvider,
    embedder: EmbeddingClient,
    batch_size: int = 100,
) -> int:
    """Embed memories that have NULL embeddings.

    Scans the provider for memories without embeddings, embeds them in
    batches, and updates each memory in-place.

    Works with both ``InMemoryProvider`` (via ``_memories`` dict) and
    ``PostgresMemoryProvider`` (via raw asyncpg connection pool ``_pool``).

    Returns the count of memories that were embedded.
    """
    total = 0

    if hasattr(provider, "_memories"):
        # InMemoryProvider: iterate the in-process dict
        unembedded = [
            m for m in provider._memories.values()
            if m.embedding is None
        ]
        for i in range(0, len(unembedded), batch_size):
            batch = unembedded[i : i + batch_size]
            texts = [m.content for m in batch]
            try:
                embeddings = await embedder.embed_batch(texts)
            except Exception:
                logger.exception(
                    "embed_batch failed for InMemoryProvider batch of %d memories",
                    len(batch),
                )
                raise
            for mem, embedding in zip(batch, embeddings):
                mem.embedding = embedding
                total += 1

    elif hasattr(provider, "_pool"):
        # PostgresMemoryProvider: use asyncpg pool directly
        pool = provider._pool
        while True:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT memory_id, content
                    FROM memory.memories
                    WHERE embedding IS NULL
                    LIMIT $1
                    """,
                    batch_size,
                )
            if not rows:
                break

            texts = [row["content"] for row in rows]
            try:
                embeddings = await embedder.embed_batch(texts)
            except Exception:
                logger.exception(
                    "embed_batch failed for Postgres batch of %d memories",
                    len(rows),
                )
                raise

            async with pool.acquire() as conn:
                for row, embedding in zip(rows, embeddings):
                    await conn.execute(
                        """
                        UPDATE memory.memories
                        SET embedding = $1::halfvec
                        WHERE memory_id = $2
                        """,
                        embedding,
                        row["memory_id"],
                    )
            total += len(rows)
            if len(rows) < batch_size:
                break

    else:
        logger.warning(
            "embed_pending_memories: unknown provider type %s — skipping",
            type(provider).__name__,
        )

    if total:
        logger.info("embed_pending_memories: embedded %d memories", total)
    return total


async def embed_memory_ids(
    provider: MemoryProvider,
    embedder: EmbeddingClient,
    memory_ids: list[uuid.UUID],
) -> int:
    """Embed a specific list of memories by ID.

    Useful when the pipeline runner already knows which memories were
    just created and wants to embed only those, rather than scanning
    all NULL embeddings.

    Returns the count of memories that were (re)embedded.
    """
    total = 0
    for memory_id in memory_ids:
        mem = await provider.get_by_id(memory_id)
        if mem is None or mem.embedding is not None:
            continue
        embedding = await embedder.embed(mem.content)
        await provider.update(memory_id, embedding=embedding)
        total += 1
    return total
