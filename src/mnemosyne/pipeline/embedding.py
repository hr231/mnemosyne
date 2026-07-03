from __future__ import annotations

import logging
import uuid

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)


def _validate_batch(
    rows: list,
    embeddings: list[list[float]],
    expected_dim: int | None,
) -> None:
    """Ensure embeddings line up 1:1 with rows and match the expected dimension."""
    if len(embeddings) != len(rows):
        raise ValueError(
            f"embed_batch returned {len(embeddings)} vectors for {len(rows)} inputs"
        )
    if expected_dim is not None:
        for vec in embeddings:
            if len(vec) != expected_dim:
                raise ValueError(
                    f"embedding dimension {len(vec)} != expected {expected_dim}"
                )


async def embed_pending_memories(
    provider: MemoryProvider,
    embedder: EmbeddingClient,
    batch_size: int = 100,
    user_id: uuid.UUID | None = None,
    expected_dim: int | None = None,
) -> int:
    """Embed memories that have NULL embeddings.

    Scans the provider for memories without embeddings, embeds them in
    batches, and updates each memory in-place. When *user_id* is given only
    that user's pending memories are embedded. *expected_dim* enables a
    response-dimension check.

    Works with both ``InMemoryProvider`` (via ``_memories`` dict) and
    ``PostgresMemoryProvider`` (via raw asyncpg connection pool ``_pool``).

    Returns the count of memories that were embedded.
    """
    total = 0

    if hasattr(provider, "_memories"):
        unembedded = [
            m
            for m in provider._memories.values()  # type: ignore[attr-defined]
            if m.embedding is None and (user_id is None or m.user_id == user_id)
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
            _validate_batch(batch, embeddings, expected_dim)
            for mem, embedding in zip(batch, embeddings, strict=True):
                mem.embedding = embedding
                total += 1

    elif hasattr(provider, "_pool"):
        pool = provider._pool  # type: ignore[attr-defined]
        params: list = [batch_size]
        user_filter = ""
        if user_id is not None:
            user_filter = "AND user_id = $2"
            params.append(user_id)

        while True:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT memory_id, content
                    FROM memory.memories
                    WHERE embedding IS NULL
                    {user_filter}
                    LIMIT $1
                    """,
                    *params,
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

            _validate_batch(rows, embeddings, expected_dim)

            update_args = [
                (embedding, row["memory_id"])
                for row, embedding in zip(rows, embeddings, strict=True)
            ]
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        """
                        UPDATE memory.memories
                        SET embedding = $1::halfvec
                        WHERE memory_id = $2
                        """,
                        update_args,
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
