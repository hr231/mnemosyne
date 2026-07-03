from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import asyncpg

# Fuzzy and semantic dedup are pairwise self-joins whose cost grows with the
# square of the active set. When a caller supplies no lower bound the scan is
# clamped to this recent window so a maintenance pass can never fan out into an
# unbounded cross-product over the whole table.
_MAX_DEDUP_WINDOW = timedelta(days=30)


def _bounded_created_after(created_after: datetime | None) -> datetime:
    """Clamp an optional lower bound to a recent window, never unbounded."""
    floor = datetime.now(timezone.utc) - _MAX_DEDUP_WINDOW
    if created_after is None:
        return floor
    return max(created_after, floor)


async def find_exact_duplicates(
    conn: asyncpg.Connection,
    user_id: uuid.UUID,
    created_after: datetime | None = None,
) -> list[dict]:
    """Find memories with identical content_hash for a user."""
    sql = """
        SELECT content_hash, array_agg(memory_id) as memory_ids,
               max(importance) as max_importance
        FROM memory.memories
        WHERE user_id = $1 AND valid_until IS NULL
    """
    params: list = [user_id]
    if created_after:
        sql += " AND created_at > $2"
        params.append(created_after)
    sql += " GROUP BY content_hash HAVING count(*) > 1"
    rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


async def find_fuzzy_duplicates(
    conn: asyncpg.Connection,
    user_id: uuid.UUID,
    threshold: float = 0.8,
    created_after: datetime | None = None,
) -> list[dict]:
    """Find memory pairs with pg_trgm similarity above threshold.

    The self-join is always bounded to a recent window (see
    ``_MAX_DEDUP_WINDOW``); a missing ``created_after`` is clamped rather than
    scanning the whole active set.
    """
    bound = _bounded_created_after(created_after)
    sql = """
        SELECT a.memory_id AS id_a, b.memory_id AS id_b,
               similarity(a.content, b.content) AS sim,
               a.importance AS imp_a, b.importance AS imp_b
        FROM memory.memories a
        JOIN memory.memories b ON a.memory_id < b.memory_id
        WHERE a.user_id = $1 AND b.user_id = $1
          AND a.valid_until IS NULL AND b.valid_until IS NULL
          AND similarity(a.content, b.content) > $2
          AND a.created_at > $3
    """
    rows = await conn.fetch(sql, user_id, threshold, bound)
    return [dict(r) for r in rows]


async def find_semantic_duplicates(
    conn: asyncpg.Connection,
    user_id: uuid.UUID,
    threshold: float = 0.90,
    created_after: datetime | None = None,
) -> list[dict]:
    """Find memory pairs with cosine similarity above threshold.

    The self-join is always bounded to a recent window (see
    ``_MAX_DEDUP_WINDOW``); a missing ``created_after`` is clamped rather than
    scanning the whole active set.
    """
    bound = _bounded_created_after(created_after)
    sql = """
        SELECT a.memory_id AS id_a, b.memory_id AS id_b,
               1 - (a.embedding <=> b.embedding) AS cosine_sim,
               a.importance AS imp_a, b.importance AS imp_b
        FROM memory.memories a
        JOIN memory.memories b ON a.memory_id < b.memory_id
        WHERE a.user_id = $1 AND b.user_id = $1
          AND a.valid_until IS NULL AND b.valid_until IS NULL
          AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
          AND 1 - (a.embedding <=> b.embedding) > $2
          AND a.created_at > $3
    """
    rows = await conn.fetch(sql, user_id, threshold, bound)
    return [dict(r) for r in rows]
