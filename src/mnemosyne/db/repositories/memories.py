from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import asyncpg


@asynccontextmanager
async def _connection(conn_or_pool: asyncpg.Connection | asyncpg.Pool):
    """Yield a connection, acquiring one from a pool when needed."""
    if hasattr(conn_or_pool, "acquire"):
        async with conn_or_pool.acquire() as conn:
            yield conn
    else:
        yield conn_or_pool


async def active_user_ids(pool: asyncpg.Connection | asyncpg.Pool) -> list[uuid.UUID]:
    """Return every user with at least one active (non-invalidated) memory."""
    async with _connection(pool) as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT user_id FROM memory.memories WHERE valid_until IS NULL"
        )
    return [r["user_id"] for r in rows if r["user_id"] is not None]
