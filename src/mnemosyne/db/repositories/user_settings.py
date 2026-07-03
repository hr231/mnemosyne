from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import asyncpg


@asynccontextmanager
async def _connection(conn_or_pool: asyncpg.Connection | asyncpg.Pool):
    """Yield a connection, acquiring one from a pool when needed."""
    if hasattr(conn_or_pool, "acquire"):
        async with conn_or_pool.acquire() as conn:
            yield conn
    else:
        yield conn_or_pool


async def get_last_reflected_at(
    pool: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
) -> datetime | None:
    """Return the reflection watermark for *user_id*, or None if never reflected."""
    async with _connection(pool) as conn:
        return await conn.fetchval(
            "SELECT last_reflected_at FROM memory.user_settings WHERE user_id = $1",
            user_id,
        )


async def set_last_reflected_at(
    pool: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    ts: datetime,
) -> None:
    """Upsert the reflection watermark, preserving any existing settings."""
    async with _connection(pool) as conn:
        await conn.execute(
            """
            INSERT INTO memory.user_settings (user_id, last_reflected_at, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (user_id) DO UPDATE
            SET last_reflected_at = EXCLUDED.last_reflected_at,
                updated_at = now()
            """,
            user_id,
            ts,
        )


async def get_extraction_enabled(
    pool: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
) -> bool | None:
    """Return the extraction opt-out flag for *user_id*, or None if no row."""
    async with _connection(pool) as conn:
        return await conn.fetchval(
            "SELECT extraction_enabled FROM memory.user_settings WHERE user_id = $1",
            user_id,
        )
