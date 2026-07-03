from __future__ import annotations

import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager

import asyncpg

_DEFAULT_STEP = "ingest"


@asynccontextmanager
async def _connection(conn_or_pool: asyncpg.Connection | asyncpg.Pool):
    """Yield a connection, acquiring one from a pool when needed."""
    if hasattr(conn_or_pool, "acquire"):
        async with conn_or_pool.acquire() as conn:
            yield conn
    else:
        yield conn_or_pool


async def insert_pending(
    conn_or_pool: asyncpg.Connection | asyncpg.Pool,
    session_id: uuid.UUID,
    user_id: uuid.UUID | None = None,
    *,
    pipeline_step: str = _DEFAULT_STEP,
) -> uuid.UUID:
    """Durably insert a pending processing-log row and return its id.

    Enqueueing is idempotent per ``(session_id, pipeline_step)``: if an
    in-flight (pending/processing) row already exists the existing id is
    returned rather than creating a duplicate that would drive redundant
    extraction work. ``processed_at`` is left NULL until a terminal
    transition stamps it.

    Accepts either a pool or an already-acquired connection so it can run
    inside a caller's transaction.
    """
    log_id = uuid.uuid4()
    async with _connection(conn_or_pool) as conn:
        inserted = await conn.fetchval(
            """
            INSERT INTO memory.processing_log
                (id, session_id, user_id, pipeline_step, status,
                 created_at, updated_at)
            VALUES ($1, $2, $3, $4, 'pending', now(), now())
            ON CONFLICT (session_id, pipeline_step)
                WHERE status IN ('pending', 'processing')
            DO NOTHING
            RETURNING id
            """,
            log_id,
            session_id,
            user_id,
            pipeline_step,
        )
        if inserted is not None:
            return inserted
        existing = await conn.fetchval(
            """
            SELECT id FROM memory.processing_log
            WHERE session_id = $1
              AND pipeline_step = $2
              AND status IN ('pending', 'processing')
            ORDER BY created_at
            LIMIT 1
            """,
            session_id,
            pipeline_step,
        )
    return existing if existing is not None else log_id


async def count_pending(pool: asyncpg.Pool) -> int:
    """Return the number of rows still awaiting processing."""
    async with _connection(pool) as conn:
        value = await conn.fetchval(
            "SELECT count(*) FROM memory.processing_log WHERE status = 'pending'"
        )
    return int(value or 0)


async def claim_pending(pool: asyncpg.Pool, batch: int) -> list[asyncpg.Record]:
    """Atomically claim up to *batch* pending rows.

    A single statement transitions the oldest pending rows to ``processing``
    using ``FOR UPDATE SKIP LOCKED`` so concurrent workers never claim the
    same row. Returns the claimed rows.
    """
    async with _connection(pool) as conn:
        rows = await conn.fetch(
            """
            UPDATE memory.processing_log
            SET status = 'processing', updated_at = now()
            WHERE id IN (
                SELECT id FROM memory.processing_log
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
            """,
            batch,
        )
    return list(rows)


async def mark_completed(
    pool: asyncpg.Pool,
    log_id: uuid.UUID,
    *,
    memories_created: Sequence[uuid.UUID] | None = None,
) -> None:
    """Mark a claimed row completed and record the memories it produced."""
    async with _connection(pool) as conn:
        await conn.execute(
            """
            UPDATE memory.processing_log
            SET status = 'completed',
                memories_created = $2,
                error_message = NULL,
                updated_at = now(),
                processed_at = now()
            WHERE id = $1
            """,
            log_id,
            list(memories_created or []),
        )


async def mark_failed(
    pool: asyncpg.Pool,
    log_id: uuid.UUID,
    *,
    error: str,
) -> None:
    """Mark a claimed row failed, store the error, and bump retry_count."""
    async with _connection(pool) as conn:
        await conn.execute(
            """
            UPDATE memory.processing_log
            SET status = 'failed',
                error_message = $2,
                retry_count = retry_count + 1,
                updated_at = now(),
                processed_at = now()
            WHERE id = $1
            """,
            log_id,
            error,
        )


async def count_failed(pool: asyncpg.Pool) -> int:
    """Return the number of rows that have exhausted a processing attempt."""
    async with _connection(pool) as conn:
        value = await conn.fetchval(
            "SELECT count(*) FROM memory.processing_log WHERE status = 'failed'"
        )
    return int(value or 0)


async def oldest_pending_age_seconds(pool: asyncpg.Pool) -> float:
    """Return the age in seconds of the oldest pending row (0 when none)."""
    async with _connection(pool) as conn:
        value = await conn.fetchval(
            "SELECT extract(epoch FROM now() - min(created_at)) "
            "FROM memory.processing_log WHERE status = 'pending'"
        )
    return float(value or 0.0)


async def requeue_failed(pool: asyncpg.Pool, max_retries: int) -> int:
    """Return failed rows below the retry ceiling to ``pending`` for another attempt.

    ``retry_count`` is left untouched — it is already incremented by
    ``mark_failed`` — so a row is retried at most *max_retries* times before it
    stays terminal. At most one row per ``(session_id, pipeline_step)`` is
    requeued and only when no in-flight sibling already covers that work, which
    keeps the requeue from colliding with the in-flight uniqueness invariant.
    Returns the number of rows requeued.
    """
    async with _connection(pool) as conn:
        rows = await conn.fetch(
            """
            UPDATE memory.processing_log
            SET status = 'pending', updated_at = now()
            WHERE id IN (
                SELECT DISTINCT ON (session_id, pipeline_step) id
                FROM memory.processing_log f
                WHERE f.status = 'failed'
                  AND f.retry_count < $1
                  AND NOT EXISTS (
                      SELECT 1 FROM memory.processing_log p
                      WHERE p.session_id = f.session_id
                        AND p.pipeline_step = f.pipeline_step
                        AND p.status IN ('pending', 'processing')
                  )
                ORDER BY session_id, pipeline_step, created_at
            )
            RETURNING id
            """,
            max_retries,
        )
    return len(rows)


async def requeue_stale(pool: asyncpg.Pool, visibility_timeout_s: float) -> int:
    """Return rows stuck in ``processing`` past the visibility timeout to pending.

    A row whose ``updated_at`` is older than *visibility_timeout_s* is assumed
    to belong to a worker that died mid-flight and is made claimable again.
    Returns the number of rows requeued.
    """
    async with _connection(pool) as conn:
        rows = await conn.fetch(
            """
            UPDATE memory.processing_log
            SET status = 'pending', updated_at = now()
            WHERE status = 'processing'
              AND updated_at < now() - make_interval(secs => $1)
            RETURNING id
            """,
            float(visibility_timeout_s),
        )
    return len(rows)
