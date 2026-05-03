from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Protocol

import asyncpg

from mnemosyne.db.models.contradiction_audit import ContradictionAudit


class ContradictionAuditStore(Protocol):
    """Persistence boundary for contradiction-resolution audit rows."""

    async def record(self, entry: ContradictionAudit) -> None: ...
    async def list_for_user(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        since: datetime | None = None,
    ) -> list[ContradictionAudit]: ...
    async def count_for_user(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> int: ...


class InMemoryContradictionAuditStore:
    """Dict-backed audit store for tests and the InMemoryProvider path."""

    def __init__(self) -> None:
        self._rows: list[ContradictionAudit] = []

    async def record(self, entry: ContradictionAudit) -> None:
        self._rows.append(entry)

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        since: datetime | None = None,
    ) -> list[ContradictionAudit]:
        rows = [r for r in self._rows if r.user_id == user_id]
        if since is not None:
            rows = [r for r in rows if r.detected_at >= since]
        rows.sort(key=lambda r: r.detected_at, reverse=True)
        return rows[offset : offset + limit]

    async def count_for_user(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> int:
        rows = (r for r in self._rows if r.user_id == user_id)
        if since is not None:
            rows = (r for r in rows if r.detected_at >= since)
        return sum(1 for _ in rows)


class PostgresContradictionAuditStore:
    """asyncpg-backed audit store writing to memory.contradiction_audit."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record(self, entry: ContradictionAudit) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory.contradiction_audit (
                    id, user_id, detected_at, new_memory_id, existing_memory_id,
                    nli_scores, llm_adjudication, resolution, reasoning, applied
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6::jsonb, $7, $8, $9, $10
                )
                """,
                entry.id,
                entry.user_id,
                entry.detected_at,
                entry.new_memory_id,
                entry.existing_memory_id,
                json.dumps(entry.nli_scores),
                entry.llm_adjudication,
                entry.resolution,
                entry.reasoning,
                entry.applied,
            )

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        since: datetime | None = None,
    ) -> list[ContradictionAudit]:
        async with self._pool.acquire() as conn:
            if since is None:
                rows = await conn.fetch(
                    """
                    SELECT * FROM memory.contradiction_audit
                    WHERE user_id = $1
                    ORDER BY detected_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    user_id,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM memory.contradiction_audit
                    WHERE user_id = $1 AND detected_at >= $2
                    ORDER BY detected_at DESC
                    LIMIT $3 OFFSET $4
                    """,
                    user_id,
                    since,
                    limit,
                    offset,
                )
        return [_row_to_audit(r) for r in rows]

    async def count_for_user(
        self, user_id: uuid.UUID, since: datetime | None = None
    ) -> int:
        async with self._pool.acquire() as conn:
            if since is None:
                row = await conn.fetchrow(
                    """
                    SELECT count(*) AS n FROM memory.contradiction_audit
                    WHERE user_id = $1
                    """,
                    user_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT count(*) AS n FROM memory.contradiction_audit
                    WHERE user_id = $1 AND detected_at >= $2
                    """,
                    user_id,
                    since,
                )
        return int(row["n"]) if row is not None else 0


def _row_to_audit(row: asyncpg.Record) -> ContradictionAudit:
    """Convert a Postgres row to a ContradictionAudit model."""
    raw_scores = row["nli_scores"]
    if isinstance(raw_scores, str):
        raw_scores = json.loads(raw_scores)
    return ContradictionAudit(
        id=row["id"],
        user_id=row["user_id"],
        detected_at=row["detected_at"],
        new_memory_id=row["new_memory_id"],
        existing_memory_id=row["existing_memory_id"],
        nli_scores=dict(raw_scores or {}),
        llm_adjudication=row["llm_adjudication"],
        resolution=row["resolution"],
        reasoning=row["reasoning"],
        applied=bool(row["applied"]),
    )
