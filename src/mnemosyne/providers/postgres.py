from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from mnemosyne.db.models.history import MemoryHistoryEntry
from mnemosyne.db.models.memory import Memory, MemoryType, ScoredMemory
from mnemosyne.errors import MemoryNotFound
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.scoring import MultiSignalScorer, ScoringWeights
from mnemosyne.utils import content_hash

# Fields that must never be overwritten via update()
_READ_ONLY_FIELDS = frozenset({"memory_id", "content_hash", "extraction_version"})

# Multiplier for HNSW over-fetch before Python-side re-rank
_OVERFETCH_FACTOR = 5


def _row_to_memory(row: asyncpg.Record) -> Memory:
    """Convert an asyncpg row from memory.memories into a Memory pydantic model."""
    embedding: list[float] | None = None
    raw_embedding = row["embedding"]
    if raw_embedding is not None:
        embedding = raw_embedding.to_list()

    source_memory_ids: list[uuid.UUID] = []
    raw_smi = row["source_memory_ids"]
    if raw_smi:
        source_memory_ids = [uuid.UUID(str(x)) for x in raw_smi]

    metadata: dict[str, Any] = {}
    raw_meta = row["metadata"]
    if raw_meta:
        if isinstance(raw_meta, str):
            metadata = json.loads(raw_meta)
        else:
            metadata = dict(raw_meta)

    return Memory(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        org_id=row.get("org_id"),
        memory_type=MemoryType(row["memory_type"]),
        content=row["content"],
        content_hash=row["content_hash"],
        embedding=embedding,
        importance=float(row["importance"]),
        access_count=int(row["access_count"]),
        last_accessed=row["last_accessed"],
        decay_rate=float(row["decay_rate"]),
        valid_from=row["valid_from"],
        valid_until=row.get("valid_until"),
        extraction_version=row["extraction_version"],
        extraction_model=row.get("extraction_model"),
        prompt_hash=row.get("prompt_hash"),
        rule_id=row.get("rule_id"),
        source_session_id=row.get("source_session_id"),
        source_memory_ids=source_memory_ids,
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PostgresMemoryProvider(MemoryProvider):
    """PostgreSQL-backed memory provider using asyncpg and pgvector.

    Use the factory method ``PostgresMemoryProvider.connect(dsn)`` to
    create an instance.  The constructor accepts an existing asyncpg pool
    for testability.

    Connection pool notes
    ---------------------
    - Pool is initialised with ``min_size=2, max_size=10``.
    - Each new connection registers the pgvector codec via
      ``pgvector.asyncpg.register_vector``.
    - Call ``await provider.close()`` to drain the pool on shutdown.

    History writes
    --------------
    Every ``add``, ``invalidate``, and ``update`` call writes an immutable
    ``MemoryHistoryEntry`` row to ``memory.memory_history`` inside the same
    transaction as the primary mutation.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def connect(cls, dsn: str) -> "PostgresMemoryProvider":
        """Create a pool, register pgvector codec, return a provider."""
        try:
            from pgvector.asyncpg import register_vector
        except ImportError as exc:
            raise ImportError(
                "pgvector package is required: pip install pgvector"
            ) from exc

        async def _init_conn(conn: asyncpg.Connection) -> None:
            await register_vector(conn)

        pool = await asyncpg.create_pool(
            dsn,
            min_size=2,
            max_size=10,
            init=_init_conn,
        )
        return cls(pool)

    async def close(self) -> None:
        """Drain and close the underlying connection pool."""
        await self._pool.close()

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------

    async def add(self, memory: Memory) -> uuid.UUID:
        """Persist *memory* and return its UUID.

        Raises ``ValueError`` if ``memory.embedding`` is ``None``.
        Returns the existing ``memory_id`` without writing a duplicate if a
        non-invalidated memory with the same ``(user_id, content_hash)``
        already exists (exact-hash dedup).
        """
        if memory.embedding is None:
            raise ValueError("caller must set embedding before add")

        mem_id = memory.memory_id
        ch = content_hash(memory.content)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Atomic dedup: attempt insert, let the DB handle the conflict
                result = await conn.fetchval(
                    """
                    INSERT INTO memory.memories (
                        memory_id, user_id, agent_id, org_id,
                        content, content_hash, embedding,
                        memory_type, importance, access_count,
                        last_accessed, decay_rate,
                        valid_from, valid_until,
                        extraction_version, extraction_model, prompt_hash, rule_id,
                        source_session_id, source_memory_ids,
                        metadata, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7::halfvec,
                        $8, $9, $10,
                        $11, $12,
                        $13, $14,
                        $15, $16, $17, $18,
                        $19, $20::uuid[],
                        $21::jsonb, $22, $23
                    )
                    ON CONFLICT (user_id, content_hash) WHERE valid_until IS NULL
                    DO NOTHING
                    RETURNING memory_id
                    """,
                    mem_id,
                    memory.user_id,
                    memory.agent_id,
                    memory.org_id,
                    memory.content,
                    ch,
                    memory.embedding,
                    memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type),
                    memory.importance,
                    memory.access_count,
                    memory.last_accessed,
                    memory.decay_rate,
                    memory.valid_from,
                    memory.valid_until,
                    memory.extraction_version,
                    memory.extraction_model,
                    memory.prompt_hash,
                    memory.rule_id,
                    memory.source_session_id,
                    [str(x) for x in memory.source_memory_ids],
                    json.dumps(memory.metadata),
                    memory.created_at,
                    memory.updated_at,
                )

                if result is None:
                    # Conflict — row already exists; return its id
                    existing = await conn.fetchval(
                        """
                        SELECT memory_id FROM memory.memories
                        WHERE user_id = $1 AND content_hash = $2 AND valid_until IS NULL
                        """,
                        memory.user_id,
                        ch,
                    )
                    return uuid.UUID(str(existing))

                # New row inserted — write history
                await conn.execute(
                    """
                    INSERT INTO memory.memory_history (
                        id, memory_id, operation, new_content,
                        new_importance, actor, actor_details, occurred_at
                    ) VALUES (
                        gen_random_uuid(), $1, 'create', $2,
                        $3, $4, $5::jsonb, now()
                    )
                    """,
                    mem_id,
                    memory.content,
                    memory.importance,
                    "pipeline_extraction",
                    "{}",
                )

        return mem_id

    async def get_by_id(self, memory_id: uuid.UUID) -> Memory | None:
        """Return the memory or ``None`` if it does not exist."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT memory_id, user_id, agent_id, org_id,
                       memory_type, content, content_hash, embedding,
                       importance, access_count, last_accessed, decay_rate,
                       valid_from, valid_until,
                       extraction_version, extraction_model, prompt_hash, rule_id,
                       source_session_id, source_memory_ids,
                       metadata, created_at, updated_at
                FROM memory.memories
                WHERE memory_id = $1
                """,
                memory_id,
            )
        if row is None:
            return None
        return _row_to_memory(row)

    async def search(
        self,
        query_embedding: list[float],
        user_id: uuid.UUID,
        limit: int = 10,
        weights: ScoringWeights | None = None,
        include_invalidated: bool = False,
    ) -> list[ScoredMemory]:
        """Return up to *limit* scored memories for *user_id*.

        Stage 1: HNSW pre-filter — fetch up to ``limit * _OVERFETCH_FACTOR``
        candidates ordered by cosine distance.
        Stage 2: Python-side re-rank — MultiSignalScorer (relevance, recency,
        importance, frequency).
        Stage 3: Slice to *limit*, bump access_count / last_accessed.

        Side-effects: increments ``access_count`` and sets ``last_accessed``
        on every memory in the returned list.
        """
        prefetch = limit * _OVERFETCH_FACTOR

        bitemporal_clause = (
            ""
            if include_invalidated
            else "AND (valid_until IS NULL OR valid_until > now())"
        )

        sql = f"""
            SELECT memory_id, user_id, agent_id, org_id,
                   memory_type, content, content_hash, embedding,
                   importance, access_count, last_accessed, decay_rate,
                   valid_from, valid_until,
                   extraction_version, extraction_model, prompt_hash, rule_id,
                   source_session_id, source_memory_ids,
                   metadata, created_at, updated_at,
                   embedding <=> $1::halfvec AS cosine_distance
            FROM memory.memories
            WHERE user_id = $2
              AND embedding IS NOT NULL
              {bitemporal_clause}
            ORDER BY embedding <=> $1::halfvec
            LIMIT $3
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, query_embedding, user_id, prefetch)

        if not rows:
            return []

        candidates: list[Memory] = [_row_to_memory(r) for r in rows]

        # Score — multi-signal: relevance, recency, importance, frequency
        now = datetime.now(timezone.utc)
        scorer = MultiSignalScorer(weights or ScoringWeights())
        scored: list[ScoredMemory] = []
        for mem in candidates:
            if mem.embedding is None:
                continue
            total, breakdown = scorer.score(mem, query_embedding, now)
            scored.append(ScoredMemory(memory=mem, score=total, score_breakdown=breakdown))

        # Deterministic sort: highest score first; ties broken by newest then
        # ascending memory_id for stability.
        scored.sort(
            key=lambda s: (
                -s.score,
                -s.memory.created_at.timestamp(),
                str(s.memory.memory_id),
            )
        )
        result = scored[:limit]

        if not result:
            return []

        # Bump access bookkeeping on returned slice
        returned_ids = [sm.memory.memory_id for sm in result]
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memory.memories
                SET access_count = access_count + 1, last_accessed = now()
                WHERE memory_id = ANY($1::uuid[])
                """,
                returned_ids,
            )

        now = datetime.now(timezone.utc)
        for sm in result:
            sm.memory.access_count += 1
            sm.memory.last_accessed = now

        return result

    async def invalidate(self, memory_id: uuid.UUID, reason: str) -> None:
        """Soft-delete *memory_id* by setting valid_until = now(UTC).

        Raises ``MemoryNotFound`` if *memory_id* does not exist or is
        already invalidated.  Records *reason* in
        ``metadata['invalidation_reason']``.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE memory.memories
                    SET valid_until = now(),
                        metadata = metadata || jsonb_build_object('invalidation_reason', $2::text)
                    WHERE memory_id = $1 AND valid_until IS NULL
                    RETURNING memory_id, importance, content
                    """,
                    memory_id,
                    reason,
                )
                if row is None:
                    raise MemoryNotFound(memory_id)

                await conn.execute(
                    """
                    INSERT INTO memory.memory_history (
                        id, memory_id, operation, old_content,
                        old_importance, actor, actor_details, occurred_at
                    ) VALUES (
                        gen_random_uuid(), $1, 'invalidate', $2,
                        $3, $4, $5::jsonb, now()
                    )
                    """,
                    memory_id,
                    row["content"],
                    float(row["importance"]),
                    "pipeline_decay",
                    "{}",
                )

    async def update(self, memory_id: uuid.UUID, **fields) -> Memory:
        """Update mutable fields on *memory_id* and return the updated memory.

        Raises ``MemoryNotFound`` if the id is unknown.
        Raises ``ValueError`` if any read-only field is present in *fields*.
        """
        bad = _READ_ONLY_FIELDS & set(fields.keys())
        if bad:
            raise ValueError(f"Cannot update read-only fields: {bad}")

        if not fields:
            mem = await self.get_by_id(memory_id)
            if mem is None:
                raise MemoryNotFound(memory_id)
            return mem

        # Fetch current row for history delta
        async with self._pool.acquire() as conn:
            current = await conn.fetchrow(
                """
                SELECT content, importance
                FROM memory.memories
                WHERE memory_id = $1
                """,
                memory_id,
            )
            if current is None:
                raise MemoryNotFound(memory_id)

        old_content: str | None = current["content"]
        old_importance: float | None = float(current["importance"])

        # Build dynamic SET clause — only allow known safe columns
        _MUTABLE_COLUMNS = {
            "content", "importance", "access_count", "last_accessed",
            "decay_rate", "valid_from", "valid_until", "extraction_model",
            "prompt_hash", "rule_id", "source_session_id", "source_memory_ids",
            "metadata", "memory_type", "agent_id", "org_id",
        }
        unknown = set(fields.keys()) - _MUTABLE_COLUMNS
        if unknown:
            raise ValueError(f"Unknown fields: {unknown}")

        set_parts: list[str] = []
        params: list[Any] = [memory_id]
        idx = 2
        for col, val in fields.items():
            if col == "metadata":
                set_parts.append(f"{col} = ${idx}::jsonb")
                params.append(json.dumps(val))
            elif col == "source_memory_ids":
                set_parts.append(f"{col} = ${idx}::uuid[]")
                params.append([str(x) for x in val])
            elif col == "memory_type":
                set_parts.append(f"{col} = ${idx}")
                params.append(val.value if hasattr(val, "value") else str(val))
            else:
                set_parts.append(f"{col} = ${idx}")
                params.append(val)
            idx += 1

        set_clause = ", ".join(set_parts)

        new_content: str | None = fields.get("content", old_content)
        new_importance: float | None = float(fields.get("importance", old_importance))

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""
                    UPDATE memory.memories
                    SET {set_clause}, updated_at = now()
                    WHERE memory_id = $1
                    RETURNING
                        memory_id, user_id, agent_id, org_id,
                        memory_type, content, content_hash, embedding,
                        importance, access_count, last_accessed, decay_rate,
                        valid_from, valid_until,
                        extraction_version, extraction_model, prompt_hash, rule_id,
                        source_session_id, source_memory_ids,
                        metadata, created_at, updated_at
                    """,
                    *params,
                )
                if row is None:
                    raise MemoryNotFound(memory_id)

                await conn.execute(
                    """
                    INSERT INTO memory.memory_history (
                        id, memory_id, operation,
                        old_content, new_content,
                        old_importance, new_importance,
                        actor, actor_details, occurred_at
                    ) VALUES (
                        gen_random_uuid(), $1, 'update',
                        $2, $3,
                        $4, $5,
                        $6, $7::jsonb, now()
                    )
                    """,
                    memory_id,
                    old_content,
                    new_content,
                    old_importance,
                    new_importance,
                    "manual",
                    "{}",
                )

        return _row_to_memory(row)

    async def get_history(self, memory_id: uuid.UUID) -> list[MemoryHistoryEntry]:
        """Return the mutation history for *memory_id*, newest first."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, memory_id, operation,
                       old_content, new_content,
                       old_importance, new_importance,
                       actor, actor_details, occurred_at
                FROM memory.memory_history
                WHERE memory_id = $1
                ORDER BY occurred_at DESC
                """,
                memory_id,
            )

        entries: list[MemoryHistoryEntry] = []
        for row in rows:
            actor_details: dict[str, Any] = {}
            raw_ad = row["actor_details"]
            if raw_ad:
                if isinstance(raw_ad, str):
                    actor_details = json.loads(raw_ad)
                else:
                    actor_details = dict(raw_ad)

            entries.append(
                MemoryHistoryEntry(
                    id=row["id"],
                    memory_id=row["memory_id"],
                    operation=row["operation"],
                    old_content=row["old_content"],
                    new_content=row["new_content"],
                    old_importance=(
                        float(row["old_importance"])
                        if row["old_importance"] is not None
                        else None
                    ),
                    new_importance=(
                        float(row["new_importance"])
                        if row["new_importance"] is not None
                        else None
                    ),
                    actor=row["actor"],
                    actor_details=actor_details,
                    occurred_at=row["occurred_at"],
                )
            )
        return entries
