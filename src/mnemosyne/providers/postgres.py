from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from mnemosyne.db.models.episode import Episode
from mnemosyne.db.models.history import MemoryHistoryEntry
from mnemosyne.db.models.memory import Memory, MemoryType, ScoredMemory
from mnemosyne.errors import MemoryNotFound
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.fusion import fuse_rrf
from mnemosyne.retrieval.scoring import MultiSignalScorer, ScoringWeights
from mnemosyne.utils import content_hash

# Fields that must never be overwritten via update()
_READ_ONLY_FIELDS = frozenset({"memory_id", "content_hash", "extraction_version"})

# Multiplier for HNSW over-fetch before Python-side re-rank
_OVERFETCH_FACTOR = 5

# Bounded retries for the add() insert/re-select concurrent-invalidate race
_ADD_MAX_ATTEMPTS = 3


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


def _row_to_episode(row: asyncpg.Record) -> Episode:
    """Convert an asyncpg row from memory.episodes into an Episode model."""
    summary_embedding: list[float] | None = None
    raw = row["summary_embedding"]
    if raw is not None:
        summary_embedding = raw.to_list()

    memory_ids: list[uuid.UUID] = []
    raw_mids = row["memory_ids"]
    if raw_mids:
        memory_ids = [uuid.UUID(str(x)) for x in raw_mids]

    metadata: dict[str, Any] = {}
    raw_meta = row["metadata"]
    if raw_meta:
        metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)

    return Episode(
        episode_id=row["episode_id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        session_id=row["session_id"],
        summary=row["summary"],
        summary_embedding=summary_embedding,
        key_topics=list(row["key_topics"] or []),
        memory_ids=memory_ids,
        outcome=row.get("outcome"),
        started_at=row.get("started_at"),
        ended_at=row.get("ended_at"),
        metadata=metadata,
        created_at=row["created_at"],
    )


class PostgresMemoryProvider(MemoryProvider):
    """PostgreSQL-backed memory provider using asyncpg and pgvector.

    Use the factory method ``PostgresMemoryProvider.connect(dsn)`` to
    create an instance.  The constructor accepts an existing asyncpg pool
    for testability.

    Connection pool notes
    ---------------------
    - Pool sizing and per-statement timeout are configurable via ``connect``
      (``min_size``, ``max_size``, ``command_timeout``); defaults are
      ``min_size=2, max_size=10`` and no command timeout.
    - Each new connection registers the pgvector codec via
      ``pgvector.asyncpg.register_vector``.
    - Call ``await provider.close()`` to drain the pool on shutdown.

    History writes
    --------------
    Every ``add``, ``invalidate``, and ``update`` call writes an immutable
    ``MemoryHistoryEntry`` row to ``memory.memory_history`` inside the same
    transaction as the primary mutation.
    """

    cascades_entities = True

    def __init__(self, pool: asyncpg.Pool, *, fts_language: str = "english") -> None:
        self._pool = pool
        if not re.fullmatch(r"[a-z_]+", fts_language):
            raise ValueError(f"Invalid text-search config name: {fts_language!r}")
        # Query-side regconfig for websearch_to_tsquery. Must match the config
        # baked into the generated content_tsv column (see migration 001);
        # changing it for multilingual content requires a column migration too.
        self._fts_language = fts_language

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 2,
        max_size: int = 10,
        command_timeout: float | None = None,
        fts_language: str = "english",
    ) -> "PostgresMemoryProvider":
        """Create a pool, register pgvector codec, return a provider.

        ``min_size``/``max_size`` size the pool; ``command_timeout`` (seconds)
        caps how long any single statement may run before asyncpg cancels it.
        ``fts_language`` selects the query-side text-search config and must
        match the config of the stored ``content_tsv`` column. All are
        optional and default to the historical behaviour (2/10, no timeout,
        english).
        """
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
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            init=_init_conn,
        )
        return cls(pool, fts_language=fts_language)

    async def close(self) -> None:
        """Drain and close the underlying connection pool."""
        await self._pool.close()

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------

    async def add(
        self, memory: Memory, actor: str = "pipeline_extraction"
    ) -> uuid.UUID:
        """Persist *memory* and return its UUID.

        Raises ``ValueError`` if ``memory.embedding`` is ``None``.
        Returns the existing ``memory_id`` without writing a duplicate if a
        non-invalidated memory with the same ``(user_id, content_hash)``
        already exists (exact-hash dedup). ``actor`` is stamped on the
        ``create`` history row.
        """
        if memory.embedding is None:
            raise ValueError("caller must set embedding before add")

        mem_id = memory.memory_id
        ch = content_hash(memory.content)
        memory_type = (
            memory.memory_type.value
            if hasattr(memory.memory_type, "value")
            else str(memory.memory_type)
        )

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Atomic dedup: attempt insert, let the DB handle the conflict.
                # A concurrent invalidate can retire the conflicting row between
                # our DO NOTHING and the re-SELECT, so retry a bounded number of
                # times rather than dereferencing a missing id.
                for _ in range(_ADD_MAX_ATTEMPTS):
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
                        memory_type,
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

                    if result is not None:
                        break

                    # Conflict — an active duplicate exists; return its id.
                    existing = await conn.fetchval(
                        """
                        SELECT memory_id FROM memory.memories
                        WHERE user_id = $1 AND content_hash = $2 AND valid_until IS NULL
                        """,
                        memory.user_id,
                        ch,
                    )
                    if existing is not None:
                        return uuid.UUID(str(existing))
                    # The duplicate was invalidated concurrently — loop and retry
                    # the insert, which can now succeed.
                else:
                    raise RuntimeError(
                        "add() could not resolve a concurrent invalidate race "
                        f"for content_hash {ch} after {_ADD_MAX_ATTEMPTS} attempts"
                    )

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
                    actor,
                    "{}",
                )

        return mem_id

    async def get_by_id(
        self, memory_id: uuid.UUID, user_id: uuid.UUID | None = None
    ) -> Memory | None:
        """Return the memory or ``None`` if it does not exist.

        When *user_id* is supplied the lookup is scoped to that owner; a memory
        owned by another user reads back as ``None``.
        """
        scope_clause = "" if user_id is None else "AND user_id = $2"
        params: list[Any] = [memory_id]
        if user_id is not None:
            params.append(user_id)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT memory_id, user_id, agent_id, org_id,
                       memory_type, content, content_hash, embedding,
                       importance, access_count, last_accessed, decay_rate,
                       valid_from, valid_until,
                       extraction_version, extraction_model, prompt_hash, rule_id,
                       source_session_id, source_memory_ids,
                       metadata, created_at, updated_at
                FROM memory.memories
                WHERE memory_id = $1
                  {scope_clause}
                """,
                *params,
            )
        if row is None:
            return None
        return _row_to_memory(row)

    async def get_by_ids(self, ids: list[uuid.UUID]) -> list[Memory]:
        """Batch-fetch active memories for *ids* in a single query.

        Filters invalidated rows in SQL (``valid_until IS NULL OR
        valid_until > now()``); unknown ids are dropped. Returns an empty list
        for an empty input without issuing a query.
        """
        if not ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT memory_id, user_id, agent_id, org_id,
                       memory_type, content, content_hash, embedding,
                       importance, access_count, last_accessed, decay_rate,
                       valid_from, valid_until,
                       extraction_version, extraction_model, prompt_hash, rule_id,
                       source_session_id, source_memory_ids,
                       metadata, created_at, updated_at
                FROM memory.memories
                WHERE memory_id = ANY($1::uuid[])
                  AND (valid_until IS NULL OR valid_until > now())
                ORDER BY created_at DESC, memory_id
                """,
                list(ids),
            )
        return [_row_to_memory(r) for r in rows]

    async def bump_access(self, ids: list[uuid.UUID]) -> None:
        """Increment access_count / advance last_accessed for *ids* in one write.

        Writes no ``memory_history`` row. Unknown ids are no-ops. This is the
        single implementation of access bookkeeping — ``search`` delegates here.
        """
        if not ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memory.memories
                SET access_count = access_count + 1,
                    last_accessed = GREATEST(now(), last_accessed)
                WHERE memory_id = ANY($1::uuid[])
                """,
                list(ids),
            )

    async def search(
        self,
        query_embedding: list[float],
        user_id: uuid.UUID,
        limit: int = 10,
        weights: ScoringWeights | None = None,
        include_invalidated: bool = False,
        explain: bool = False,
        source_message_id: uuid.UUID | None = None,
        query_text: str | None = None,
        *,
        bump_access: bool = True,
    ) -> list[ScoredMemory]:
        """Return up to *limit* scored memories for *user_id*.

        Stage 1: HNSW pre-filter — fetch up to ``limit * _OVERFETCH_FACTOR``
        candidates ordered by cosine distance, with ``hnsw.ef_search`` tuned to
        the prefetch size for recall. When ``query_text`` is supplied a second
        full-text candidate set (``websearch_to_tsquery``) is fused with the
        vector set via Reciprocal Rank Fusion.
        Stage 2: Python-side re-rank — MultiSignalScorer (relevance, recency,
        importance, frequency).
        Stage 3: Slice to *limit*; bump access_count / last_accessed when
        ``bump_access`` is True.

        When ``source_message_id`` is set, the candidate set is restricted
        to memories whose ``metadata['source_message_ids']`` array contains
        that UUID.
        """
        prefetch = limit * _OVERFETCH_FACTOR

        bitemporal_clause = (
            ""
            if include_invalidated
            else "AND (valid_until IS NULL OR valid_until > now())"
        )

        provenance_clause = ""
        params: list[Any] = [query_embedding, user_id, prefetch]
        if source_message_id is not None:
            provenance_clause = "AND metadata -> 'source_message_ids' ? $4"
            params.append(str(source_message_id))

        vector_sql = f"""
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
              {provenance_clause}
            ORDER BY embedding <=> $1::halfvec
            LIMIT $3
        """

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {int(prefetch)}")
                try:
                    await conn.execute(
                        "SET LOCAL hnsw.iterative_scan = 'relaxed_order'"
                    )
                except asyncpg.PostgresError:
                    # Older pgvector builds lack iterative_scan — tolerate it.
                    pass
                vector_rows = await conn.fetch(vector_sql, *params)

                fts_rows: list[asyncpg.Record] = []
                if query_text:
                    fts_params: list[Any] = [user_id, query_text, prefetch]
                    fts_provenance = ""
                    if source_message_id is not None:
                        fts_provenance = "AND metadata -> 'source_message_ids' ? $4"
                        fts_params.append(str(source_message_id))
                    fts_sql = f"""
                        SELECT memory_id, user_id, agent_id, org_id,
                               memory_type, content, content_hash, embedding,
                               importance, access_count, last_accessed, decay_rate,
                               valid_from, valid_until,
                               extraction_version, extraction_model, prompt_hash,
                               rule_id, source_session_id, source_memory_ids,
                               metadata, created_at, updated_at,
                               ts_rank(content_tsv, q) AS fts_rank
                        FROM memory.memories,
                             websearch_to_tsquery('{self._fts_language}', $2) AS q
                        WHERE user_id = $1
                          AND embedding IS NOT NULL
                          AND content_tsv @@ q
                          {bitemporal_clause}
                          {fts_provenance}
                        ORDER BY fts_rank DESC
                        LIMIT $3
                    """
                    fts_rows = await conn.fetch(fts_sql, *fts_params)

        if not vector_rows and not fts_rows:
            return []

        now = datetime.now(timezone.utc)
        scorer = MultiSignalScorer(weights or ScoringWeights())
        query_norm = math.sqrt(sum(x * x for x in query_embedding))

        def _score(mem: Memory, cosine_distance: float | None = None) -> ScoredMemory:
            # Reuse the DB-computed cosine distance for the vector leg; FTS-only
            # candidates fall back to a Python cosine against the same query.
            relevance = (
                None if cosine_distance is None else 1.0 - float(cosine_distance)
            )
            total, second = scorer.score(
                mem,
                query_embedding,
                now,
                explain=explain,
                relevance=relevance,
                query_norm=query_norm,
            )
            if explain:
                return ScoredMemory(
                    memory=mem, score=total, score_breakdown_explain=second
                )
            return ScoredMemory(memory=mem, score=total, score_breakdown=second)

        vector_scored: list[ScoredMemory] = [
            _score(_row_to_memory(r), r["cosine_distance"]) for r in vector_rows
        ]
        vector_scored.sort(
            key=lambda s: (
                -s.score,
                -s.memory.created_at.timestamp(),
                str(s.memory.memory_id),
            )
        )

        # Fusion runs in explain mode too, so explain reflects the real ranking.
        if fts_rows:
            by_id = {s.memory.memory_id: s for s in vector_scored}
            fts_scored: list[ScoredMemory] = []
            for r in fts_rows:
                mem = _row_to_memory(r)
                existing = by_id.get(mem.memory_id)
                # Reuse the already-scored object to keep one identity per id.
                fts_scored.append(existing if existing is not None else _score(mem))
            fused = fuse_rrf(vector_scored, fts_scored, max(limit, len(vector_scored)))
            result = fused[:limit]
        else:
            result = vector_scored[:limit]

        if not result:
            return []

        if bump_access:
            await self.bump_access([sm.memory.memory_id for sm in result])
            bumped_at = datetime.now(timezone.utc)
            for sm in result:
                sm.memory.access_count += 1
                sm.memory.last_accessed = max(bumped_at, sm.memory.last_accessed)

        return result

    async def invalidate(
        self,
        memory_id: uuid.UUID,
        reason: str,
        user_id: uuid.UUID | None = None,
        actor: str = "pipeline_decay",
    ) -> Memory:
        """Soft-delete *memory_id* by setting valid_until = now(UTC).

        Returns the invalidated :class:`Memory`. When *user_id* is supplied the
        operation is scoped: a memory owned by another user raises
        ``MemoryNotFound``.

        Raises ``MemoryNotFound`` if *memory_id* does not exist or is already
        invalidated. Records *reason* in ``metadata['invalidation_reason']`` and
        stamps ``actor`` on the ``invalidate`` history row.
        """
        scope_clause = "" if user_id is None else "AND user_id = $3"
        params: list[Any] = [memory_id, reason]
        if user_id is not None:
            params.append(user_id)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""
                    UPDATE memory.memories
                    SET valid_until = now(),
                        metadata = metadata || jsonb_build_object('invalidation_reason', $2::text)
                    WHERE memory_id = $1 AND valid_until IS NULL
                      {scope_clause}
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
                    actor,
                    "{}",
                )

        return _row_to_memory(row)

    async def update(
        self, memory_id: uuid.UUID, *, actor: str = "manual", **fields
    ) -> Memory:
        """Update mutable fields on *memory_id* and return the updated memory.

        When ``content`` changes the ``content_hash`` is recomputed in the same
        statement so the ``(user_id, content_hash)`` dedup index stays true.
        ``actor`` is stamped on the ``update`` history row.

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
            "metadata", "memory_type", "agent_id", "org_id", "embedding",
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
            elif col == "embedding":
                set_parts.append(f"{col} = ${idx}::halfvec")
                params.append(val)
            else:
                set_parts.append(f"{col} = ${idx}")
                params.append(val)
            idx += 1

        # Recompute content_hash whenever content is mutated so the dedup unique
        # index does not go stale against the new content.
        if "content" in fields:
            set_parts.append(f"content_hash = ${idx}")
            params.append(content_hash(fields["content"]))
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
                    actor,
                    "{}",
                )

        return _row_to_memory(row)

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        include_invalidated: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Return memories for *user_id*, newest first.

        ``limit=None`` returns every row; supply ``limit``/``offset`` to page.
        """
        bitemporal = "" if include_invalidated else "AND valid_until IS NULL"
        params: list[Any] = [user_id]
        page_clause = ""
        if limit is not None:
            params.append(limit)
            params.append(offset)
            page_clause = f"LIMIT ${len(params) - 1} OFFSET ${len(params)}"
        sql = f"""
            SELECT memory_id, user_id, agent_id, org_id,
                   memory_type, content, content_hash, embedding,
                   importance, access_count, last_accessed, decay_rate,
                   valid_from, valid_until,
                   extraction_version, extraction_model, prompt_hash, rule_id,
                   source_session_id, source_memory_ids,
                   metadata, created_at, updated_at
            FROM memory.memories
            WHERE user_id = $1
              {bitemporal}
            ORDER BY created_at DESC, memory_id
            {page_clause}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_memory(r) for r in rows]

    async def add_episode(self, episode: Episode) -> Episode:
        """Persist *episode*, upserting on ``session_id``."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memory.episodes (
                    episode_id, user_id, agent_id, session_id,
                    summary, summary_embedding, key_topics, memory_ids,
                    outcome, started_at, ended_at, metadata, created_at
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6::halfvec, $7::text[], $8::uuid[],
                    $9, $10, $11, $12::jsonb, $13
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    agent_id = EXCLUDED.agent_id,
                    summary = EXCLUDED.summary,
                    summary_embedding = EXCLUDED.summary_embedding,
                    key_topics = EXCLUDED.key_topics,
                    memory_ids = EXCLUDED.memory_ids,
                    outcome = EXCLUDED.outcome,
                    started_at = EXCLUDED.started_at,
                    ended_at = EXCLUDED.ended_at,
                    metadata = EXCLUDED.metadata
                RETURNING
                    episode_id, user_id, agent_id, session_id,
                    summary, summary_embedding, key_topics, memory_ids,
                    outcome, started_at, ended_at, metadata, created_at
                """,
                episode.episode_id,
                episode.user_id,
                episode.agent_id,
                episode.session_id,
                episode.summary,
                episode.summary_embedding,
                list(episode.key_topics),
                [str(x) for x in episode.memory_ids],
                episode.outcome,
                episode.started_at,
                episode.ended_at,
                json.dumps(episode.metadata),
                episode.created_at,
            )
        return _row_to_episode(row)

    async def list_episodes(
        self, user_id: uuid.UUID, limit: int = 20, offset: int = 0
    ) -> list[Episode]:
        """Return episodes for *user_id* ordered by created_at DESC."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT episode_id, user_id, agent_id, session_id,
                       summary, summary_embedding, key_topics, memory_ids,
                       outcome, started_at, ended_at, metadata, created_at
                FROM memory.episodes
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                user_id,
                limit,
                offset,
            )
        return [_row_to_episode(r) for r in rows]

    async def physical_delete_user(
        self, user_id: uuid.UUID, requestor: str, dry_run: bool = False
    ) -> int:
        """Physically delete every row owned by *user_id*.

        Purges memories, memory_history, entities, entity_mentions, episodes,
        session_buffer, contradiction_audit, reextraction_jobs, user_settings
        and the user's processing_log rows (matched by user_id or by the user's
        session ids), all in one transaction. Cross-user reflections derived
        from the erased memories are soft-invalidated and have their personal
        content/embedding scrubbed in place. Writes an audit row into
        ``memory.gdpr_deletions`` BEFORE any destructive delete. On ``dry_run``
        only the audit row is written (row counts are captured) and no data is
        removed. Returns the number of memory rows.
        """
        if not requestor:
            raise ValueError("requestor is required")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Only count what the gdpr_deletions audit row can store.
                counts = await conn.fetchrow(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM memory.memories WHERE user_id = $1) AS rows_memories,
                        (SELECT COUNT(*) FROM memory.entities WHERE user_id = $1) AS rows_entities,
                        (SELECT COUNT(*) FROM memory.entity_mentions em
                            JOIN memory.entities e ON em.entity_id = e.entity_id
                            WHERE e.user_id = $1) AS rows_mentions,
                        (SELECT COUNT(*) FROM memory.episodes WHERE user_id = $1) AS rows_episodes,
                        (SELECT COUNT(*) FROM memory.memory_history mh
                            JOIN memory.memories m ON mh.memory_id = m.memory_id
                            WHERE m.user_id = $1) AS rows_history
                    """,
                    user_id,
                )

                await conn.execute(
                    """
                    INSERT INTO memory.gdpr_deletions
                        (id, user_id, requestor, reason,
                         rows_memories, rows_entities, rows_mentions,
                         rows_episodes, rows_history,
                         occurred_at, dry_run)
                    VALUES ($1, $2, $3, $4,
                            $5, $6, $7,
                            $8, $9,
                            $10, $11)
                    """,
                    uuid.uuid4(),
                    user_id,
                    requestor,
                    "user_request",
                    int(counts["rows_memories"]),
                    int(counts["rows_entities"]),
                    int(counts["rows_mentions"]),
                    int(counts["rows_episodes"]),
                    int(counts["rows_history"]),
                    datetime.now(timezone.utc),
                    dry_run,
                )

                if dry_run:
                    return int(counts["rows_memories"])

                # Cross-user reflections derived from this user's memories are
                # retained for the audit trail but scrubbed of personal data:
                # soft-invalidate and clear content/hash/embedding. content is
                # NOT NULL, so it is blanked rather than nulled.
                await conn.execute(
                    """
                    UPDATE memory.memories
                    SET valid_until = now(),
                        content = '',
                        content_hash = NULL,
                        embedding = NULL,
                        metadata = metadata
                            || jsonb_build_object('invalidation_reason', 'gdpr_source_deleted')
                    WHERE user_id <> $1
                      AND source_memory_ids && (
                          SELECT COALESCE(array_agg(memory_id), ARRAY[]::uuid[])
                          FROM memory.memories WHERE user_id = $1
                      )
                      AND valid_until IS NULL
                    """,
                    user_id,
                )

                await conn.execute(
                    """
                    DELETE FROM memory.memory_history
                    WHERE memory_id IN (
                        SELECT memory_id FROM memory.memories WHERE user_id = $1
                    )
                    """,
                    user_id,
                )
                await conn.execute(
                    """
                    DELETE FROM memory.entity_mentions
                    WHERE entity_id IN (
                        SELECT entity_id FROM memory.entities WHERE user_id = $1
                    )
                    """,
                    user_id,
                )
                await conn.execute(
                    """
                    DELETE FROM memory.entity_mentions
                    WHERE memory_id IN (
                        SELECT memory_id FROM memory.memories WHERE user_id = $1
                    )
                    """,
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM memory.entities WHERE user_id = $1",
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM memory.episodes WHERE user_id = $1",
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM memory.contradiction_audit WHERE user_id = $1",
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM memory.reextraction_jobs WHERE user_id = $1",
                    user_id,
                )
                # processing_log references the user's sessions; delete it
                # before session_buffer so the session lookup still resolves.
                await conn.execute(
                    """
                    DELETE FROM memory.processing_log
                    WHERE user_id = $1
                       OR session_id IN (
                           SELECT session_id FROM memory.session_buffer
                           WHERE user_id = $1
                       )
                    """,
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM memory.session_buffer WHERE user_id = $1",
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM memory.user_settings WHERE user_id = $1",
                    user_id,
                )
                await conn.execute(
                    "DELETE FROM memory.memories WHERE user_id = $1",
                    user_id,
                )

        return int(counts["rows_memories"])

    async def select_by_extraction_version_below(
        self, user_id: uuid.UUID, target_version: str
    ) -> list[Memory]:
        """Return active memories for *user_id* whose extraction_version is
        strictly less than *target_version* (semver tuple order)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT memory_id, user_id, agent_id, org_id,
                       memory_type, content, content_hash, embedding,
                       importance, access_count, last_accessed, decay_rate,
                       valid_from, valid_until,
                       extraction_version, extraction_model, prompt_hash, rule_id,
                       source_session_id, source_memory_ids,
                       metadata, created_at, updated_at
                FROM memory.memories
                WHERE user_id = $1
                  AND extraction_version IS NOT NULL
                  AND extraction_version ~ '^[0-9]+\\.[0-9]+\\.[0-9]+$'
                  AND valid_until IS NULL
                  AND string_to_array(extraction_version, '.')::int[]
                      < string_to_array($2, '.')::int[]
                ORDER BY created_at ASC
                """,
                user_id,
                target_version,
            )
        return [_row_to_memory(r) for r in rows]

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
