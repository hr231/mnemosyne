from __future__ import annotations

import json
import uuid
from typing import Protocol

import asyncpg

from mnemosyne.db.models.entity import Entity, EntityMention


class EntityStore(Protocol):
    """Protocol for entity storage backends."""

    async def upsert_entity(self, entity: Entity) -> uuid.UUID: ...
    async def find_by_name(self, user_id: uuid.UUID, name: str, entity_type: str) -> Entity | None: ...
    async def find_by_embedding(
        self, user_id: uuid.UUID, embedding: list[float], threshold: float, limit: int,
    ) -> list[Entity]: ...
    async def add_mention(self, mention: EntityMention) -> None: ...
    async def find_mentions_for_entity(self, entity_id: uuid.UUID) -> list[uuid.UUID]: ...
    async def find_entities_for_memory(self, memory_id: uuid.UUID) -> list[Entity]: ...


class PostgresEntityStore:
    """PostgreSQL-backed entity store using asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_entity(self, entity: Entity) -> uuid.UUID:
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                INSERT INTO memory.entities (
                    entity_id, user_id, agent_id, entity_name, entity_type,
                    normalized_name, embedding, facts, confidence,
                    mention_count, source_memory_ids, metadata,
                    created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7::halfvec, $8::jsonb, $9,
                    $10, $11::uuid[], $12::jsonb,
                    $13, $14
                )
                ON CONFLICT (user_id, normalized_name, entity_type) DO UPDATE SET
                    confidence = GREATEST(memory.entities.confidence, EXCLUDED.confidence),
                    mention_count = memory.entities.mention_count + 1,
                    facts = memory.entities.facts || EXCLUDED.facts,
                    updated_at = now()
                RETURNING entity_id
                """,
                entity.entity_id,
                entity.user_id,
                entity.agent_id,
                entity.entity_name,
                entity.entity_type,
                entity.normalized_name,
                entity.embedding,
                json.dumps(entity.facts),
                entity.confidence,
                entity.mention_count,
                [str(x) for x in entity.source_memory_ids],
                json.dumps(entity.metadata),
                entity.created_at,
                entity.updated_at,
            )
            return uuid.UUID(str(result))

    async def find_by_name(
        self, user_id: uuid.UUID, name: str, entity_type: str,
    ) -> Entity | None:
        normalized = name.strip().lower()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM memory.entities
                WHERE user_id = $1 AND normalized_name = $2 AND entity_type = $3
                """,
                user_id, normalized, entity_type,
            )
        if row is None:
            return None
        return _row_to_entity(row)

    async def find_by_embedding(
        self,
        user_id: uuid.UUID,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 10,
    ) -> list[Entity]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *, 1 - (embedding <=> $1::halfvec) AS similarity
                FROM memory.entities
                WHERE user_id = $2
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> $1::halfvec) > $3
                ORDER BY embedding <=> $1::halfvec
                LIMIT $4
                """,
                embedding, user_id, threshold, limit,
            )
        return [_row_to_entity(r) for r in rows]

    async def add_mention(self, mention: EntityMention) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory.entity_mentions
                    (id, entity_id, memory_id, mention_text, context, occurred_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                mention.id,
                mention.entity_id,
                mention.memory_id,
                mention.mention_text,
                mention.context,
                mention.occurred_at,
            )

    async def find_mentions_for_entity(self, entity_id: uuid.UUID) -> list[uuid.UUID]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT memory_id FROM memory.entity_mentions
                WHERE entity_id = $1
                ORDER BY occurred_at DESC
                """,
                entity_id,
            )
        return [row["memory_id"] for row in rows]

    async def find_entities_for_memory(self, memory_id: uuid.UUID) -> list[Entity]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.* FROM memory.entities e
                JOIN memory.entity_mentions em ON e.entity_id = em.entity_id
                WHERE em.memory_id = $1
                """,
                memory_id,
            )
        return [_row_to_entity(r) for r in rows]


def _row_to_entity(row: asyncpg.Record) -> Entity:
    """Convert a Postgres row to an Entity model."""
    embedding = None
    raw_emb = row.get("embedding")
    if raw_emb is not None:
        embedding = raw_emb.to_list()

    source_ids: list[uuid.UUID] = []
    raw_smi = row.get("source_memory_ids")
    if raw_smi:
        source_ids = [uuid.UUID(str(x)) for x in raw_smi]

    facts: dict = {}
    raw_facts = row.get("facts")
    if raw_facts:
        facts = dict(raw_facts) if not isinstance(raw_facts, str) else json.loads(raw_facts)

    metadata: dict = {}
    raw_meta = row.get("metadata")
    if raw_meta:
        metadata = dict(raw_meta) if not isinstance(raw_meta, str) else json.loads(raw_meta)

    return Entity(
        entity_id=row["entity_id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        entity_name=row["entity_name"],
        entity_type=row["entity_type"],
        normalized_name=row["normalized_name"],
        embedding=embedding,
        facts=facts,
        confidence=float(row["confidence"]),
        mention_count=int(row["mention_count"]),
        source_memory_ids=source_ids,
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
