from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.entity import Entity, EntityMention


class InMemoryEntityStore:
    """In-memory entity store for development and testing."""

    def __init__(self) -> None:
        self._entities: dict[uuid.UUID, Entity] = {}
        self._mentions: list[EntityMention] = []

    async def upsert_entity(self, entity: Entity) -> uuid.UUID:
        """Insert or update an entity.

        Conflict key is (user_id, normalized_name, entity_type).
        On conflict: take max confidence, increment mention_count, merge facts.
        """
        for existing in self._entities.values():
            if (
                existing.user_id == entity.user_id
                and existing.normalized_name == entity.normalized_name
                and existing.entity_type == entity.entity_type
            ):
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.mention_count += 1
                existing.facts.update(entity.facts)
                existing.updated_at = datetime.now(timezone.utc)
                return existing.entity_id

        self._entities[entity.entity_id] = entity
        return entity.entity_id

    async def find_by_name(
        self, user_id: uuid.UUID, name: str, entity_type: str,
    ) -> Entity | None:
        normalized = name.strip().lower()
        for e in self._entities.values():
            if (
                e.user_id == user_id
                and e.normalized_name == normalized
                and e.entity_type == entity_type
            ):
                return e
        return None

    async def find_by_embedding(
        self,
        user_id: uuid.UUID,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 10,
    ) -> list[Entity]:
        results: list[tuple[float, Entity]] = []
        for e in self._entities.values():
            if e.user_id != user_id or e.embedding is None:
                continue
            dot = sum(a * b for a, b in zip(embedding, e.embedding))
            norm_a = math.sqrt(sum(x * x for x in embedding))
            norm_b = math.sqrt(sum(x * x for x in e.embedding))
            if norm_a == 0 or norm_b == 0:
                continue
            sim = dot / (norm_a * norm_b)
            if sim >= threshold:
                results.append((sim, e))
        results.sort(key=lambda x: -x[0])
        return [e for _, e in results[:limit]]

    async def add_mention(self, mention: EntityMention) -> None:
        self._mentions.append(mention)

    async def find_mentions_for_entity(self, entity_id: uuid.UUID) -> list[uuid.UUID]:
        return [
            m.memory_id
            for m in sorted(
                (m for m in self._mentions if m.entity_id == entity_id),
                key=lambda m: m.occurred_at,
                reverse=True,
            )
        ]

    async def find_entities_for_memory(self, memory_id: uuid.UUID) -> list[Entity]:
        entity_ids = {m.entity_id for m in self._mentions if m.memory_id == memory_id}
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
