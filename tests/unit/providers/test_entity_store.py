"""Parameterized tests for EntityStore — runs against InMemoryEntityStore and PostgresEntityStore."""
from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.entity import Entity, EntityMention


def _make_entity(
    user_id: uuid.UUID,
    entity_name: str,
    entity_type: str = "organization",
    **kwargs,
) -> Entity:
    return Entity(
        user_id=user_id,
        entity_name=entity_name,
        entity_type=entity_type,
        **kwargs,
    )


class TestUpsertEntity:
    async def test_upsert_creates_new_entity(self, entity_store):
        """Upserting a new entity makes it findable by name."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "Anthropic")

        await entity_store.upsert_entity(entity)

        found = await entity_store.find_by_name(user_id, "Anthropic", "organization")
        assert found is not None
        assert found.entity_name == "Anthropic"
        assert found.normalized_name == "anthropic"

    async def test_upsert_updates_existing(self, entity_store):
        """Upserting the same (user_id, name, type) twice increments mention_count and merges facts."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "Nike", entity_type="brand", facts={"country": "USA"})
        await entity_store.upsert_entity(entity)

        entity2 = _make_entity(user_id, "Nike", entity_type="brand", facts={"sport": "running"})
        await entity_store.upsert_entity(entity2)

        found = await entity_store.find_by_name(user_id, "Nike", "brand")
        assert found is not None
        assert found.mention_count == 1  # started at 0, incremented once on second upsert
        assert "country" in found.facts
        assert "sport" in found.facts

    async def test_upsert_returns_entity_id(self, entity_store):
        """upsert_entity returns the entity_id (UUID) of the persisted entity."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "OpenAI")

        returned_id = await entity_store.upsert_entity(entity)
        assert isinstance(returned_id, uuid.UUID)

    async def test_upsert_conflict_returns_existing_id(self, entity_store):
        """Second upsert of same key returns the original entity_id, not a new one."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "Google", entity_type="organization")
        first_id = await entity_store.upsert_entity(entity)

        entity2 = _make_entity(user_id, "Google", entity_type="organization")
        second_id = await entity_store.upsert_entity(entity2)

        assert first_id == second_id


class TestFindByName:
    async def test_find_by_name_not_found(self, entity_store):
        """find_by_name returns None for an entity that does not exist."""
        user_id = uuid.uuid4()
        result = await entity_store.find_by_name(user_id, "NonExistent", "brand")
        assert result is None

    async def test_find_by_name_case_insensitive(self, entity_store):
        """Upsert 'Nike', then find_by_name('NIKE') returns the same entity."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "Nike", entity_type="brand")
        await entity_store.upsert_entity(entity)

        found = await entity_store.find_by_name(user_id, "NIKE", "brand")
        assert found is not None
        assert found.normalized_name == "nike"

    async def test_find_by_name_with_whitespace(self, entity_store):
        """find_by_name strips and lowercases the lookup name."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "  Nike Inc. ", entity_type="brand")
        await entity_store.upsert_entity(entity)

        found = await entity_store.find_by_name(user_id, "  NIKE INC.  ", "brand")
        assert found is not None

    async def test_find_by_name_different_type_returns_none(self, entity_store):
        """The same name under a different entity_type is a separate entity."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "Amazon", entity_type="organization")
        await entity_store.upsert_entity(entity)

        result = await entity_store.find_by_name(user_id, "Amazon", "brand")
        assert result is None


class TestMentions:
    async def test_add_mention_and_find(self, entity_store, stub_memory_id):
        """After adding a mention, find_mentions_for_entity returns the memory_id."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "Apple", entity_type="brand")
        entity_id = await entity_store.upsert_entity(entity)

        mention = EntityMention(
            entity_id=entity_id,
            memory_id=stub_memory_id,
            mention_text="Apple",
            context="User likes Apple products",
        )
        await entity_store.add_mention(mention)

        memory_ids = await entity_store.find_mentions_for_entity(entity_id)
        assert stub_memory_id in memory_ids

    async def test_add_multiple_mentions_ordered_by_recency(self, entity_store, stub_memory_id):
        """find_mentions_for_entity returns memory_ids ordered newest-first."""
        from datetime import timedelta

        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "Tesla", entity_type="organization")
        entity_id = await entity_store.upsert_entity(entity)

        # stub_memory_id is the "new" one; generate a second stub for "old"
        from mnemosyne.providers.in_memory_entity_store import InMemoryEntityStore
        if isinstance(entity_store, InMemoryEntityStore):
            mem_old = uuid.uuid4()
        else:
            from tests.conftest import _insert_stub_memory
            mem_old = uuid.uuid4()
            await _insert_stub_memory(entity_store._pool, mem_old)

        mem_new = stub_memory_id

        from datetime import datetime, timezone
        base = datetime.now(timezone.utc)

        await entity_store.add_mention(EntityMention(
            entity_id=entity_id,
            memory_id=mem_old,
            occurred_at=base - timedelta(hours=2),
        ))
        await entity_store.add_mention(EntityMention(
            entity_id=entity_id,
            memory_id=mem_new,
            occurred_at=base,
        ))

        memory_ids = await entity_store.find_mentions_for_entity(entity_id)
        assert memory_ids[0] == mem_new
        assert memory_ids[1] == mem_old

    async def test_find_entities_for_memory(self, entity_store, stub_memory_id):
        """After adding a mention, find_entities_for_memory returns the linked entity."""
        user_id = uuid.uuid4()
        entity = _make_entity(user_id, "SpaceX", entity_type="organization")
        entity_id = await entity_store.upsert_entity(entity)

        mention = EntityMention(entity_id=entity_id, memory_id=stub_memory_id)
        await entity_store.add_mention(mention)

        entities = await entity_store.find_entities_for_memory(stub_memory_id)
        assert len(entities) == 1
        assert entities[0].normalized_name == "spacex"

    async def test_find_entities_for_memory_no_mentions(self, entity_store):
        """find_entities_for_memory returns empty list for a memory with no mentions."""
        memory_id = uuid.uuid4()
        entities = await entity_store.find_entities_for_memory(memory_id)
        assert entities == []

    async def test_find_mentions_for_entity_no_mentions(self, entity_store):
        """find_mentions_for_entity returns empty list when no mentions recorded."""
        entity_id = uuid.uuid4()
        memory_ids = await entity_store.find_mentions_for_entity(entity_id)
        assert memory_ids == []


class TestUserIsolation:
    async def test_user_isolation(self, entity_store):
        """An entity belonging to user_a is not returned when querying as user_b."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        entity = _make_entity(user_a, "Nike", entity_type="brand")
        await entity_store.upsert_entity(entity)

        result = await entity_store.find_by_name(user_b, "Nike", "brand")
        assert result is None

    async def test_same_name_different_users_are_distinct(self, entity_store):
        """The same (name, type) for two different users creates two independent entities."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        entity_a = _make_entity(user_a, "Meta", entity_type="organization", facts={"ceo": "Zuckerberg"})
        entity_b = _make_entity(user_b, "Meta", entity_type="organization", facts={"ceo": "different"})

        id_a = await entity_store.upsert_entity(entity_a)
        id_b = await entity_store.upsert_entity(entity_b)

        assert id_a != id_b

        found_a = await entity_store.find_by_name(user_a, "Meta", "organization")
        found_b = await entity_store.find_by_name(user_b, "Meta", "organization")

        assert found_a is not None
        assert found_b is not None
        assert found_a.entity_id != found_b.entity_id
