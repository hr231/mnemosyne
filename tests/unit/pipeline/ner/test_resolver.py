from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.entity import Entity, EntityMention
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.pipeline.ner.resolver import resolve_entities
from mnemosyne.pipeline.ner.spacy_extractor import RawEntity
from mnemosyne.providers.in_memory_entity_store import InMemoryEntityStore


def _raw(name: str, entity_type: str = "person") -> RawEntity:
    return RawEntity(
        name=name,
        entity_type=entity_type,
        mention_text=name,
        context=f"Context for {name}.",
        source="spacy",
    )


@pytest.fixture
def user_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def memory_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def store() -> InMemoryEntityStore:
    return InMemoryEntityStore()


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient(dim=768)


@pytest.mark.asyncio
async def test_creates_new_entity(user_id, memory_id, store, embedder):
    results = await resolve_entities(
        raw_entities=[_raw("John Smith")],
        user_id=user_id,
        memory_id=memory_id,
        entity_store=store,
        embedder=embedder,
    )
    assert len(results) == 1
    entity = results[0]
    assert entity.entity_name == "John Smith"
    assert entity.entity_type == "person"
    assert entity.user_id == user_id

    # Verify it was actually stored
    found = await store.find_by_name(user_id, "John Smith", "person")
    assert found is not None


@pytest.mark.asyncio
async def test_resolves_to_existing(user_id, memory_id, store, embedder):
    """Resolving the same name twice should return the same entity_id."""
    first_memory = uuid.uuid4()
    second_memory = uuid.uuid4()

    first_results = await resolve_entities(
        raw_entities=[_raw("Alice")],
        user_id=user_id,
        memory_id=first_memory,
        entity_store=store,
        embedder=embedder,
    )
    first_id = first_results[0].entity_id

    second_results = await resolve_entities(
        raw_entities=[_raw("Alice")],
        user_id=user_id,
        memory_id=second_memory,
        entity_store=store,
        embedder=embedder,
    )
    second_id = second_results[0].entity_id

    assert first_id == second_id


@pytest.mark.asyncio
async def test_case_insensitive_resolution(user_id, memory_id, store, embedder):
    """Creating 'Nike' and then resolving 'NIKE' should hit the same entity."""
    first_memory = uuid.uuid4()
    second_memory = uuid.uuid4()

    await resolve_entities(
        raw_entities=[_raw("Nike", "organization")],
        user_id=user_id,
        memory_id=first_memory,
        entity_store=store,
        embedder=embedder,
    )

    second_results = await resolve_entities(
        raw_entities=[_raw("NIKE", "organization")],
        user_id=user_id,
        memory_id=second_memory,
        entity_store=store,
        embedder=embedder,
    )
    # find_by_name normalizes before lookup, so NIKE -> nike -> hits "nike" stored entry
    found = await store.find_by_name(user_id, "Nike", "organization")
    assert found is not None
    # The second resolve should find the existing entity (same normalized key)
    assert second_results[0].entity_id == found.entity_id


@pytest.mark.asyncio
async def test_creates_mention(user_id, memory_id, store, embedder):
    await resolve_entities(
        raw_entities=[_raw("Bob")],
        user_id=user_id,
        memory_id=memory_id,
        entity_store=store,
        embedder=embedder,
    )
    entity = await store.find_by_name(user_id, "Bob", "person")
    assert entity is not None
    memory_ids = await store.find_mentions_for_entity(entity.entity_id)
    assert memory_id in memory_ids


@pytest.mark.asyncio
async def test_multiple_entities_resolved(user_id, memory_id, store, embedder):
    raw = [
        _raw("Alice", "person"),
        _raw("Google", "organization"),
        _raw("London", "location"),
    ]
    results = await resolve_entities(
        raw_entities=raw,
        user_id=user_id,
        memory_id=memory_id,
        entity_store=store,
        embedder=embedder,
    )
    assert len(results) == 3
    names = {e.entity_name for e in results}
    assert "Alice" in names
    assert "Google" in names
    assert "London" in names


@pytest.mark.asyncio
async def test_mention_text_stored_correctly(user_id, memory_id, store, embedder):
    raw = [RawEntity(
        name="Apple",
        entity_type="organization",
        mention_text="Apple Inc.",
        context="Apple Inc. is a tech company.",
        source="spacy",
    )]
    await resolve_entities(
        raw_entities=raw,
        user_id=user_id,
        memory_id=memory_id,
        entity_store=store,
        embedder=embedder,
    )
    entity = await store.find_by_name(user_id, "Apple", "organization")
    assert entity is not None
    mentions = store._mentions
    apple_mentions = [m for m in mentions if m.entity_id == entity.entity_id]
    assert len(apple_mentions) >= 1
    assert apple_mentions[0].mention_text == "Apple Inc."


@pytest.mark.asyncio
async def test_empty_input_returns_empty(user_id, memory_id, store, embedder):
    results = await resolve_entities(
        raw_entities=[],
        user_id=user_id,
        memory_id=memory_id,
        entity_store=store,
        embedder=embedder,
    )
    assert results == []
