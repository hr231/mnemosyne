"""Bi-temporal filtering tests for InMemoryProvider."""
from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.memory import Memory
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.errors import MemoryNotFound
from mnemosyne.providers.in_memory import InMemoryProvider


async def _make_memory(
    embedder: FakeEmbeddingClient,
    user_id: uuid.UUID,
    content: str,
    **kwargs,
) -> Memory:
    """Helper: build a Memory with an embedding already set."""
    embedding = await embedder.embed(content)
    return Memory(user_id=user_id, content=content, embedding=embedding, **kwargs)


@pytest.fixture
def provider():
    return InMemoryProvider()


@pytest.fixture
def embedder():
    return FakeEmbeddingClient(dim=1536)


class TestBitemporalFiltering:
    async def test_invalidate_excludes_from_search(self, provider, embedder):
        """An invalidated memory should not appear in default search results."""
        user_id = uuid.uuid4()
        mem = await _make_memory(embedder, user_id, "User is allergic to peanuts")
        memory_id = await provider.add(mem)

        # Confirm it appears before invalidation
        query_vec = await embedder.embed("peanut allergy")
        hits_before = await provider.search(query_vec, user_id=user_id, limit=5)
        assert any(h.memory.memory_id == memory_id for h in hits_before)

        # Invalidate and verify it no longer appears
        await provider.invalidate(memory_id, reason="user corrected record")
        hits_after = await provider.search(query_vec, user_id=user_id, limit=5)
        assert not any(h.memory.memory_id == memory_id for h in hits_after)

    async def test_invalidate_sets_valid_until(self, provider, embedder):
        """After invalidation, valid_until must be set (not None)."""
        user_id = uuid.uuid4()
        mem = await _make_memory(embedder, user_id, "User lives in New York")
        memory_id = await provider.add(mem)

        await provider.invalidate(memory_id, reason="moved cities")
        stored = await provider.get_by_id(memory_id)

        assert stored is not None
        assert stored.valid_until is not None

    async def test_invalidate_stores_reason(self, provider, embedder):
        """The invalidation reason string must be persisted in metadata."""
        user_id = uuid.uuid4()
        mem = await _make_memory(embedder, user_id, "User drives a Tesla")
        memory_id = await provider.add(mem)

        reason = "user sold the car"
        await provider.invalidate(memory_id, reason=reason)
        stored = await provider.get_by_id(memory_id)

        assert stored is not None
        assert stored.metadata.get("invalidation_reason") == reason

    async def test_invalidate_nonexistent_raises(self, provider):
        """Invalidating an unknown memory_id must raise MemoryNotFound."""
        nonexistent_id = uuid.uuid4()
        with pytest.raises(MemoryNotFound):
            await provider.invalidate(nonexistent_id, reason="does not matter")

    async def test_historical_search_includes_invalidated(self, provider, embedder):
        """include_invalidated=True must return invalidated memories."""
        user_id = uuid.uuid4()
        mem = await _make_memory(embedder, user_id, "User prefers morning workouts")
        memory_id = await provider.add(mem)

        await provider.invalidate(memory_id, reason="schedule changed")

        # Default search: excluded
        query_vec = await embedder.embed("morning workout preference")
        hits_default = await provider.search(query_vec, user_id=user_id, limit=5)
        assert not any(h.memory.memory_id == memory_id for h in hits_default)

        # Historical search: included
        hits_historical = await provider.search(
            query_vec, user_id=user_id, limit=5, include_invalidated=True
        )
        assert any(h.memory.memory_id == memory_id for h in hits_historical)

    async def test_active_memories_still_returned_after_some_invalidated(
        self, provider, embedder
    ):
        """Invalidating one memory should not affect other active memories."""
        user_id = uuid.uuid4()
        mem_a = await _make_memory(embedder, user_id, "User likes coffee")
        mem_b = await _make_memory(embedder, user_id, "User likes tea")

        id_a = await provider.add(mem_a)
        id_b = await provider.add(mem_b)

        await provider.invalidate(id_a, reason="changed preference")

        query_vec = await embedder.embed("tea preference")
        hits = await provider.search(query_vec, user_id=user_id, limit=5)

        returned_ids = {h.memory.memory_id for h in hits}
        assert id_a not in returned_ids
        assert id_b in returned_ids
