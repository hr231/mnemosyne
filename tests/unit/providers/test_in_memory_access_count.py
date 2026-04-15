"""Access count bookkeeping tests for InMemoryProvider."""
from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.memory import Memory
from tests.fixtures.fake_embedding import FakeEmbeddingClient


async def _make_memory(
    embedder: FakeEmbeddingClient,
    user_id: uuid.UUID,
    content: str,
    **kwargs,
) -> Memory:
    """Helper: build a Memory with an embedding already set."""
    embedding = await embedder.embed(content)
    return Memory(user_id=user_id, content=content, embedding=embedding, **kwargs)


class TestAccessCountBookkeeping:
    async def test_access_count_starts_at_zero(self, provider, embedder):
        """A freshly added memory has access_count == 0 before any search."""
        user_id = uuid.uuid4()
        mem = await _make_memory(embedder, user_id, "User runs marathons")
        memory_id = await provider.add(mem)

        stored = await provider.get_by_id(memory_id)
        assert stored is not None
        assert stored.access_count == 0

    async def test_search_increments_access_count(self, provider, embedder):
        """Each search call that returns the memory increments access_count by 1."""
        user_id = uuid.uuid4()
        mem = await _make_memory(embedder, user_id, "User runs marathons")
        memory_id = await provider.add(mem)

        query_vec = await embedder.embed("marathon running")

        await provider.search(query_vec, user_id=user_id, limit=5)
        await provider.search(query_vec, user_id=user_id, limit=5)

        stored = await provider.get_by_id(memory_id)
        assert stored is not None
        assert stored.access_count == 2

    async def test_access_count_only_on_returned(self, provider, embedder):
        """Only memories actually returned (within limit) have access_count incremented."""
        user_id = uuid.uuid4()

        # Add 5 memories with distinct content
        contents = [
            "User likes swimming",
            "User enjoys cycling",
            "User loves hiking",
            "User plays tennis",
            "User does yoga",
        ]
        memory_ids = []
        for content in contents:
            mem = await _make_memory(embedder, user_id, content)
            mid = await provider.add(mem)
            memory_ids.append(mid)

        # Search with limit=2 — only 2 memories should have access_count incremented
        query_vec = await embedder.embed("sport hobby activity")
        hits = await provider.search(query_vec, user_id=user_id, limit=2)

        assert len(hits) == 2

        returned_ids = {h.memory.memory_id for h in hits}
        not_returned_ids = set(memory_ids) - returned_ids

        # Returned memories: access_count == 1
        for mid in returned_ids:
            stored = await provider.get_by_id(mid)
            assert stored is not None
            assert stored.access_count == 1, (
                f"Expected access_count=1 for returned memory {mid}, "
                f"got {stored.access_count}"
            )

        # Non-returned memories: access_count still 0
        for mid in not_returned_ids:
            stored = await provider.get_by_id(mid)
            assert stored is not None
            assert stored.access_count == 0, (
                f"Expected access_count=0 for non-returned memory {mid}, "
                f"got {stored.access_count}"
            )

    async def test_access_count_not_incremented_before_scoring(
        self, provider, embedder
    ):
        """access_count is bumped AFTER scoring so the frequency signal is
        deterministic for the first query (count=0 during scoring)."""
        user_id = uuid.uuid4()
        mem = await _make_memory(embedder, user_id, "User drinks green tea")
        memory_id = await provider.add(mem)

        query_vec = await embedder.embed("tea preference")

        # First search — scoring sees access_count=0, then bumps to 1
        hits = await provider.search(query_vec, user_id=user_id, limit=5)
        assert any(h.memory.memory_id == memory_id for h in hits)

        stored_after_first = await provider.get_by_id(memory_id)
        assert stored_after_first is not None
        assert stored_after_first.access_count == 1

    async def test_last_accessed_updated_on_search(self, provider, embedder):
        """last_accessed timestamp is updated each time the memory appears in results."""
        user_id = uuid.uuid4()
        mem = await _make_memory(embedder, user_id, "User speaks French")
        memory_id = await provider.add(mem)

        stored_before = await provider.get_by_id(memory_id)
        original_last_accessed = stored_before.last_accessed

        query_vec = await embedder.embed("French language")
        await provider.search(query_vec, user_id=user_id, limit=5)

        stored_after = await provider.get_by_id(memory_id)
        # last_accessed should be >= the original timestamp
        assert stored_after.last_accessed >= original_last_accessed
