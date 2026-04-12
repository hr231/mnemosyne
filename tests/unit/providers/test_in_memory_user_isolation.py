"""SECURITY BOUNDARY: User isolation tests for InMemoryProvider.

These tests enforce that memories from one user NEVER appear in another
user's search results. This is a regression test for a security boundary
violation — if any of these tests fail, the provider is leaking data
across user boundaries and must be treated as a P0 bug.
"""
from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.memory import Memory
from mnemosyne.embedding.fake import FakeEmbeddingClient
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


class TestUserIsolation:
    async def test_user_a_cannot_see_user_b_memories(self, provider, embedder):
        """A search as user_a must return zero results when only user_b has memories."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        # Only user_b has a memory
        mem_b = await _make_memory(embedder, user_b, "User B is allergic to peanuts")
        await provider.add(mem_b)

        # user_a searches — should see nothing
        query_vec = await embedder.embed("peanut allergy")
        hits = await provider.search(query_vec, user_id=user_a, limit=10)

        assert len(hits) == 0, (
            f"SECURITY VIOLATION: user_a received {len(hits)} results "
            f"that belong to user_b"
        )

    async def test_user_b_cannot_see_user_a_memories(self, provider, embedder):
        """A search as user_b must return zero results when only user_a has memories."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        # Only user_a has a memory
        mem_a = await _make_memory(embedder, user_a, "User A drives a red car")
        await provider.add(mem_a)

        # user_b searches — should see nothing
        query_vec = await embedder.embed("red car driving")
        hits = await provider.search(query_vec, user_id=user_b, limit=10)

        assert len(hits) == 0, (
            f"SECURITY VIOLATION: user_b received {len(hits)} results "
            f"that belong to user_a"
        )

    async def test_bidirectional_isolation(self, provider, embedder):
        """Each user must only see their own memories, never the other's."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        content_a = "User A prefers Italian cuisine"
        content_b = "User B prefers Japanese cuisine"

        mem_a = await _make_memory(embedder, user_a, content_a)
        mem_b = await _make_memory(embedder, user_b, content_b)

        id_a = await provider.add(mem_a)
        id_b = await provider.add(mem_b)

        # Use a generic query that should match both contents equally
        query_vec = await embedder.embed("food cuisine preference")

        # user_a sees only their own memory
        hits_a = await provider.search(query_vec, user_id=user_a, limit=10)
        ids_returned_to_a = {h.memory.memory_id for h in hits_a}
        assert id_a in ids_returned_to_a, "user_a should see their own memory"
        assert id_b not in ids_returned_to_a, (
            "SECURITY VIOLATION: user_a received user_b's memory"
        )

        # user_b sees only their own memory
        hits_b = await provider.search(query_vec, user_id=user_b, limit=10)
        ids_returned_to_b = {h.memory.memory_id for h in hits_b}
        assert id_b in ids_returned_to_b, "user_b should see their own memory"
        assert id_a not in ids_returned_to_b, (
            "SECURITY VIOLATION: user_b received user_a's memory"
        )

    async def test_multiple_users_many_memories_isolation(self, provider, embedder):
        """Isolation holds with many users and many memories in the store."""
        users = [uuid.uuid4() for _ in range(5)]
        user_memories: dict[uuid.UUID, list[uuid.UUID]] = {}

        # Each user gets 3 memories
        for user in users:
            user_memories[user] = []
            for i in range(3):
                mem = await _make_memory(
                    embedder, user, f"Memory {i} for user {str(user)[:8]}"
                )
                mid = await provider.add(mem)
                user_memories[user].append(mid)

        # Each user's search must only return their own memory IDs
        query_vec = await embedder.embed("memory recall")
        for user in users:
            hits = await provider.search(query_vec, user_id=user, limit=20)
            returned_ids = {h.memory.memory_id for h in hits}

            own_ids = set(user_memories[user])
            other_ids = {
                mid
                for other_user, ids in user_memories.items()
                for mid in ids
                if other_user != user
            }

            leaked = returned_ids & other_ids
            assert not leaked, (
                f"SECURITY VIOLATION: user {str(user)[:8]} received "
                f"{len(leaked)} memories belonging to other users: {leaked}"
            )

    async def test_get_by_id_returns_any_user_memory(self, provider, embedder):
        """get_by_id does not enforce user isolation (it is an admin-level lookup).
        This test documents the expected behavior: get_by_id returns the memory
        regardless of which user_id it belongs to, since the caller provides the UUID.
        """
        user_a = uuid.uuid4()
        mem_a = await _make_memory(embedder, user_a, "Private fact for user A")
        id_a = await provider.add(mem_a)

        # get_by_id is not user-scoped — returns by primary key
        result = await provider.get_by_id(id_a)
        assert result is not None
        assert result.memory_id == id_a

    async def test_isolation_preserved_with_include_invalidated(
        self, provider, embedder
    ):
        """User isolation must hold even when include_invalidated=True is set."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        mem_b = await _make_memory(embedder, user_b, "User B owns a startup")
        id_b = await provider.add(mem_b)
        await provider.invalidate(id_b, reason="history test")

        query_vec = await embedder.embed("startup company")
        # user_a should not see user_b's invalidated memory even in historical mode
        hits = await provider.search(
            query_vec, user_id=user_a, limit=10, include_invalidated=True
        )
        ids = {h.memory.memory_id for h in hits}
        assert id_b not in ids, (
            "SECURITY VIOLATION: user_a received user_b's memory "
            "via include_invalidated=True"
        )
