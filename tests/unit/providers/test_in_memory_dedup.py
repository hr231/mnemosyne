"""Content-hash deduplication tests for InMemoryProvider."""
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


class TestContentHashDedup:
    async def test_duplicate_add_returns_same_id(self, provider, embedder):
        """Adding the same content twice for the same user returns the original memory_id."""
        user_id = uuid.uuid4()
        content = "User is allergic to shellfish"

        mem1 = await _make_memory(embedder, user_id, content)
        mem2 = await _make_memory(embedder, user_id, content)

        id1 = await provider.add(mem1)
        id2 = await provider.add(mem2)

        assert id1 == id2

    async def test_duplicate_does_not_create_extra_row(self, provider, embedder):
        """After two identical adds, only one memory exists in the store."""
        user_id = uuid.uuid4()
        content = "User prefers window seats"

        mem1 = await _make_memory(embedder, user_id, content)
        mem2 = await _make_memory(embedder, user_id, content)

        await provider.add(mem1)
        await provider.add(mem2)

        # Search should return exactly one result
        query_vec = await embedder.embed("window seat preference")
        hits = await provider.search(query_vec, user_id=user_id, limit=10)
        matching = [h for h in hits if h.memory.content == content]
        assert len(matching) == 1

    async def test_different_content_different_ids(self, provider, embedder):
        """Different content strings produce different memory_ids."""
        user_id = uuid.uuid4()

        mem_a = await _make_memory(embedder, user_id, "User likes pizza")
        mem_b = await _make_memory(embedder, user_id, "User dislikes broccoli")

        id_a = await provider.add(mem_a)
        id_b = await provider.add(mem_b)

        assert id_a != id_b

    async def test_different_user_same_content_not_deduped(self, provider, embedder):
        """The same content for two different users must produce two separate memories."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        content = "Prefers early morning meetings"

        mem_a = await _make_memory(embedder, user_a, content)
        mem_b = await _make_memory(embedder, user_b, content)

        id_a = await provider.add(mem_a)
        id_b = await provider.add(mem_b)

        assert id_a != id_b

    async def test_invalidated_then_readd(self, provider, embedder):
        """After invalidating a memory, re-adding the same content creates a new memory."""
        user_id = uuid.uuid4()
        content = "User owns a mountain bike"

        mem1 = await _make_memory(embedder, user_id, content)
        original_id = await provider.add(mem1)

        # Invalidate the original
        await provider.invalidate(original_id, reason="sold the bike")

        # Re-add the same content — should NOT dedup against the invalidated row
        mem2 = await _make_memory(embedder, user_id, content)
        new_id = await provider.add(mem2)

        assert new_id != original_id

    async def test_dedup_is_case_insensitive_and_strips_whitespace(
        self, provider, embedder
    ):
        """Content hash normalises to strip+lower, so near-identical strings dedup."""
        user_id = uuid.uuid4()

        # First add with one form
        mem1 = await _make_memory(embedder, user_id, "User likes Jazz Music")
        id1 = await provider.add(mem1)

        # Second add with stripped/lowercased equivalent (provider normalises)
        # We must test the normalisation that the provider applies internally.
        # The content itself is "user likes jazz music" — same hash after normalisation.
        mem2 = await _make_memory(embedder, user_id, "  User Likes Jazz Music  ")
        id2 = await provider.add(mem2)

        assert id1 == id2

    async def test_add_without_embedding_raises_value_error(self, provider):
        """add() must raise ValueError when memory.embedding is None."""
        user_id = uuid.uuid4()
        mem = Memory(user_id=user_id, content="No embedding set")
        # embedding is None by default

        with pytest.raises(ValueError, match="embedding"):
            await provider.add(mem)
