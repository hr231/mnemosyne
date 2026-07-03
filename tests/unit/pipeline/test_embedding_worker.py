"""Unit tests for the batch embedding worker."""
from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.memory import Memory
from tests.fixtures.fake_embedding import FakeEmbeddingClient
from mnemosyne.pipeline.embedding import embed_memory_ids, embed_pending_memories
from tests.unit.pipeline.conftest import build_in_memory_provider


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient(dim=768)


@pytest.fixture
def provider():
    return build_in_memory_provider()


async def _add_memory(provider, embedder: FakeEmbeddingClient, content: str, user_id: uuid.UUID) -> uuid.UUID:
    """Helper: embed and add a memory, return its ID."""
    embedding = await embedder.embed(content)
    mem = Memory(user_id=user_id, content=content, embedding=embedding)
    return await provider.add(mem)


# ---------------------------------------------------------------------------
# embed_pending_memories
# ---------------------------------------------------------------------------


class TestEmbedPendingMemories:
    @pytest.mark.asyncio
    async def test_embeds_memories_with_null_embedding(self, provider, embedder):
        user_id = uuid.uuid4()
        # Inject a memory directly without embedding
        mem = Memory(user_id=user_id, content="I like blue", embedding=None)
        # Bypass provider.add because it raises ValueError for None embedding
        mem_with_hash = mem.model_copy(update={"content_hash": "fake"})
        provider._memories[mem.memory_id] = mem_with_hash

        count = await embed_pending_memories(provider, embedder)

        assert count == 1
        stored = provider._memories[mem.memory_id]
        assert stored.embedding is not None
        assert len(stored.embedding) == 768

    @pytest.mark.asyncio
    async def test_skips_already_embedded_memories(self, provider, embedder):
        user_id = uuid.uuid4()
        mem_id = await _add_memory(provider, embedder, "already embedded", user_id)
        assert provider._memories[mem_id].embedding is not None

        count = await embed_pending_memories(provider, embedder)

        assert count == 0

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_provider(self, provider, embedder):
        count = await embed_pending_memories(provider, embedder)
        assert count == 0

    @pytest.mark.asyncio
    async def test_embeds_multiple_null_memories(self, provider, embedder):
        user_id = uuid.uuid4()
        for i in range(3):
            mem = Memory(user_id=user_id, content=f"content {i}", embedding=None)
            mem_with_hash = mem.model_copy(update={"content_hash": f"hash{i}"})
            provider._memories[mem.memory_id] = mem_with_hash

        count = await embed_pending_memories(provider, embedder)

        assert count == 3
        for mem in provider._memories.values():
            assert mem.embedding is not None

    @pytest.mark.asyncio
    async def test_respects_batch_size(self, provider, embedder):
        user_id = uuid.uuid4()
        for i in range(5):
            mem = Memory(user_id=user_id, content=f"item {i}", embedding=None)
            mem_with_hash = mem.model_copy(update={"content_hash": f"h{i}"})
            provider._memories[mem.memory_id] = mem_with_hash

        # batch_size=2 should still embed all 5 in multiple iterations
        count = await embed_pending_memories(provider, embedder, batch_size=2)
        assert count == 5

    @pytest.mark.asyncio
    async def test_user_id_filter_only_embeds_that_user(self, provider, embedder):
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        for user_id, tag in ((user_a, "a"), (user_b, "b")):
            mem = Memory(user_id=user_id, content=f"item {tag}", embedding=None)
            mem = mem.model_copy(update={"content_hash": f"h{tag}"})
            provider._memories[mem.memory_id] = mem

        count = await embed_pending_memories(provider, embedder, user_id=user_a)

        assert count == 1
        unembedded = [
            m for m in provider._memories.values() if m.embedding is None
        ]
        assert len(unembedded) == 1
        assert unembedded[0].user_id == user_b

    @pytest.mark.asyncio
    async def test_dim_mismatch_raises(self, provider):
        class WrongDimEmbedder(FakeEmbeddingClient):
            async def embed_batch(self, texts):
                return [[0.0] * 5 for _ in texts]

        user_id = uuid.uuid4()
        mem = Memory(user_id=user_id, content="x", embedding=None)
        mem = mem.model_copy(update={"content_hash": "hx"})
        provider._memories[mem.memory_id] = mem

        with pytest.raises(ValueError, match="dimension"):
            await embed_pending_memories(
                provider, WrongDimEmbedder(dim=768), expected_dim=768
            )

    @pytest.mark.asyncio
    async def test_length_mismatch_raises(self, provider):
        class ShortBatchEmbedder(FakeEmbeddingClient):
            async def embed_batch(self, texts):
                return [await self.embed(texts[0])]  # returns fewer than requested

        user_id = uuid.uuid4()
        for i in range(2):
            mem = Memory(user_id=user_id, content=f"c{i}", embedding=None)
            mem = mem.model_copy(update={"content_hash": f"h{i}"})
            provider._memories[mem.memory_id] = mem

        with pytest.raises(ValueError):
            await embed_pending_memories(provider, ShortBatchEmbedder(dim=768))


# ---------------------------------------------------------------------------
# embed_memory_ids
# ---------------------------------------------------------------------------


class TestEmbedMemoryIds:
    @pytest.mark.asyncio
    async def test_embeds_specified_ids_only(self, provider, embedder):
        user_id = uuid.uuid4()
        # One memory with embedding (should be skipped), one without
        embedded_id = await _add_memory(provider, embedder, "already done", user_id)
        unembedded_mem = Memory(user_id=user_id, content="needs embedding", embedding=None)
        unembedded_mem = unembedded_mem.model_copy(update={"content_hash": "uh"})
        provider._memories[unembedded_mem.memory_id] = unembedded_mem

        count = await embed_memory_ids(provider, embedder, [embedded_id, unembedded_mem.memory_id])

        assert count == 1
        stored = provider._memories[unembedded_mem.memory_id]
        assert stored.embedding is not None

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_list(self, provider, embedder):
        count = await embed_memory_ids(provider, embedder, [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_ignores_missing_ids(self, provider, embedder):
        count = await embed_memory_ids(provider, embedder, [uuid.uuid4()])
        assert count == 0
