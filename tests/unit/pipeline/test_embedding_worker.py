"""Unit tests for the batch embedding worker."""
from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.memory import Memory
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.pipeline.embedding import embed_memory_ids, embed_pending_memories
from mnemosyne.providers.in_memory import InMemoryProvider


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient(dim=768)


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


async def _add_memory(provider: InMemoryProvider, embedder: FakeEmbeddingClient, content: str, user_id: uuid.UUID) -> uuid.UUID:
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
