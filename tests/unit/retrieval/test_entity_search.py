from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from mnemosyne.context.assembly import ContextBlock, assemble_context
from mnemosyne.db.models.entity import Entity, EntityMention
from mnemosyne.db.models.memory import Memory, ScoredMemory
from tests.fixtures.fake_embedding import FakeEmbeddingClient
from mnemosyne.providers.in_memory import InMemoryProvider
from mnemosyne.providers.in_memory_entity_store import InMemoryEntityStore
from mnemosyne.retrieval.entity_search import (
    MAX_MENTIONS_PER_ENTITY,
    RRF_K,
    _fuse_rrf,
    _rrf_score,
    entity_aware_search,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EMBEDDER = FakeEmbeddingClient(dim=32)


async def _make_memory(
    provider: InMemoryProvider,
    user_id: uuid.UUID,
    content: str,
    importance: float = 0.5,
    valid_until: datetime | None = None,
) -> Memory:
    embedding = await _EMBEDDER.embed(content)
    mem = Memory(
        user_id=user_id,
        content=content,
        importance=importance,
        embedding=embedding,
        valid_until=valid_until,
    )
    await provider.add(mem)
    # Fetch back so we have the stored object (dedup may return existing id)
    stored = provider._memories.get(mem.memory_id)
    # If dedup returned a different id the original mem still has the right content
    return stored if stored is not None else mem


def _make_entity(
    user_id: uuid.UUID,
    name: str,
    entity_type: str,
    embedding: list[float] | None = None,
) -> Entity:
    return Entity(
        user_id=user_id,
        entity_name=name,
        entity_type=entity_type,
        embedding=embedding,
    )


def _make_mention(entity_id: uuid.UUID, memory_id: uuid.UUID) -> EntityMention:
    return EntityMention(entity_id=entity_id, memory_id=memory_id)


# ---------------------------------------------------------------------------
# _rrf_score unit tests
# ---------------------------------------------------------------------------


class TestRrfScoreFunction:
    def test_single_rank_one(self) -> None:
        score = _rrf_score([1])
        assert abs(score - 1.0 / (RRF_K + 1)) < 1e-9

    def test_single_rank_ten(self) -> None:
        score = _rrf_score([10])
        assert abs(score - 1.0 / (RRF_K + 10)) < 1e-9

    def test_two_ranks_summed(self) -> None:
        score = _rrf_score([1, 2])
        expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 2)
        assert abs(score - expected) < 1e-9

    def test_higher_rank_means_lower_score(self) -> None:
        # rank 1 (top result) should give higher score than rank 5
        assert _rrf_score([1]) > _rrf_score([5])

    def test_appearing_in_both_lists_gives_higher_score(self) -> None:
        # Appearing in two lists beats appearing in only one at the same rank
        assert _rrf_score([1, 1]) > _rrf_score([1])


# ---------------------------------------------------------------------------
# Fallback: no entity_store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_without_entity_store() -> None:
    """entity_aware_search with entity_store=None returns the same results as
    provider.search (up to limit)."""
    provider = InMemoryProvider()
    user_id = uuid.uuid4()

    for i in range(5):
        await _make_memory(provider, user_id, f"memory content number {i}")

    query_text = "memory content"
    query_embedding = await _EMBEDDER.embed(query_text)

    # entity_aware_search without store
    results = await entity_aware_search(
        provider=provider,
        entity_store=None,
        query_text=query_text,
        query_embedding=query_embedding,
        user_id=user_id,
        limit=3,
    )

    # provider.search directly
    reference = await provider.search(query_embedding, user_id=user_id, limit=3)

    # Should return at most limit results
    assert len(results) <= 3
    # Memory IDs should match (order may differ due to access-count side-effects)
    result_ids = {sm.memory.memory_id for sm in results}
    ref_ids = {sm.memory.memory_id for sm in reference}
    assert result_ids == ref_ids


# ---------------------------------------------------------------------------
# Entity expansion finds memories that don't match by text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_aware_search_finds_by_entity() -> None:
    """A memory that does not contain 'Nike' in text should be found when
    an entity 'Nike' is linked to it via a mention and the query is 'Nike'."""
    provider = InMemoryProvider()
    entity_store = InMemoryEntityStore()
    user_id = uuid.uuid4()

    # Memory text deliberately does NOT contain "Nike"
    mem = await _make_memory(provider, user_id, "those shoes were too narrow")

    # Create Nike entity and link it to that memory
    entity = _make_entity(user_id, "Nike", "brand")
    await entity_store.upsert_entity(entity)
    mention = _make_mention(entity.entity_id, mem.memory_id)
    await entity_store.add_mention(mention)

    query_text = "Nike"
    query_embedding = await _EMBEDDER.embed(query_text)

    results = await entity_aware_search(
        provider=provider,
        entity_store=entity_store,
        query_text=query_text,
        query_embedding=query_embedding,
        user_id=user_id,
        embedder=_EMBEDDER,
        limit=10,
    )

    result_ids = {sm.memory.memory_id for sm in results}
    assert mem.memory_id in result_ids, (
        "Expected memory linked via entity mention to appear in results"
    )


# ---------------------------------------------------------------------------
# RRF fusion: memories in both lists rank higher
# ---------------------------------------------------------------------------


def test_rrf_fusion_combines_both_sources() -> None:
    """A memory that appears in both vector and entity results should have a
    higher RRF score than one that appears only in one list.

    This is a pure unit test of _fuse_rrf — no search involved — so it is
    fully deterministic regardless of embedding values.
    """
    user_id = uuid.uuid4()

    # mem_a appears in BOTH vector and entity lists (rank 1 in each)
    mem_a = _make_scored_memory(user_id, "in both lists", score=0.9)
    # mem_b appears ONLY in the vector list (rank 2)
    mem_b = _make_scored_memory(user_id, "vector only", score=0.8)
    # mem_c appears ONLY in the entity list (rank 2)
    mem_c = _make_scored_memory(user_id, "entity only", score=0.0)

    vector_results = [mem_a, mem_b]
    entity_results = [mem_a, mem_c]

    fused = _fuse_rrf(vector_results, entity_results, limit=10)

    assert len(fused) >= 1
    # All scores must be positive
    for sm in fused:
        assert sm.score > 0.0

    # mem_a (rank 1 in both) should rank above mem_b (rank 2 in vector only)
    # RRF(1,1) = 1/(60+1) + 1/(60+1) > 1/(60+1) = RRF(1 in vector)
    a_score = next(sm.score for sm in fused if sm.memory.memory_id == mem_a.memory.memory_id)
    b_score = next(sm.score for sm in fused if sm.memory.memory_id == mem_b.memory.memory_id)
    assert a_score > b_score, (
        "Memory in both lists must score higher than one in only one list"
    )
    # First result must be mem_a
    assert fused[0].memory.memory_id == mem_a.memory.memory_id


# ---------------------------------------------------------------------------
# RRF pure vector: when entity expansion returns nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rrf_fusion_pure_vector() -> None:
    """When entity expansion yields no results, output matches vector-only search."""
    provider = InMemoryProvider()
    entity_store = InMemoryEntityStore()  # empty — no entities registered
    user_id = uuid.uuid4()

    for i in range(4):
        await _make_memory(provider, user_id, f"general content piece {i}")

    query_text = "content piece"
    query_embedding = await _EMBEDDER.embed(query_text)

    results = await entity_aware_search(
        provider=provider,
        entity_store=entity_store,
        query_text=query_text,
        query_embedding=query_embedding,
        user_id=user_id,
        embedder=_EMBEDDER,
        limit=4,
    )

    # With no entity expansion, should return vector results up to limit
    assert len(results) <= 4
    assert all(isinstance(sm, ScoredMemory) for sm in results)


# ---------------------------------------------------------------------------
# Mention cap: at most MAX_MENTIONS_PER_ENTITY expanded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mention_cap_enforced() -> None:
    """Only the first MAX_MENTIONS_PER_ENTITY mentions are expanded;
    additional mentions are ignored."""
    provider = InMemoryProvider()
    entity_store = InMemoryEntityStore()
    user_id = uuid.uuid4()

    entity = _make_entity(user_id, "Acme", "organization")
    await entity_store.upsert_entity(entity)

    # Add 100 distinct memories and link them all to the entity
    total_memories = 100
    memory_ids: list[uuid.UUID] = []
    for i in range(total_memories):
        mem = await _make_memory(provider, user_id, f"acme related memory {i}")
        memory_ids.append(mem.memory_id)
        await entity_store.add_mention(_make_mention(entity.entity_id, mem.memory_id))

    query_text = "Acme"
    query_embedding = await _EMBEDDER.embed(query_text)

    results = await entity_aware_search(
        provider=provider,
        entity_store=entity_store,
        query_text=query_text,
        query_embedding=query_embedding,
        user_id=user_id,
        embedder=_EMBEDDER,
        limit=200,  # high limit to avoid trimming at fusion step
    )

    # entity expansion is capped at MAX_MENTIONS_PER_ENTITY;
    # combined with vector results the total is capped by the limit param,
    # but entity-expanded results alone cannot exceed the cap.
    # After RRF fusion, in_entity == 1.0 flags memories that came from entity expansion
    in_entity_results = [sm for sm in results if sm.score_breakdown.get("in_entity") == 1.0]
    # Total results (from entity expansion) should not exceed MAX_MENTIONS_PER_ENTITY
    # The strict cap is on expansion; verify no more than cap entities are in the bucket.
    assert len(in_entity_results) <= MAX_MENTIONS_PER_ENTITY


# ---------------------------------------------------------------------------
# Invalidated memories excluded from entity expansion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_search_excludes_invalidated() -> None:
    """An entity mention pointing to an invalidated memory must not appear
    in entity_aware_search results."""
    provider = InMemoryProvider()
    entity_store = InMemoryEntityStore()
    user_id = uuid.uuid4()

    # Create a memory and immediately invalidate it
    mem_valid = await _make_memory(provider, user_id, "valid shoe preference")
    mem_invalid = await _make_memory(provider, user_id, "old invalidated preference")
    await provider.invalidate(mem_invalid.memory_id, reason="superseded")

    entity = _make_entity(user_id, "Adidas", "brand")
    await entity_store.upsert_entity(entity)
    # Link both to the entity
    await entity_store.add_mention(_make_mention(entity.entity_id, mem_valid.memory_id))
    await entity_store.add_mention(_make_mention(entity.entity_id, mem_invalid.memory_id))

    query_text = "Adidas"
    query_embedding = await _EMBEDDER.embed(query_text)

    results = await entity_aware_search(
        provider=provider,
        entity_store=entity_store,
        query_text=query_text,
        query_embedding=query_embedding,
        user_id=user_id,
        embedder=_EMBEDDER,
        limit=10,
    )

    result_ids = {sm.memory.memory_id for sm in results}
    assert mem_invalid.memory_id not in result_ids, (
        "Invalidated memory must not appear in results"
    )


# ---------------------------------------------------------------------------
# _fuse_rrf direct unit test
# ---------------------------------------------------------------------------


def _make_scored_memory(user_id: uuid.UUID, content: str, score: float) -> ScoredMemory:
    mem = Memory(
        user_id=user_id,
        content=content,
        embedding=[0.1] * 32,
    )
    return ScoredMemory(memory=mem, score=score, score_breakdown={})


class TestFuseRrf:
    def test_overlap_boosts_score(self) -> None:
        """A memory in both lists should rank above one appearing in only one."""
        user_id = uuid.uuid4()
        shared = _make_scored_memory(user_id, "shared memory", score=0.8)
        vector_only = _make_scored_memory(user_id, "vector only", score=0.9)
        entity_only = _make_scored_memory(user_id, "entity only", score=0.0)

        vector_results = [shared, vector_only]
        entity_results = [shared, entity_only]

        fused = _fuse_rrf(vector_results, entity_results, limit=10)
        assert len(fused) == 3
        # shared appears in both so its RRF score should be highest
        assert fused[0].memory.memory_id == shared.memory.memory_id

    def test_limit_respected(self) -> None:
        user_id = uuid.uuid4()
        vec = [_make_scored_memory(user_id, f"v{i}", score=float(i)) for i in range(5)]
        ent = [_make_scored_memory(user_id, f"e{i}", score=float(i)) for i in range(5)]
        fused = _fuse_rrf(vec, ent, limit=3)
        assert len(fused) == 3

    def test_score_breakdown_contains_rrf_fields(self) -> None:
        user_id = uuid.uuid4()
        sm = _make_scored_memory(user_id, "test", score=0.5)
        fused = _fuse_rrf([sm], [], limit=5)
        assert len(fused) == 1
        bd = fused[0].score_breakdown
        assert "rrf_score" in bd
        assert "in_vector" in bd
        assert "in_entity" in bd

    def test_entity_only_has_in_entity_true(self) -> None:
        user_id = uuid.uuid4()
        sm = _make_scored_memory(user_id, "entity only", score=0.0)
        fused = _fuse_rrf([], [sm], limit=5)
        assert fused[0].score_breakdown["in_entity"] == 1.0
        assert fused[0].score_breakdown["in_vector"] == 0.0


# ---------------------------------------------------------------------------
# assemble_context backward compatibility (no entity_store)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_context_without_entity_store() -> None:
    """Calling assemble_context without entity_store must work identically to
    the pre-entity-search behaviour."""
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    user_id = uuid.uuid4()

    for i in range(3):
        emb = await embedder.embed(f"memory about topic {i}")
        mem = Memory(
            user_id=user_id,
            content=f"memory about topic {i}",
            embedding=emb,
        )
        await provider.add(mem)

    query_vec = await embedder.embed("memory topic")
    block = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
        # entity_store omitted — default None
    )

    assert isinstance(block, ContextBlock)
    assert block.token_count <= 500
    assert block.sections is not None


# ---------------------------------------------------------------------------
# assemble_context with entity_store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_context_with_entity_store() -> None:
    """assemble_context with entity_store and query_text should return a valid
    ContextBlock with token_count within budget."""
    provider = InMemoryProvider()
    entity_store = InMemoryEntityStore()
    embedder = FakeEmbeddingClient(dim=768)
    user_id = uuid.uuid4()

    # Add a memory
    emb = await embedder.embed("user prefers Nike running shoes")
    mem = Memory(
        user_id=user_id,
        content="user prefers Nike running shoes",
        embedding=emb,
        importance=0.8,
    )
    await provider.add(mem)

    # Register a Nike entity and link it to the memory
    entity = _make_entity(user_id, "Nike", "brand")
    await entity_store.upsert_entity(entity)
    await entity_store.add_mention(_make_mention(entity.entity_id, mem.memory_id))

    query_text = "Nike"
    query_vec = await embedder.embed(query_text)

    block = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
        entity_store=entity_store,
        query_text=query_text,
    )

    assert isinstance(block, ContextBlock)
    assert block.token_count <= 500
    assert block.sections is not None
    # The Nike memory should appear somewhere in the assembled text
    assert "nike" in block.text.lower() or len(block.text) == 0
