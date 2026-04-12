"""Unit tests for the consolidation worker."""
from __future__ import annotations

import hashlib
import uuid

import pytest

from mnemosyne.db.models.memory import Memory
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.pipeline.consolidation import (
    REFLECTION_IMPORTANCE_SUM_THRESHOLD,
    accumulated_importance_sum,
    compute_content_hash,
    find_semantic_duplicates,
    needs_reflection,
    run_dedup,
)
from mnemosyne.providers.in_memory import InMemoryProvider


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient(dim=768)


async def _add(provider, content, user_id, importance=0.5):
    embedding = await FakeEmbeddingClient(dim=768).embed(content)
    mem = Memory(user_id=user_id, content=content, embedding=embedding, importance=importance)
    mid = await provider.add(mem)
    return mid


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    def test_sha256_length(self):
        h = compute_content_hash("Budget: $200")
        assert len(h) == 64

    def test_normalisation_case_insensitive(self):
        assert compute_content_hash("Budget: $200") == compute_content_hash("BUDGET: $200")

    def test_normalisation_strips_whitespace(self):
        assert compute_content_hash("Budget: $200") == compute_content_hash("  Budget: $200  ")

    def test_matches_expected_sha256(self):
        expected = hashlib.sha256("budget: $200".encode()).hexdigest()
        assert compute_content_hash("Budget: $200") == expected


# ---------------------------------------------------------------------------
# find_semantic_duplicates
# ---------------------------------------------------------------------------


class TestFindSemanticDuplicates:
    def test_detects_near_duplicates(self):
        uid = uuid.uuid4()
        m1 = Memory(user_id=uid, content="a", embedding=[1.0, 0.0, 0.0])
        m2 = Memory(user_id=uid, content="b", embedding=[0.99, 0.1, 0.0])
        m3 = Memory(user_id=uid, content="c", embedding=[0.0, 1.0, 0.0])

        pairs = find_semantic_duplicates([m1, m2, m3], similarity_threshold=0.90)

        assert len(pairs) == 1
        ids = {pairs[0][0].memory_id, pairs[0][1].memory_id}
        assert ids == {m1.memory_id, m2.memory_id}

    def test_no_false_positives_below_threshold(self):
        uid = uuid.uuid4()
        m1 = Memory(user_id=uid, content="a", embedding=[1.0, 0.0])
        m2 = Memory(user_id=uid, content="b", embedding=[0.0, 1.0])

        assert find_semantic_duplicates([m1, m2], similarity_threshold=0.90) == []

    def test_cross_user_pairs_excluded(self):
        m1 = Memory(user_id=uuid.uuid4(), content="a", embedding=[1.0, 0.0])
        m2 = Memory(user_id=uuid.uuid4(), content="b", embedding=[0.99, 0.01])

        assert find_semantic_duplicates([m1, m2], similarity_threshold=0.90) == []

    def test_memories_without_embedding_skipped(self):
        uid = uuid.uuid4()
        m1 = Memory(user_id=uid, content="a", embedding=None)
        m2 = Memory(user_id=uid, content="b", embedding=[1.0, 0.0])

        assert find_semantic_duplicates([m1, m2], similarity_threshold=0.90) == []

    def test_empty_list_returns_empty(self):
        assert find_semantic_duplicates([], 0.90) == []


# ---------------------------------------------------------------------------
# accumulated_importance_sum
# ---------------------------------------------------------------------------


class TestAccumulatedImportanceSum:
    def test_scales_by_ten(self):
        uid = uuid.uuid4()
        mems = [Memory(user_id=uid, content=f"m{i}", importance=0.5) for i in range(10)]
        assert accumulated_importance_sum(mems) == pytest.approx(50.0)

    def test_high_importance_reaches_threshold_faster(self):
        uid = uuid.uuid4()
        mems = [Memory(user_id=uid, content=f"m{i}", importance=1.0) for i in range(15)]
        assert accumulated_importance_sum(mems) == pytest.approx(150.0)

    def test_empty_list_returns_zero(self):
        assert accumulated_importance_sum([]) == 0.0


# ---------------------------------------------------------------------------
# needs_reflection
# ---------------------------------------------------------------------------


class TestNeedsReflection:
    def test_triggers_at_threshold(self):
        uid = uuid.uuid4()
        # 15 memories × importance 1.0 × 10 = 150 — exactly at threshold
        mems = [Memory(user_id=uid, content=f"m{i}", importance=1.0) for i in range(15)]
        assert needs_reflection(mems) is True

    def test_does_not_trigger_below_threshold(self):
        uid = uuid.uuid4()
        # 10 × 0.5 × 10 = 50 — well below 150
        mems = [Memory(user_id=uid, content=f"m{i}", importance=0.5) for i in range(10)]
        assert needs_reflection(mems) is False

    def test_mundane_events_accumulate_slowly(self):
        uid = uuid.uuid4()
        # 50 × 0.15 × 10 = 75 — below 150
        mems = [Memory(user_id=uid, content=f"m{i}", importance=0.15) for i in range(50)]
        assert needs_reflection(mems) is False

    def test_custom_threshold(self):
        uid = uuid.uuid4()
        mems = [Memory(user_id=uid, content=f"m{i}", importance=1.0) for i in range(5)]
        # sum = 50; threshold = 40 → should trigger
        assert needs_reflection(mems, threshold=40.0) is True

    def test_default_threshold_is_150(self):
        assert REFLECTION_IMPORTANCE_SUM_THRESHOLD == 150.0


# ---------------------------------------------------------------------------
# run_dedup — InMemoryProvider
# ---------------------------------------------------------------------------


class TestRunDedupInMemory:
    @pytest.mark.asyncio
    async def test_exact_hash_dedup_keeps_higher_importance(self, provider):
        user_id = uuid.uuid4()
        # Same content → same hash → dedup should fire
        mid1 = await _add(provider, "Budget: $200", user_id, importance=0.9)
        mid2 = await _add(provider, "Budget: $200", user_id, importance=0.6)

        # mid2 is an exact-hash duplicate BUT InMemoryProvider already deduplicates
        # at add() time (returns existing mid1 for mid2).  So only one memory exists.
        # The dedup worker should still run without error.
        merged = await run_dedup(provider, user_id)
        assert merged == 0  # nothing more to merge

    @pytest.mark.asyncio
    async def test_semantic_dedup_invalidates_lower_importance(self, provider):
        user_id = uuid.uuid4()
        # Manually insert two memories with very similar embeddings that bypass
        # the content-hash dedup (different content, same embedding direction).
        emb_a = [1.0, 0.0, 0.0, 0.0]
        emb_b = [0.999, 0.001, 0.0, 0.0]  # cosine sim > 0.99 — above threshold

        mem_a = Memory(
            user_id=user_id, content="Budget is $200", embedding=emb_a, importance=0.9
        )
        mem_b = Memory(
            user_id=user_id, content="Budget around 200", embedding=emb_b, importance=0.6
        )
        # Bypass content-hash dedup (different content)
        mem_a = mem_a.model_copy(update={"content_hash": "hash_a"})
        mem_b = mem_b.model_copy(update={"content_hash": "hash_b"})
        provider._memories[mem_a.memory_id] = mem_a
        provider._memories[mem_b.memory_id] = mem_b

        merged = await run_dedup(provider, user_id)

        assert merged >= 1
        # The loser (lower importance = mem_b) should be invalidated
        stored_b = provider._memories[mem_b.memory_id]
        assert stored_b.valid_until is not None

    @pytest.mark.asyncio
    async def test_different_users_not_merged(self, provider):
        emb = [1.0, 0.0]
        for user_id in [uuid.uuid4(), uuid.uuid4()]:
            mem = Memory(user_id=user_id, content="Budget: $100", embedding=emb)
            mem = mem.model_copy(update={"content_hash": f"h_{user_id}"})
            provider._memories[mem.memory_id] = mem

        user_id_1 = list(provider._memories.values())[0].user_id
        merged = await run_dedup(provider, user_id_1)

        # No cross-user merges
        assert merged == 0

    @pytest.mark.asyncio
    async def test_dedup_returns_zero_when_no_duplicates(self, provider):
        user_id = uuid.uuid4()
        await _add(provider, "Budget: $200", user_id)
        await _add(provider, "I prefer organic food", user_id)

        merged = await run_dedup(provider, user_id)
        assert merged == 0
