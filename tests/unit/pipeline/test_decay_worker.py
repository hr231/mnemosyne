"""Unit tests for the decay worker."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne.db.models.memory import Memory
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.pipeline.decay import (
    apply_decay,
    compute_decayed_importance,
    should_archive,
)
from mnemosyne.providers.in_memory import InMemoryProvider


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient(dim=768)


async def _add(provider, content, user_id, importance=0.8, days_ago=0):
    embedding = await FakeEmbeddingClient(dim=768).embed(content)
    last_accessed = datetime.now(timezone.utc) - timedelta(days=days_ago)
    mem = Memory(
        user_id=user_id,
        content=content,
        embedding=embedding,
        importance=importance,
        last_accessed=last_accessed,
    )
    mid = await provider.add(mem)
    # Set last_accessed directly because add() doesn't update it
    provider._memories[mid].last_accessed = last_accessed
    return mid


# ---------------------------------------------------------------------------
# compute_decayed_importance
# ---------------------------------------------------------------------------


class TestComputeDecayedImportance:
    def test_recent_memory_barely_decays(self):
        mem = Memory(
            user_id=uuid.uuid4(),
            content="recent",
            importance=0.8,
            last_accessed=datetime.now(timezone.utc),
            decay_rate=0.01,
        )
        new_imp = compute_decayed_importance(mem)
        assert new_imp >= 0.78  # effectively no change

    def test_old_memory_decays_significantly(self):
        mem = Memory(
            user_id=uuid.uuid4(),
            content="old",
            importance=0.8,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=90),
            decay_rate=0.01,
        )
        new_imp = compute_decayed_importance(mem)
        assert new_imp < 0.7

    def test_zero_decay_rate_no_change(self):
        mem = Memory(
            user_id=uuid.uuid4(),
            content="stable",
            importance=0.8,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=365),
            decay_rate=0.0,
        )
        assert compute_decayed_importance(mem) == pytest.approx(0.8)

    def test_result_clamped_to_non_negative(self):
        mem = Memory(
            user_id=uuid.uuid4(),
            content="extreme",
            importance=0.1,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=1000),
            decay_rate=1.0,
        )
        assert compute_decayed_importance(mem) >= 0.0


# ---------------------------------------------------------------------------
# should_archive
# ---------------------------------------------------------------------------


class TestShouldArchive:
    def test_low_importance_old_memory_archived(self):
        mem = Memory(
            user_id=uuid.uuid4(),
            content="forgotten",
            importance=0.03,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=100),
        )
        assert should_archive(mem) is True

    def test_high_importance_recent_memory_not_archived(self):
        mem = Memory(
            user_id=uuid.uuid4(),
            content="important",
            importance=0.9,
            last_accessed=datetime.now(timezone.utc),
        )
        assert should_archive(mem) is False

    def test_low_importance_recent_memory_not_archived(self):
        # Below threshold but recently accessed — not yet eligible
        mem = Memory(
            user_id=uuid.uuid4(),
            content="low but recent",
            importance=0.02,
            last_accessed=datetime.now(timezone.utc),
        )
        assert should_archive(mem) is False

    def test_already_invalidated_never_archived(self):
        mem = Memory(
            user_id=uuid.uuid4(),
            content="invalidated",
            importance=0.01,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=200),
            valid_until=datetime.now(timezone.utc),
        )
        assert should_archive(mem) is False

    def test_custom_thresholds(self):
        mem = Memory(
            user_id=uuid.uuid4(),
            content="custom",
            importance=0.1,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=10),
        )
        # With threshold=0.2 and archive_after_days=5, this should archive
        assert should_archive(mem, archive_threshold=0.2, archive_after_days=5) is True
        # With default thresholds it should not
        assert should_archive(mem) is False


# ---------------------------------------------------------------------------
# apply_decay
# ---------------------------------------------------------------------------


class TestApplyDecay:
    @pytest.mark.asyncio
    async def test_importance_decreases_for_old_memory(self, provider):
        user_id = uuid.uuid4()
        mid = await _add(provider, "old memory", user_id, importance=0.8, days_ago=100)
        original = provider._memories[mid].importance

        stats = await apply_decay(provider, user_id)

        updated = provider._memories[mid].importance
        assert updated < original
        assert stats["decayed"] >= 1

    @pytest.mark.asyncio
    async def test_recent_memory_not_significantly_changed(self, provider):
        user_id = uuid.uuid4()
        mid = await _add(provider, "recent memory", user_id, importance=0.8, days_ago=0)
        original_importance = provider._memories[mid].importance

        await apply_decay(provider, user_id)

        updated = provider._memories[mid].importance
        # Very small or no change for a brand-new memory
        assert abs(updated - original_importance) < 0.01

    @pytest.mark.asyncio
    async def test_archives_stale_low_importance_memory(self, provider):
        user_id = uuid.uuid4()
        mid = await _add(provider, "stale memory", user_id, importance=0.03, days_ago=100)

        await apply_decay(provider, user_id, archive_threshold=0.05, archive_after_days=90)

        mem = provider._memories[mid]
        assert mem.metadata.get("archived") is True

    @pytest.mark.asyncio
    async def test_dry_run_makes_no_changes(self, provider):
        user_id = uuid.uuid4()
        mid = await _add(provider, "dry run target", user_id, importance=0.8, days_ago=100)
        original = provider._memories[mid].importance

        stats = await apply_decay(provider, user_id, dry_run=True)

        assert provider._memories[mid].importance == original
        assert stats["decayed"] >= 1  # counted but not applied

    @pytest.mark.asyncio
    async def test_returns_decay_and_archive_counts(self, provider):
        user_id = uuid.uuid4()
        await _add(provider, "memory A", user_id, importance=0.8, days_ago=100)
        await _add(provider, "memory B", user_id, importance=0.03, days_ago=200)

        stats = await apply_decay(provider, user_id, archive_threshold=0.05, archive_after_days=90)

        assert "decayed" in stats
        assert "archived" in stats
        assert stats["decayed"] >= 1
        assert stats["archived"] >= 1

    @pytest.mark.asyncio
    async def test_user_isolation(self, provider):
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        mid_a = await _add(provider, "user A memory", user_a, importance=0.8, days_ago=100)
        mid_b = await _add(provider, "user B memory", user_b, importance=0.8, days_ago=100)

        await apply_decay(provider, user_id=user_a)

        imp_a = provider._memories[mid_a].importance
        imp_b = provider._memories[mid_b].importance
        assert imp_a < 0.8      # user A memory decayed
        assert imp_b == pytest.approx(0.8)  # user B memory untouched

    @pytest.mark.asyncio
    async def test_empty_provider_returns_zero_stats(self, provider):
        stats = await apply_decay(provider, uuid.uuid4())
        assert stats["decayed"] == 0
        assert stats["archived"] == 0
