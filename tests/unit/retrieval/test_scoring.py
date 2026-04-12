from __future__ import annotations

import asyncio
import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from mnemosyne.db.models.memory import Memory
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.retrieval.scoring import MultiSignalScorer, ScoringWeights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = uuid.uuid4()


def _make_memory(**kwargs) -> Memory:
    """Return a Memory with sensible defaults; override via kwargs."""
    defaults = dict(
        user_id=_USER_ID,
        content="test memory",
        embedding=None,
        importance=0.5,
        access_count=0,
        last_accessed=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Memory(**defaults)


def _embed_sync(text: str) -> list[float]:
    """Run FakeEmbeddingClient.embed synchronously for use in tests."""
    client = FakeEmbeddingClient(dim=32)
    return asyncio.run(client.embed(text))


# ---------------------------------------------------------------------------
# ScoringWeights tests
# ---------------------------------------------------------------------------


class TestScoringWeights:
    def test_default_weights_sum_to_one(self) -> None:
        weights = ScoringWeights()
        total = weights.relevance + weights.recency + weights.importance + weights.frequency
        assert abs(total - 1.0) < 0.01

    def test_custom_weights_valid(self) -> None:
        weights = ScoringWeights(relevance=0.4, recency=0.3, importance=0.2, frequency=0.1)
        total = weights.relevance + weights.recency + weights.importance + weights.frequency
        assert abs(total - 1.0) < 0.01

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            ScoringWeights(relevance=0.9, recency=0.9, importance=0.1, frequency=0.1)

    def test_weights_are_frozen(self) -> None:
        weights = ScoringWeights()
        with pytest.raises(Exception):
            weights.relevance = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MultiSignalScorer tests
# ---------------------------------------------------------------------------


class TestMultiSignalScorer:
    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    # -- relevance signal ----------------------------------------------------

    def test_relevance_signal(self) -> None:
        """Memory whose embedding matches the query should score higher on relevance."""
        query = _embed_sync("coffee preferences")

        # A memory that should embed closer to the query
        close_vec = _embed_sync("coffee preferences")
        far_vec = _embed_sync("completely unrelated random gibberish xyz")

        memory_close = _make_memory(embedding=close_vec)
        memory_far = _make_memory(embedding=far_vec)

        # Use relevance-only weights to isolate the signal
        weights = ScoringWeights(relevance=1.0, recency=0.0, importance=0.0, frequency=0.0)
        scorer = MultiSignalScorer(weights=weights)
        now = self._now()

        score_close, breakdown_close = scorer.score(memory_close, query, now)
        score_far, breakdown_far = scorer.score(memory_far, query, now)

        assert breakdown_close["relevance"] > breakdown_far["relevance"]
        assert score_close > score_far

    def test_relevance_identical_embeddings_gives_one(self) -> None:
        vec = _embed_sync("hello world")
        memory = _make_memory(embedding=vec)
        weights = ScoringWeights(relevance=1.0, recency=0.0, importance=0.0, frequency=0.0)
        scorer = MultiSignalScorer(weights=weights)
        _, breakdown = scorer.score(memory, vec, self._now())
        assert abs(breakdown["relevance"] - 1.0) < 1e-6

    def test_relevance_empty_embedding_returns_zero(self) -> None:
        memory = _make_memory(embedding=None)
        scorer = MultiSignalScorer()
        _, breakdown = scorer.score(memory, _embed_sync("test"), self._now())
        assert breakdown["relevance"] == 0.0

    # -- recency signal ------------------------------------------------------

    def test_recency_signal(self) -> None:
        """Memory accessed more recently should have a higher recency score."""
        now = self._now()
        recent = _make_memory(last_accessed=now - timedelta(hours=1))
        old = _make_memory(last_accessed=now - timedelta(days=90))

        weights = ScoringWeights(relevance=0.0, recency=1.0, importance=0.0, frequency=0.0)
        scorer = MultiSignalScorer(weights=weights)
        query = _embed_sync("anything")

        _, breakdown_recent = scorer.score(recent, query, now)
        _, breakdown_old = scorer.score(old, query, now)

        assert breakdown_recent["recency"] > breakdown_old["recency"]

    def test_recency_just_accessed_approaches_one(self) -> None:
        now = self._now()
        memory = _make_memory(last_accessed=now)
        scorer = MultiSignalScorer()
        _, breakdown = scorer.score(memory, _embed_sync("x"), now)
        # exp(-0.01 * 0) = 1.0
        assert abs(breakdown["recency"] - 1.0) < 1e-6

    def test_recency_naive_datetime_handled(self) -> None:
        """Scorer must not raise when last_accessed or now lacks tzinfo."""
        naive_now = datetime.utcnow()
        naive_accessed = naive_now - timedelta(days=1)
        memory = _make_memory(last_accessed=naive_accessed)
        scorer = MultiSignalScorer()
        # Should not raise; just compute
        total, breakdown = scorer.score(memory, _embed_sync("x"), naive_now)
        assert math.isfinite(total)
        assert breakdown["recency"] > 0.0

    # -- importance signal ---------------------------------------------------

    def test_importance_signal(self) -> None:
        """Higher stored importance should yield a higher importance signal."""
        now = self._now()
        query = _embed_sync("x")
        high = _make_memory(importance=0.9)
        low = _make_memory(importance=0.1)

        weights = ScoringWeights(relevance=0.0, recency=0.0, importance=1.0, frequency=0.0)
        scorer = MultiSignalScorer(weights=weights)

        _, breakdown_high = scorer.score(high, query, now)
        _, breakdown_low = scorer.score(low, query, now)

        assert breakdown_high["importance"] > breakdown_low["importance"]
        assert abs(breakdown_high["importance"] - 0.9) < 1e-9
        assert abs(breakdown_low["importance"] - 0.1) < 1e-9

    # -- frequency signal ----------------------------------------------------

    def test_frequency_signal(self) -> None:
        """Memory with higher access_count should score higher on frequency."""
        now = self._now()
        query = _embed_sync("x")
        frequent = _make_memory(access_count=50)
        rare = _make_memory(access_count=1)

        weights = ScoringWeights(relevance=0.0, recency=0.0, importance=0.0, frequency=1.0)
        scorer = MultiSignalScorer(weights=weights)

        _, breakdown_frequent = scorer.score(frequent, query, now)
        _, breakdown_rare = scorer.score(rare, query, now)

        assert breakdown_frequent["frequency"] > breakdown_rare["frequency"]

    def test_frequency_zero_access_count(self) -> None:
        memory = _make_memory(access_count=0)
        scorer = MultiSignalScorer()
        _, breakdown = scorer.score(memory, _embed_sync("x"), self._now())
        # log1p(0) / log1p(100) == 0
        assert breakdown["frequency"] == 0.0

    def test_frequency_capped_at_100(self) -> None:
        """access_count >= 100 should produce frequency near or at 1.0."""
        memory_100 = _make_memory(access_count=100)
        memory_200 = _make_memory(access_count=200)
        scorer = MultiSignalScorer()
        now = self._now()
        query = _embed_sync("x")
        _, b100 = scorer.score(memory_100, query, now)
        _, b200 = scorer.score(memory_200, query, now)
        assert abs(b100["frequency"] - 1.0) < 1e-9
        # Above the cap the signal can exceed 1 but should be finite
        assert b200["frequency"] > 1.0 or b200["frequency"] >= 1.0

    # -- weight configuration ------------------------------------------------

    def test_weight_configuration_relevance_only(self) -> None:
        """When relevance=1.0 and others=0.0, total must equal the relevance signal."""
        query = _embed_sync("some query text")
        memory_vec = _embed_sync("some query text")
        memory = _make_memory(embedding=memory_vec, importance=0.8, access_count=10)

        weights = ScoringWeights(relevance=1.0, recency=0.0, importance=0.0, frequency=0.0)
        scorer = MultiSignalScorer(weights=weights)
        now = self._now()

        total, breakdown = scorer.score(memory, query, now)
        assert abs(total - breakdown["relevance"]) < 1e-9

    def test_weight_configuration_importance_only(self) -> None:
        memory = _make_memory(importance=0.75, access_count=20)
        weights = ScoringWeights(relevance=0.0, recency=0.0, importance=1.0, frequency=0.0)
        scorer = MultiSignalScorer(weights=weights)
        total, breakdown = scorer.score(memory, _embed_sync("q"), self._now())
        assert abs(total - breakdown["importance"]) < 1e-9
        assert abs(total - 0.75) < 1e-9

    # -- combined scoring ----------------------------------------------------

    def test_combined_scoring_is_weighted_sum(self) -> None:
        """total must equal w_r*r + w_e*e + w_i*i + w_f*f exactly."""
        weights = ScoringWeights(relevance=0.5, recency=0.2, importance=0.2, frequency=0.1)
        scorer = MultiSignalScorer(weights=weights)

        query = _embed_sync("combined test")
        now = self._now()
        memory = _make_memory(
            embedding=_embed_sync("combined test"),
            importance=0.6,
            access_count=5,
            last_accessed=now - timedelta(days=3),
        )

        total, breakdown = scorer.score(memory, query, now)

        expected = (
            weights.relevance * breakdown["relevance"]
            + weights.recency * breakdown["recency"]
            + weights.importance * breakdown["importance"]
            + weights.frequency * breakdown["frequency"]
        )
        assert abs(total - expected) < 1e-9

    def test_combined_scoring_all_signals_present(self) -> None:
        """breakdown dict must contain all four keys."""
        scorer = MultiSignalScorer()
        memory = _make_memory()
        _, breakdown = scorer.score(memory, _embed_sync("test"), self._now())
        assert set(breakdown.keys()) == {"relevance", "recency", "importance", "frequency"}

    def test_default_scorer_uses_default_weights(self) -> None:
        scorer = MultiSignalScorer()
        assert scorer.weights == ScoringWeights()
