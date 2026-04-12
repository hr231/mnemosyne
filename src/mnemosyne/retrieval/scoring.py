from __future__ import annotations

import math
from datetime import datetime, timezone

from pydantic import BaseModel, model_validator

from mnemosyne.db.models.memory import Memory


class ScoringWeights(BaseModel, frozen=True):
    """Per-deployment configurable weights for the four retrieval signals.

    All four weights must sum to approximately 1.0 (within ±0.01 tolerance
    to accommodate floating-point representation).
    """

    relevance: float = 0.5
    recency: float = 0.2
    importance: float = 0.2
    frequency: float = 0.1

    @model_validator(mode="after")
    def _sum_to_one(self) -> "ScoringWeights":
        s = self.relevance + self.recency + self.importance + self.frequency
        if not 0.99 <= s <= 1.01:
            raise ValueError(f"weights must sum to ~1.0, got {s}")
        return self


class MultiSignalScorer:
    """Computes a weighted combination of four retrieval signals for a memory.

    Signals:
      - relevance:  cosine similarity between query and memory embeddings
      - recency:    exponential decay from last_accessed (lambda=0.01/day)
      - importance: raw importance value stored on the memory (0.0–1.0)
      - frequency:  log-scaled access count, normalised against a cap of 100

    Usage::

        scorer = MultiSignalScorer()
        total, breakdown = scorer.score(memory, query_embedding, now)
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score(
        self,
        memory: Memory,
        query_embedding: list[float],
        now: datetime,
    ) -> tuple[float, dict[str, float]]:
        """Return (total_score, per-signal breakdown) for *memory*.

        Args:
            memory: the Memory object to score
            query_embedding: the query vector (unit-normalised recommended)
            now: the reference datetime used to compute recency decay

        Returns:
            A tuple of (total weighted score, dict with individual signal values)
        """
        # 1. Relevance: cosine similarity between query and stored embedding
        relevance = self._cosine_sim(query_embedding, memory.embedding or [])

        # 2. Recency: exponential decay from last_accessed
        #    Make both datetimes timezone-aware for safe subtraction.
        last_accessed = memory.last_accessed
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        ref_now = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        days_since = (ref_now - last_accessed).total_seconds() / 86400.0
        recency = math.exp(-0.01 * days_since)

        # 3. Importance: raw value from memory (clamped to [0, 1] by model)
        importance = memory.importance

        # 4. Frequency: log-scaled access count normalised against cap of 100
        frequency = math.log1p(memory.access_count) / math.log1p(100)

        breakdown: dict[str, float] = {
            "relevance": relevance,
            "recency": recency,
            "importance": importance,
            "frequency": frequency,
        }

        total = (
            self.weights.relevance * relevance
            + self.weights.recency * recency
            + self.weights.importance * importance
            + self.weights.frequency * frequency
        )

        return total, breakdown

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Return the cosine similarity between vectors *a* and *b*.

        Returns 0.0 for empty or zero-magnitude vectors.
        """
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
