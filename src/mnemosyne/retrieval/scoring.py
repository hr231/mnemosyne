from __future__ import annotations

import math
from datetime import datetime, timezone

from pydantic import BaseModel, model_validator

from mnemosyne.db.models.memory import Memory


class ScoreBreakdown(BaseModel):
    """Per-signal contribution explanation for a single scored memory.

    ``weights`` mirrors the ``ScoringWeights`` used at scoring time so the
    breakdown is self-describing: a consumer can recompute ``raw_total`` from
    the four components and the weights without round-tripping into config.
    """

    relevance: float
    recency: float
    importance: float
    frequency: float
    weights: dict[str, float]
    raw_total: float
    final_score: float


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
        explain: bool = False,
        *,
        relevance: float | None = None,
        query_norm: float | None = None,
    ) -> tuple[float, dict[str, float]] | tuple[float, ScoreBreakdown]:
        """Return ``(total, breakdown)`` for *memory*.

        When ``explain=False`` (the default) the second element is the
        legacy ``dict[str, float]`` of raw signal values. When ``explain=True``
        it is a :class:`ScoreBreakdown` carrying the four components, the
        weights used, and both raw and final totals.

        ``relevance`` may be supplied precomputed (e.g. from the DB-side
        ``embedding <=> query`` cosine distance) to skip the Python cosine
        recomputation. ``query_norm`` lets the caller precompute the query
        vector's L2 norm once and reuse it across candidates.
        """
        # 1. Relevance: cosine similarity between query and stored embedding.
        #    Clamp to [0, 1] — raw cosine can be negative and the DB leg may
        #    hand back a value at the boundary.
        if relevance is None:
            relevance = self._cosine_sim(
                query_embedding, memory.embedding or [], query_norm=query_norm
            )
        relevance = max(0.0, min(1.0, relevance))

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

        # 4. Frequency: log-scaled access count normalised against cap of 100,
        #    clamped so counts above the cap cannot exceed 1.0.
        frequency = min(1.0, math.log1p(memory.access_count) / math.log1p(100))

        total = (
            self.weights.relevance * relevance
            + self.weights.recency * recency
            + self.weights.importance * importance
            + self.weights.frequency * frequency
        )

        if not explain:
            return total, {
                "relevance": relevance,
                "recency": recency,
                "importance": importance,
                "frequency": frequency,
            }

        breakdown = ScoreBreakdown(
            relevance=relevance,
            recency=recency,
            importance=importance,
            frequency=frequency,
            weights={
                "relevance": self.weights.relevance,
                "recency": self.weights.recency,
                "importance": self.weights.importance,
                "frequency": self.weights.frequency,
            },
            raw_total=total,
            final_score=total,
        )
        return total, breakdown

    @staticmethod
    def _cosine_sim(
        a: list[float], b: list[float], query_norm: float | None = None
    ) -> float:
        """Return the cosine similarity between vectors *a* and *b*.

        Returns 0.0 for empty or zero-magnitude vectors. Raises ``ValueError``
        when the two vectors have different lengths — a dimension mismatch is a
        configuration error, not a similarity worth silently truncating.

        ``query_norm`` may carry the precomputed L2 norm of *a* so a caller
        scoring many candidates against one query does not recompute it.
        """
        if not a or not b:
            return 0.0
        if len(a) != len(b):
            raise ValueError(
                f"embedding dimension mismatch: {len(a)} vs {len(b)}"
            )
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = query_norm if query_norm is not None else math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
