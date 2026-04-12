from __future__ import annotations

from pydantic import BaseModel, model_validator


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
