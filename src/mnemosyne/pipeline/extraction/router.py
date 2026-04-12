from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractionStats:
    extracted_count: int
    total_chars: int
    chars_matched_by_rules: int

    @property
    def unstructured_ratio(self) -> float:
        if self.total_chars == 0:
            return 0.0
        return 1.0 - (self.chars_matched_by_rules / self.total_chars)


def should_route_to_llm(
    stats: ExtractionStats,
    unstructured_threshold: float = 0.7,
) -> bool:
    """Return True if the text should be escalated to LLM extraction.

    Two signals (OR gate):
    - extracted_count == 0 (rules found nothing)
    - unstructured_ratio > threshold (most text was not covered by rules)
    """
    if stats.extracted_count == 0:
        return True
    if stats.unstructured_ratio > unstructured_threshold:
        return True
    return False
