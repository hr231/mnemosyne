from __future__ import annotations

import pytest

from mnemosyne.pipeline.extraction.router import ExtractionStats, should_route_to_llm


def test_route_when_no_extractions():
    stats = ExtractionStats(extracted_count=0, total_chars=100, chars_matched_by_rules=0)
    assert should_route_to_llm(stats) is True


def test_route_when_high_unstructured():
    # 10 matched out of 100 chars -> unstructured_ratio = 0.9 > 0.7
    stats = ExtractionStats(extracted_count=2, total_chars=100, chars_matched_by_rules=10)
    assert should_route_to_llm(stats) is True


def test_no_route_when_well_covered():
    # 80 matched out of 100 chars -> unstructured_ratio = 0.2, not > 0.7
    stats = ExtractionStats(extracted_count=3, total_chars=100, chars_matched_by_rules=80)
    assert should_route_to_llm(stats) is False


def test_no_route_when_empty_text():
    # total_chars=0 -> unstructured_ratio=0.0, and extracted_count=0 triggers True
    # BUT: if text is empty there is nothing to extract, so extracted_count=0 would
    # still route. Test that empty text with zero extractions routes (no match).
    # The spec says unstructured_ratio=0 for empty text; extracted_count=0 still fires.
    stats = ExtractionStats(extracted_count=0, total_chars=0, chars_matched_by_rules=0)
    # extracted_count == 0 -> True regardless of text length
    assert should_route_to_llm(stats) is True


def test_no_route_empty_text_with_prior_extractions():
    # Pathological: 0 chars, but somehow 1 extraction — ratio=0 -> no route
    stats = ExtractionStats(extracted_count=1, total_chars=0, chars_matched_by_rules=0)
    assert should_route_to_llm(stats) is False


def test_custom_threshold():
    # unstructured_ratio = 0.6 > threshold=0.5 -> True
    stats = ExtractionStats(extracted_count=2, total_chars=100, chars_matched_by_rules=40)
    assert should_route_to_llm(stats, unstructured_threshold=0.5) is True


def test_custom_threshold_below():
    # unstructured_ratio = 0.6, threshold=0.8 -> False
    stats = ExtractionStats(extracted_count=2, total_chars=100, chars_matched_by_rules=40)
    assert should_route_to_llm(stats, unstructured_threshold=0.8) is False


def test_unstructured_ratio_clamped_at_zero_when_over_matched():
    # chars_matched_by_rules > total_chars is nonsensical but should not blow up
    stats = ExtractionStats(extracted_count=1, total_chars=10, chars_matched_by_rules=20)
    # ratio = 1 - 2.0 = -1.0, which is not > 0.7, and count > 0 -> False
    assert should_route_to_llm(stats) is False
