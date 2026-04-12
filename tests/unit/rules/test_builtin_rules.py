"""Tests for builtin YAML rules shipped in rules/core/."""
from __future__ import annotations

from pathlib import Path

import pytest

from mnemosyne.db.models.memory import MemoryType
from mnemosyne.rules.rule_loader import RuleLoader
from mnemosyne.rules.rule_registry import RuleRegistry

# Resolve the rules/core directory relative to the project root
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_BUILTIN_DIR = _REPO_ROOT / "rules" / "core"


@pytest.fixture(scope="module")
def registry() -> RuleRegistry:
    loader = RuleLoader()
    extractors = loader.load_from_directory(_BUILTIN_DIR)
    reg = RuleRegistry()
    reg.register_all(extractors)
    return reg


# ---------------------------------------------------------------------------
# Budget rules
# ---------------------------------------------------------------------------


class TestBudgetRules:
    def test_dollar_amount_my_budget(self, registry):
        results = registry.extract("My budget is $200")
        assert any("200" in r.content for r in results), (
            f"Expected '200' in some result content, got: {[r.content for r in results]}"
        )

    def test_budget_under_amount(self, registry):
        results = registry.extract("I want something under $150")
        assert any("150" in r.content for r in results), (
            f"Expected '150' in some result, got: {[r.content for r in results]}"
        )

    def test_spend_around(self, registry):
        results = registry.extract("Looking to spend around $500")
        assert any("500" in r.content for r in results), (
            f"Expected '500' in some result, got: {[r.content for r in results]}"
        )

    def test_budget_memory_type_is_fact(self, registry):
        results = registry.extract("My budget is $300")
        budget_results = [r for r in results if "300" in r.content]
        assert all(r.memory_type == MemoryType.FACT for r in budget_results)

    def test_budget_importance_is_high(self, registry):
        results = registry.extract("My budget is $200")
        budget_results = [r for r in results if "200" in r.content]
        assert all(r.importance >= 0.7 for r in budget_results)


# ---------------------------------------------------------------------------
# Preference rules
# ---------------------------------------------------------------------------


class TestPreferenceRules:
    def test_prefer_organic(self, registry):
        results = registry.extract("I prefer organic products.")
        assert any(
            r.memory_type == MemoryType.PREFERENCE and "organic" in r.content.lower()
            for r in results
        ), f"Got: {[r.content for r in results]}"

    def test_like_minimalist(self, registry):
        results = registry.extract("I really like minimalist design.")
        assert any(r.memory_type == MemoryType.PREFERENCE for r in results), (
            f"Got: {[r.content for r in results]}"
        )

    def test_dont_like_synthetic(self, registry):
        results = registry.extract("I don't like synthetic materials.")
        assert any(
            r.memory_type == MemoryType.PREFERENCE and "synthetic" in r.content.lower()
            for r in results
        ), f"Got: {[r.content for r in results]}"

    def test_preference_importance(self, registry):
        results = registry.extract("I prefer organic products.")
        pref_results = [r for r in results if r.memory_type == MemoryType.PREFERENCE]
        assert all(r.importance >= 0.6 for r in pref_results)


# ---------------------------------------------------------------------------
# No extraction cases
# ---------------------------------------------------------------------------


class TestNoExtraction:
    def test_greeting_produces_no_results(self, registry):
        results = registry.extract("hi")
        assert results == [], f"Expected no results for 'hi', got: {results}"

    def test_thanks_produces_no_results(self, registry):
        results = registry.extract("thanks")
        assert results == [], f"Expected no results for 'thanks', got: {results}"

    def test_empty_string_produces_no_results(self, registry):
        assert registry.extract("") == []
