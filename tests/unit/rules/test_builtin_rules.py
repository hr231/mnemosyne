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
# Size rules
# ---------------------------------------------------------------------------


class TestSizeRules:
    def test_wear_size_number(self, registry):
        results = registry.extract("I wear size 10")
        assert any("10" in r.content for r in results), (
            f"Expected '10' in some result content, got: {[r.content for r in results]}"
        )

    def test_wear_size_letter(self, registry):
        results = registry.extract("I wear size M")
        assert any("M" in r.content for r in results), (
            f"Expected 'M' in some result content, got: {[r.content for r in results]}"
        )

    def test_my_size_is(self, registry):
        results = registry.extract("my size is M")
        assert any("M" in r.content for r in results), (
            f"Expected 'M' in some result content, got: {[r.content for r in results]}"
        )

    def test_size_type_is_fact(self, registry):
        results = registry.extract("I wear size 10")
        size_results = [r for r in results if "10" in r.content]
        assert all(r.memory_type == MemoryType.FACT for r in size_results)

    def test_size_importance_is_high(self, registry):
        results = registry.extract("I wear size M")
        size_results = [r for r in results if "M" in r.content]
        assert all(r.importance >= 0.8 for r in size_results)


# ---------------------------------------------------------------------------
# Keyword trigger rules
# ---------------------------------------------------------------------------


class TestKeywordTriggerRules:
    def test_allergic_to_latex(self, registry):
        results = registry.extract("I'm allergic to latex")
        assert len(results) >= 1, f"Expected at least 1 result, got: {results}"
        allergic_results = [r for r in results if "allergic" in r.content.lower()]
        assert len(allergic_results) >= 1

    def test_allergic_importance_is_very_high(self, registry):
        results = registry.extract("I'm allergic to nuts")
        allergic_results = [r for r in results if "allergic" in r.content.lower()]
        assert all(r.importance >= 0.9 for r in allergic_results)

    def test_remember_keyword(self, registry):
        results = registry.extract("Please remember I take the train to work")
        assert any("remember" in r.content.lower() for r in results), (
            f"Expected 'remember' keyword result, got: {[r.content for r in results]}"
        )

    def test_always_keyword_is_procedural(self, registry):
        results = registry.extract("I always check reviews before buying")
        always_results = [r for r in results if "always" in r.content.lower()]
        assert len(always_results) >= 1
        assert all(r.memory_type == MemoryType.PROCEDURAL for r in always_results)

    def test_never_keyword_is_procedural(self, registry):
        results = registry.extract("I never buy without reading the reviews first")
        never_results = [r for r in results if "never" in r.content.lower()]
        assert len(never_results) >= 1
        assert all(r.memory_type == MemoryType.PROCEDURAL for r in never_results)

    def test_important_keyword(self, registry):
        results = registry.extract("It's important that you remember my dietary needs")
        assert any("important" in r.content.lower() for r in results), (
            f"Expected 'important' keyword result, got: {[r.content for r in results]}"
        )

    def test_important_importance_high(self, registry):
        results = registry.extract("This is important: I am vegan")
        important_results = [r for r in results if "important" in r.content.lower()]
        assert all(r.importance >= 0.8 for r in important_results)


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
