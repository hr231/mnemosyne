"""Tests for RuleRegistry."""
from __future__ import annotations

import pytest

from mnemosyne.db.models.memory import ExtractionResult, MemoryType
from mnemosyne.rules.base_extractor import BaseExtractor
from mnemosyne.rules.rule_registry import RuleRegistry


# ---------------------------------------------------------------------------
# Minimal stub extractor for testing
# ---------------------------------------------------------------------------


class StubRule(BaseExtractor):
    def __init__(self, rid: str, trigger: str, enabled: bool = True):
        self.rule_id = rid
        self.id = rid
        self.category = MemoryType.FACT
        self.importance = 0.5
        self.enabled = enabled
        self._trigger = trigger

    def extract(self, text: str) -> list[ExtractionResult]:
        if self._trigger in text:
            return [
                ExtractionResult(
                    content=f"matched {self._trigger}",
                    memory_type=MemoryType.FACT,
                    importance=0.5,
                    rule_id=self.rule_id,
                    matched_chars=len(self._trigger),
                )
            ]
        return []


class ErrorRule(BaseExtractor):
    """A rule that always raises — used to test isolation."""

    def __init__(self):
        self.rule_id = "error_rule"
        self.id = "error_rule"
        self.category = MemoryType.FACT
        self.importance = 0.5
        self.enabled = True

    def extract(self, text: str) -> list[ExtractionResult]:
        raise RuntimeError("intentional rule error")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterAndList:
    def test_register_and_list(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        reg.register(StubRule("r2", "bar"))
        assert len(reg.all()) == 2

    def test_duplicate_id_raises(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(StubRule("r1", "bar"))

    def test_register_all(self):
        reg = RuleRegistry()
        reg.register_all([StubRule("r1", "foo"), StubRule("r2", "bar")])
        assert len(reg.all()) == 2

    def test_get_by_id_found(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        found = reg.get("r1")
        assert found is not None
        assert found.rule_id == "r1"

    def test_get_by_id_not_found(self):
        reg = RuleRegistry()
        assert reg.get("nonexistent") is None

    def test_empty_registry_all_returns_empty_list(self):
        reg = RuleRegistry()
        assert reg.all() == []


class TestExtract:
    def test_extract_runs_all_rules(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        reg.register(StubRule("r2", "bar"))
        results = reg.extract("foo and bar")
        assert len(results) == 2
        assert {r.rule_id for r in results} == {"r1", "r2"}

    def test_extract_empty_text_returns_empty(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        assert reg.extract("") == []
        assert reg.extract("   ") == []

    def test_extract_no_match_returns_empty(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        assert reg.extract("nothing here") == []

    def test_disabled_rule_is_skipped(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo", enabled=False))
        results = reg.extract("foo")
        assert results == []

    def test_disabled_rule_does_not_affect_enabled_rules(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo", enabled=False))
        reg.register(StubRule("r2", "bar", enabled=True))
        results = reg.extract("foo and bar")
        assert len(results) == 1
        assert results[0].rule_id == "r2"

    def test_error_in_one_rule_does_not_stop_others(self):
        reg = RuleRegistry()
        reg.register(ErrorRule())
        reg.register(StubRule("r2", "bar"))
        # Should not raise, and r2 should still fire
        results = reg.extract("bar")
        assert len(results) == 1
        assert results[0].rule_id == "r2"

    def test_empty_registry_extract_returns_empty(self):
        reg = RuleRegistry()
        assert reg.extract("anything") == []
