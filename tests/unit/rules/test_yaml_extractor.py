"""Tests for YamlRuleExtractor and load_yaml_rules."""
from __future__ import annotations

import pytest

from mnemosyne.db.models.memory import MemoryType
from mnemosyne.rules.yaml_extractor import YamlRuleExtractor, load_yaml_rules


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _regex_rule(
    rule_id: str = "test_regex",
    pattern: str = r"budget is \$(\d+)",
    template: str = "Budget: $${1}",
    importance: float = 0.8,
    enabled: bool = True,
) -> dict:
    return {
        "id": rule_id,
        "category": "fact",
        "type": "regex",
        "pattern": pattern,
        "template": template,
        "importance": importance,
        "enabled": enabled,
    }


def _keyword_rule(
    rule_id: str = "test_keyword",
    keywords: list[str] | None = None,
    importance: float = 0.6,
    enabled: bool = True,
) -> dict:
    return {
        "id": rule_id,
        "category": "preference",
        "type": "keyword",
        "keywords": keywords or ["organic"],
        "importance": importance,
        "enabled": enabled,
    }


def _keyword_context_rule(
    rule_id: str = "test_kw_ctx",
    keywords: list[str] | None = None,
    importance: float = 0.6,
    enabled: bool = True,
) -> dict:
    return {
        "id": rule_id,
        "category": "preference",
        "type": "keyword_context",
        "keywords": keywords or ["prefer"],
        "importance": importance,
        "enabled": enabled,
    }


# ---------------------------------------------------------------------------
# Regex extractor
# ---------------------------------------------------------------------------


class TestRegexExtractor:
    def test_match_returns_result(self):
        extractor = YamlRuleExtractor(_regex_rule())
        results = extractor.extract("My budget is $200")
        assert len(results) == 1
        assert "200" in results[0].content

    def test_no_match_returns_empty(self):
        extractor = YamlRuleExtractor(_regex_rule())
        results = extractor.extract("I like apples")
        assert results == []

    def test_template_group_interpolation(self):
        rule = _regex_rule(pattern=r"size (\w+)", template="Size: ${1}")
        extractor = YamlRuleExtractor(rule)
        results = extractor.extract("I wear size 10")
        assert len(results) >= 1
        assert "10" in results[0].content

    def test_multiple_matches(self):
        rule = _regex_rule(
            pattern=r"\$(\d+)",
            template="Amount: $${1}",
        )
        extractor = YamlRuleExtractor(rule)
        results = extractor.extract("Pay $100 now and $50 later")
        assert len(results) == 2

    def test_memory_type_is_set(self):
        extractor = YamlRuleExtractor(_regex_rule())
        results = extractor.extract("My budget is $200")
        assert results[0].memory_type == MemoryType.FACT

    def test_rule_id_stamped_on_result(self):
        extractor = YamlRuleExtractor(_regex_rule(rule_id="budget_rule"))
        results = extractor.extract("My budget is $200")
        assert results[0].rule_id == "budget_rule"

    def test_importance_propagated(self):
        extractor = YamlRuleExtractor(_regex_rule(importance=0.9))
        results = extractor.extract("My budget is $200")
        assert results[0].importance == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Keyword extractor
# ---------------------------------------------------------------------------


class TestKeywordExtractor:
    def test_keyword_present_returns_result(self):
        extractor = YamlRuleExtractor(_keyword_rule(keywords=["organic"]))
        results = extractor.extract("I prefer organic products")
        assert len(results) == 1
        assert "organic" in results[0].content

    def test_keyword_absent_returns_empty(self):
        extractor = YamlRuleExtractor(_keyword_rule(keywords=["organic"]))
        results = extractor.extract("I like everything")
        assert results == []

    def test_keyword_case_insensitive(self):
        extractor = YamlRuleExtractor(_keyword_rule(keywords=["organic"]))
        results = extractor.extract("I buy ORGANIC food")
        assert len(results) == 1

    def test_multiple_keywords_each_match_separately(self):
        extractor = YamlRuleExtractor(_keyword_rule(keywords=["organic", "natural"]))
        results = extractor.extract("I prefer organic and natural foods")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Keyword-context extractor
# ---------------------------------------------------------------------------


class TestKeywordContextExtractor:
    def test_extracts_containing_sentence(self):
        extractor = YamlRuleExtractor(_keyword_context_rule(keywords=["prefer"]))
        results = extractor.extract("I prefer organic products. Also I like coffee.")
        assert len(results) == 1
        assert "prefer" in results[0].content.lower()

    def test_no_match_returns_empty(self):
        extractor = YamlRuleExtractor(_keyword_context_rule(keywords=["prefer"]))
        results = extractor.extract("Nothing relevant here.")
        assert results == []

    def test_sentence_content_not_just_keyword(self):
        extractor = YamlRuleExtractor(_keyword_context_rule(keywords=["prefer"]))
        results = extractor.extract("I really prefer dark chocolate.")
        assert len(results[0].content) > len("prefer")


# ---------------------------------------------------------------------------
# Disabled rule
# ---------------------------------------------------------------------------


class TestDisabledRule:
    def test_disabled_regex_returns_empty(self):
        extractor = YamlRuleExtractor(_regex_rule(enabled=False))
        results = extractor.extract("My budget is $200")
        assert results == []

    def test_disabled_keyword_returns_empty(self):
        extractor = YamlRuleExtractor(_keyword_rule(enabled=False))
        results = extractor.extract("I prefer organic products")
        assert results == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string_returns_empty(self):
        extractor = YamlRuleExtractor(_regex_rule())
        assert extractor.extract("") == []

    def test_very_short_string_returns_empty(self):
        extractor = YamlRuleExtractor(_regex_rule())
        assert extractor.extract("x") == []

    def test_validation_missing_required_fields(self):
        with pytest.raises(ValueError, match="missing required fields"):
            YamlRuleExtractor({"id": "x", "type": "regex"})

    def test_validation_regex_missing_pattern(self):
        with pytest.raises(ValueError):
            YamlRuleExtractor(
                {"id": "x", "category": "fact", "type": "regex", "importance": 0.5}
            )

    def test_validation_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown rule type"):
            YamlRuleExtractor(
                {"id": "x", "category": "fact", "type": "fuzzy", "importance": 0.5, "pattern": "foo", "template": "foo"}
            )


# ---------------------------------------------------------------------------
# load_yaml_rules function
# ---------------------------------------------------------------------------


class TestLoadYamlRules:
    def test_loads_multiple_rules(self, tmp_path):
        yaml_content = """
rules:
  - id: rule_one
    category: fact
    type: regex
    pattern: 'foo'
    template: 'found foo'
    importance: 0.5

  - id: rule_two
    category: preference
    type: keyword
    keywords: ['bar']
    importance: 0.6
"""
        path = tmp_path / "test_rules.yaml"
        path.write_text(yaml_content)
        extractors = load_yaml_rules(path)
        assert len(extractors) == 2
        assert extractors[0].rule_id == "rule_one"
        assert extractors[1].rule_id == "rule_two"

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert load_yaml_rules(path) == []

    def test_missing_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "nonexistent.yaml"
        assert load_yaml_rules(path) == []

    def test_rule_id_attribute_set(self, tmp_path):
        yaml_content = """
rules:
  - id: my_rule
    category: fact
    type: keyword
    keywords: ['hello']
    importance: 0.5
"""
        path = tmp_path / "r.yaml"
        path.write_text(yaml_content)
        extractors = load_yaml_rules(path)
        assert extractors[0].rule_id == "my_rule"
