"""Tests for RuleLoader."""
from __future__ import annotations

import pytest

from mnemosyne.rules.rule_loader import RuleLoader


_VALID_REGEX_RULE = """
rules:
  - id: {rule_id}
    category: fact
    type: regex
    pattern: 'foo'
    template: 'found foo'
    importance: 0.5
"""

_VALID_KEYWORD_RULE = """
rules:
  - id: {rule_id}
    category: preference
    type: keyword
    keywords: ['bar']
    importance: 0.6
"""


class TestLoadFromDirectory:
    def test_loads_yaml_files(self, tmp_path):
        (tmp_path / "rule1.yaml").write_text(_VALID_REGEX_RULE.format(rule_id="rule_a"))
        (tmp_path / "rule2.yaml").write_text(_VALID_KEYWORD_RULE.format(rule_id="rule_b"))

        loader = RuleLoader()
        extractors = loader.load_from_directory(tmp_path)

        assert len(extractors) == 2
        ids = {e.rule_id for e in extractors}
        assert "rule_a" in ids
        assert "rule_b" in ids

    def test_empty_directory_returns_empty(self, tmp_path):
        loader = RuleLoader()
        assert loader.load_from_directory(tmp_path) == []

    def test_ignores_non_yaml_files(self, tmp_path):
        (tmp_path / "rule.yaml").write_text(_VALID_REGEX_RULE.format(rule_id="rule_a"))
        (tmp_path / "readme.md").write_text("not a rule file")
        (tmp_path / "notes.txt").write_text("also not a rule file")

        loader = RuleLoader()
        extractors = loader.load_from_directory(tmp_path)
        assert len(extractors) == 1

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        loader = RuleLoader()
        result = loader.load_from_directory(tmp_path / "no_such_dir")
        assert result == []

    def test_malformed_yaml_is_skipped_not_raised(self, tmp_path):
        """A bad YAML file should be logged and skipped, not crash the loader."""
        # This file has a rule missing required fields
        (tmp_path / "bad.yaml").write_text("rules:\n  - id: only_id\n    importance: 0.5\n")
        (tmp_path / "good.yaml").write_text(_VALID_REGEX_RULE.format(rule_id="good_rule"))

        loader = RuleLoader()
        # Should not raise — bad file is skipped
        extractors = loader.load_from_directory(tmp_path)
        # Only the good rule should be loaded
        assert len(extractors) == 1
        assert extractors[0].rule_id == "good_rule"

    def test_empty_yaml_file_produces_no_rules(self, tmp_path):
        (tmp_path / "empty.yaml").write_text("")
        loader = RuleLoader()
        extractors = loader.load_from_directory(tmp_path)
        assert extractors == []

    def test_python_files_are_ignored(self, tmp_path):
        """Python files in the rules directory are not loaded."""
        (tmp_path / "plugin.py").write_text("# not loaded yet")
        (tmp_path / "rule.yaml").write_text(_VALID_REGEX_RULE.format(rule_id="yaml_rule"))

        loader = RuleLoader()
        extractors = loader.load_from_directory(tmp_path)
        assert len(extractors) == 1
        assert extractors[0].rule_id == "yaml_rule"


class TestLoadFromPaths:
    def test_merges_rules_from_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "core"
        dir2 = tmp_path / "ecommerce"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "a.yaml").write_text(_VALID_REGEX_RULE.format(rule_id="core_a"))
        (dir2 / "b.yaml").write_text(_VALID_KEYWORD_RULE.format(rule_id="ecom_b"))

        loader = RuleLoader()
        extractors = loader.load_from_paths([dir1, dir2])
        assert len(extractors) == 2
        assert {e.rule_id for e in extractors} == {"core_a", "ecom_b"}

    def test_empty_paths_list_returns_empty(self):
        loader = RuleLoader()
        assert loader.load_from_paths([]) == []
