from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from mnemosyne.db.models.memory import ExtractionResult, MemoryType
from mnemosyne.rules.base_extractor import BaseExtractor


class YamlRuleExtractor(BaseExtractor):
    """Executes a YAML-defined rule against input text.

    Supports three rule types:
    - ``regex``: Compiled regex with ``${N}`` capture-group template interpolation.
    - ``keyword``: Simple keyword presence check; emits a match-notification result.
    - ``keyword_context``: Keyword presence with sentence-context extraction.

    Every instance exposes ``rule_id`` (same as ``id``) so the registry and
    orchestrator can identify it without inspecting the underlying rule dict.
    """

    def __init__(self, rule_def: dict[str, Any]) -> None:
        self._validate(rule_def)
        self._rule = rule_def

        self.rule_id: str = rule_def["id"]
        self.id: str = rule_def["id"]
        self.category: MemoryType = MemoryType(rule_def["category"])
        self.importance: float = float(rule_def["importance"])
        self.enabled: bool = bool(rule_def.get("enabled", True))
        self.version: int = int(rule_def.get("version", 1))
        self._type: str = rule_def["type"]

        # Pre-compile pattern/keywords for runtime efficiency
        if self._type == "regex":
            self._pattern = re.compile(rule_def["pattern"], re.IGNORECASE)
            self._template: str = rule_def["template"]
        elif self._type in ("keyword", "keyword_context"):
            self._keywords: list[str] = [k.lower() for k in rule_def["keywords"]]

    # Validation
    @staticmethod
    def _validate(rule_def: dict[str, Any]) -> None:
        required = {"id", "category", "type", "importance"}
        missing = required - set(rule_def.keys())
        if missing:
            raise ValueError(f"YAML rule missing required fields: {missing}")

        rule_type = rule_def["type"]
        if rule_type == "regex":
            if "pattern" not in rule_def or "template" not in rule_def:
                raise ValueError(
                    f"Regex rule '{rule_def['id']}' requires 'pattern' and 'template'"
                )
        elif rule_type in ("keyword", "keyword_context"):
            if "keywords" not in rule_def:
                raise ValueError(
                    f"Keyword rule '{rule_def['id']}' requires 'keywords'"
                )
        else:
            raise ValueError(f"Unknown rule type: '{rule_type}'")

    # Public extract interface
    def extract(self, text: str) -> list[ExtractionResult]:
        if not self.enabled or not text or len(text.strip()) < 2:
            return []

        if self._type == "regex":
            return self._extract_regex(text)
        elif self._type == "keyword_context":
            return self._extract_keyword_context(text)
        elif self._type == "keyword":
            return self._extract_keyword(text)
        return []

    # Internal extraction helpers
    def _extract_regex(self, text: str) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for match in self._pattern.finditer(text):
            try:
                content = self._apply_template(self._template, match)
            except (IndexError, KeyError):
                continue
            results.append(
                ExtractionResult(
                    content=content,
                    memory_type=self.category,
                    importance=self.importance,
                    matched_chars=match.end() - match.start(),
                    rule_id=self.rule_id,
                )
            )
        return results

    @staticmethod
    def _apply_template(template: str, match: re.Match) -> str:  # type: ignore[type-arg]
        """Replace ``${1}``, ``${2}`` … with the corresponding capture groups."""

        def sub(m: re.Match) -> str:  # type: ignore[type-arg]
            idx = int(m.group(1))
            return match.group(idx)

        return re.sub(r"\$\{(\d+)\}", sub, template).strip()

    def _extract_keyword_context(self, text: str) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        text_lower = text.lower()
        for keyword in self._keywords:
            if keyword not in text_lower:
                continue
            # Return the first sentence that contains the keyword
            for sentence in re.split(r"[.!?\n]+", text):
                if keyword in sentence.lower() and len(sentence.strip()) > 3:
                    results.append(
                        ExtractionResult(
                            content=sentence.strip(),
                            memory_type=self.category,
                            importance=self.importance,
                            matched_chars=len(sentence.strip()),
                            rule_id=self.rule_id,
                        )
                    )
                    break
        return results

    def _extract_keyword(self, text: str) -> list[ExtractionResult]:
        text_lower = text.lower()
        results: list[ExtractionResult] = []
        for keyword in self._keywords:
            if keyword in text_lower:
                results.append(
                    ExtractionResult(
                        content=f"Keyword match: {keyword}",
                        memory_type=self.category,
                        importance=self.importance,
                        matched_chars=len(keyword),
                        rule_id=self.rule_id,
                    )
                )
        return results


# File-level loader

def load_yaml_rules(path: Path) -> list[YamlRuleExtractor]:
    """Load YAML rules from *path*.  Returns ``[]`` if file is missing or empty."""
    if not path.exists():
        return []
    content = yaml.safe_load(path.read_text()) or {}
    rules_list = content.get("rules", [])
    return [YamlRuleExtractor(rule_def) for rule_def in rules_list]
