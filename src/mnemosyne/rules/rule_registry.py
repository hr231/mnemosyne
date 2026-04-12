from __future__ import annotations

import logging

from mnemosyne.db.models.memory import ExtractionResult
from mnemosyne.rules.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class RuleRegistry:
    """Central holder for loaded extractors.

    Rules are keyed by ``rule_id`` (falling back to ``id`` for compatibility
    with subclasses that only set the class-level ``id`` attribute).

    Duplicate rule IDs raise ``ValueError`` at registration time so
    misconfigured rule packs are caught early rather than silently shadowing
    each other.

    ``extract()`` provides per-rule error isolation: a single rule raising an
    exception is logged and skipped; the remaining rules still run.
    """

    def __init__(self) -> None:
        self._rules: dict[str, BaseExtractor] = {}

#registration

    def register(self, extractor: BaseExtractor) -> None:
        """Register *extractor*.

        Raises
        ------
        ValueError
            If a rule with the same id is already registered.
        """
        rule_id = getattr(extractor, "rule_id", None) or getattr(extractor, "id", "")
        if not rule_id:
            raise ValueError(f"Extractor {extractor!r} has no rule_id or id attribute")
        if rule_id in self._rules:
            raise ValueError(
                f"Rule '{rule_id}' already registered — duplicate rule ids are not allowed"
            )
        self._rules[rule_id] = extractor

    def register_all(self, extractors: list[BaseExtractor]) -> None:
        """Register a batch of extractors in order."""
        for ex in extractors:
            self.register(ex)

    # Lookup
    def get(self, rule_id: str) -> BaseExtractor | None:
        """Return the extractor registered under *rule_id*, or ``None``."""
        return self._rules.get(rule_id)

    def all(self) -> list[BaseExtractor]:
        """Return all registered extractors in insertion order."""
        return list(self._rules.values())

    # Extraction
    def extract(self, text: str) -> list[ExtractionResult]:
        """Run all enabled rules against *text* and return combined results.

        Per-rule errors are isolated: a failing rule is logged and skipped;
        the remaining rules still execute.
        """
        if not text or not text.strip():
            return []

        results: list[ExtractionResult] = []
        for rule in self._rules.values():
            if not getattr(rule, "enabled", True):
                continue
            try:
                results.extend(rule.extract(text))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Rule %r raised an exception during extract — skipping: %s",
                    getattr(rule, "rule_id", rule),
                    exc,
                )
        return results
