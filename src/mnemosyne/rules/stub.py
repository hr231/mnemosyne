from __future__ import annotations

import re

from mnemosyne.db.models.memory import ExtractionResult, MemoryType
from mnemosyne.rules.base_extractor import BaseExtractor


class StubRegexExtractor(BaseExtractor):
    """Walking-skeleton regex extractor used in v0.1.0.

    Matches simple preference expressions ("I like/prefer/love/want/need …")
    up to the first sentence boundary.  Replaced by the full YAML rule
    registry (``RuleRegistry.load()``) in Day 1 afternoon — the orchestrator
    swaps the extractor list without touching this class.

    ``extraction_version`` is read from the constructor argument, which
    the orchestrator sets from ``settings.extraction_version``.  No
    version strings are hardcoded here.

    Provenance stamped on every result:
    - ``rule_id="walking_skeleton_stub"``
    - ``extraction_version`` from constructor
    """

    id = "walking_skeleton_stub"
    category = MemoryType.PREFERENCE
    importance = 0.6

    _PATTERN = re.compile(
        r"(?i)\b(?:i (?:like|prefer|love|want|need))\b[^.]*",
        re.IGNORECASE,
    )

    def __init__(self, extraction_version: str = "0.1.0") -> None:
        self._extraction_version = extraction_version

    def extract(self, text: str) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for match in self._PATTERN.finditer(text):
            matched_text = match.group(0).strip()
            results.append(
                ExtractionResult(
                    content=matched_text,
                    memory_type=self.category,
                    importance=self.importance,
                    matched_chars=len(matched_text),
                    rule_id=self.id,
                    extraction_version=self._extraction_version,
                )
            )
        return results
