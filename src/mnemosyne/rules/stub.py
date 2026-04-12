from __future__ import annotations

import re

from mnemosyne.db.models.memory import ExtractionResult, MemoryType
from mnemosyne.rules.base_extractor import BaseExtractor


class StubRegexExtractor(BaseExtractor):
    """Fallback regex extractor matching simple preference expressions.

    Matches "I like/prefer/love/want/need ..." up to the first sentence
    boundary. Used when no YAML rules are found in the configured rules
    directory.
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
