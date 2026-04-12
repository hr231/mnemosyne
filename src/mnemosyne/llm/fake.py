from __future__ import annotations

from mnemosyne.db.models.memory import ExtractionResult
from mnemosyne.errors import CannedResponseMissing
from mnemosyne.llm.base import LLMClient


class FakeLLMClient(LLMClient):
    """In-memory canned-response LLM client for use in tests.

    Register expected responses with ``add_canned(fragment, results)``.
    Any ``extract_memories`` call whose input contains the registered
    fragment (case-insensitive) returns the pre-loaded results.

    Raises ``CannedResponseMissing`` on a cache miss so tests fail loudly
    rather than silently returning empty results.

    ``complete`` always returns ``"fake completion"``.
    """

    def __init__(self) -> None:
        self._canned: dict[str, list[ExtractionResult]] = {}

    def add_canned(self, text_fragment: str, results: list[ExtractionResult]) -> None:
        """Register *results* to be returned when *text_fragment* appears in input."""
        self._canned[text_fragment.lower()] = results

    async def complete(self, prompt: str, **kwargs) -> str:
        return "fake completion"

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        text_lower = text.lower()
        for fragment, results in self._canned.items():
            if fragment in text_lower:
                return results
        raise CannedResponseMissing(text)
