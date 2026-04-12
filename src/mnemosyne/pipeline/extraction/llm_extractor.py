from __future__ import annotations

import logging

from mnemosyne.db.models.memory import ExtractionResult
from mnemosyne.llm.base import LLMClient

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Delegates memory extraction to an LLM client."""

    def __init__(self, llm_client: LLMClient, extraction_version: str = "0.1.0"):
        self._client = llm_client
        self._extraction_version = extraction_version

    async def extract(self, text: str) -> list[ExtractionResult]:
        results = await self._client.extract_memories(text)
        stamped = []
        for r in results:
            stamped.append(r.model_copy(update={
                "extraction_version": self._extraction_version,
                "rule_id": r.rule_id or "llm_extractor",
            }))
        return stamped
