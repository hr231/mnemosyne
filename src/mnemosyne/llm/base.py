from __future__ import annotations

from abc import ABC, abstractmethod

from mnemosyne.db.models.memory import ExtractionResult


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Raw completion. Used by reflection generation in Sprint 2."""
        ...

    @abstractmethod
    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        """Extract structured memories from text."""
        ...
