from __future__ import annotations

from abc import ABC, abstractmethod

from mnemosyne.db.models.memory import ExtractionResult


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Raw text completion."""
        ...

    @abstractmethod
    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        """Extract structured memories from text."""
        ...

    @classmethod
    def from_config(cls, config: dict) -> "LLMClient":
        """Create an LLMClient from a config dict.

        Config keys:
            provider: "fake" | "openai_compatible"
            base_url: str (required for openai_compatible)
            model: str (required for openai_compatible)
            api_key: str | None
        """
        provider = config.get("provider", "fake")

        if provider == "fake":
            from mnemosyne.llm.fake import FakeLLMClient
            return FakeLLMClient()

        if provider == "openai_compatible":
            from mnemosyne.llm.openai_compatible import OpenAICompatibleClient
            return OpenAICompatibleClient(
                base_url=config["base_url"],
                model=config["model"],
                api_key=config.get("api_key"),
            )

        raise ValueError(f"Unknown LLM provider: {provider!r}")
