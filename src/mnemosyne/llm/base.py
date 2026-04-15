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

        Supported providers:
            "openai"              — OpenAI SDK (direct)
            "azure"               — Azure OpenAI (same SDK)
            "anthropic"           — Anthropic SDK
            "google"              — Google GenAI SDK
            "openai_compatible"   — any /v1/chat/completions endpoint
        """
        provider = config.get("provider")
        if not provider:
            raise ValueError("LLM provider not specified in config")

        if provider == "openai":
            from mnemosyne.llm.openai_sdk import OpenAILLMClient
            return OpenAILLMClient(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config.get("api_key"),
            )

        if provider == "azure":
            from mnemosyne.llm.openai_sdk import OpenAILLMClient
            return OpenAILLMClient(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config.get("api_key"),
                azure_endpoint=config["azure_endpoint"],
                api_version=config.get("api_version"),
            )

        if provider == "anthropic":
            from mnemosyne.llm.anthropic_sdk import AnthropicLLMClient
            return AnthropicLLMClient(
                model=config.get("model", "claude-sonnet-4-20250514"),
                api_key=config.get("api_key"),
            )

        if provider == "google":
            from mnemosyne.llm.google_sdk import GoogleLLMClient
            return GoogleLLMClient(
                model=config.get("model", "gemini-2.0-flash"),
                api_key=config.get("api_key"),
            )

        if provider == "openai_compatible":
            from mnemosyne.llm.openai_compatible import OpenAICompatibleClient
            return OpenAICompatibleClient(
                base_url=config["base_url"],
                model=config["model"],
                api_key=config.get("api_key"),
            )

        raise ValueError(f"Unknown LLM provider: {provider!r}")
