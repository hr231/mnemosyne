from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @classmethod
    def from_config(cls, config: dict) -> "EmbeddingClient":
        """Create an EmbeddingClient from a config dict.

        Config keys:
            provider: "fake" | "ollama" | "openai_compatible"
            base_url: str (required for ollama and openai_compatible)
            model: str (required for ollama and openai_compatible)
            api_key: str | None
            dimensions: int | None
        """
        provider = config.get("provider", "fake")

        if provider == "fake":
            from mnemosyne.embedding.fake import FakeEmbeddingClient
            return FakeEmbeddingClient(dim=config.get("dimensions") or 768)

        if provider == "ollama":
            from mnemosyne.embedding.ollama import OllamaEmbeddingClient
            return OllamaEmbeddingClient(
                base_url=config.get("base_url", "http://localhost:11434"),
                model=config.get("model", "nomic-embed-text"),
                expected_dim=config.get("dimensions"),
            )

        if provider == "openai_compatible":
            from mnemosyne.embedding.openai_compatible import OpenAICompatibleEmbeddingClient
            return OpenAICompatibleEmbeddingClient(
                base_url=config["base_url"],
                model=config["model"],
                api_key=config.get("api_key"),
                dimensions=config.get("dimensions"),
            )

        if provider == "fastembed":
            from mnemosyne.embedding.fastembed import FastEmbedClient
            return FastEmbedClient(
                model_name=config.get("model", "BAAI/bge-small-en-v1.5"),
            )

        raise ValueError(f"Unknown embedding provider: {provider!r}")
