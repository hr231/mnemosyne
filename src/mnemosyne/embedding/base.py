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

        Supported providers:
            "openai"              — OpenAI SDK (direct)
            "azure"               — Azure OpenAI (same SDK)
            "google"              — Google GenAI SDK
            "openai_compatible"   — any /v1/embeddings endpoint
            "ollama"              — Ollama /api/embed endpoint
            "fastembed"           — local FastEmbed (zero API)
        """
        provider = config.get("provider")
        if not provider:
            raise ValueError("Embedding provider not specified in config")

        if provider == "openai":
            from mnemosyne.embedding.openai_sdk import OpenAIEmbeddingClient
            return OpenAIEmbeddingClient(
                model=config.get("model", "text-embedding-3-small"),
                api_key=config.get("api_key"),
                dimensions=config.get("dimensions"),
            )

        if provider == "azure":
            from mnemosyne.embedding.openai_sdk import OpenAIEmbeddingClient
            return OpenAIEmbeddingClient(
                model=config.get("model", "text-embedding-3-small"),
                api_key=config.get("api_key"),
                dimensions=config.get("dimensions"),
                azure_endpoint=config["azure_endpoint"],
                api_version=config.get("api_version"),
            )

        if provider == "google":
            from mnemosyne.embedding.google_sdk import GoogleEmbeddingClient
            return GoogleEmbeddingClient(
                model=config.get("model", "text-embedding-004"),
                api_key=config.get("api_key"),
            )

        if provider == "openai_compatible":
            from mnemosyne.embedding.openai_compatible import OpenAICompatibleEmbeddingClient
            return OpenAICompatibleEmbeddingClient(
                base_url=config["base_url"],
                model=config["model"],
                api_key=config.get("api_key"),
                dimensions=config.get("dimensions"),
            )

        if provider == "ollama":
            from mnemosyne.embedding.ollama import OllamaEmbeddingClient
            return OllamaEmbeddingClient(
                base_url=config.get("base_url", "http://localhost:11434"),
                model=config.get("model", "nomic-embed-text"),
                expected_dim=config.get("dimensions"),
            )

        if provider == "fastembed":
            from mnemosyne.embedding.fastembed import FastEmbedClient
            return FastEmbedClient(
                model_name=config.get("model", "BAAI/bge-small-en-v1.5"),
            )

        raise ValueError(f"Unknown embedding provider: {provider!r}")
