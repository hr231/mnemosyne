from __future__ import annotations

import logging

import httpx

from mnemosyne.embedding.base import EmbeddingClient

logger = logging.getLogger(__name__)


class OllamaEmbeddingClient(EmbeddingClient):
    """Embedding client for Ollama's /api/embed endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
        expected_dim: int | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._expected_dim = expected_dim
        self._dim_validated = False

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()

        embedding = data["embeddings"][0]

        if self._expected_dim and not self._dim_validated:
            if len(embedding) != self._expected_dim:
                raise ValueError(
                    f"Expected {self._expected_dim}-dim embeddings from {self._model}, "
                    f"got {len(embedding)}"
                )
            self._dim_validated = True

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()

        embeddings = data["embeddings"]

        if self._expected_dim and not self._dim_validated:
            if embeddings and len(embeddings[0]) != self._expected_dim:
                raise ValueError(
                    f"Expected {self._expected_dim}-dim embeddings, got {len(embeddings[0])}"
                )
            self._dim_validated = True

        return embeddings
