from __future__ import annotations

import logging

import httpx

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.utils import retry_async

logger = logging.getLogger(__name__)


class OllamaEmbeddingClient(EmbeddingClient):
    """Embedding client for Ollama's /api/embed endpoint.

    Uses a single persistent ``httpx.AsyncClient`` per instance and retries
    transient failures (timeouts, connection errors, 429/5xx) with exponential
    backoff.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
        expected_dim: int | None = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._expected_dim = expected_dim
        self._dim_validated = False
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the persistent HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _validate_dim(self, embedding: list[float]) -> None:
        if self._expected_dim and not self._dim_validated:
            if len(embedding) != self._expected_dim:
                raise ValueError(
                    f"Expected {self._expected_dim}-dim embeddings from "
                    f"{self._model}, got {len(embedding)}"
                )
            self._dim_validated = True

    async def _post(self, payload: dict) -> dict:
        client = self._get_client()

        async def _call() -> dict:
            resp = await client.post(f"{self._base_url}/api/embed", json=payload)
            resp.raise_for_status()
            return resp.json()

        return await retry_async(
            _call,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            retry_on=(httpx.HTTPError,),
        )

    async def embed(self, text: str) -> list[float]:
        data = await self._post({"model": self._model, "input": text})
        embedding = data["embeddings"][0]
        self._validate_dim(embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post({"model": self._model, "input": texts})
        embeddings = data["embeddings"]
        if embeddings:
            self._validate_dim(embeddings[0])
        return embeddings
