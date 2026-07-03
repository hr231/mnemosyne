from __future__ import annotations

import logging

import httpx

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.utils import retry_async

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """Embedding client for OpenAI-compatible /v1/embeddings endpoints.

    Uses a single persistent ``httpx.AsyncClient`` per instance and retries
    transient failures (timeouts, connection errors, 429/5xx) with backoff.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._timeout = timeout
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

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, value: str | list[str]) -> dict:
        payload: dict = {"model": self._model, "input": value}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        return payload

    async def _post(self, payload: dict) -> dict:
        client = self._get_client()

        async def _call() -> dict:
            resp = await client.post(
                f"{self._base_url}/v1/embeddings",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

        return await retry_async(
            _call,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            retry_on=(httpx.HTTPError,),
        )

    def _validate_dim(self, embedding: list[float]) -> None:
        if self._dimensions is not None and len(embedding) != self._dimensions:
            raise ValueError(
                f"Expected {self._dimensions}-dim embeddings from {self._model}, "
                f"got {len(embedding)}"
            )

    async def embed(self, text: str) -> list[float]:
        data = await self._post(self._payload(text))
        embedding = data["data"][0]["embedding"]
        self._validate_dim(embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post(self._payload(texts))
        sorted_data = sorted(data["data"], key=lambda d: d["index"])
        embeddings = [d["embedding"] for d in sorted_data]
        if embeddings:
            self._validate_dim(embeddings[0])
        return embeddings
