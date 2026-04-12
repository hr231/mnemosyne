from __future__ import annotations

import logging

import httpx

from mnemosyne.embedding.base import EmbeddingClient

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """Embedding client for OpenAI-compatible /v1/embeddings endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def embed(self, text: str) -> list[float]:
        payload: dict = {
            "model": self._model,
            "input": text,
        }
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/embeddings",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        return data["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/embeddings",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        # OpenAI returns data sorted by index
        sorted_data = sorted(data["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in sorted_data]
