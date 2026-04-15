from __future__ import annotations

import logging

from mnemosyne.embedding.base import EmbeddingClient

logger = logging.getLogger(__name__)


class GoogleEmbeddingClient(EmbeddingClient):
    """Embedding client using the official Google GenAI SDK."""

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: str | None = None,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
        self._kwargs = kwargs
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Install mnemosyne[google] for Google GenAI support: pip install 'mnemosyne[google]'"
            ) from exc
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.aio.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(response.embeddings[0].values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
        return results
