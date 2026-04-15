from __future__ import annotations

import logging

from mnemosyne.embedding.base import EmbeddingClient

logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding client using the official OpenAI SDK.

    Supports both OpenAI direct and Azure OpenAI.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._kwargs = kwargs
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "Install mnemosyne[openai] for OpenAI support: pip install 'mnemosyne[openai]'"
            ) from exc

        if self._azure_endpoint:
            self._client = openai.AsyncAzureOpenAI(
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version or "2024-02-01",
                api_key=self._api_key,
            )
        else:
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        kwargs: dict = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = await client.embeddings.create(**kwargs)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        kwargs: dict = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = await client.embeddings.create(**kwargs)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]
