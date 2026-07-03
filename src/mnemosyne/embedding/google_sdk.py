from __future__ import annotations

import asyncio
import logging

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.utils import retry_async

logger = logging.getLogger(__name__)


class GoogleEmbeddingClient(EmbeddingClient):
    """Embedding client using the official Google GenAI SDK.

    The SDK client is created once per instance with the configured per-request
    timeout and reused across calls. The Google embedding endpoint takes one
    input per request, so ``embed_batch`` fans out concurrent requests bounded
    by ``max_concurrency``. Transient failures are retried with backoff.
    """

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_concurrency: int = 5,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_concurrency = max_concurrency
        self._kwargs = kwargs
        self._client = None
        self._retry_on: tuple[type[Exception], ...] | None = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Install mnemosyne[google] for Google GenAI support: pip install 'mnemosyne[google]'"
            ) from exc
        # Google SDK expects the per-request timeout in milliseconds.
        http_options = types.HttpOptions(timeout=int(self._timeout * 1000))
        self._client = genai.Client(api_key=self._api_key, http_options=http_options)
        return self._client

    def _retryable(self) -> tuple[type[Exception], ...]:
        if self._retry_on is not None:
            return self._retry_on
        return _retryable_exceptions()

    def _validate_dim(self, embedding: list[float]) -> None:
        if self._dimensions is not None and len(embedding) != self._dimensions:
            raise ValueError(
                f"Expected {self._dimensions}-dim embeddings from {self._model}, "
                f"got {len(embedding)}"
            )

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()

        async def _do() -> list[float]:
            response = await client.aio.models.embed_content(
                model=self._model,
                contents=text,
            )
            return list(response.embeddings[0].values)

        embedding = await retry_async(
            _do,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            retry_on=self._retryable(),
        )
        self._validate_dim(embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _one(text: str) -> list[float]:
            async with semaphore:
                return await self.embed(text)

        return await asyncio.gather(*(_one(t) for t in texts))


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Best-effort tuple of SDK transient errors plus generic fallbacks."""
    exceptions: list[type[Exception]] = [TimeoutError, ConnectionError]
    try:
        from google.genai import errors as genai_errors

        for name in ("APIError", "ServerError", "ClientError"):
            exc = getattr(genai_errors, name, None)
            if isinstance(exc, type):
                exceptions.append(exc)
    except ImportError:
        pass
    return tuple(exceptions)
