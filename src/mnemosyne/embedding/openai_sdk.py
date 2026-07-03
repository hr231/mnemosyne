from __future__ import annotations

import logging

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.utils import retry_async

logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding client using the official OpenAI SDK.

    Supports both OpenAI direct and Azure OpenAI. The SDK client is created
    once per instance with the configured timeout and reused across calls;
    transient failures (timeouts, connection errors, rate limits, 5xx) are
    retried with exponential backoff.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._kwargs = kwargs
        self._client = None
        self._retry_on: tuple[type[Exception], ...] | None = None

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
                timeout=self._timeout,
            )
        else:
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key, timeout=self._timeout
            )
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
        kwargs: dict = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        async def _do() -> list[float]:
            response = await client.embeddings.create(**kwargs)
            return response.data[0].embedding

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
        client = self._get_client()
        kwargs: dict = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        async def _do() -> list[list[float]]:
            response = await client.embeddings.create(**kwargs)
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]

        embeddings = await retry_async(
            _do,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            retry_on=self._retryable(),
        )
        if embeddings:
            self._validate_dim(embeddings[0])
        return embeddings


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Best-effort tuple of SDK transient errors plus generic fallbacks."""
    exceptions: list[type[Exception]] = [TimeoutError, ConnectionError]
    try:
        import openai

        for name in (
            "APITimeoutError",
            "APIConnectionError",
            "RateLimitError",
            "InternalServerError",
        ):
            exc = getattr(openai, name, None)
            if isinstance(exc, type):
                exceptions.append(exc)
    except ImportError:
        pass
    return tuple(exceptions)
