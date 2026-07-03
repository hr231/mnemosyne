from __future__ import annotations

import logging

from mnemosyne.db.models.memory import ExtractionResult
from mnemosyne.errors import MalformedLLMResponse  # noqa: F401 — re-exported via openai_sdk
from mnemosyne.llm.base import LLMClient, llm_semaphore, record_llm_usage
from mnemosyne.llm.hardening import render_with_untrusted
from mnemosyne.utils import retry_async

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract structured memories from the following text.
Return a JSON array of objects, each with:
- "content": the memory text
- "memory_type": one of "fact", "preference", "entity", "procedural"
- "importance": float 0.0-1.0

Text: $input

Respond with ONLY valid JSON array."""


class GoogleLLMClient(LLMClient):
    """LLM client using the official Google GenAI SDK."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._kwargs = kwargs
        self._client = None

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

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        usage: dict[str, int | None] = {"tokens": None}

        async def _do() -> str:
            response = await client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                **kwargs,
            )
            usage["tokens"] = _extract_total_tokens(response)
            return response.text or ""

        async with llm_semaphore():
            text = await retry_async(
                _do, max_retries=self._max_retries, retry_on=_retryable_exceptions()
            )
        record_llm_usage(usage["tokens"])
        return text

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        prompt = render_with_untrusted(EXTRACTION_PROMPT, text)
        raw = await self.complete(prompt)
        from mnemosyne.llm.openai_sdk import _parse_extraction_response

        return _parse_extraction_response(raw)


def _extract_total_tokens(response: object) -> int | None:
    """Best-effort read of total token usage from a Google GenAI response."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    total = getattr(usage, "total_token_count", None)
    return int(total) if total is not None else None


def _retryable_exceptions() -> tuple[type[Exception], ...]:
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
