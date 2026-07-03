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


class AnthropicLLMClient(LLMClient):
    """LLM client using the official Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._kwargs = kwargs
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Install mnemosyne[anthropic] for Anthropic support: pip install 'mnemosyne[anthropic]'"
            ) from exc
        self._client = anthropic.AsyncAnthropic(
            api_key=self._api_key, timeout=self._timeout
        )
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        usage: dict[str, int | None] = {"tokens": None}

        async def _do() -> str:
            response = await client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            usage["tokens"] = _extract_total_tokens(response)
            return response.content[0].text

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
    """Best-effort read of total token usage from an Anthropic response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None) or 0
    output_tokens = getattr(usage, "output_tokens", None) or 0
    total = int(input_tokens) + int(output_tokens)
    return total or None


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    exceptions: list[type[Exception]] = [TimeoutError, ConnectionError]
    try:
        import anthropic

        for name in ("APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"):
            exc = getattr(anthropic, name, None)
            if isinstance(exc, type):
                exceptions.append(exc)
    except ImportError:
        pass
    return tuple(exceptions)
