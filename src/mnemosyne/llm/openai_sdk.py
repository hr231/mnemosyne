from __future__ import annotations

import json
import logging

from mnemosyne.db.models.memory import ExtractionResult
from mnemosyne.errors import MalformedLLMResponse
from mnemosyne.llm.base import LLMClient, llm_semaphore, record_llm_usage
from mnemosyne.llm.hardening import (
    clamp_importance,
    render_with_untrusted,
    safe_memory_type,
)
from mnemosyne.utils import retry_async

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract structured memories from the following text.
Return a JSON array of objects, each with:
- "content": the memory text
- "memory_type": one of "fact", "preference", "entity", "procedural"
- "importance": float 0.0-1.0

Text: $input

Respond with ONLY valid JSON array."""


class OpenAILLMClient(LLMClient):
    """LLM client using the official OpenAI SDK.

    Supports both OpenAI direct and Azure OpenAI (same SDK, different constructor).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._timeout = timeout
        self._max_retries = max_retries
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
                timeout=self._timeout,
            )
        else:
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key, timeout=self._timeout
            )
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        usage: dict[str, int | None] = {"tokens": None}

        async def _do() -> str:
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            usage["tokens"] = _extract_total_tokens(response)
            return response.choices[0].message.content or ""

        async with llm_semaphore():
            text = await retry_async(
                _do, max_retries=self._max_retries, retry_on=_retryable_exceptions()
            )
        record_llm_usage(usage["tokens"])
        return text

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        prompt = render_with_untrusted(EXTRACTION_PROMPT, text)
        raw = await self.complete(prompt)
        return _parse_extraction_response(raw)


def _extract_total_tokens(response: object) -> int | None:
    """Best-effort read of total token usage from an OpenAI-style response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    total = getattr(usage, "total_tokens", None)
    return int(total) if total is not None else None


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Best-effort tuple of SDK transient errors plus generic fallbacks."""
    exceptions: list[type[Exception]] = [TimeoutError, ConnectionError]
    try:
        import openai

        for name in ("APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"):
            exc = getattr(openai, name, None)
            if isinstance(exc, type):
                exceptions.append(exc)
    except ImportError:
        pass
    return tuple(exceptions)


def _parse_extraction_response(raw: str) -> list[ExtractionResult]:
    """Parse and validate an LLM extraction response into ExtractionResults."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise MalformedLLMResponse(f"Invalid JSON from LLM: {e}") from e

    if not isinstance(items, list):
        raise MalformedLLMResponse(f"Expected JSON array, got {type(items).__name__}")

    results = []
    for item in items:
        if not isinstance(item, dict) or "content" not in item:
            continue
        results.append(
            ExtractionResult(
                content=item["content"],
                memory_type=safe_memory_type(item.get("memory_type")),
                importance=clamp_importance(item.get("importance", 0.5)),
                rule_id="llm_extractor",
            )
        )
    return results
