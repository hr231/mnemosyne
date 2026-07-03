from __future__ import annotations

import json
import logging

import httpx

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


class OpenAICompatibleClient(LLMClient):
    """LLM client for OpenAI-compatible APIs (OpenAI, Ollama, vLLM)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        client = self._get_client()
        usage: dict[str, int | None] = {"tokens": None}

        async def _do() -> str:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            usage["tokens"] = _extract_total_tokens(data)
            return data["choices"][0]["message"]["content"]

        async with llm_semaphore():
            text = await retry_async(
                _do,
                max_retries=self._max_retries,
                retry_on=(httpx.HTTPError,),
            )
        record_llm_usage(usage["tokens"])
        return text

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        prompt = render_with_untrusted(EXTRACTION_PROMPT, text)
        raw = await self.complete(prompt)
        return _parse_extraction_response(raw)


def _extract_total_tokens(data: object) -> int | None:
    """Best-effort read of total token usage from an OpenAI-compatible payload."""
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    return int(total) if total is not None else None


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
