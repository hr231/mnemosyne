from __future__ import annotations

import json
import logging

from mnemosyne.db.models.memory import ExtractionResult, MemoryType
from mnemosyne.errors import MalformedLLMResponse
from mnemosyne.llm.base import LLMClient

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract structured memories from the following text.
Return a JSON array of objects, each with:
- "content": the memory text
- "memory_type": one of "fact", "preference", "entity", "procedural"
- "importance": float 0.0-1.0

Text: {text}

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
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
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

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        prompt = EXTRACTION_PROMPT.format(text=text)
        raw = await self.complete(prompt)
        return _parse_extraction_response(raw)


def _parse_extraction_response(raw: str) -> list[ExtractionResult]:
    """Parse LLM extraction response into ExtractionResult list."""
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
        results.append(ExtractionResult(
            content=item["content"],
            memory_type=MemoryType(item.get("memory_type", "fact")),
            importance=float(item.get("importance", 0.5)),
            rule_id="llm_extractor",
        ))
    return results
