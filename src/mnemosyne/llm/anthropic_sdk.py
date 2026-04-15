from __future__ import annotations

import logging

from mnemosyne.db.models.memory import ExtractionResult
from mnemosyne.errors import MalformedLLMResponse  # noqa: F401 — re-exported via openai_sdk
from mnemosyne.llm.base import LLMClient

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract structured memories from the following text.
Return a JSON array of objects, each with:
- "content": the memory text
- "memory_type": one of "fact", "preference", "entity", "procedural"
- "importance": float 0.0-1.0

Text: {text}

Respond with ONLY valid JSON array."""


class AnthropicLLMClient(LLMClient):
    """LLM client using the official Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        **kwargs,
    ):
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
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
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=kwargs.pop("max_tokens", self._max_tokens),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content[0].text

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        prompt = EXTRACTION_PROMPT.format(text=text)
        raw = await self.complete(prompt)
        from mnemosyne.llm.openai_sdk import _parse_extraction_response
        return _parse_extraction_response(raw)
