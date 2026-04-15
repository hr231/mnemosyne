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


class GoogleLLMClient(LLMClient):
    """LLM client using the official Google GenAI SDK."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
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

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            **kwargs,
        )
        return response.text or ""

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        prompt = EXTRACTION_PROMPT.format(text=text)
        raw = await self.complete(prompt)
        from mnemosyne.llm.openai_sdk import _parse_extraction_response
        return _parse_extraction_response(raw)
