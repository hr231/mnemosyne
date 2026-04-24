from __future__ import annotations

import json
import logging

import httpx

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


class OpenAICompatibleClient(LLMClient):
    """LLM client for OpenAI-compatible APIs (OpenAI, Ollama, vLLM)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def complete(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        prompt = EXTRACTION_PROMPT.format(text=text)
        raw = await self.complete(prompt)

        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
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
