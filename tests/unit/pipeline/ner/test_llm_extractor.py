from __future__ import annotations

import json

import pytest

from mnemosyne.db.models.memory import ExtractionResult
from mnemosyne.llm.base import LLMClient
from mnemosyne.pipeline.ner.llm_extractor import extract_entities_llm
from mnemosyne.pipeline.ner.spacy_extractor import RawEntity


class _ConfigurableLLMClient(LLMClient):
    """Test double: returns a fixed string from complete(), raises on demand."""

    def __init__(self, response: str = "", raise_on_complete: bool = False) -> None:
        self._response = response
        self._raise = raise_on_complete

    async def complete(self, prompt: str, **kwargs) -> str:
        if self._raise:
            raise RuntimeError("simulated LLM failure")
        return self._response

    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        return []


@pytest.mark.asyncio
async def test_extract_entities():
    payload = json.dumps([
        {"name": "John Smith", "type": "person", "context": "John Smith attended the meeting."},
        {"name": "Acme Corp", "type": "organization", "context": "He works at Acme Corp."},
    ])
    client = _ConfigurableLLMClient(response=payload)

    results = await extract_entities_llm("John Smith attended the meeting at Acme Corp.", client)

    assert len(results) == 2
    names = {e.name for e in results}
    assert "John Smith" in names
    assert "Acme Corp" in names
    types = {e.entity_type for e in results}
    assert "person" in types
    assert "organization" in types
    assert all(isinstance(e, RawEntity) for e in results)
    assert all(e.source == "llm" for e in results)


@pytest.mark.asyncio
async def test_handles_malformed_response():
    client = _ConfigurableLLMClient(response="this is not json at all!!!")
    results = await extract_entities_llm("some text", client)
    assert results == []


@pytest.mark.asyncio
async def test_handles_exception():
    client = _ConfigurableLLMClient(raise_on_complete=True)
    results = await extract_entities_llm("some text", client)
    assert results == []


@pytest.mark.asyncio
async def test_handles_non_list_json():
    client = _ConfigurableLLMClient(response='{"name": "Alice"}')
    results = await extract_entities_llm("Alice said hello", client)
    assert results == []


@pytest.mark.asyncio
async def test_skips_items_missing_name():
    payload = json.dumps([
        {"type": "person", "context": "missing name field"},
        {"name": "Bob", "type": "person", "context": "Bob arrived."},
    ])
    client = _ConfigurableLLMClient(response=payload)
    results = await extract_entities_llm("Bob arrived.", client)
    assert len(results) == 1
    assert results[0].name == "Bob"


@pytest.mark.asyncio
async def test_strips_markdown_fences():
    inner = json.dumps([{"name": "Paris", "type": "location", "context": "Visit Paris."}])
    fenced = f"```json\n{inner}\n```"
    client = _ConfigurableLLMClient(response=fenced)
    results = await extract_entities_llm("Visit Paris.", client)
    assert len(results) == 1
    assert results[0].name == "Paris"


@pytest.mark.asyncio
async def test_defaults_type_to_concept_when_missing():
    payload = json.dumps([{"name": "Zorg", "context": "Zorg appeared."}])
    client = _ConfigurableLLMClient(response=payload)
    results = await extract_entities_llm("Zorg appeared.", client)
    assert len(results) == 1
    assert results[0].entity_type == "concept"
