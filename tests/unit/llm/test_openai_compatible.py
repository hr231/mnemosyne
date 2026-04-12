from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemosyne.errors import MalformedLLMResponse
from mnemosyne.llm.openai_compatible import EXTRACTION_PROMPT, OpenAICompatibleClient


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_chat_response(content: str) -> dict:
    """Build a minimal OpenAI-compatible /chat/completions response dict."""
    return {
        "choices": [
            {"message": {"content": content}}
        ]
    }


def _mock_httpx_post(response_content: str):
    """Return a context-manager mock that yields a response with *response_content*."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_chat_response(response_content)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extraction_prompt_format():
    """EXTRACTION_PROMPT must contain a {text} placeholder."""
    assert "{text}" in EXTRACTION_PROMPT


@pytest.mark.asyncio
async def test_parse_valid_json():
    payload = json.dumps([
        {"content": "User prefers dark mode", "memory_type": "preference", "importance": 0.8},
        {"content": "User is based in London", "memory_type": "fact", "importance": 0.6},
    ])

    client = OpenAICompatibleClient()
    with patch("httpx.AsyncClient", return_value=_mock_httpx_post(payload)):
        results = await client.extract_memories("User prefers dark mode, based in London")

    assert len(results) == 2
    assert results[0].content == "User prefers dark mode"
    assert results[0].memory_type.value == "preference"
    assert results[0].importance == pytest.approx(0.8)
    assert results[1].content == "User is based in London"


@pytest.mark.asyncio
async def test_parse_markdown_fenced_json():
    payload_items = [{"content": "User likes jazz", "memory_type": "preference", "importance": 0.7}]
    fenced = f"```json\n{json.dumps(payload_items)}\n```"

    client = OpenAICompatibleClient()
    with patch("httpx.AsyncClient", return_value=_mock_httpx_post(fenced)):
        results = await client.extract_memories("User likes jazz music")

    assert len(results) == 1
    assert results[0].content == "User likes jazz"


@pytest.mark.asyncio
async def test_parse_invalid_json_raises():
    client = OpenAICompatibleClient()
    with patch("httpx.AsyncClient", return_value=_mock_httpx_post("not json at all")):
        with pytest.raises(MalformedLLMResponse, match="Invalid JSON"):
            await client.extract_memories("some text")


@pytest.mark.asyncio
async def test_parse_non_array_raises():
    payload = json.dumps({"key": "val"})

    client = OpenAICompatibleClient()
    with patch("httpx.AsyncClient", return_value=_mock_httpx_post(payload)):
        with pytest.raises(MalformedLLMResponse, match="Expected JSON array"):
            await client.extract_memories("some text")


@pytest.mark.asyncio
async def test_items_missing_content_are_skipped():
    """Items without a 'content' key must be silently skipped."""
    payload = json.dumps([
        {"memory_type": "fact", "importance": 0.5},          # no content
        {"content": "User owns a cat", "memory_type": "fact", "importance": 0.6},
    ])

    client = OpenAICompatibleClient()
    with patch("httpx.AsyncClient", return_value=_mock_httpx_post(payload)):
        results = await client.extract_memories("User owns a cat")

    assert len(results) == 1
    assert results[0].content == "User owns a cat"


@pytest.mark.asyncio
async def test_default_memory_type_and_importance():
    """Items with missing optional fields use defaults (fact, 0.5)."""
    payload = json.dumps([{"content": "User drinks coffee"}])

    client = OpenAICompatibleClient()
    with patch("httpx.AsyncClient", return_value=_mock_httpx_post(payload)):
        results = await client.extract_memories("User drinks coffee")

    assert results[0].memory_type.value == "fact"
    assert results[0].importance == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_api_key_added_to_headers():
    """When api_key is set the Authorization header is included."""
    payload = json.dumps([{"content": "test", "memory_type": "fact", "importance": 0.5}])
    mock_client_instance = _mock_httpx_post(payload)

    client = OpenAICompatibleClient(api_key="sk-test-key")
    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        await client.extract_memories("test")

    call_kwargs = mock_client_instance.post.call_args
    headers = call_kwargs.kwargs.get("headers", call_kwargs.args[1] if len(call_kwargs.args) > 1 else {})
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer sk-test-key"
