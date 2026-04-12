from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mnemosyne.embedding.ollama import OllamaEmbeddingClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed_response(embeddings: list[list[float]], model: str = "nomic-embed-text") -> dict:
    return {"model": model, "embeddings": embeddings}


def _mock_httpx_client(response_data: dict) -> MagicMock:
    """Return an async context-manager mock that yields a client whose .post()
    returns a response with *response_data* as its JSON payload."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response_data

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_single():
    """embed() returns the first vector from the Ollama response."""
    vector = [0.1] * 768
    mock_client = _mock_httpx_client(_make_embed_response([vector]))

    client = OllamaEmbeddingClient()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.embed("hello world")

    assert result == vector
    assert len(result) == 768


@pytest.mark.asyncio
async def test_embed_batch():
    """embed_batch() returns all vectors from the Ollama response."""
    vectors = [[float(i)] * 768 for i in range(3)]
    mock_client = _mock_httpx_client(_make_embed_response(vectors))

    client = OllamaEmbeddingClient()
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await client.embed_batch(["a", "b", "c"])

    assert len(results) == 3
    assert results == vectors


@pytest.mark.asyncio
async def test_embed_empty_batch():
    """embed_batch([]) returns [] without making any HTTP request."""
    client = OllamaEmbeddingClient()
    with patch("httpx.AsyncClient") as mock_cls:
        results = await client.embed_batch([])

    assert results == []
    mock_cls.assert_not_called()


@pytest.mark.asyncio
async def test_dimension_validation_passes():
    """No error raised when the returned vector has the expected dimension."""
    vector = [0.5] * 768
    mock_client = _mock_httpx_client(_make_embed_response([vector]))

    client = OllamaEmbeddingClient(expected_dim=768)
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.embed("test")

    assert len(result) == 768


@pytest.mark.asyncio
async def test_dimension_validation_fails():
    """ValueError raised when returned vector dimension does not match expected_dim."""
    vector = [0.5] * 512
    mock_client = _mock_httpx_client(_make_embed_response([vector]))

    client = OllamaEmbeddingClient(expected_dim=768)
    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ValueError, match="Expected 768-dim"):
            await client.embed("test")


@pytest.mark.asyncio
async def test_connection_error_propagates():
    """httpx.ConnectError from the HTTP layer propagates to the caller."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    client = OllamaEmbeddingClient()
    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.ConnectError):
            await client.embed("hello")


@pytest.mark.asyncio
async def test_model_name_passed_in_request():
    """The configured model name is included in the POST body sent to Ollama."""
    vector = [0.1] * 768
    mock_client = _mock_httpx_client(_make_embed_response([vector], model="all-minilm"))

    client = OllamaEmbeddingClient(model="all-minilm")
    with patch("httpx.AsyncClient", return_value=mock_client):
        await client.embed("test text")

    call_kwargs = mock_client.post.call_args
    sent_json = call_kwargs.kwargs.get("json", {})
    assert sent_json.get("model") == "all-minilm"
    assert sent_json.get("input") == "test text"
