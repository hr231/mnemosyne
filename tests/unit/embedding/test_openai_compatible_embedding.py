from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mnemosyne.embedding.openai_compatible import OpenAICompatibleEmbeddingClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings_response(items: list[dict]) -> dict:
    """Build a minimal OpenAI-compatible /v1/embeddings response dict."""
    return {"data": items, "model": "text-embedding-3-small", "object": "list"}


def _make_item(embedding: list[float], index: int) -> dict:
    return {"embedding": embedding, "index": index, "object": "embedding"}


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
    """embed() returns the first embedding from the /v1/embeddings response."""
    vector = [0.1] * 768
    response = _make_embeddings_response([_make_item(vector, 0)])
    mock_client = _mock_httpx_client(response)

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await client.embed("hello world")

    assert result == vector
    assert len(result) == 768


@pytest.mark.asyncio
async def test_embed_batch():
    """embed_batch() returns all embeddings in correct order."""
    vectors = [[float(i)] * 768 for i in range(3)]
    items = [_make_item(v, i) for i, v in enumerate(vectors)]
    response = _make_embeddings_response(items)
    mock_client = _mock_httpx_client(response)

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await client.embed_batch(["a", "b", "c"])

    assert len(results) == 3
    assert results == vectors


@pytest.mark.asyncio
async def test_embed_batch_sorts_by_index():
    """embed_batch() reorders items by their index field when returned out of order."""
    vec0 = [0.0] * 4
    vec1 = [1.0] * 4
    vec2 = [2.0] * 4
    # Returned out of order: 2, 0, 1
    items = [_make_item(vec2, 2), _make_item(vec0, 0), _make_item(vec1, 1)]
    response = _make_embeddings_response(items)
    mock_client = _mock_httpx_client(response)

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await client.embed_batch(["x", "y", "z"])

    assert results == [vec0, vec1, vec2]


@pytest.mark.asyncio
async def test_embed_empty_batch():
    """embed_batch([]) returns [] without making any HTTP request."""
    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
    )
    with patch("httpx.AsyncClient") as mock_cls:
        results = await client.embed_batch([])

    assert results == []
    mock_cls.assert_not_called()


@pytest.mark.asyncio
async def test_api_key_in_headers():
    """Authorization header is set when api_key is provided."""
    vector = [0.1] * 4
    response = _make_embeddings_response([_make_item(vector, 0)])
    mock_client = _mock_httpx_client(response)

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
        api_key="sk-test-key",
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        await client.embed("test")

    call_kwargs = mock_client.post.call_args
    headers = call_kwargs.kwargs.get("headers", {})
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer sk-test-key"


@pytest.mark.asyncio
async def test_no_auth_header_when_no_key():
    """No Authorization header is sent when api_key is None."""
    vector = [0.1] * 4
    response = _make_embeddings_response([_make_item(vector, 0)])
    mock_client = _mock_httpx_client(response)

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
        api_key=None,
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        await client.embed("test")

    call_kwargs = mock_client.post.call_args
    headers = call_kwargs.kwargs.get("headers", {})
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_dimensions_param_sent():
    """When dimensions is set, it appears in the request JSON payload."""
    vector = [0.1] * 256
    response = _make_embeddings_response([_make_item(vector, 0)])
    mock_client = _mock_httpx_client(response)

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
        dimensions=256,
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        await client.embed("test")

    call_kwargs = mock_client.post.call_args
    sent_json = call_kwargs.kwargs.get("json", {})
    assert sent_json.get("dimensions") == 256


@pytest.mark.asyncio
async def test_dimensions_param_omitted():
    """When dimensions is None, the 'dimensions' key is not present in the request JSON."""
    vector = [0.1] * 768
    response = _make_embeddings_response([_make_item(vector, 0)])
    mock_client = _mock_httpx_client(response)

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
        dimensions=None,
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        await client.embed("test")

    call_kwargs = mock_client.post.call_args
    sent_json = call_kwargs.kwargs.get("json", {})
    assert "dimensions" not in sent_json


@pytest.mark.asyncio
async def test_connection_error_propagates():
    """httpx.ConnectError from the HTTP layer propagates to the caller."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    client = OpenAICompatibleEmbeddingClient(
        base_url="http://api.openai.com",
        model="text-embedding-3-small",
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.ConnectError):
            await client.embed("hello")
