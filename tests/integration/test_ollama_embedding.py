from __future__ import annotations

import os

import httpx
import pytest

from mnemosyne.embedding.ollama import OllamaEmbeddingClient

_OLLAMA_BASE = "http://localhost:11434"


def _embedding_integration_enabled() -> bool:
    return os.environ.get("MNEMOSYNE_EMBEDDING_INTEGRATION", "") in ("1", "true", "yes")


def _ollama_reachable() -> bool:
    try:
        resp = httpx.get(f"{_OLLAMA_BASE}/api/tags", timeout=5.0)
        resp.raise_for_status()
        return True
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return False


@pytest.fixture(autouse=True)
def require_ollama_integration():
    if not _embedding_integration_enabled():
        pytest.skip("MNEMOSYNE_EMBEDDING_INTEGRATION not enabled")
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at http://localhost:11434")


@pytest.mark.asyncio
async def test_embed_returns_float_list():
    """embed() returns a list of floats of length 768 for nomic-embed-text."""
    client = OllamaEmbeddingClient(base_url=_OLLAMA_BASE, model="nomic-embed-text")
    result = await client.embed("hello world")

    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_embed_batch_returns_two_vectors():
    """embed_batch() returns one vector per input text."""
    client = OllamaEmbeddingClient(base_url=_OLLAMA_BASE, model="nomic-embed-text")
    results = await client.embed_batch(["hello", "world"])

    assert len(results) == 2
    for vec in results:
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)
