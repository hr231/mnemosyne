from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.memory import MemoryType
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.integration.save_memory_tool import handle_save_memory, save_memory_tool_spec
from mnemosyne.providers.in_memory import InMemoryProvider


class FakeEmbeddingClient(EmbeddingClient):
    """Returns a deterministic fixed-length embedding for any input."""

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


USER_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# Tool spec tests
# ---------------------------------------------------------------------------


def test_tool_spec_structure():
    spec = save_memory_tool_spec()
    assert spec["name"] == "save_memory"
    assert "description" in spec
    params = spec["parameters"]
    assert params["type"] == "object"
    assert "content" in params["required"]
    props = params["properties"]
    assert "content" in props
    assert "memory_type" in props
    assert "importance" in props


def test_tool_spec_has_source_session_id():
    spec = save_memory_tool_spec()
    props = spec["parameters"]["properties"]
    assert "source_session_id" in props
    assert props["source_session_id"]["type"] == "string"


def test_tool_spec_memory_type_enum():
    spec = save_memory_tool_spec()
    enum_values = spec["parameters"]["properties"]["memory_type"]["enum"]
    expected = {t.value for t in MemoryType}
    assert set(enum_values) == expected


# ---------------------------------------------------------------------------
# Handler happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_happy_path():
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient()
    result = await handle_save_memory(
        provider,
        embedder,
        USER_ID,
        {"content": "User prefers dark mode", "memory_type": "preference", "importance": 0.8},
    )
    assert result["status"] == "saved"
    assert "memory_id" in result
    # memory_id must be a valid UUID string
    uuid.UUID(result["memory_id"])


@pytest.mark.asyncio
async def test_handle_with_session_id():
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient()
    session_id = uuid.uuid4()
    result = await handle_save_memory(
        provider,
        embedder,
        USER_ID,
        {
            "content": "User lives in Berlin",
            "memory_type": "fact",
            "importance": 0.6,
            "source_session_id": str(session_id),
        },
    )
    assert result["status"] == "saved"
    mem_id = uuid.UUID(result["memory_id"])
    stored = await provider.get_by_id(mem_id)
    assert stored is not None
    assert stored.source_session_id == session_id


# ---------------------------------------------------------------------------
# Handler error-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_empty_content_error():
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient()
    result = await handle_save_memory(provider, embedder, USER_ID, {"content": ""})
    assert result["status"] == "error"
    assert "content" in result["error"]


@pytest.mark.asyncio
async def test_handle_missing_content_error():
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient()
    result = await handle_save_memory(provider, embedder, USER_ID, {})
    assert result["status"] == "error"
    assert "content" in result["error"]


@pytest.mark.asyncio
async def test_handle_invalid_memory_type_error():
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient()
    result = await handle_save_memory(
        provider,
        embedder,
        USER_ID,
        {"content": "x", "memory_type": "invalid"},
    )
    assert result["status"] == "error"
    assert "memory_type" in result["error"]


@pytest.mark.asyncio
async def test_handle_importance_out_of_range():
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient()
    result = await handle_save_memory(
        provider,
        embedder,
        USER_ID,
        {"content": "x", "importance": 1.5},
    )
    assert result["status"] == "error"
    assert "importance" in result["error"]


@pytest.mark.asyncio
async def test_handle_invalid_session_id():
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient()
    result = await handle_save_memory(
        provider,
        embedder,
        USER_ID,
        {"content": "x", "source_session_id": "not-a-uuid"},
    )
    assert result["status"] == "error"
    assert "source_session_id" in result["error"]
