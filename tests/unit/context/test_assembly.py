from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne.context.assembly import assemble_context
from mnemosyne.db.models.memory import Memory, MemoryType
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.providers.in_memory import InMemoryProvider


async def _add_memory(
    provider: InMemoryProvider,
    embedder: FakeEmbeddingClient,
    user_id: uuid.UUID,
    content: str,
    importance: float = 0.5,
    created_at: datetime | None = None,
) -> Memory:
    embedding = await embedder.embed(content)
    mem = Memory(
        user_id=user_id,
        content=content,
        importance=importance,
        embedding=embedding,
        created_at=created_at or datetime.now(timezone.utc),
    )
    await provider.add(mem)
    return mem


@pytest.mark.asyncio
async def test_assembles_within_budget() -> None:
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    user_id = uuid.uuid4()

    for i in range(5):
        await _add_memory(provider, embedder, user_id, f"Memory content number {i} about various things")

    query_vec = await embedder.embed("memory content")
    block = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
    )
    assert block.token_count <= 500


@pytest.mark.asyncio
async def test_sections_populated() -> None:
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    user_id = uuid.uuid4()

    await _add_memory(provider, embedder, user_id, "User prefers dark mode", importance=0.8)
    await _add_memory(provider, embedder, user_id, "User is a software engineer", importance=0.6)

    query_vec = await embedder.embed("user preferences")
    block = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
    )
    assert block.sections is not None
    assert len(block.sections) >= 1


@pytest.mark.asyncio
async def test_profile_section_has_high_importance() -> None:
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    user_id = uuid.uuid4()

    high_imp_content = "User is a professional chef"
    low_imp_content = "User sometimes watches TV"

    await _add_memory(provider, embedder, user_id, high_imp_content, importance=0.9)
    await _add_memory(provider, embedder, user_id, low_imp_content, importance=0.3)

    query_vec = await embedder.embed("chef")
    block = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
    )
    assert block.sections is not None
    profile_sections = [s for s in block.sections if s.name == "profile"]
    assert len(profile_sections) == 1
    assert high_imp_content in profile_sections[0].content


@pytest.mark.asyncio
async def test_small_budget_truncates() -> None:
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    user_id = uuid.uuid4()

    # Add memories with long content that will exceed a tiny budget
    for i in range(10):
        await _add_memory(
            provider, embedder, user_id,
            f"This is a fairly detailed memory about topic number {i} with extra words to fill tokens",
            importance=0.5,
        )

    query_vec = await embedder.embed("memory about topic")
    block = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=50,
    )
    assert block.token_count <= 50


@pytest.mark.asyncio
async def test_empty_provider() -> None:
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    user_id = uuid.uuid4()

    query_vec = await embedder.embed("anything")
    block = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
    )
    assert block.text == ""
    assert block.token_count == 0


@pytest.mark.asyncio
async def test_different_users_isolated() -> None:
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()

    await _add_memory(provider, embedder, user_a, "User A loves cycling")
    await _add_memory(provider, embedder, user_b, "User B loves painting")

    query_vec = await embedder.embed("hobbies")

    block_a = await assemble_context(
        provider=provider,
        user_id=user_a,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
    )
    block_b = await assemble_context(
        provider=provider,
        user_id=user_b,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
    )

    assert "cycling" in block_a.text.lower()
    assert "cycling" not in block_b.text.lower()
    assert "painting" in block_b.text.lower()
    assert "painting" not in block_a.text.lower()
