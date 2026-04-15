"""Unit tests for the episode creation worker."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from mnemosyne.db.models.memory import Memory
from tests.fixtures.fake_embedding import FakeEmbeddingClient
from mnemosyne.pipeline.episodes import create_episode
from mnemosyne.providers.in_memory import InMemoryProvider


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient(dim=768)


async def _add_memory(provider: InMemoryProvider, content: str, user_id: uuid.UUID) -> uuid.UUID:
    embedding = await FakeEmbeddingClient(dim=768).embed(content)
    mem = Memory(user_id=user_id, content=content, embedding=embedding)
    return await provider.add(mem)


class TestCreateEpisode:
    @pytest.mark.asyncio
    async def test_uses_provided_summary(self, provider):
        user_id = uuid.uuid4()
        session_id = uuid.uuid4()

        episode = await create_episode(
            provider=provider,
            session_id=session_id,
            user_id=user_id,
            memory_ids=[],
            summary="User searched for running shoes.",
        )

        assert episode.summary == "User searched for running shoes."
        assert episode.session_id == session_id
        assert episode.user_id == user_id

    @pytest.mark.asyncio
    async def test_fallback_summary_from_memory_contents(self, provider):
        user_id = uuid.uuid4()
        session_id = uuid.uuid4()

        mid1 = await _add_memory(provider, "Budget: $200", user_id)
        mid2 = await _add_memory(provider, "Prefers organic", user_id)

        episode = await create_episode(
            provider=provider,
            session_id=session_id,
            user_id=user_id,
            memory_ids=[mid1, mid2],
        )

        assert "Budget: $200" in episode.summary
        assert "Prefers organic" in episode.summary
        assert len(episode.memory_ids) == 2

    @pytest.mark.asyncio
    async def test_empty_session_default_summary(self, provider):
        user_id = uuid.uuid4()
        session_id = uuid.uuid4()

        episode = await create_episode(
            provider=provider,
            session_id=session_id,
            user_id=user_id,
            memory_ids=[],
        )

        assert episode.summary == "Empty session"
        assert episode.memory_ids == []

    @pytest.mark.asyncio
    async def test_episode_has_correct_ids(self, provider):
        user_id = uuid.uuid4()
        session_id = uuid.uuid4()
        mid = await _add_memory(provider, "I wear size 10", user_id)

        episode = await create_episode(
            provider=provider,
            session_id=session_id,
            user_id=user_id,
            memory_ids=[mid],
            summary="User wears size 10.",
        )

        assert episode.user_id == user_id
        assert episode.session_id == session_id
        assert mid in episode.memory_ids

    @pytest.mark.asyncio
    async def test_episode_id_is_unique(self, provider):
        user_id = uuid.uuid4()

        ep1 = await create_episode(
            provider=provider,
            session_id=uuid.uuid4(),
            user_id=user_id,
            memory_ids=[],
            summary="First",
        )
        ep2 = await create_episode(
            provider=provider,
            session_id=uuid.uuid4(),
            user_id=user_id,
            memory_ids=[],
            summary="Second",
        )

        assert ep1.episode_id != ep2.episode_id

    @pytest.mark.asyncio
    async def test_episode_embedding_generated_when_embedder_provided(self, provider, embedder):
        user_id = uuid.uuid4()

        episode = await create_episode(
            provider=provider,
            session_id=uuid.uuid4(),
            user_id=user_id,
            memory_ids=[],
            summary="User asked about shoes.",
            embedder=embedder,
        )

        assert episode.summary_embedding is not None
        assert len(episode.summary_embedding) == 768

    @pytest.mark.asyncio
    async def test_started_at_defaults_to_now(self, provider):
        before = datetime.now(timezone.utc)
        episode = await create_episode(
            provider=provider,
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            memory_ids=[],
        )
        after = datetime.now(timezone.utc)

        assert before <= episode.started_at <= after

    @pytest.mark.asyncio
    async def test_ended_at_passed_through(self, provider):
        ended = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        episode = await create_episode(
            provider=provider,
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            memory_ids=[],
            ended_at=ended,
        )
        assert episode.ended_at == ended
