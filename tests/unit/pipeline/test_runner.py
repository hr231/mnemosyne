"""Unit tests for the pipeline runner."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from mnemosyne.config.settings import Settings
from mnemosyne.db.models.memory import ExtractionResult, MemoryType
from tests.fixtures.fake_embedding import FakeEmbeddingClient
from mnemosyne.pipeline.runner import SessionProcessingResult, process_session
from mnemosyne.providers.in_memory import InMemoryProvider
from mnemosyne.rules.stub import StubRegexExtractor


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient(dim=768)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(rules_dir=tmp_path / "rules", extraction_version="0.1.0")


class TestProcessSession:
    @pytest.mark.asyncio
    async def test_returns_session_processing_result(self, provider, embedder, settings):
        session_id = uuid.uuid4()
        user_id = uuid.uuid4()

        result = await process_session(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            embedder=embedder,
            settings=settings,
        )

        assert isinstance(result, SessionProcessingResult)
        assert result.session_id == session_id

    @pytest.mark.asyncio
    async def test_extraction_from_text(self, provider, embedder, settings):
        session_id = uuid.uuid4()
        user_id = uuid.uuid4()

        result = await process_session(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            embedder=embedder,
            settings=settings,
            text="I like Nike running shoes",
            extractors=[StubRegexExtractor()],
        )

        assert result.memories_created >= 1

    @pytest.mark.asyncio
    async def test_extraction_from_pre_extracted_results(self, provider, embedder, settings):
        session_id = uuid.uuid4()
        user_id = uuid.uuid4()

        pre_extracted = [
            ExtractionResult(
                content="Budget: $300",
                memory_type=MemoryType.FACT,
                importance=0.8,
                rule_id="budget_explicit",
            ),
            ExtractionResult(
                content="Size: M",
                memory_type=MemoryType.FACT,
                importance=0.9,
                rule_id="size_explicit",
            ),
        ]

        result = await process_session(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            embedder=embedder,
            settings=settings,
            extraction_results=pre_extracted,
        )

        assert result.memories_created == 2

    @pytest.mark.asyncio
    async def test_episode_always_created(self, provider, embedder, settings):
        result = await process_session(
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider=provider,
            embedder=embedder,
            settings=settings,
        )

        assert result.episode_created is True

    @pytest.mark.asyncio
    async def test_memories_searchable_after_processing(self, provider, embedder, settings):
        user_id = uuid.uuid4()
        session_id = uuid.uuid4()

        pre_extracted = [
            ExtractionResult(
                content="Budget: $300",
                memory_type=MemoryType.FACT,
                importance=0.8,
                rule_id="budget_explicit",
            ),
        ]

        await process_session(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            embedder=embedder,
            settings=settings,
            extraction_results=pre_extracted,
        )

        query_emb = await embedder.embed("budget")
        results = await provider.search(query_emb, user_id=user_id, limit=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_stats_includes_all_stage_keys(self, provider, embedder, settings):
        result = await process_session(
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider=provider,
            embedder=embedder,
            settings=settings,
        )

        assert hasattr(result, "memories_created")
        assert hasattr(result, "embedded")
        assert hasattr(result, "episode_created")
        assert hasattr(result, "deduped")
        assert hasattr(result, "decay_stats")

    @pytest.mark.asyncio
    async def test_decay_stats_structure(self, provider, embedder, settings):
        result = await process_session(
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider=provider,
            embedder=embedder,
            settings=settings,
        )

        assert "decayed" in result.decay_stats
        assert "archived" in result.decay_stats

    @pytest.mark.asyncio
    async def test_idempotent_no_explosion_on_repeat(self, provider, embedder, settings):
        session_id = uuid.uuid4()
        user_id = uuid.uuid4()
        pre_extracted = [
            ExtractionResult(
                content="I prefer organic products",
                memory_type=MemoryType.PREFERENCE,
                importance=0.7,
                rule_id="pref_organic",
            )
        ]

        # Run twice — should not raise and should not double the memories
        # (content-hash dedup in add() prevents duplication)
        r1 = await process_session(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            embedder=embedder,
            settings=settings,
            extraction_results=pre_extracted,
        )
        r2 = await process_session(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            embedder=embedder,
            settings=settings,
            extraction_results=pre_extracted,
        )

        active = [m for m in provider._memories.values() if m.valid_until is None]
        assert len(active) == 1  # second run deduped at add() time

    @pytest.mark.asyncio
    async def test_session_id_preserved_in_result(self, provider, embedder, settings):
        sid = uuid.uuid4()
        result = await process_session(
            session_id=sid,
            user_id=uuid.uuid4(),
            provider=provider,
            embedder=embedder,
            settings=settings,
        )
        assert result.session_id == sid
