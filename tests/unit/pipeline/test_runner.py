"""Unit tests for the pipeline runner."""
from __future__ import annotations

import uuid

import pytest

from mnemosyne.config.settings import Settings
from mnemosyne.db.models.memory import ExtractionResult, MemoryType
from mnemosyne.pipeline.runner import SessionProcessingResult, process_session
from mnemosyne.rules.stub import StubRegexExtractor
from tests.fixtures.fake_embedding import FakeEmbeddingClient
from tests.unit.pipeline.conftest import build_in_memory_provider


@pytest.fixture
def provider():
    return build_in_memory_provider()


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
        result = await process_session(
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider=provider,
            embedder=embedder,
            settings=settings,
            text="I like Nike running shoes",
            extractors=[StubRegexExtractor()],
        )

        assert result.memories_created >= 1

    @pytest.mark.asyncio
    async def test_extraction_from_pre_extracted_results(self, provider, embedder, settings):
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
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider=provider,
            embedder=embedder,
            settings=settings,
            extraction_results=pre_extracted,
        )

        assert result.memories_created == 2

    @pytest.mark.asyncio
    async def test_episode_always_persisted(self, provider, embedder, settings):
        user_id = uuid.uuid4()
        session_id = uuid.uuid4()
        result = await process_session(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            embedder=embedder,
            settings=settings,
        )

        assert result.episode_created is True
        episodes = await provider.list_episodes(user_id)
        assert any(e.session_id == session_id for e in episodes)

    @pytest.mark.asyncio
    async def test_episode_upserts_on_rerun(self, provider, embedder, settings):
        user_id = uuid.uuid4()
        session_id = uuid.uuid4()
        for _ in range(2):
            await process_session(
                session_id=session_id,
                user_id=user_id,
                provider=provider,
                embedder=embedder,
                settings=settings,
            )

        episodes = await provider.list_episodes(user_id)
        same_session = [e for e in episodes if e.session_id == session_id]
        assert len(same_session) == 1

    @pytest.mark.asyncio
    async def test_memories_searchable_after_processing(self, provider, embedder, settings):
        user_id = uuid.uuid4()
        pre_extracted = [
            ExtractionResult(
                content="Budget: $300",
                memory_type=MemoryType.FACT,
                importance=0.8,
                rule_id="budget_explicit",
            ),
        ]

        await process_session(
            session_id=uuid.uuid4(),
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
    async def test_result_has_expected_fields(self, provider, embedder, settings):
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
        assert hasattr(result, "contradictions_resolved")

    @pytest.mark.asyncio
    async def test_no_per_session_decay_or_dedup(self, provider, embedder, settings):
        """process_session must not run decay or dedup — those are maintenance only."""
        from unittest.mock import patch

        with patch("mnemosyne.pipeline.runner.apply_decay") as decay, patch(
            "mnemosyne.pipeline.runner.run_dedup"
        ) as dedup:
            await process_session(
                session_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                provider=provider,
                embedder=embedder,
                settings=settings,
            )
        decay.assert_not_called()
        dedup.assert_not_called()

    @pytest.mark.asyncio
    async def test_contradiction_check_runs_when_llm_present(self, provider, embedder, settings):
        from unittest.mock import AsyncMock, patch

        from tests.fixtures.fake_llm import FakeLLMClient

        with patch(
            "mnemosyne.pipeline.runner.run_contradiction_check",
            new=AsyncMock(return_value=0),
        ) as check:
            await process_session(
                session_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                provider=provider,
                embedder=embedder,
                settings=settings,
                extraction_results=[
                    ExtractionResult(content="a fact", importance=0.5, rule_id="r")
                ],
                llm_client=FakeLLMClient(),
            )
        check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_contradiction_check_skipped_without_llm(self, provider, embedder, settings):
        from unittest.mock import AsyncMock, patch

        with patch(
            "mnemosyne.pipeline.runner.run_contradiction_check",
            new=AsyncMock(return_value=0),
        ) as check:
            await process_session(
                session_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                provider=provider,
                embedder=embedder,
                settings=settings,
                extraction_results=[
                    ExtractionResult(content="a fact", importance=0.5, rule_id="r")
                ],
                llm_client=None,
            )
        check.assert_not_called()

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

        await process_session(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            embedder=embedder,
            settings=settings,
            extraction_results=pre_extracted,
        )
        await process_session(
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
