"""Tests for ExtractionPipeline.from_settings with real YAML rules."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mnemosyne.config.settings import Settings
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.providers.in_memory import InMemoryProvider

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_RULES_CORE_DIR = _REPO_ROOT / "rules" / "core"


@pytest.fixture
def settings():
    return Settings(rules_dir=_RULES_CORE_DIR, extraction_version="0.1.0")


@pytest.fixture
def provider():
    return InMemoryProvider()


@pytest.fixture
def embedder():
    return FakeEmbeddingClient(dim=1536)


@pytest.mark.asyncio
async def test_from_settings_loads_rules_and_produces_budget_result(
    settings, provider, embedder
):
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)
    user_id = uuid4()

    results = await pipeline.process(user_id=user_id, text="My budget is $300")

    assert len(results) >= 1
    assert any("300" in r.content for r in results), (
        f"Expected a budget extraction containing '300'. Got: {[r.content for r in results]}"
    )


@pytest.mark.asyncio
async def test_from_settings_loads_rules_and_produces_preference_result(
    settings, provider, embedder
):
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)
    user_id = uuid4()

    results = await pipeline.process(user_id=user_id, text="I prefer organic products.")

    assert len(results) >= 1
    assert any("organic" in r.content.lower() for r in results), (
        f"Expected preference extraction with 'organic'. Got: {[r.content for r in results]}"
    )


@pytest.mark.asyncio
async def test_from_settings_extraction_version_stamped(settings, provider, embedder):
    """Every result must carry the extraction_version from settings."""
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)
    user_id = uuid4()

    results = await pipeline.process(
        user_id=user_id,
        text="I like Nike running shoes, budget under $150",
    )
    assert len(results) >= 1
    assert all(r.extraction_version == "0.1.0" for r in results), (
        f"Some results missing correct extraction_version: "
        f"{[(r.rule_id, r.extraction_version) for r in results]}"
    )


@pytest.mark.asyncio
async def test_from_settings_rule_id_set_on_results(settings, provider, embedder):
    """Every result must have a non-empty rule_id for provenance."""
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)
    user_id = uuid4()

    results = await pipeline.process(
        user_id=user_id,
        text="My budget is $200",
    )
    assert all(r.rule_id for r in results), (
        f"Some results missing rule_id: {results}"
    )


@pytest.mark.asyncio
async def test_from_settings_falls_back_to_stub_when_rules_dir_missing(
    provider, embedder
):
    """When rules_dir does not exist the stub extractor is used as fallback."""
    settings = Settings(
        rules_dir=Path("/nonexistent/path/to/rules"),
        extraction_version="0.1.0",
    )
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)
    user_id = uuid4()

    # Stub matches "I like/prefer/love/want/need"
    results = await pipeline.process(user_id=user_id, text="I like Nike running shoes")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_from_settings_combined_budget_and_preference(settings, provider, embedder):
    """Both budget and preference rules should fire on the canonical walking-skeleton text."""
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)
    user_id = uuid4()

    results = await pipeline.process(
        user_id=user_id,
        text="I like Nike running shoes, size 10, budget under $150",
    )
    assert len(results) >= 2, (
        f"Expected at least 2 extractions (budget + preference). Got: "
        f"{[r.content for r in results]}"
    )
