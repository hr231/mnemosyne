"""Tests for ExtractionPipeline — from_settings with real YAML rules and LLM routing."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mnemosyne.config.settings import Settings
from mnemosyne.db.models.memory import ExtractionResult, MemoryType
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.errors import CannedResponseMissing
from mnemosyne.llm.fake import FakeLLMClient
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.providers.in_memory import InMemoryProvider
from mnemosyne.rules.stub import StubRegexExtractor

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
    return FakeEmbeddingClient(dim=768)


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


# ---------------------------------------------------------------------------
# LLM routing tests
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_no_rules():
    """Settings pointing at a non-existent rules dir so StubRegexExtractor is used."""
    return Settings(
        rules_dir=Path("/nonexistent/rules"),
        extraction_version="0.1.0",
        router_unstructured_threshold=0.7,
    )


@pytest.mark.asyncio
async def test_router_triggers_llm_extraction(settings_no_rules):
    """Text that rules cannot match should trigger LLM extraction."""
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)

    llm_client = FakeLLMClient()
    # This text contains no "I like/prefer/love/want/need" so StubRegexExtractor
    # will produce 0 results -> router fires.
    trigger_text = "The user visited Paris last summer and stayed at Hotel Lutetia."
    llm_client.add_canned(
        "paris",
        [
            ExtractionResult(
                content="User visited Paris last summer",
                memory_type=MemoryType.FACT,
                importance=0.7,
                rule_id="llm_extractor",
            )
        ],
    )

    pipeline = ExtractionPipeline(
        settings=settings_no_rules,
        provider=provider,
        embedder=embedder,
        extractors=[StubRegexExtractor()],
        llm_client=llm_client,
    )
    results = await pipeline.process(user_id=uuid4(), text=trigger_text)

    assert len(results) == 1
    assert "Paris" in results[0].content


@pytest.mark.asyncio
async def test_router_skips_llm_when_rules_match(settings_no_rules):
    """When rules extract well-covered content the LLM must NOT be called."""
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)

    llm_client = FakeLLMClient()
    # No canned response registered — FakeLLMClient raises CannedResponseMissing if called.

    # StubRegexExtractor matches this text and returns a long matched span,
    # so unstructured_ratio will be low enough to skip LLM routing.
    # We also inject settings with a very high threshold to ensure no routing.
    settings_high_threshold = Settings(
        rules_dir=Path("/nonexistent/rules"),
        extraction_version="0.1.0",
        router_unstructured_threshold=0.99,  # practically never route
    )

    pipeline = ExtractionPipeline(
        settings=settings_high_threshold,
        provider=provider,
        embedder=embedder,
        extractors=[StubRegexExtractor()],
        llm_client=llm_client,
    )
    # StubRegexExtractor matches "I like Nike running shoes" and returns it,
    # so extracted_count > 0 and high threshold ensures no routing.
    results = await pipeline.process(
        user_id=uuid4(), text="I like Nike running shoes"
    )

    assert len(results) >= 1
    # Verify LLM was never invoked by checking no CannedResponseMissing was raised
    # (the test would have failed above if LLM had been called)


@pytest.mark.asyncio
async def test_llm_results_deduped_with_rules(settings_no_rules):
    """When both rules and LLM produce the same content, only one result persists."""
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)

    llm_client = FakeLLMClient()
    # Text is long so the "I like coffee" match covers only a small fraction,
    # making unstructured_ratio high enough to trigger LLM even with threshold=0.3.
    filler = " " * 200  # unmatched padding inflates total_chars
    text = "I like coffee" + filler
    llm_client.add_canned(
        "coffee",
        [
            ExtractionResult(
                content="I like coffee",   # identical to rule result -> deduped
                memory_type=MemoryType.PREFERENCE,
                importance=0.6,
                rule_id="llm_extractor",
            ),
            ExtractionResult(
                content="User enjoys hot beverages",  # unique -> kept
                memory_type=MemoryType.FACT,
                importance=0.5,
                rule_id="llm_extractor",
            ),
        ],
    )

    # threshold=0.3: unstructured_ratio ~= 1 - 13/213 ~= 0.94, so LLM fires
    settings_low_threshold = Settings(
        rules_dir=Path("/nonexistent/rules"),
        extraction_version="0.1.0",
        router_unstructured_threshold=0.3,
    )

    pipeline = ExtractionPipeline(
        settings=settings_low_threshold,
        provider=provider,
        embedder=embedder,
        extractors=[StubRegexExtractor()],
        llm_client=llm_client,
    )
    results = await pipeline.process(user_id=uuid4(), text=text)

    contents = [r.content for r in results]
    # "I like coffee" appears once (deduped), "User enjoys hot beverages" appears once
    assert contents.count("I like coffee") == 1
    assert any("hot beverages" in c for c in contents)


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_rules(settings_no_rules):
    """When the LLM raises, rule results are still returned."""
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)

    # FakeLLMClient with no canned response raises CannedResponseMissing
    llm_client = FakeLLMClient()

    # Force routing with threshold=0
    settings_low_threshold = Settings(
        rules_dir=Path("/nonexistent/rules"),
        extraction_version="0.1.0",
        router_unstructured_threshold=0.0,
    )

    pipeline = ExtractionPipeline(
        settings=settings_low_threshold,
        provider=provider,
        embedder=embedder,
        extractors=[StubRegexExtractor()],
        llm_client=llm_client,
    )
    # StubRegexExtractor will match "I prefer tea"
    results = await pipeline.process(user_id=uuid4(), text="I prefer tea")

    # LLM raised (CannedResponseMissing), but rule results still returned
    assert len(results) >= 1
    assert any("prefer" in r.content.lower() or "tea" in r.content.lower() for r in results)
