import pytest
from uuid import uuid4

from mnemosyne.config.settings import Settings
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.llm.openai_compatible import OpenAICompatibleClient
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.providers.in_memory import InMemoryProvider


@pytest.mark.asyncio
async def test_llm_extraction_with_ollama(real_llm_or_skip):
    settings = Settings.from_env()
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    llm_client = OpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        model="gemma4:26b",
        timeout=60.0,
    )
    pipeline = ExtractionPipeline.from_settings(
        settings, provider, embedder, llm_client=llm_client
    )

    user_id = uuid4()
    text = (
        "Last time those shoes felt off, kind of pinched in the middle, "
        "like what I had before but updated somehow — I just want "
        "something that doesn't make me think about my feet."
    )

    results = await pipeline.process(user_id=user_id, text=text)
    assert len(results) >= 1
    assert any(r.rule_id == "llm_extractor" for r in results)
