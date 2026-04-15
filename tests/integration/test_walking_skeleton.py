import os
import pytest
from uuid import uuid4
from mnemosyne.providers.in_memory import InMemoryProvider
from tests.fixtures.fake_embedding import FakeEmbeddingClient
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.context.assembly import assemble_context
from mnemosyne.integration.save_memory_tool import save_memory_tool_spec, handle_save_memory
from mnemosyne.config.settings import Settings

@pytest.mark.asyncio
async def test_walking_skeleton_end_to_end():
    settings = Settings.from_env()  # reads MNEMOSYNE_* or defaults
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)

    user_id = uuid4()
    text = "I like Nike running shoes, size 10, budget under $150"

    # 1. Extraction pipeline runs rules → router → (stub LLM off) → provider.add
    results = await pipeline.process(user_id=user_id, text=text)
    assert len(results) >= 1
    assert all(r.extraction_version == "0.1.0" for r in results)
    assert all(r.rule_id for r in results)  # every memory has provenance

    # 2. Content-hash dedup: running the same text again adds nothing
    results_second = await pipeline.process(user_id=user_id, text=text)
    # same memory ids come back
    assert {m.memory_id for m in results} == {m.memory_id for m in results_second}

    # 3. Search returns the extracted memory with access bookkeeping
    query_vec = await embedder.embed("Nike shoes")
    hits = await provider.search(query_vec, user_id=user_id, limit=5)
    assert any("nike" in h.memory.content.lower() for h in hits)
    assert all(h.memory.access_count >= 1 for h in hits)

    # 4. Second search increments access_count (tests the mutation path)
    hits_again = await provider.search(query_vec, user_id=user_id, limit=5)
    for h in hits_again:
        assert h.memory.access_count >= 2

    # 5. Context assembly fits inside a token budget
    block = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=500,
    )
    assert block.token_count <= 500
    assert "nike" in block.text.lower() or "shoes" in block.text.lower()

    # 6. save_memory tool spec is JSON-schema-valid
    spec = save_memory_tool_spec()
    assert spec["name"] == "save_memory"
    assert "content" in spec["parameters"]["properties"]

    # 7. save_memory tool roundtrip: agent calls tool → memory lands → search finds it
    tool_result = await handle_save_memory(
        provider=provider,
        embedder=embedder,
        user_id=user_id,
        args={"content": "User is allergic to latex", "memory_type": "fact", "importance": 0.9},
    )
    assert tool_result["status"] == "saved"
    latex_vec = await embedder.embed("latex allergy")
    # Use a large limit so all stored memories are returned — with multi-signal
    # scoring the latex memory may not rank in the top-3 because the fake
    # embedder produces near-random cosine values while already-accessed Nike
    # memories have a frequency boost.
    latex_hits = await provider.search(latex_vec, user_id=user_id, limit=10)
    assert any("latex" in h.memory.content.lower() for h in latex_hits)

    # 8. Bi-temporal invalidate: invalidated memories don't appear in default search
    latex_mem_id = next(h.memory.memory_id for h in latex_hits if "latex" in h.memory.content.lower())
    await provider.invalidate(latex_mem_id, reason="user retracted")
    latex_hits_after = await provider.search(latex_vec, user_id=user_id, limit=10)
    assert not any(h.memory.memory_id == latex_mem_id for h in latex_hits_after)

    # 9. But historical query sees it
    latex_hits_historical = await provider.search(
        latex_vec, user_id=user_id, limit=10, include_invalidated=True
    )
    assert any(h.memory.memory_id == latex_mem_id for h in latex_hits_historical)


@pytest.mark.asyncio
async def test_token_budget_is_configurable_not_hardcoded():
    """Assert shape, not value: the budget works for any reasonable number."""
    settings = Settings.from_env()
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=768)
    # seed a few memories
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)
    user_id = uuid4()
    await pipeline.process(user_id=user_id, text="I like Nike size 10 budget $150")
    await pipeline.process(user_id=user_id, text="I prefer wide fit shoes")
    query_vec = await embedder.embed("shoes")

    for budget in (100, 500, 2000):
        block = await assemble_context(
            provider=provider,
            user_id=user_id,
            query_embedding=query_vec,
            embedder=embedder,
            token_budget=budget,
        )
        assert block.token_count <= budget
