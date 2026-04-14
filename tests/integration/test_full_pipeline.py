from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("MNEMOSYNE_FULL_INTEGRATION") == "1"
        and os.environ.get("MNEMOSYNE_PG_DSN")
    ),
    reason="MNEMOSYNE_FULL_INTEGRATION=1 and MNEMOSYNE_PG_DSN required",
)


@pytest.mark.asyncio
async def test_full_pipeline_extract_search_context():
    """End-to-end: extract -> embed -> store -> entity extract -> search -> context."""
    from mnemosyne.config.settings import Settings
    from mnemosyne.embedding.ollama import OllamaEmbeddingClient
    from mnemosyne.providers.postgres import PostgresMemoryProvider
    from mnemosyne.db.repositories.entity import PostgresEntityStore
    from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
    from mnemosyne.pipeline.ner.spacy_extractor import extract_entities_spacy
    from mnemosyne.pipeline.ner.router import merge_entities
    from mnemosyne.pipeline.ner.resolver import resolve_entities
    from mnemosyne.retrieval.entity_search import entity_aware_search
    from mnemosyne.context.assembly import assemble_context

    dsn = os.environ["MNEMOSYNE_PG_DSN"]
    settings = Settings.from_env()

    provider = await PostgresMemoryProvider.connect(dsn)
    entity_store = PostgresEntityStore(provider._pool)
    embedder = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
        expected_dim=768,
    )
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)

    try:
        async with provider._pool.acquire() as conn:
            await conn.execute("TRUNCATE memory.entity_mentions CASCADE")
            await conn.execute("TRUNCATE memory.entities CASCADE")
            await conn.execute("TRUNCATE memory.memory_history CASCADE")
            await conn.execute("TRUNCATE memory.memories CASCADE")

        user_id = uuid.uuid4()
        text = "I love Nike running shoes, size 10, budget under $150. I'm allergic to latex."

        # Step 1: Extract memories
        results = await pipeline.process(user_id=user_id, text=text)
        assert len(results) >= 1

        # Step 2: Extract and resolve entities
        memory_id = results[0].memory_id
        raw_entities = extract_entities_spacy(text)
        # Even if spaCy is unavailable, continue to test the rest of the flow
        if raw_entities:
            merged = merge_entities(raw_entities, [], [])
            resolved = await resolve_entities(
                merged, user_id, memory_id, entity_store, embedder
            )
            assert len(resolved) >= 1

        # Step 3: Search via entity-aware search
        query = "Nike shoes"
        query_vec = await embedder.embed(query)
        search_results = await entity_aware_search(
            provider, entity_store, query, query_vec, user_id, embedder, limit=10
        )
        assert len(search_results) >= 1

        # Step 4: Context assembly
        context = await assemble_context(
            provider, user_id, query_vec, embedder,
            token_budget=2000,
            entity_store=entity_store,
            query_text=query,
        )
        assert context.token_count > 0
        assert context.text.strip() != ""

    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_full_pipeline_dedup_idempotency():
    """Running extraction twice on identical text yields the same memory set."""
    from mnemosyne.config.settings import Settings
    from mnemosyne.embedding.ollama import OllamaEmbeddingClient
    from mnemosyne.providers.postgres import PostgresMemoryProvider
    from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline

    dsn = os.environ["MNEMOSYNE_PG_DSN"]
    settings = Settings.from_env()

    provider = await PostgresMemoryProvider.connect(dsn)
    embedder = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
        expected_dim=768,
    )
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)

    try:
        async with provider._pool.acquire() as conn:
            await conn.execute("TRUNCATE memory.memory_history CASCADE")
            await conn.execute("TRUNCATE memory.memories CASCADE")

        user_id = uuid.uuid4()
        text = "I prefer wide-fit running shoes with good arch support."

        first = await pipeline.process(user_id=user_id, text=text)
        second = await pipeline.process(user_id=user_id, text=text)

        assert len(first) >= 1
        # Idempotency: same memory IDs returned on second call
        assert {r.memory_id for r in first} == {r.memory_id for r in second}

    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_full_pipeline_bitemporal_invalidation():
    """Invalidated memories are excluded from default search but visible historically."""
    from mnemosyne.config.settings import Settings
    from mnemosyne.embedding.ollama import OllamaEmbeddingClient
    from mnemosyne.providers.postgres import PostgresMemoryProvider
    from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline

    dsn = os.environ["MNEMOSYNE_PG_DSN"]
    settings = Settings.from_env()

    provider = await PostgresMemoryProvider.connect(dsn)
    embedder = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
        expected_dim=768,
    )
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)

    try:
        async with provider._pool.acquire() as conn:
            await conn.execute("TRUNCATE memory.memory_history CASCADE")
            await conn.execute("TRUNCATE memory.memories CASCADE")

        user_id = uuid.uuid4()
        text = "I am allergic to peanuts."
        results = await pipeline.process(user_id=user_id, text=text)
        assert len(results) >= 1

        memory_id = results[0].memory_id

        query_vec = await embedder.embed("peanut allergy")
        hits_before = await provider.search(query_vec, user_id=user_id, limit=10)
        assert any(h.memory.memory_id == memory_id for h in hits_before)

        await provider.invalidate(memory_id, reason="allergy resolved")

        hits_after = await provider.search(query_vec, user_id=user_id, limit=10)
        assert not any(h.memory.memory_id == memory_id for h in hits_after)

        hits_historical = await provider.search(
            query_vec, user_id=user_id, limit=10, include_invalidated=True
        )
        assert any(h.memory.memory_id == memory_id for h in hits_historical)

    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_full_pipeline_reflection_after_accumulation():
    """Accumulate enough memories and verify reflection check runs without error."""
    from mnemosyne.config.settings import Settings
    from mnemosyne.embedding.ollama import OllamaEmbeddingClient
    from mnemosyne.providers.postgres import PostgresMemoryProvider
    from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
    from mnemosyne.pipeline.reflection import should_generate_reflection

    dsn = os.environ["MNEMOSYNE_PG_DSN"]
    settings = Settings.from_env()

    provider = await PostgresMemoryProvider.connect(dsn)
    embedder = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
        expected_dim=768,
    )
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)

    try:
        async with provider._pool.acquire() as conn:
            await conn.execute("TRUNCATE memory.memory_history CASCADE")
            await conn.execute("TRUNCATE memory.memories CASCADE")

        user_id = uuid.uuid4()

        texts = [
            "I prefer Nike running shoes",
            "My budget is always under $200",
            "I'm a size 10 in most brands",
            "I like minimalist design",
            "I always check reviews before buying",
            "I prefer comfort over style",
            "I don't like synthetic materials",
            "My favorite color for shoes is black",
            "I run 5 miles every morning",
            "I need good arch support",
            "I'm training for a marathon",
            "I prefer lightweight shoes",
            "I've been wearing Nike for 10 years",
            "I switched from Adidas last year",
            "I care about durability",
            "I hate shoes that squeak",
            "I need waterproof shoes for winter",
            "I prefer lace-up over slip-on",
            "I buy new shoes every 6 months",
            "I trust recommendations from Runner's World",
        ]
        for t in texts:
            await pipeline.process(user_id=user_id, text=t)

        should_reflect = await should_generate_reflection(provider, user_id)
        assert isinstance(should_reflect, bool)

    finally:
        await provider.close()
