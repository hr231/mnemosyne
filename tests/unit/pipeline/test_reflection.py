from __future__ import annotations

import json
import uuid

import pytest

from mnemosyne.db.models.memory import Memory, MemoryType
from tests.fixtures.fake_embedding import FakeEmbeddingClient
from tests.fixtures.fake_llm import FakeLLMClient
from mnemosyne.providers.in_memory import InMemoryProvider
from mnemosyne.pipeline.reflection import (
    generate_reflections,
    should_generate_reflection,
    MAX_REFLECTION_DEPTH,
)


@pytest.fixture
def provider():
    return InMemoryProvider()


@pytest.fixture
def embedder():
    return FakeEmbeddingClient(dim=768)


@pytest.fixture
def llm_client():
    client = FakeLLMClient()
    client.add_canned("memories about a user", [])
    return client


async def _seed_memories(provider, embedder, user_id, count=20, importance=0.8):
    """Seed provider with test memories."""
    for i in range(count):
        emb = await embedder.embed(f"memory {i}")
        mem = Memory(
            user_id=user_id,
            content=f"Test memory {i} about user preferences",
            memory_type=MemoryType.FACT,
            importance=importance,
            embedding=emb,
        )
        await provider.add(mem)


class TestShouldGenerateReflection:
    async def test_triggers_when_threshold_reached(self, provider, embedder):
        user_id = uuid.uuid4()
        # 20 memories at importance 0.8 -> scaled sum = 20 * 8.0 = 160 >= 150
        await _seed_memories(provider, embedder, user_id, count=20, importance=0.8)
        assert await should_generate_reflection(provider, user_id) is True

    async def test_does_not_trigger_below_threshold(self, provider, embedder):
        user_id = uuid.uuid4()
        # 10 memories at importance 0.5 -> scaled sum = 10 * 5.0 = 50 < 150
        await _seed_memories(provider, embedder, user_id, count=10, importance=0.5)
        assert await should_generate_reflection(provider, user_id) is False

    async def test_excludes_deep_reflections(self, provider, embedder):
        user_id = uuid.uuid4()
        # Seed 20 memories at depth 2 with importance 0.9 -> should be excluded
        for i in range(20):
            emb = await embedder.embed(f"reflection {i}")
            mem = Memory(
                user_id=user_id,
                content=f"Deep reflection {i}",
                memory_type=MemoryType.REFLECTION,
                importance=0.9,
                embedding=emb,
                metadata={"reflection_depth": 2},
            )
            await provider.add(mem)
        # Even though total importance is high, depth-2 reflections are excluded
        assert await should_generate_reflection(provider, user_id) is False


class TestGenerateReflections:
    async def test_generates_reflections(self, provider, embedder):
        user_id = uuid.uuid4()
        await _seed_memories(provider, embedder, user_id, count=5)

        insights = ["User prefers comfortable shoes", "User has moderate budget"]

        class ReflectionLLM(FakeLLMClient):
            async def complete(self, prompt, **kwargs):
                return json.dumps(insights)

        result = await generate_reflections(provider, user_id, ReflectionLLM(), embedder)
        assert len(result) == 2
        assert all(r.memory_type == MemoryType.REFLECTION for r in result)
        assert all(r.importance == 0.9 for r in result)
        assert all(r.metadata.get("reflection_depth") == 1 for r in result)

    async def test_reflection_depth_increments(self, provider, embedder):
        user_id = uuid.uuid4()
        # Seed depth-1 reflections
        for i in range(3):
            emb = await embedder.embed(f"reflection {i}")
            mem = Memory(
                user_id=user_id,
                content=f"Reflection {i}",
                memory_type=MemoryType.REFLECTION,
                importance=0.9,
                embedding=emb,
                metadata={"reflection_depth": 1},
            )
            await provider.add(mem)

        class DepthLLM(FakeLLMClient):
            async def complete(self, prompt, **kwargs):
                return json.dumps(["Meta-reflection about patterns"])

        result = await generate_reflections(provider, user_id, DepthLLM(), embedder)
        assert len(result) >= 1
        assert result[0].metadata["reflection_depth"] == 2

    async def test_handles_llm_failure(self, provider, embedder):
        user_id = uuid.uuid4()
        await _seed_memories(provider, embedder, user_id, count=5)

        class FailingLLM(FakeLLMClient):
            async def complete(self, prompt, **kwargs):
                raise RuntimeError("LLM down")

        result = await generate_reflections(provider, user_id, FailingLLM(), embedder)
        assert result == []

    async def test_no_memories_returns_empty(self, provider, embedder):
        user_id = uuid.uuid4()

        class NoopLLM(FakeLLMClient):
            async def complete(self, prompt, **kwargs):
                return "[]"

        result = await generate_reflections(provider, user_id, NoopLLM(), embedder)
        assert result == []


class TestMaybeRunReflection:
    async def test_fires_when_threshold_crossed(self, provider, embedder):
        from mnemosyne.pipeline.reflection import maybe_run_reflection

        class InsightLLM(FakeLLMClient):
            async def complete(self, prompt, **kwargs):
                return json.dumps(["User values comfort", "User prefers running"])

        user_id = uuid.uuid4()
        # 20 memories * 0.8 importance = 160 scaled sum >= 150 threshold
        await _seed_memories(provider, embedder, user_id, count=20, importance=0.8)

        count = await maybe_run_reflection(provider, InsightLLM(), embedder, user_id)
        assert count == 2

        persisted_reflections = [
            m
            for m in provider._memories.values()
            if m.user_id == user_id and m.memory_type == MemoryType.REFLECTION
        ]
        assert len(persisted_reflections) == 2
        for r in persisted_reflections:
            assert r.metadata.get("reflection_depth") == 1
            assert r.source_memory_ids

    async def test_skips_below_threshold(self, provider, embedder):
        from mnemosyne.pipeline.reflection import maybe_run_reflection

        class TrackingLLM(FakeLLMClient):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def complete(self, prompt, **kwargs):
                self.calls += 1
                return "[]"

        user_id = uuid.uuid4()
        # 3 memories * 0.1 importance = 3 scaled sum << 150
        await _seed_memories(provider, embedder, user_id, count=3, importance=0.1)

        llm = TrackingLLM()
        count = await maybe_run_reflection(provider, llm, embedder, user_id)
        assert count == 0
        assert llm.calls == 0, "LLM must NOT be called when the trigger does not fire"

    async def test_depth_cap_blocks_trigger(self, provider, embedder):
        from mnemosyne.pipeline.reflection import maybe_run_reflection

        class TrackingLLM(FakeLLMClient):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def complete(self, prompt, **kwargs):
                self.calls += 1
                return json.dumps(["ignored"])

        user_id = uuid.uuid4()
        # Seed 20 depth-2 reflections at importance 0.9 — total importance
        # is very high but depth-2 memories are excluded from the sum.
        for i in range(20):
            emb = await embedder.embed(f"deep {i}")
            mem = Memory(
                user_id=user_id,
                content=f"deep reflection {i}",
                memory_type=MemoryType.REFLECTION,
                importance=0.9,
                embedding=emb,
                metadata={"reflection_depth": MAX_REFLECTION_DEPTH},
            )
            await provider.add(mem)

        llm = TrackingLLM()
        count = await maybe_run_reflection(provider, llm, embedder, user_id)
        assert count == 0
        assert llm.calls == 0, "depth-2 memories must not contribute to the trigger"
