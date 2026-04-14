from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.memory import Memory, MemoryType
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.llm.fake import FakeLLMClient
from mnemosyne.providers.in_memory import InMemoryProvider
from mnemosyne.pipeline.contradiction import (
    ContradictionAction,
    CONTRADICTION_SIMILARITY_MIN,
    CONTRADICTION_SIMILARITY_MAX,
    _parse_action,
    _execute_action,
    detect_contradictions,
    resolve_contradiction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_memory(
    provider: InMemoryProvider,
    embedder: FakeEmbeddingClient,
    user_id: uuid.UUID,
    content: str,
    importance: float = 0.5,
    memory_type: MemoryType = MemoryType.FACT,
) -> Memory:
    """Create a memory with a real embedding and persist it."""
    emb = await embedder.embed(content)
    mem = Memory(
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        importance=importance,
        embedding=emb,
    )
    mem_id = await provider.add(mem)
    # Return the stored copy so memory_id is canonical.
    return provider._memories[mem_id]


class FakeActionLLM(FakeLLMClient):
    """LLM client that returns a fixed action token from complete()."""

    def __init__(self, action_token: str) -> None:
        super().__init__()
        self._action_token = action_token

    async def complete(self, prompt: str, **kwargs) -> str:
        return self._action_token


class FailingLLM(FakeLLMClient):
    """LLM client whose complete() always raises."""

    async def complete(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("LLM unavailable")


# ---------------------------------------------------------------------------
# _parse_action
# ---------------------------------------------------------------------------


class TestParseAction:
    def test_supersede_exact(self):
        assert _parse_action("SUPERSEDE") == ContradictionAction.SUPERSEDE

    def test_keep_both_exact(self):
        assert _parse_action("KEEP_BOTH") == ContradictionAction.KEEP_BOTH

    def test_merge_exact(self):
        assert _parse_action("MERGE") == ContradictionAction.MERGE

    def test_keep_old_exact(self):
        assert _parse_action("KEEP_OLD") == ContradictionAction.KEEP_OLD

    def test_parse_action_with_extra_text(self):
        """Action token embedded in prose should still parse correctly."""
        assert _parse_action("The answer is SUPERSEDE because B is more recent") == ContradictionAction.SUPERSEDE

    def test_parse_action_default(self):
        """Unrecognised response defaults to KEEP_BOTH."""
        assert _parse_action("I don't know what to do here.") == ContradictionAction.KEEP_BOTH

    def test_parse_action_case_insensitive(self):
        assert _parse_action("supersede") == ContradictionAction.SUPERSEDE

    def test_parse_action_lowercase_keep_old(self):
        assert _parse_action("keep_old is the right choice") == ContradictionAction.KEEP_OLD


# ---------------------------------------------------------------------------
# _execute_action
# ---------------------------------------------------------------------------


class TestExecuteAction:
    """Tests for _execute_action in isolation, bypassing detection."""

    async def test_supersede_action(self):
        """SUPERSEDE invalidates old, keeps new."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)

        old_mem = await _make_memory(provider, embedder, user_id, "Budget is $200", importance=0.6)
        new_mem = await _make_memory(provider, embedder, user_id, "Budget is $300", importance=0.7)

        await _execute_action(ContradictionAction.SUPERSEDE, new_mem, old_mem, provider, embedder)

        stored_old = await provider.get_by_id(old_mem.memory_id)
        stored_new = await provider.get_by_id(new_mem.memory_id)

        assert stored_old is not None and stored_old.valid_until is not None, "old should be invalidated"
        assert stored_old.metadata.get("invalidation_reason") == "contradiction_superseded"
        assert stored_new is not None and stored_new.valid_until is None, "new should remain active"

    async def test_keep_both_action(self):
        """KEEP_BOTH leaves both memories active."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)

        old_mem = await _make_memory(provider, embedder, user_id, "Prefers blue", importance=0.5)
        new_mem = await _make_memory(provider, embedder, user_id, "Prefers red", importance=0.5)

        await _execute_action(ContradictionAction.KEEP_BOTH, new_mem, old_mem, provider, embedder)

        stored_old = await provider.get_by_id(old_mem.memory_id)
        stored_new = await provider.get_by_id(new_mem.memory_id)

        assert stored_old is not None and stored_old.valid_until is None, "old should remain active"
        assert stored_new is not None and stored_new.valid_until is None, "new should remain active"

    async def test_merge_action(self):
        """MERGE creates a merged memory and invalidates both originals."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)

        old_mem = await _make_memory(provider, embedder, user_id, "Budget is $200", importance=0.6)
        new_mem = await _make_memory(provider, embedder, user_id, "Budget is $250", importance=0.7)

        initial_count = len(provider._memories)
        await _execute_action(ContradictionAction.MERGE, new_mem, old_mem, provider, embedder)

        # Both originals invalidated.
        stored_old = await provider.get_by_id(old_mem.memory_id)
        stored_new = await provider.get_by_id(new_mem.memory_id)
        assert stored_old is not None and stored_old.valid_until is not None
        assert stored_old.metadata.get("invalidation_reason") == "contradiction_merged"
        assert stored_new is not None and stored_new.valid_until is not None
        assert stored_new.metadata.get("invalidation_reason") == "contradiction_merged"

        # A new merged memory was created.
        assert len(provider._memories) == initial_count + 1

        merged = next(
            m for m in provider._memories.values()
            if m.memory_id not in (old_mem.memory_id, new_mem.memory_id)
        )
        assert merged.valid_until is None, "merged memory should be active"
        assert "Budget is $200" in merged.content
        assert "Budget is $250" in merged.content
        assert merged.importance == max(old_mem.importance, new_mem.importance)
        assert old_mem.memory_id in merged.source_memory_ids
        assert new_mem.memory_id in merged.source_memory_ids

    async def test_keep_old_action(self):
        """KEEP_OLD invalidates new, keeps old."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)

        old_mem = await _make_memory(provider, embedder, user_id, "Shoe size 10", importance=0.7)
        new_mem = await _make_memory(provider, embedder, user_id, "Shoe size 11", importance=0.5)

        await _execute_action(ContradictionAction.KEEP_OLD, new_mem, old_mem, provider, embedder)

        stored_old = await provider.get_by_id(old_mem.memory_id)
        stored_new = await provider.get_by_id(new_mem.memory_id)

        assert stored_old is not None and stored_old.valid_until is None, "old should remain active"
        assert stored_new is not None and stored_new.valid_until is not None, "new should be invalidated"
        assert stored_new.metadata.get("invalidation_reason") == "contradiction_rejected"


# ---------------------------------------------------------------------------
# resolve_contradiction
# ---------------------------------------------------------------------------


class TestResolveContradiction:
    async def test_supersede_via_llm(self):
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)
        llm = FakeActionLLM("SUPERSEDE")

        old_mem = await _make_memory(provider, embedder, user_id, "Budget is $200")
        new_mem = await _make_memory(provider, embedder, user_id, "Budget is $300")

        action = await resolve_contradiction(new_mem, old_mem, 0.75, provider, llm, embedder)

        assert action == ContradictionAction.SUPERSEDE
        stored_old = await provider.get_by_id(old_mem.memory_id)
        assert stored_old is not None and stored_old.valid_until is not None

    async def test_llm_failure_defaults_to_keep_both(self):
        """When LLM raises, resolve_contradiction falls back to KEEP_BOTH."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)
        llm = FailingLLM()

        old_mem = await _make_memory(provider, embedder, user_id, "Budget is $200")
        new_mem = await _make_memory(provider, embedder, user_id, "Budget is $300")

        action = await resolve_contradiction(new_mem, old_mem, 0.75, provider, llm, embedder)

        assert action == ContradictionAction.KEEP_BOTH
        # Both should still be valid.
        assert (await provider.get_by_id(old_mem.memory_id)).valid_until is None
        assert (await provider.get_by_id(new_mem.memory_id)).valid_until is None

    async def test_merge_embeds_content(self):
        """MERGE action must call embedder.embed with the merged text."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        llm = FakeActionLLM("MERGE")

        embed_calls: list[str] = []

        class TrackingEmbedder(FakeEmbeddingClient):
            async def embed(self, text: str) -> list[float]:
                embed_calls.append(text)
                return await super().embed(text)

        embedder = TrackingEmbedder(dim=768)

        old_mem = await _make_memory(provider, embedder, user_id, "Budget is $200", importance=0.6)
        new_mem = await _make_memory(provider, embedder, user_id, "Budget is $250", importance=0.7)

        # Reset tracking after seeding — only interested in MERGE embed call.
        embed_calls.clear()

        await resolve_contradiction(new_mem, old_mem, 0.75, provider, llm, embedder)

        assert any("Budget is $200" in c and "Budget is $250" in c for c in embed_calls), (
            "embedder.embed must be called with the merged content string"
        )


# ---------------------------------------------------------------------------
# detect_contradictions
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    async def test_detect_no_contradictions_unrelated(self):
        """Two completely unrelated memories should return an empty candidate list."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)

        # These texts hash to very different vectors — similarity will be far
        # below the 0.70 threshold.
        mem_a = await _make_memory(provider, embedder, user_id, "The capital of France is Paris")
        mem_b = await _make_memory(provider, embedder, user_id, "My dog is named Rex")

        candidates = await detect_contradictions(mem_a, provider, embedder, use_nli=False)

        # mem_b's cosine similarity to mem_a is not in [0.70, 0.89], so no candidates.
        assert candidates == []

    async def test_no_embedding_returns_empty(self):
        """Memory without an embedding produces no candidates."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)

        mem = Memory(
            user_id=user_id,
            content="Budget is $200",
            embedding=None,
        )
        candidates = await detect_contradictions(mem, provider, embedder, use_nli=False)
        assert candidates == []

    async def test_self_not_returned(self):
        """The new memory itself is never returned as its own candidate."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)

        mem = await _make_memory(provider, embedder, user_id, "Budget is $200")
        candidates = await detect_contradictions(mem, provider, embedder, use_nli=False)

        ids = [m.memory_id for m, _ in candidates]
        assert mem.memory_id not in ids

    async def test_similarity_band_filter(self):
        """Only memories in the [0.70, 0.89] band are returned as candidates."""
        user_id = uuid.uuid4()
        provider = InMemoryProvider()
        embedder = FakeEmbeddingClient(dim=768)

        # Create the target memory.
        target_content = "User budget is two hundred dollars"
        target_emb = await embedder.embed(target_content)
        target = Memory(
            user_id=user_id,
            content=target_content,
            embedding=target_emb,
        )

        # Inject a memory whose embedding is synthetically positioned inside band.
        import math

        def _make_vector_at_sim(base: list[float], target_sim: float) -> list[float]:
            """Construct a unit vector with a known cosine similarity to base."""
            n = len(base)
            # Orthogonal component
            perp = [0.0] * n
            for i in range(1, n):
                perp[i - 1] += base[i]
                perp[i] -= base[i - 1]
            norm_perp = math.sqrt(sum(x * x for x in perp)) or 1.0
            perp = [x / norm_perp for x in perp]
            # Combine: cos * base_unit + sin * perp
            sin_val = math.sqrt(max(0.0, 1.0 - target_sim ** 2))
            v = [target_sim * b + sin_val * p for b, p in zip(base, perp)]
            norm_v = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / norm_v for x in v]

        # Memory inside band (sim ~ 0.80).
        in_band_emb = _make_vector_at_sim(target_emb, 0.80)
        in_band_mem = Memory(
            user_id=user_id,
            content="User's spending limit is two hundred dollars",
            embedding=in_band_emb,
        )
        provider._memories[in_band_mem.memory_id] = in_band_mem

        # Memory above band (sim ~ 0.95 — dedup territory, not contradiction).
        above_band_emb = _make_vector_at_sim(target_emb, 0.95)
        above_band_mem = Memory(
            user_id=user_id,
            content="User budget is 200 dollars",
            embedding=above_band_emb,
        )
        provider._memories[above_band_mem.memory_id] = above_band_mem

        # Memory below band (sim ~ 0.30 — unrelated).
        below_band_emb = _make_vector_at_sim(target_emb, 0.30)
        below_band_mem = Memory(
            user_id=user_id,
            content="Completely unrelated topic about weather",
            embedding=below_band_emb,
        )
        provider._memories[below_band_mem.memory_id] = below_band_mem

        candidates = await detect_contradictions(target, provider, embedder, use_nli=False)
        candidate_ids = {m.memory_id for m, _ in candidates}

        assert in_band_mem.memory_id in candidate_ids, "in-band memory should be a candidate"
        assert above_band_mem.memory_id not in candidate_ids, "above-band memory should be skipped"
        assert below_band_mem.memory_id not in candidate_ids, "below-band memory should be skipped"
