"""Unit tests for Memory, ScoredMemory, and ExtractionResult models."""
from __future__ import annotations

import uuid

import pytest

from mnemosyne.db.models.memory import (
    ExtractionResult,
    Memory,
    MemoryType,
    ScoredMemory,
)


class TestMemoryDefaults:
    def test_memory_defaults(self):
        """Memory created with just user_id + content should have correct defaults."""
        user_id = uuid.uuid4()
        mem = Memory(user_id=user_id, content="I like coffee")

        assert mem.memory_type == MemoryType.FACT
        assert mem.importance == 0.5
        assert mem.access_count == 0
        assert mem.valid_until is None
        assert mem.extraction_version == "0.1.0"
        # auto-generated fields should be populated
        assert mem.memory_id is not None
        assert mem.created_at is not None
        assert mem.updated_at is not None
        assert mem.user_id == user_id
        assert mem.content == "I like coffee"

    def test_memory_default_agent_id(self):
        """Default agent_id is the zero UUID."""
        mem = Memory(user_id=uuid.uuid4(), content="test")
        assert mem.agent_id == uuid.UUID("00000000-0000-0000-0000-000000000000")

    def test_memory_default_embedding_is_none(self):
        """Embedding defaults to None — caller must set it before calling provider.add."""
        mem = Memory(user_id=uuid.uuid4(), content="test")
        assert mem.embedding is None

    def test_memory_default_content_hash_is_none(self):
        """content_hash starts as None; provider stamps it on add."""
        mem = Memory(user_id=uuid.uuid4(), content="some content")
        assert mem.content_hash is None


class TestMemoryImportanceClamping:
    def test_importance_clamped_above_one(self):
        """importance=1.5 should be clamped to 1.0."""
        mem = Memory(user_id=uuid.uuid4(), content="test", importance=1.5)
        assert mem.importance == 1.0

    def test_importance_clamped_below_zero(self):
        """importance=-0.5 should be clamped to 0.0."""
        mem = Memory(user_id=uuid.uuid4(), content="test", importance=-0.5)
        assert mem.importance == 0.0

    def test_importance_valid_value_unchanged(self):
        """A valid importance value in [0, 1] passes through unchanged."""
        mem = Memory(user_id=uuid.uuid4(), content="test", importance=0.75)
        assert mem.importance == 0.75

    def test_importance_zero_allowed(self):
        mem = Memory(user_id=uuid.uuid4(), content="test", importance=0.0)
        assert mem.importance == 0.0

    def test_importance_one_allowed(self):
        mem = Memory(user_id=uuid.uuid4(), content="test", importance=1.0)
        assert mem.importance == 1.0


class TestMemoryAllFields:
    def test_memory_all_fields(self):
        """Creating a Memory with all fields set should round-trip correctly."""
        memory_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session_id = uuid.uuid4()
        source_mem_id = uuid.uuid4()
        embedding = [0.1] * 1536

        mem = Memory(
            memory_id=memory_id,
            user_id=user_id,
            agent_id=agent_id,
            org_id=org_id,
            memory_type=MemoryType.PREFERENCE,
            content="User prefers dark roast coffee",
            content_hash="abc123",
            embedding=embedding,
            importance=0.8,
            access_count=5,
            decay_rate=0.02,
            extraction_version="0.2.0",
            extraction_model="gemma3:4b",
            prompt_hash="deadbeef",
            rule_id="preference_extractor",
            source_session_id=session_id,
            source_memory_ids=[source_mem_id],
            metadata={"key": "value"},
        )

        assert mem.memory_id == memory_id
        assert mem.user_id == user_id
        assert mem.agent_id == agent_id
        assert mem.org_id == org_id
        assert mem.memory_type == MemoryType.PREFERENCE
        assert mem.content == "User prefers dark roast coffee"
        assert mem.content_hash == "abc123"
        assert mem.embedding == embedding
        assert mem.importance == 0.8
        assert mem.access_count == 5
        assert mem.decay_rate == 0.02
        assert mem.extraction_version == "0.2.0"
        assert mem.extraction_model == "gemma3:4b"
        assert mem.prompt_hash == "deadbeef"
        assert mem.rule_id == "preference_extractor"
        assert mem.source_session_id == session_id
        assert mem.source_memory_ids == [source_mem_id]
        assert mem.metadata == {"key": "value"}


class TestScoredMemory:
    def test_scored_memory(self):
        """ScoredMemory wraps a Memory with a score and score_breakdown."""
        mem = Memory(user_id=uuid.uuid4(), content="Test memory")
        breakdown = {"relevance": 0.9, "recency": 0.7, "importance": 0.5, "frequency": 0.2}
        scored = ScoredMemory(memory=mem, score=0.85, score_breakdown=breakdown)

        assert scored.memory is mem
        assert scored.score == 0.85
        assert scored.score_breakdown == breakdown

    def test_scored_memory_default_breakdown(self):
        """score_breakdown defaults to empty dict."""
        mem = Memory(user_id=uuid.uuid4(), content="Test memory")
        scored = ScoredMemory(memory=mem, score=0.5)
        assert scored.score_breakdown == {}

    def test_scored_memory_preserves_memory_fields(self):
        """ScoredMemory provides access to all underlying Memory fields."""
        user_id = uuid.uuid4()
        mem = Memory(user_id=user_id, content="hello", importance=0.9)
        scored = ScoredMemory(memory=mem, score=0.7)

        assert scored.memory.user_id == user_id
        assert scored.memory.importance == 0.9
        assert scored.memory.content == "hello"


class TestExtractionResultDefaults:
    def test_extraction_result_defaults(self):
        """ExtractionResult with just content should have correct defaults."""
        result = ExtractionResult(content="User likes running")

        assert result.memory_type == MemoryType.FACT
        assert result.importance == 0.5
        assert result.rule_id == ""
        assert result.memory_id is None
        assert result.matched_chars == 0
        assert result.confidence == 1.0
        assert result.extraction_version == "0.1.0"
        assert result.metadata == {}

    def test_extraction_result_with_memory_type(self):
        result = ExtractionResult(
            content="User prefers Nike",
            memory_type=MemoryType.PREFERENCE,
            importance=0.7,
            rule_id="preference_rule",
        )
        assert result.memory_type == MemoryType.PREFERENCE
        assert result.importance == 0.7
        assert result.rule_id == "preference_rule"

    def test_extraction_result_memory_id_settable(self):
        """memory_id can be set after the pipeline calls provider.add."""
        mem_id = uuid.uuid4()
        result = ExtractionResult(content="test", memory_id=mem_id)
        assert result.memory_id == mem_id
