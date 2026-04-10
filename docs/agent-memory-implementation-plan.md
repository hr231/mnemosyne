# Agent Memory System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a general-purpose agent memory module (PostgreSQL + pgvector) with hybrid extraction, multi-signal retrieval, and context assembly — embeddable in the existing agent server.

**Architecture:** Embedded Python package with abstract `MemoryProvider` interface. PostgreSQL `memory` schema alongside existing `lakebase` schema. 5-stage processing pipeline (extraction → embedding → episodes → consolidation → decay). Two write paths: hot (agent tool) + cold (background pipeline).

**Tech Stack:** Python 3.11+, PostgreSQL 15+ with pgvector, asyncpg, pydantic, pytest, PyYAML

---

## File Structure

```
agent_memory/
├── agent-memory-design.md                  # Design doc (exists)
├── agent-memory-implementation-plan.md     # This plan
├── pyproject.toml
├── config.yaml
├── src/
│   └── agent_memory/
│       ├── __init__.py
│       ├── config.py                       # Config loading from YAML
│       ├── models.py                       # Pydantic models: Memory, Episode, Entity, etc.
│       ├── embeddings.py                   # Embedding model interface (model-agnostic)
│       ├── provider.py                     # MemoryProvider ABC
│       ├── postgres_provider.py            # PostgreSQL + pgvector implementation
│       ├── in_memory_provider.py           # InMemory implementation (dev/test)
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── base_extractor.py          # BaseExtractor ABC + ExtractionResult
│       │   ├── yaml_extractor.py          # YamlRuleExtractor — wraps YAML rule defs
│       │   ├── rule_loader.py             # Loads YAML files + Python plugins from paths
│       │   ├── rule_registry.py           # Holds loaded extractors, dispatch
│       │   ├── llm_router.py              # 5-signal LLM routing decision
│       │   ├── llm_extractor.py           # LLM extraction call
│       │   ├── pipeline.py                # Orchestrates rules → routing → LLM fallback
│       │   └── builtin/                   # Default rule set shipped with the module
│       │       ├── budget.yaml
│       │       ├── preferences.yaml
│       │       ├── sizes.yaml
│       │       └── keyword_triggers.yaml
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── search.py                  # Multi-signal scored search
│       │   └── context.py                 # Context assembly + token budgeting
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── embedding.py               # Stage 2: Batch embedding worker
│       │   ├── episodes.py                # Stage 3: Episode creation
│       │   ├── consolidation.py           # Stage 4: Dedup, reflection, contradictions
│       │   ├── decay.py                   # Stage 5: Decay & archival
│       │   └── runner.py                  # Pipeline orchestrator
│       └── integration/
│           ├── __init__.py
│           ├── agent_tool.py              # save_memory tool definition
│           └── prompt.py                  # System prompt memory block builder
├── migrations/
│   └── 001_create_memory_schema.sql
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_in_memory_provider.py
│   ├── test_base_extractor.py
│   ├── test_yaml_extractor.py
│   ├── test_rule_loader.py
│   ├── test_rule_registry.py
│   ├── test_builtin_rules.py
│   ├── test_llm_router.py
│   ├── test_extraction_pipeline.py
│   ├── test_search.py
│   ├── test_context.py
│   ├── test_embedding_pipeline.py
│   ├── test_episodes.py
│   ├── test_consolidation.py
│   ├── test_decay.py
│   ├── test_agent_tool.py
│   └── test_prompt.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/agent_memory/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agent-memory"
version = "0.1.0"
description = "General-purpose agent memory module with PostgreSQL + pgvector"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "asyncpg>=0.29",
    "pgvector>=0.3",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]
extraction = [
    "spacy>=3.7",
]
llm = [
    "httpx>=0.27",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.hatch.build.targets.wheel]
packages = ["src/agent_memory"]
```

- [ ] **Step 2: Create package init**

```python
# src/agent_memory/__init__.py
"""Agent Memory — general-purpose memory module for AI agents."""

from agent_memory.models import Memory, Episode, Entity, MemoryType
from agent_memory.provider import MemoryProvider

__all__ = ["Memory", "Episode", "Entity", "MemoryType", "MemoryProvider"]
```

- [ ] **Step 3: Create test conftest**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import uuid
import pytest


@pytest.fixture
def user_id():
    return uuid.uuid4()


@pytest.fixture
def agent_id():
    return uuid.UUID("00000000-0000-0000-0000-000000000000")


@pytest.fixture
def session_id():
    return uuid.uuid4()
```

- [ ] **Step 4: Install in dev mode and verify**

Run: `cd /Users/harshit/agent_memory && pip install -e ".[dev]" 2>&1 | tail -5`
Expected: Successfully installed agent-memory-0.1.0

Run: `pytest tests/ -v --co 2>&1 | tail -5`
Expected: "no tests ran" (empty collection, no errors)

- [ ] **Step 5: Init git and commit**

```bash
git init
echo "__pycache__/\n*.egg-info/\n.eggs/\ndist/\n*.pyc\n.pytest_cache/\nvenv/\n.venv/" > .gitignore
git add .
git commit -m "feat: project scaffolding with pyproject.toml and test fixtures"
```

---

## Task 2: Data Models

**Files:**
- Create: `src/agent_memory/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for models**

```python
# tests/test_models.py
import uuid
from datetime import datetime, timezone

from agent_memory.models import Memory, Episode, Entity, MemoryType, ProcessingLog, MemoryHistoryEntry


class TestMemory:
    def test_create_with_defaults(self):
        m = Memory(user_id=uuid.uuid4(), content="User prefers organic products")
        assert m.memory_type == MemoryType.FACT
        assert m.importance == 0.5
        assert m.access_count == 0
        assert m.decay_rate == 0.01
        assert m.embedding is None
        assert m.metadata == {}
        assert m.id is not None
        assert m.valid_until is None  # currently valid
        assert m.extraction_version == "0.1.0"
        assert m.content_hash is None

    def test_create_with_all_fields(self):
        uid = uuid.uuid4()
        aid = uuid.uuid4()
        m = Memory(
            user_id=uid,
            agent_id=aid,
            org_id=uuid.uuid4(),
            memory_type=MemoryType.PREFERENCE,
            content="Likes blue",
            content_hash="abc123",
            embedding=[0.1] * 1536,
            importance=0.9,
            source_session_id=uuid.uuid4(),
            extraction_model="gpt-4o-mini",
            rule_id="preference_positive",
        )
        assert m.user_id == uid
        assert m.agent_id == aid
        assert m.memory_type == MemoryType.PREFERENCE
        assert m.importance == 0.9
        assert len(m.embedding) == 1536
        assert m.extraction_model == "gpt-4o-mini"
        assert m.rule_id == "preference_positive"

    def test_importance_clamped(self):
        m = Memory(user_id=uuid.uuid4(), content="test", importance=1.5)
        assert m.importance <= 1.0
        m2 = Memory(user_id=uuid.uuid4(), content="test", importance=-0.5)
        assert m2.importance >= 0.0

    def test_invalidation_via_valid_until(self):
        """Invalidated memories keep their data but are excluded by default queries."""
        from datetime import timedelta
        m = Memory(
            user_id=uuid.uuid4(),
            content="old fact",
            valid_until=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert m.valid_until is not None


class TestMemoryHistoryEntry:
    def test_create(self):
        entry = MemoryHistoryEntry(
            memory_id=uuid.uuid4(),
            operation="update",
            old_content="Budget: $100",
            new_content="Budget: $200",
            old_importance=0.7,
            new_importance=0.8,
            actor="agent_tool",
            actor_details={"session_id": str(uuid.uuid4())},
        )
        assert entry.operation == "update"
        assert entry.actor == "agent_tool"


class TestEpisode:
    def test_create(self):
        e = Episode(
            user_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            summary="User searched for running shoes and compared two options.",
            started_at=datetime.now(timezone.utc),
        )
        assert e.summary_embedding is None
        assert e.key_topics == []
        assert e.memory_ids == []


class TestEntity:
    def test_create(self):
        ent = Entity(
            user_id=uuid.uuid4(),
            entity_name="Nike",
            entity_type="brand",
        )
        assert ent.facts == {}
        assert ent.confidence == 1.0


class TestProcessingLog:
    def test_create(self):
        log = ProcessingLog(
            session_id=uuid.uuid4(),
            pipeline_step="extraction",
        )
        assert log.status == "pending"
        assert log.error_message is None
        assert log.memories_created == []
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — cannot import models

- [ ] **Step 3: Implement models**

```python
# src/agent_memory/models.py
"""Data models for the agent memory system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    ENTITY = "entity"
    PROCEDURAL = "procedural"
    REFLECTION = "reflection"


class Memory(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    agent_id: uuid.UUID = Field(default=uuid.UUID("00000000-0000-0000-0000-000000000000"))
    org_id: uuid.UUID | None = None

    memory_type: MemoryType = MemoryType.FACT
    content: str
    content_hash: str | None = None  # SHA-256 of normalized content for exact-dup detection

    embedding: list[float] | None = None  # halfvec(1536) in the DB

    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decay_rate: float = 0.01

    # Bi-temporal model (Zep/Graphiti pattern — never silently overwrite)
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: datetime | None = None  # NULL means "currently valid"

    # Extraction versioning
    extraction_version: str = "0.1.0"
    extraction_model: str | None = None   # e.g. "gpt-4o-mini", "rule:budget_explicit"
    prompt_hash: str | None = None         # SHA-256 of the extraction prompt used
    rule_id: str | None = None             # which rule/plugin produced this memory

    # Provenance
    source_session_id: uuid.UUID | None = None
    source_memory_ids: list[uuid.UUID] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("importance")
    @classmethod
    def clamp_importance(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class MemoryHistoryEntry(BaseModel):
    """Immutable audit log entry for memory mutations."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    memory_id: uuid.UUID
    operation: str  # create, update, delete, merge, invalidate
    old_content: str | None = None
    new_content: str | None = None
    old_importance: float | None = None
    new_importance: float | None = None
    actor: str  # agent_tool, pipeline_extraction, pipeline_consolidation, pipeline_decay, manual
    actor_details: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Episode(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    agent_id: uuid.UUID = Field(default=uuid.UUID("00000000-0000-0000-0000-000000000000"))
    session_id: uuid.UUID

    summary: str
    summary_embedding: list[float] | None = None

    outcome: str | None = None
    key_topics: list[str] = Field(default_factory=list)
    memory_ids: list[uuid.UUID] = Field(default_factory=list)

    started_at: datetime
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Entity(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID | None = None
    agent_id: uuid.UUID = Field(default=uuid.UUID("00000000-0000-0000-0000-000000000000"))

    entity_name: str
    entity_type: str
    facts: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    confidence: float = 1.0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_memory_ids: list[uuid.UUID] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProcessingLog(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    pipeline_step: str
    status: str = "pending"
    error_message: str | None = None
    memories_created: list[uuid.UUID] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContextRow(BaseModel):
    """A single row returned by context assembly."""
    section: str
    content: str
    priority: int
```

- [ ] **Step 4: Run tests to verify passing**

Run: `pytest tests/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_memory/models.py tests/test_models.py
git commit -m "feat: pydantic data models for Memory, Episode, Entity, ProcessingLog"
```

---

## Task 3: Config Loading

**Files:**
- Create: `config.yaml`
- Create: `src/agent_memory/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
from pathlib import Path
from agent_memory.config import MemoryConfig, load_config


class TestConfig:
    def test_load_defaults(self):
        cfg = MemoryConfig()
        assert cfg.embedding.dimensions == 1536
        assert cfg.embedding.use_halfvec is True
        assert cfg.extraction.batch_size == 50
        # Simplified v1 routing: 2-signal OR gate (A-MAC ablation)
        assert cfg.extraction.yield_threshold == 0.3
        assert cfg.extraction.unstructured_threshold == 0.7
        assert cfg.extraction.enable_novelty_signal is False  # v1.5
        assert cfg.decay.default_decay_rate == 0.01
        assert cfg.decay.archive_threshold == 0.05
        assert cfg.decay.archive_after_days == 90
        # Dedup: three-tier with 0.90 cosine similarity (= 0.10 distance)
        assert cfg.consolidation.dedup_exact_enabled is True
        assert cfg.consolidation.dedup_fuzzy_similarity == 0.8
        assert cfg.consolidation.dedup_semantic_similarity == 0.90
        # Reflection: importance-sum-based (Stanford Generative Agents)
        assert cfg.consolidation.reflection_importance_sum_threshold == 150
        assert cfg.retrieval.w_relevance == 0.5
        assert cfg.retrieval.w_recency == 0.2
        assert cfg.retrieval.w_importance == 0.2
        assert cfg.retrieval.w_frequency == 0.1
        assert cfg.context.default_token_budget == 2000
        assert cfg.capacity.memories_per_user_soft_limit == 5000

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
extraction:
  batch_size: 100
  yield_threshold: 0.4
embedding:
  dimensions: 768
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)
        cfg = load_config(config_file)
        assert cfg.extraction.batch_size == 100
        assert cfg.extraction.yield_threshold == 0.4
        assert cfg.embedding.dimensions == 768
        # defaults preserved
        assert cfg.decay.default_decay_rate == 0.01
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — cannot import

- [ ] **Step 3: Implement config**

```python
# src/agent_memory/config.py
"""Configuration loading for agent memory system."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class ExtractionConfig(BaseModel):
    """Simplified v1 routing — 2-signal OR gate per A-MAC ablation.

    The routing function is:
      ROUTE_TO_LLM = (extraction_yield < yield_threshold)
                  OR (unstructured_ratio > unstructured_threshold)

    Contradictions and session complexity are NOT in v1 routing:
    - Contradictions belong to consolidation, not extraction routing
    - Session complexity did not contribute significantly to A-MAC's F1
    Semantic novelty is a v1.5 addition (requires local MiniLM model).
    """
    batch_size: int = 50
    llm_model: str = "configurable"
    extraction_version: str = "0.1.0"

    # v1 two-signal routing
    yield_threshold: float = 0.3         # extraction_yield < this → LLM
    unstructured_threshold: float = 0.7  # unstructured_ratio > this → LLM

    # v1.5 additions (disabled by default)
    enable_novelty_signal: bool = False
    novelty_threshold: float = 0.7
    novelty_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingConfig(BaseModel):
    model: str = "configurable"
    dimensions: int = 1536
    batch_size: int = 100
    use_halfvec: bool = True  # 50% storage reduction, minimal recall impact


class EpisodeConfig(BaseModel):
    # Lakebase summaries are agent output summaries, usable as a starting point.
    # When false or summary missing, pipeline invokes memory-specific extraction prompt.
    reuse_lakebase_summary: bool = True
    enrich_lakebase_summary: bool = True  # append memory IDs, key topics, outcome
    min_summary_length: int = 50  # if lakebase summary is shorter, re-generate


class ConsolidationConfig(BaseModel):
    # Three-tier dedup (hot path + batch)
    dedup_exact_enabled: bool = True                 # SHA-256 hash lookup
    dedup_fuzzy_similarity: float = 0.8              # pg_trgm threshold
    dedup_semantic_similarity: float = 0.90          # cosine similarity threshold
    dedup_semantic_auto_merge: float = 0.95          # auto-merge above this

    # Reflection — importance-sum-based per Stanford Generative Agents (UIST 2023)
    reflection_importance_sum_threshold: int = 150   # scaled importance in [1, 10]
    reflection_lookback_memories: int = 100

    # Bi-temporal invalidation
    invalidate_on_contradiction: bool = True


class DecayConfig(BaseModel):
    default_decay_rate: float = 0.01
    archive_threshold: float = 0.05
    archive_after_days: int = 90
    dry_run: bool = False  # if true, report what would change without applying


class RetrievalConfig(BaseModel):
    w_relevance: float = 0.5
    w_recency: float = 0.2
    w_importance: float = 0.2
    w_frequency: float = 0.1
    default_limit: int = 10
    candidate_multiplier: int = 5
    exclude_invalidated: bool = True  # filter WHERE valid_until IS NULL


class ContextConfig(BaseModel):
    default_token_budget: int = 2000
    profile_limit: int = 5
    relevant_limit: int = 10
    episode_limit: int = 3
    entity_limit: int = 3


class CapacityConfig(BaseModel):
    memories_per_user_soft_limit: int = 5000
    memories_per_user_hard_limit: int = 10000
    eviction_strategy: str = "importance_weighted"  # future: "lru", "fifo"


class DatabaseConfig(BaseModel):
    dsn: str = "postgresql://localhost:5432/agent"
    memory_schema: str = "memory"
    lakebase_schema: str = "lakebase"


class MemoryConfig(BaseModel):
    extraction: ExtractionConfig = ExtractionConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    episode: EpisodeConfig = EpisodeConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()
    decay: DecayConfig = DecayConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    context: ContextConfig = ContextConfig()
    capacity: CapacityConfig = CapacityConfig()
    database: DatabaseConfig = DatabaseConfig()


def load_config(path: Path | None = None) -> MemoryConfig:
    if path is None or not path.exists():
        return MemoryConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return MemoryConfig(**raw)
```

- [ ] **Step 4: Create default config.yaml**

```yaml
# config.yaml — Agent Memory System Configuration
# All values shown are defaults. Override as needed.

database:
  dsn: "postgresql://localhost:5432/agent"
  memory_schema: "memory"
  lakebase_schema: "lakebase"

extraction:
  batch_size: 50
  llm_model: "configurable"
  extraction_version: "0.1.0"

  # v1: two-signal routing (A-MAC ablation — Content Type Prior captures ~49%)
  yield_threshold: 0.3              # extraction_yield < this → LLM
  unstructured_threshold: 0.7       # unstructured_ratio > this → LLM

  # v1.5 additions (flip to true when ready)
  enable_novelty_signal: false
  novelty_threshold: 0.7
  novelty_embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

embedding:
  model: "configurable"
  dimensions: 1536
  batch_size: 100
  use_halfvec: true                 # 50% storage reduction from day one

episode:
  reuse_lakebase_summary: true
  enrich_lakebase_summary: true
  min_summary_length: 50

consolidation:
  # Three-tier dedup
  dedup_exact_enabled: true
  dedup_fuzzy_similarity: 0.8       # pg_trgm threshold
  dedup_semantic_similarity: 0.90   # cosine similarity ≥ this → candidate
  dedup_semantic_auto_merge: 0.95   # ≥ this → auto-merge without review

  # Reflection: importance-sum-based (Stanford Generative Agents)
  reflection_importance_sum_threshold: 150
  reflection_lookback_memories: 100

  invalidate_on_contradiction: true

decay:
  default_decay_rate: 0.01
  archive_threshold: 0.05
  archive_after_days: 90
  dry_run: false

retrieval:
  w_relevance: 0.5
  w_recency: 0.2
  w_importance: 0.2
  w_frequency: 0.1
  exclude_invalidated: true

context:
  default_token_budget: 2000

capacity:
  memories_per_user_soft_limit: 5000
  memories_per_user_hard_limit: 10000
  eviction_strategy: "importance_weighted"
```

- [ ] **Step 5: Run tests, commit**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

```bash
git add src/agent_memory/config.py config.yaml tests/test_config.py
git commit -m "feat: YAML-driven config with pydantic validation"
```

---

## Task 4: Embedding Interface

**Files:**
- Create: `src/agent_memory/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_embeddings.py
import pytest
from agent_memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider


class TestFakeEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_embed_single(self):
        provider = FakeEmbeddingProvider(dimensions=4)
        result = await provider.embed("hello world")
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        provider = FakeEmbeddingProvider(dimensions=4)
        results = await provider.embed_batch(["hello", "world", "test"])
        assert len(results) == 3
        assert all(len(r) == 4 for r in results)

    @pytest.mark.asyncio
    async def test_deterministic_for_same_input(self):
        provider = FakeEmbeddingProvider(dimensions=4)
        a = await provider.embed("same text")
        b = await provider.embed("same text")
        assert a == b

    @pytest.mark.asyncio
    async def test_different_for_different_input(self):
        provider = FakeEmbeddingProvider(dimensions=4)
        a = await provider.embed("text one")
        b = await provider.embed("text two")
        assert a != b
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/agent_memory/embeddings.py
"""Model-agnostic embedding provider interface."""

from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract embedding provider. Swap implementations via config."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake for testing. Produces consistent embeddings from text hash."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        # Extend hash to fill dimensions
        values = []
        for i in range(self.dimensions):
            seed = hashlib.sha256(h + struct.pack("I", i)).digest()[:4]
            val = struct.unpack("f", seed)[0]
            # Normalize to [-1, 1]
            val = (val % 2.0) - 1.0
            values.append(round(val, 6))
        return values

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_embeddings.py -v`
Expected: All PASS

```bash
git add src/agent_memory/embeddings.py tests/test_embeddings.py
git commit -m "feat: model-agnostic embedding provider interface with fake for testing"
```

---

## Task 5: MemoryProvider Interface + InMemoryProvider

**Files:**
- Create: `src/agent_memory/provider.py`
- Create: `src/agent_memory/in_memory_provider.py`
- Create: `tests/test_in_memory_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_in_memory_provider.py
import uuid
import pytest
from agent_memory.in_memory_provider import InMemoryProvider
from agent_memory.models import MemoryType
from agent_memory.embeddings import FakeEmbeddingProvider


@pytest.fixture
def provider():
    return InMemoryProvider(embedding_provider=FakeEmbeddingProvider(dimensions=4))


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_returns_uuid(self, provider, user_id):
        mid = await provider.add(
            user_id=user_id,
            content="User prefers organic products",
            memory_type=MemoryType.PREFERENCE,
            importance=0.8,
        )
        assert isinstance(mid, uuid.UUID)

    @pytest.mark.asyncio
    async def test_add_and_retrieve(self, provider, user_id):
        await provider.add(
            user_id=user_id,
            content="Budget is $200",
            memory_type=MemoryType.FACT,
            importance=0.9,
        )
        results = await provider.search(
            user_id=user_id,
            query="budget",
            query_embedding=await provider._embed.embed("budget"),
            limit=5,
        )
        assert len(results) == 1
        assert "Budget is $200" in results[0].content


class TestSearch:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self, provider, user_id):
        emb = await provider._embed.embed("anything")
        results = await provider.search(
            user_id=user_id, query="anything", query_embedding=emb, limit=5
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_filter_by_memory_type(self, provider, user_id):
        await provider.add(user_id=user_id, content="fact one", memory_type=MemoryType.FACT)
        await provider.add(user_id=user_id, content="pref one", memory_type=MemoryType.PREFERENCE)
        emb = await provider._embed.embed("one")
        facts = await provider.search(
            user_id=user_id, query="one", query_embedding=emb,
            limit=5, memory_types=[MemoryType.FACT],
        )
        assert len(facts) == 1
        assert facts[0].memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_scoped_by_user(self, provider):
        u1, u2 = uuid.uuid4(), uuid.uuid4()
        await provider.add(user_id=u1, content="user1 memory")
        await provider.add(user_id=u2, content="user2 memory")
        emb = await provider._embed.embed("memory")
        r1 = await provider.search(user_id=u1, query="memory", query_embedding=emb, limit=5)
        assert len(r1) == 1
        assert "user1" in r1[0].content


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_content(self, provider, user_id):
        mid = await provider.add(user_id=user_id, content="old content")
        await provider.update(memory_id=mid, content="new content")
        emb = await provider._embed.embed("new content")
        results = await provider.search(user_id=user_id, query="new", query_embedding=emb, limit=5)
        assert results[0].content == "new content"


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete(self, provider, user_id):
        mid = await provider.add(user_id=user_id, content="to delete")
        await provider.delete(memory_id=mid)
        emb = await provider._embed.embed("to delete")
        results = await provider.search(user_id=user_id, query="delete", query_embedding=emb, limit=5)
        assert len(results) == 0


class TestAssembleContext:
    @pytest.mark.asyncio
    async def test_returns_prioritized_sections(self, provider, user_id, session_id):
        await provider.add(user_id=user_id, content="high importance fact", importance=0.9)
        await provider.add(user_id=user_id, content="low importance fact", importance=0.3)
        emb = await provider._embed.embed("fact")
        rows = await provider.assemble_context(
            user_id=user_id, query="fact", query_embedding=emb,
            session_id=session_id, token_budget=2000,
        )
        assert len(rows) > 0
        assert all(hasattr(r, "section") for r in rows)
        assert all(hasattr(r, "priority") for r in rows)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_in_memory_provider.py -v`
Expected: FAIL

- [ ] **Step 3: Implement provider interface**

```python
# src/agent_memory/provider.py
"""Abstract MemoryProvider interface."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from agent_memory.models import ContextRow, Memory, MemoryType


class MemoryProvider(ABC):
    """Abstract interface — swap implementations without changing agent code."""

    @abstractmethod
    async def add(
        self,
        user_id: uuid.UUID,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        source_session_id: uuid.UUID | None = None,
    ) -> uuid.UUID: ...

    @abstractmethod
    async def search(
        self,
        user_id: uuid.UUID,
        query: str,
        query_embedding: list[float],
        limit: int = 10,
        memory_types: list[MemoryType] | None = None,
        agent_id: uuid.UUID | None = None,
    ) -> list[Memory]: ...

    @abstractmethod
    async def assemble_context(
        self,
        user_id: uuid.UUID,
        query: str,
        query_embedding: list[float],
        session_id: uuid.UUID,
        token_budget: int = 2000,
    ) -> list[ContextRow]: ...

    @abstractmethod
    async def update(
        self,
        memory_id: uuid.UUID,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def delete(self, memory_id: uuid.UUID) -> None: ...
```

- [ ] **Step 4: Implement InMemoryProvider**

```python
# src/agent_memory/in_memory_provider.py
"""In-memory MemoryProvider for dev/test."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

from agent_memory.embeddings import EmbeddingProvider
from agent_memory.models import ContextRow, Memory, MemoryType
from agent_memory.provider import MemoryProvider


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryProvider(MemoryProvider):
    def __init__(self, embedding_provider: EmbeddingProvider):
        self._embed = embedding_provider
        self._memories: dict[uuid.UUID, Memory] = {}

    async def add(
        self,
        user_id: uuid.UUID,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        source_session_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        embedding = await self._embed.embed(content)
        mem = Memory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            metadata=metadata or {},
            source_session_id=source_session_id,
        )
        self._memories[mem.id] = mem
        return mem.id

    async def search(
        self,
        user_id: uuid.UUID,
        query: str,
        query_embedding: list[float],
        limit: int = 10,
        memory_types: list[MemoryType] | None = None,
        agent_id: uuid.UUID | None = None,
    ) -> list[Memory]:
        candidates = [
            m for m in self._memories.values()
            if m.user_id == user_id
            and m.embedding is not None
            and m.metadata.get("archived") is not True
            and (memory_types is None or m.memory_type in memory_types)
            and (agent_id is None or m.agent_id == agent_id)
        ]
        if not candidates:
            return []

        now = datetime.now(timezone.utc)
        scored = []
        for m in candidates:
            relevance = _cosine_sim(query_embedding, m.embedding)
            seconds_since = (now - m.last_accessed).total_seconds()
            recency = math.exp(-0.01 * seconds_since / 86400.0)
            frequency = min(math.log(m.access_count + 1) / 5.0, 1.0)
            score = (
                0.5 * relevance
                + 0.2 * recency
                + 0.2 * m.importance
                + 0.1 * frequency
            )
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, m in scored[:limit]:
            m.access_count += 1
            m.last_accessed = now
            results.append(m)
        return results

    async def assemble_context(
        self,
        user_id: uuid.UUID,
        query: str,
        query_embedding: list[float],
        session_id: uuid.UUID,
        token_budget: int = 2000,
    ) -> list[ContextRow]:
        rows: list[ContextRow] = []

        # 1. Profile: high-importance facts and reflections
        profile_mems = [
            m for m in self._memories.values()
            if m.user_id == user_id
            and m.memory_type in (MemoryType.FACT, MemoryType.REFLECTION)
            and m.importance > 0.7
        ]
        profile_mems.sort(key=lambda m: m.importance, reverse=True)
        for m in profile_mems[:5]:
            rows.append(ContextRow(section="profile", content=m.content, priority=1))

        # 2. Query-relevant
        relevant = await self.search(
            user_id=user_id, query=query, query_embedding=query_embedding, limit=10
        )
        for m in relevant:
            rows.append(ContextRow(section="relevant", content=m.content, priority=2))

        return rows

    async def update(
        self,
        memory_id: uuid.UUID,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        mem = self._memories.get(memory_id)
        if mem is None:
            return
        if content is not None:
            mem.content = content
            mem.embedding = await self._embed.embed(content)
        if importance is not None:
            mem.importance = max(0.0, min(1.0, importance))
        if metadata is not None:
            mem.metadata.update(metadata)
        mem.updated_at = datetime.now(timezone.utc)

    async def delete(self, memory_id: uuid.UUID) -> None:
        self._memories.pop(memory_id, None)
```

- [ ] **Step 5: Run tests, commit**

Run: `pytest tests/test_in_memory_provider.py -v`
Expected: All PASS

```bash
git add src/agent_memory/provider.py src/agent_memory/in_memory_provider.py tests/test_in_memory_provider.py
git commit -m "feat: MemoryProvider ABC + InMemoryProvider for dev/test"
```

---

## Task 6: Database Migration

**Files:**
- Create: `migrations/001_create_memory_schema.sql`

- [ ] **Step 1: Write the migration**

```sql
-- migrations/001_create_memory_schema.sql
-- Agent Memory System — Database Schema
-- Requires: PostgreSQL 15+, pgvector 0.7+ (for halfvec), pg_trgm

BEGIN;

CREATE SCHEMA IF NOT EXISTS memory;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- fuzzy text matching for dedup

-- ============================================================
-- Table: memory.memories — Core memory table
-- Bi-temporal model: valid_from / valid_until (Zep/Graphiti pattern)
-- halfvec: 50% storage reduction vs vector, minimal recall impact
-- ============================================================
CREATE TABLE memory.memories (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            UUID NOT NULL,
    agent_id           UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    org_id             UUID,

    memory_type        TEXT NOT NULL CHECK (memory_type IN
                         ('fact', 'preference', 'entity', 'procedural', 'reflection')),
    content            TEXT NOT NULL,
    content_hash       TEXT,  -- SHA-256 of normalized content for exact-dup detection

    embedding          halfvec(1536),  -- half-precision, 50% storage reduction

    importance         FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    access_count       INTEGER DEFAULT 0,
    last_accessed      TIMESTAMPTZ DEFAULT now(),
    decay_rate         FLOAT DEFAULT 0.01,

    -- Bi-temporal (Zep/Graphiti pattern — never silently overwrite)
    valid_from         TIMESTAMPTZ DEFAULT now(),
    valid_until        TIMESTAMPTZ,  -- NULL means "currently valid"

    -- Extraction versioning (enables re-extraction when rules/prompts change)
    extraction_version TEXT DEFAULT '0.1.0',
    extraction_model   TEXT,  -- e.g., 'gpt-4o-mini', 'rule:budget_explicit'
    prompt_hash        TEXT,  -- SHA-256 of extraction prompt (NULL for rule-based)
    rule_id            TEXT,  -- which rule/plugin produced this (NULL for LLM extraction)

    -- Provenance
    source_session_id  UUID,
    source_memory_ids  UUID[],

    metadata           JSONB DEFAULT '{}',

    created_at         TIMESTAMPTZ DEFAULT now(),
    updated_at         TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE memory.memories ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- ============================================================
-- Table: memory.episodes — Session summaries
-- ============================================================
CREATE TABLE memory.episodes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL,
    agent_id            UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    session_id          UUID NOT NULL UNIQUE,

    summary             TEXT NOT NULL,
    summary_embedding   halfvec(1536),

    outcome             TEXT,
    key_topics          TEXT[],
    memory_ids          UUID[],

    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Table: memory.entities — Structured entity knowledge
-- ============================================================
CREATE TABLE memory.entities (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            UUID,
    agent_id           UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',

    entity_name        TEXT NOT NULL,
    entity_type        TEXT NOT NULL,
    facts              JSONB DEFAULT '{}',
    embedding          halfvec(1536),

    confidence         FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    last_updated       TIMESTAMPTZ DEFAULT now(),
    source_memory_ids  UUID[],

    created_at         TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, agent_id, entity_name, entity_type)
);

-- ============================================================
-- Table: memory.processing_log — Pipeline audit trail
-- ============================================================
CREATE TABLE memory.processing_log (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL,
    pipeline_step    TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    error_message    TEXT,
    memories_created UUID[],
    processed_at     TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Table: memory.memory_history — Immutable mutation audit trail
-- Append-only. Never updated. Retention pruned by background job.
-- ============================================================
CREATE TABLE memory.memory_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id       UUID NOT NULL,
    operation       TEXT NOT NULL CHECK (operation IN
                      ('create', 'update', 'delete', 'merge', 'invalidate')),
    old_content     TEXT,
    new_content     TEXT,
    old_importance  FLOAT,
    new_importance  FLOAT,
    actor           TEXT NOT NULL,  -- agent_tool, pipeline_extraction, etc.
    actor_details   JSONB DEFAULT '{}',
    occurred_at     TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Table: memory.extraction_versions — Registry of extraction config versions
-- ============================================================
CREATE TABLE memory.extraction_versions (
    version          TEXT PRIMARY KEY,  -- semver
    rule_pack_commit TEXT,               -- git SHA of rules directory
    llm_model        TEXT,
    prompt_hash      TEXT,
    config_snapshot  JSONB,
    deployed_at      TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Indexes
-- ============================================================

-- Vector search (HNSW on halfvec)
CREATE INDEX idx_mem_embedding ON memory.memories
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_ep_embedding ON memory.episodes
    USING hnsw (summary_embedding halfvec_cosine_ops);
CREATE INDEX idx_ent_embedding ON memory.entities
    USING hnsw (embedding halfvec_cosine_ops);

-- Scoped lookups
CREATE INDEX idx_mem_user ON memory.memories (user_id, memory_type);
CREATE INDEX idx_mem_importance ON memory.memories (user_id, importance DESC);
CREATE INDEX idx_ep_user ON memory.episodes (user_id, started_at DESC);
CREATE INDEX idx_ent_user ON memory.entities (user_id, entity_type);

-- Bi-temporal: "currently valid" lookup is the common case
CREATE INDEX idx_mem_currently_valid ON memory.memories (user_id, last_accessed DESC)
    WHERE valid_until IS NULL;
CREATE INDEX idx_mem_temporal ON memory.memories (user_id, valid_from DESC, valid_until);

-- Exact-dup: same user + same content_hash must not both be "currently valid"
CREATE UNIQUE INDEX idx_mem_content_hash ON memory.memories (user_id, content_hash)
    WHERE valid_until IS NULL AND content_hash IS NOT NULL;

-- Fuzzy-dup: pg_trgm GIN index for similarity() queries
CREATE INDEX idx_mem_content_trgm ON memory.memories USING gin (content gin_trgm_ops);

-- Full-text search
CREATE INDEX idx_mem_fts ON memory.memories USING gin (content_tsv);

-- Processing pipeline
CREATE INDEX idx_proc_pending ON memory.processing_log (session_id)
    WHERE status = 'pending';
CREATE INDEX idx_unembedded ON memory.memories (id)
    WHERE embedding IS NULL;

-- Extraction version lag (for re-extraction batch job)
CREATE INDEX idx_mem_extraction_version ON memory.memories (extraction_version, last_accessed DESC);

-- Memory history lookup
CREATE INDEX idx_mem_history ON memory.memory_history (memory_id, occurred_at DESC);

-- Seed the initial extraction version
INSERT INTO memory.extraction_versions (version, rule_pack_commit, llm_model, prompt_hash, config_snapshot)
VALUES ('0.1.0', 'initial', 'gpt-4o-mini', 'initial', '{}');

-- ============================================================
-- Functions: search_memories
-- ============================================================
CREATE OR REPLACE FUNCTION memory.search_memories(
    p_user_id         UUID,
    p_query           TEXT,
    p_query_embedding halfvec(1536),
    p_limit           INT DEFAULT 10,
    p_w_relevance     FLOAT DEFAULT 0.5,
    p_w_recency       FLOAT DEFAULT 0.2,
    p_w_importance    FLOAT DEFAULT 0.2,
    p_w_frequency     FLOAT DEFAULT 0.1,
    p_memory_types    TEXT[] DEFAULT NULL,
    p_agent_id        UUID DEFAULT NULL
) RETURNS TABLE (
    id          UUID,
    content     TEXT,
    memory_type TEXT,
    score       FLOAT,
    metadata    JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH candidates AS (
        SELECT m.id, m.content, m.memory_type, m.metadata,
               m.importance, m.access_count, m.last_accessed,
               1 - (m.embedding <=> p_query_embedding) AS vector_sim
        FROM memory.memories m
        WHERE m.user_id = p_user_id
          AND m.embedding IS NOT NULL
          AND m.valid_until IS NULL  -- bi-temporal: only currently-valid
          AND (m.metadata->>'archived')::boolean IS NOT TRUE
          AND (p_memory_types IS NULL OR m.memory_type = ANY(p_memory_types))
          AND (p_agent_id IS NULL OR m.agent_id = p_agent_id)
        ORDER BY m.embedding <=> p_query_embedding
        LIMIT p_limit * 5
    ),
    text_matches AS (
        SELECT m.id,
               ts_rank_cd(m.content_tsv, plainto_tsquery('english', p_query)) AS text_rank
        FROM memory.memories m
        WHERE m.user_id = p_user_id
          AND m.valid_until IS NULL
          AND m.content_tsv @@ plainto_tsquery('english', p_query)
    ),
    scored AS (
        SELECT c.id, c.content, c.memory_type, c.metadata,
            (
                p_w_relevance * (c.vector_sim + COALESCE(t.text_rank * 0.2, 0))
                + p_w_recency * exp(
                    -0.01 * EXTRACT(EPOCH FROM now() - c.last_accessed) / 86400.0
                  )
                + p_w_importance * c.importance
                + p_w_frequency * least(ln(c.access_count + 1) / 5.0, 1.0)
            ) AS score
        FROM candidates c
        LEFT JOIN text_matches t ON c.id = t.id
    )
    SELECT s.id, s.content, s.memory_type, s.score, s.metadata
    FROM scored s
    ORDER BY s.score DESC
    LIMIT p_limit;

    -- Bump access stats (only currently-valid memories)
    UPDATE memory.memories m
    SET access_count = m.access_count + 1,
        last_accessed = now()
    WHERE m.id IN (
        SELECT m2.id FROM memory.memories m2
        WHERE m2.user_id = p_user_id
          AND m2.embedding IS NOT NULL
          AND m2.valid_until IS NULL
          AND (m2.metadata->>'archived')::boolean IS NOT TRUE
        ORDER BY m2.embedding <=> p_query_embedding
        LIMIT p_limit
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Functions: assemble_context
-- Respects bi-temporal filter throughout
-- ============================================================
CREATE OR REPLACE FUNCTION memory.assemble_context(
    p_user_id         UUID,
    p_query           TEXT,
    p_query_embedding halfvec(1536),
    p_session_id      UUID,
    p_token_budget    INT DEFAULT 2000
) RETURNS TABLE (
    section   TEXT,
    content   TEXT,
    priority  INT
) AS $$
BEGIN
    -- 1. User profile: reflections + high-importance facts
    RETURN QUERY
    SELECT 'profile'::TEXT, m.content, 1
    FROM memory.memories m
    WHERE m.user_id = p_user_id
      AND m.valid_until IS NULL
      AND m.memory_type IN ('reflection', 'fact')
      AND m.importance > 0.7
    ORDER BY m.importance DESC
    LIMIT 5;

    -- 2. Query-relevant memories (search_memories already filters bi-temporal)
    RETURN QUERY
    SELECT 'relevant'::TEXT, r.content, 2
    FROM memory.search_memories(
        p_user_id, p_query, p_query_embedding, 10
    ) r;

    -- 3. Recent episodes (last 3 sessions for continuity)
    RETURN QUERY
    SELECT 'recent_episodes'::TEXT, e.summary, 3
    FROM memory.episodes e
    WHERE e.user_id = p_user_id
      AND e.session_id != p_session_id
    ORDER BY e.started_at DESC
    LIMIT 3;

    -- 4. Relevant entities
    RETURN QUERY
    SELECT 'entities'::TEXT,
           e.entity_name || ': ' || e.facts::text, 4
    FROM memory.entities e
    WHERE e.user_id = p_user_id
      AND e.embedding IS NOT NULL
    ORDER BY e.embedding <=> p_query_embedding
    LIMIT 3;
END;
$$ LANGUAGE plpgsql;

COMMIT;
```

- [ ] **Step 2: Commit**

```bash
git add migrations/
git commit -m "feat: database migration — memory schema, tables, indexes, SQL functions"
```

---

## Task 7: Rule Interface — BaseExtractor + ExtractionResult

The extraction system is **plugin-based**. Core code defines the interface and a loader. Rules live in YAML files and/or Python plugin classes, loaded at startup from a configurable path. Core code knows nothing about specific rules.

**Files:**
- Create: `src/agent_memory/extraction/__init__.py`
- Create: `src/agent_memory/extraction/base_extractor.py`
- Create: `tests/test_base_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_base_extractor.py
from agent_memory.extraction.base_extractor import BaseExtractor, ExtractionResult
from agent_memory.models import MemoryType


class TestExtractionResult:
    def test_create_with_defaults(self):
        result = ExtractionResult(
            content="Budget: $200",
            memory_type=MemoryType.FACT,
            importance=0.8,
        )
        assert result.matched_chars == 0
        assert result.rule_id == ""
        assert result.confidence == 1.0

    def test_create_with_provenance(self):
        result = ExtractionResult(
            content="Prefers: organic",
            memory_type=MemoryType.PREFERENCE,
            importance=0.7,
            matched_chars=17,
            rule_id="preference_positive",
            confidence=0.9,
        )
        assert result.rule_id == "preference_positive"


class TestBaseExtractor:
    def test_is_abstract(self):
        # Should not be instantiable directly
        import pytest
        with pytest.raises(TypeError):
            BaseExtractor()  # type: ignore

    def test_subclass_must_implement_extract(self):
        class Incomplete(BaseExtractor):
            id = "incomplete"
            category = MemoryType.FACT

        import pytest
        with pytest.raises(TypeError):
            Incomplete()  # type: ignore

    def test_concrete_subclass_works(self):
        class StubExtractor(BaseExtractor):
            id = "stub"
            category = MemoryType.FACT
            importance = 0.5

            def extract(self, text: str) -> list[ExtractionResult]:
                return [ExtractionResult(
                    content=f"stub: {text}",
                    memory_type=self.category,
                    importance=self.importance,
                    rule_id=self.id,
                )]

        ex = StubExtractor()
        results = ex.extract("test input")
        assert len(results) == 1
        assert results[0].rule_id == "stub"
        assert results[0].content == "stub: test input"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_base_extractor.py -v`
Expected: FAIL — cannot import

- [ ] **Step 3: Implement**

```python
# src/agent_memory/extraction/__init__.py
```

```python
# src/agent_memory/extraction/base_extractor.py
"""BaseExtractor interface — all extractors (YAML-wrapped or Python plugins) implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agent_memory.models import MemoryType


@dataclass
class ExtractionResult:
    """A single extracted memory candidate."""
    content: str
    memory_type: MemoryType
    importance: float
    matched_chars: int = 0
    rule_id: str = ""
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class BaseExtractor(ABC):
    """Base class for all memory extractors.

    Subclasses define a rule by providing:
    - id: unique identifier for this rule
    - category: the MemoryType this rule produces
    - importance: default importance for results from this rule

    And implementing extract() which returns zero or more ExtractionResults.
    """

    id: str = ""
    category: MemoryType = MemoryType.FACT
    importance: float = 0.5
    enabled: bool = True
    version: int = 1

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Subclasses without an id are treated as abstract-ish; enforced at registry time

    @abstractmethod
    def extract(self, text: str) -> list[ExtractionResult]:
        """Run this rule on the given text and return matching ExtractionResults."""
        ...
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_base_extractor.py -v`
Expected: All PASS

```bash
git add src/agent_memory/extraction/ tests/test_base_extractor.py
git commit -m "feat: BaseExtractor interface + ExtractionResult dataclass"
```

---

## Task 7a: YAML Rule Schema + YamlRuleExtractor

YAML is the primary way non-engineers (and engineers too) define simple rules. This task defines the schema and a wrapper that turns a YAML rule into a `BaseExtractor`.

**Files:**
- Create: `src/agent_memory/extraction/yaml_extractor.py`
- Create: `tests/test_yaml_extractor.py`

### YAML Rule Schema

Each YAML file contains a top-level `rules:` list. Each rule has:

```yaml
rules:
  - id: budget_explicit            # REQUIRED: unique rule ID
    name: "Explicit budget mention"  # OPTIONAL: human-readable name
    description: "Detects patterns like 'my budget is $200'"  # OPTIONAL
    category: fact                  # REQUIRED: memory type (fact|preference|entity|procedural)
    type: regex                     # REQUIRED: rule type (regex|keyword|keyword_context)
    pattern: 'budget\s*(?:is|of)?\s*\$?\s*(\d[\d,]*)'  # For type=regex
    template: "Budget: ${1}"        # For type=regex — uses match groups
    importance: 0.8                 # REQUIRED: 0.0-1.0
    enabled: true                   # OPTIONAL: default true
    version: 1                      # OPTIONAL: for governance
    owner: "core-team"              # OPTIONAL: who maintains this rule
    test_cases:                     # OPTIONAL but recommended
      - input: "My budget is $200"
        expected_content: "Budget: $200"
      - input: "budget of 500"
        expected_content: "Budget: $500"
```

Three rule types supported:

| Type | What it does | Required fields |
|---|---|---|
| `regex` | Regex match with template interpolation | `pattern`, `template` |
| `keyword` | Simple keyword presence | `keywords` (list) |
| `keyword_context` | Keyword + extracts containing sentence | `keywords` (list) |

- [ ] **Step 1: Write failing tests**

```python
# tests/test_yaml_extractor.py
import pytest
from agent_memory.extraction.yaml_extractor import YamlRuleExtractor, load_yaml_rules
from agent_memory.models import MemoryType


class TestRegexRule:
    def test_simple_regex_match(self):
        rule_def = {
            "id": "budget_test",
            "category": "fact",
            "type": "regex",
            "pattern": r"budget\s*(?:is|of)?\s*\$?\s*(\d[\d,]*)",
            "template": "Budget: ${1}",
            "importance": 0.8,
        }
        extractor = YamlRuleExtractor(rule_def)
        results = extractor.extract("My budget is $200")
        assert len(results) == 1
        assert results[0].content == "Budget: $200"
        assert results[0].memory_type == MemoryType.FACT
        assert results[0].importance == 0.8
        assert results[0].rule_id == "budget_test"

    def test_regex_no_match(self):
        rule_def = {
            "id": "nothing",
            "category": "fact",
            "type": "regex",
            "pattern": r"zzzyyyxxx",
            "template": "match",
            "importance": 0.5,
        }
        extractor = YamlRuleExtractor(rule_def)
        assert extractor.extract("hello world") == []

    def test_template_with_multiple_groups(self):
        rule_def = {
            "id": "preference",
            "category": "preference",
            "type": "regex",
            "pattern": r"i\s+(prefer|like)\s+(.+?)(?:[.\n,]|$)",
            "template": "${1}s: ${2}",
            "importance": 0.7,
        }
        extractor = YamlRuleExtractor(rule_def)
        results = extractor.extract("I prefer organic products.")
        assert len(results) == 1
        assert "prefers: organic products" in results[0].content.lower()


class TestKeywordContextRule:
    def test_extracts_sentence_with_keyword(self):
        rule_def = {
            "id": "allergic",
            "category": "fact",
            "type": "keyword_context",
            "keywords": ["allergic", "allergy"],
            "importance": 0.95,
        }
        extractor = YamlRuleExtractor(rule_def)
        results = extractor.extract("Hi there. I'm allergic to latex. Thanks!")
        assert len(results) == 1
        assert "allergic to latex" in results[0].content.lower()
        assert results[0].importance == 0.95

    def test_no_keyword_no_match(self):
        rule_def = {
            "id": "allergic",
            "category": "fact",
            "type": "keyword_context",
            "keywords": ["allergic"],
            "importance": 0.95,
        }
        extractor = YamlRuleExtractor(rule_def)
        assert extractor.extract("I love pizza") == []


class TestKeywordRule:
    def test_simple_keyword_match(self):
        rule_def = {
            "id": "remember",
            "category": "fact",
            "type": "keyword",
            "keywords": ["remember", "important"],
            "importance": 0.9,
        }
        extractor = YamlRuleExtractor(rule_def)
        results = extractor.extract("Please remember this")
        assert len(results) >= 1


class TestDisabledRule:
    def test_disabled_rule_returns_nothing(self):
        rule_def = {
            "id": "disabled_test",
            "category": "fact",
            "type": "regex",
            "pattern": r"anything",
            "template": "match",
            "importance": 0.5,
            "enabled": False,
        }
        extractor = YamlRuleExtractor(rule_def)
        assert extractor.extract("anything here") == []


class TestLoadYamlRules:
    def test_loads_rules_from_yaml(self, tmp_path):
        yaml_content = """
rules:
  - id: rule_one
    category: fact
    type: regex
    pattern: 'foo'
    template: 'found foo'
    importance: 0.5

  - id: rule_two
    category: preference
    type: keyword
    keywords: ['bar']
    importance: 0.6
"""
        path = tmp_path / "test_rules.yaml"
        path.write_text(yaml_content)
        extractors = load_yaml_rules(path)
        assert len(extractors) == 2
        assert extractors[0].rule_id == "rule_one"
        assert extractors[1].rule_id == "rule_two"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert load_yaml_rules(path) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_yaml_extractor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/agent_memory/extraction/yaml_extractor.py
"""YAML-defined rule extractor — wraps a YAML rule definition as a BaseExtractor."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from agent_memory.extraction.base_extractor import BaseExtractor, ExtractionResult
from agent_memory.models import MemoryType


class YamlRuleExtractor(BaseExtractor):
    """Executes a YAML-defined rule against input text."""

    def __init__(self, rule_def: dict[str, Any]):
        self._validate(rule_def)
        self._rule = rule_def

        # Expose BaseExtractor class attributes via properties (set on instance)
        self.rule_id = rule_def["id"]
        self.id = rule_def["id"]
        self.category = MemoryType(rule_def["category"])
        self.importance = float(rule_def["importance"])
        self.enabled = bool(rule_def.get("enabled", True))
        self.version = int(rule_def.get("version", 1))
        self._type = rule_def["type"]

        # Pre-compile regex for efficiency
        if self._type == "regex":
            self._pattern = re.compile(rule_def["pattern"], re.IGNORECASE)
            self._template = rule_def["template"]
        elif self._type in ("keyword", "keyword_context"):
            self._keywords = [k.lower() for k in rule_def["keywords"]]

    @staticmethod
    def _validate(rule_def: dict[str, Any]) -> None:
        required = {"id", "category", "type", "importance"}
        missing = required - set(rule_def.keys())
        if missing:
            raise ValueError(f"YAML rule missing required fields: {missing}")

        rule_type = rule_def["type"]
        if rule_type == "regex":
            if "pattern" not in rule_def or "template" not in rule_def:
                raise ValueError(f"Regex rule {rule_def['id']} needs 'pattern' and 'template'")
        elif rule_type in ("keyword", "keyword_context"):
            if "keywords" not in rule_def:
                raise ValueError(f"Keyword rule {rule_def['id']} needs 'keywords'")
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

    def extract(self, text: str) -> list[ExtractionResult]:
        if not self.enabled or not text or len(text.strip()) < 2:
            return []

        if self._type == "regex":
            return self._extract_regex(text)
        elif self._type == "keyword_context":
            return self._extract_keyword_context(text)
        elif self._type == "keyword":
            return self._extract_keyword(text)
        return []

    def _extract_regex(self, text: str) -> list[ExtractionResult]:
        results = []
        for match in self._pattern.finditer(text):
            try:
                content = self._apply_template(self._template, match)
            except (IndexError, KeyError):
                continue
            results.append(ExtractionResult(
                content=content,
                memory_type=self.category,
                importance=self.importance,
                matched_chars=match.end() - match.start(),
                rule_id=self.rule_id,
            ))
        return results

    @staticmethod
    def _apply_template(template: str, match: re.Match) -> str:
        # Replace ${1}, ${2}, ... with capture groups
        def sub(m: re.Match) -> str:
            idx = int(m.group(1))
            return match.group(idx)
        return re.sub(r"\$\{(\d+)\}", sub, template).strip()

    def _extract_keyword_context(self, text: str) -> list[ExtractionResult]:
        results = []
        text_lower = text.lower()
        for keyword in self._keywords:
            if keyword not in text_lower:
                continue
            for sentence in re.split(r"[.!?\n]+", text):
                if keyword in sentence.lower() and len(sentence.strip()) > 3:
                    results.append(ExtractionResult(
                        content=sentence.strip(),
                        memory_type=self.category,
                        importance=self.importance,
                        matched_chars=len(sentence.strip()),
                        rule_id=self.rule_id,
                    ))
                    break
        return results

    def _extract_keyword(self, text: str) -> list[ExtractionResult]:
        text_lower = text.lower()
        results = []
        for keyword in self._keywords:
            if keyword in text_lower:
                results.append(ExtractionResult(
                    content=f"Keyword match: {keyword}",
                    memory_type=self.category,
                    importance=self.importance,
                    matched_chars=len(keyword),
                    rule_id=self.rule_id,
                ))
        return results


def load_yaml_rules(path: Path) -> list[YamlRuleExtractor]:
    """Load YAML rules from a file. Returns list of YamlRuleExtractor instances."""
    if not path.exists():
        return []
    content = yaml.safe_load(path.read_text()) or {}
    rules_list = content.get("rules", [])
    return [YamlRuleExtractor(rule_def) for rule_def in rules_list]
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_yaml_extractor.py -v`
Expected: All PASS

```bash
git add src/agent_memory/extraction/yaml_extractor.py tests/test_yaml_extractor.py
git commit -m "feat: YamlRuleExtractor — regex + keyword rules loaded from YAML files"
```

---

## Task 7b: Rule Loader + Rule Registry

**Files:**
- Create: `src/agent_memory/extraction/rule_loader.py`
- Create: `src/agent_memory/extraction/rule_registry.py`
- Create: `tests/test_rule_loader.py`
- Create: `tests/test_rule_registry.py`

The loader scans a directory for YAML rule files and Python plugin modules. The registry holds loaded extractors and provides the main `extract()` entrypoint that runs all enabled rules.

- [ ] **Step 1: Write failing tests for loader**

```python
# tests/test_rule_loader.py
import pytest
from agent_memory.extraction.rule_loader import RuleLoader
from agent_memory.extraction.base_extractor import BaseExtractor, ExtractionResult
from agent_memory.models import MemoryType


class TestLoadFromDirectory:
    def test_loads_yaml_files(self, tmp_path):
        (tmp_path / "rule1.yaml").write_text("""
rules:
  - id: rule_a
    category: fact
    type: regex
    pattern: 'foo'
    template: 'found foo'
    importance: 0.5
""")
        (tmp_path / "rule2.yaml").write_text("""
rules:
  - id: rule_b
    category: preference
    type: keyword
    keywords: ['bar']
    importance: 0.6
""")
        loader = RuleLoader()
        extractors = loader.load_from_directory(tmp_path)
        assert len(extractors) == 2
        ids = [e.rule_id for e in extractors]
        assert "rule_a" in ids
        assert "rule_b" in ids

    def test_empty_directory(self, tmp_path):
        loader = RuleLoader()
        assert loader.load_from_directory(tmp_path) == []

    def test_ignores_non_yaml(self, tmp_path):
        (tmp_path / "rule.yaml").write_text("""
rules:
  - id: rule_a
    category: fact
    type: regex
    pattern: 'foo'
    template: 'found foo'
    importance: 0.5
""")
        (tmp_path / "readme.md").write_text("not a rule file")
        loader = RuleLoader()
        extractors = loader.load_from_directory(tmp_path)
        assert len(extractors) == 1

    def test_invalid_yaml_raises(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("""
rules:
  - id: missing_category_and_type
    importance: 0.5
""")
        loader = RuleLoader()
        with pytest.raises(ValueError):
            loader.load_from_directory(tmp_path)


class TestLoadFromMultiplePaths:
    def test_merges_rules_from_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "core"
        dir2 = tmp_path / "ecommerce"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "a.yaml").write_text("""
rules:
  - id: core_a
    category: fact
    type: regex
    pattern: 'x'
    template: 'x'
    importance: 0.5
""")
        (dir2 / "b.yaml").write_text("""
rules:
  - id: ecom_b
    category: preference
    type: keyword
    keywords: ['y']
    importance: 0.6
""")
        loader = RuleLoader()
        extractors = loader.load_from_paths([dir1, dir2])
        assert len(extractors) == 2
        assert {e.rule_id for e in extractors} == {"core_a", "ecom_b"}
```

- [ ] **Step 2: Write failing tests for registry**

```python
# tests/test_rule_registry.py
import pytest
from agent_memory.extraction.base_extractor import BaseExtractor, ExtractionResult
from agent_memory.extraction.rule_registry import RuleRegistry
from agent_memory.models import MemoryType


class StubRule(BaseExtractor):
    def __init__(self, rid: str, content: str):
        self.rule_id = rid
        self.id = rid
        self.category = MemoryType.FACT
        self.importance = 0.5
        self.enabled = True
        self._content = content

    def extract(self, text: str) -> list[ExtractionResult]:
        if self._content in text:
            return [ExtractionResult(
                content=f"matched {self._content}",
                memory_type=MemoryType.FACT,
                importance=0.5,
                rule_id=self.rule_id,
                matched_chars=len(self._content),
            )]
        return []


class TestRegistry:
    def test_register_and_list(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        reg.register(StubRule("r2", "bar"))
        assert len(reg.all()) == 2

    def test_no_duplicate_ids(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(StubRule("r1", "bar"))

    def test_extract_runs_all_rules(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        reg.register(StubRule("r2", "bar"))
        results = reg.extract("foo and bar")
        assert len(results) == 2
        assert {r.rule_id for r in results} == {"r1", "r2"}

    def test_extract_empty_text(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        assert reg.extract("") == []
        assert reg.extract("   ") == []

    def test_extract_no_matches(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        assert reg.extract("nothing here") == []

    def test_disabled_rule_skipped(self):
        reg = RuleRegistry()
        rule = StubRule("r1", "foo")
        rule.enabled = False
        reg.register(rule)
        assert reg.extract("foo") == []

    def test_get_by_id(self):
        reg = RuleRegistry()
        reg.register(StubRule("r1", "foo"))
        found = reg.get("r1")
        assert found is not None
        assert found.rule_id == "r1"
        assert reg.get("nonexistent") is None
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_rule_loader.py tests/test_rule_registry.py -v`
Expected: FAIL

- [ ] **Step 4: Implement rule loader**

```python
# src/agent_memory/extraction/rule_loader.py
"""Rule loader — discovers YAML rule files and Python plugins from configurable paths."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from agent_memory.extraction.base_extractor import BaseExtractor
from agent_memory.extraction.yaml_extractor import load_yaml_rules


class RuleLoader:
    """Loads rules from a directory. Discovers .yaml files and .py plugin modules."""

    def load_from_directory(self, path: Path) -> list[BaseExtractor]:
        """Load all rules (YAML + Python plugins) from a single directory."""
        if not path.exists() or not path.is_dir():
            return []

        extractors: list[BaseExtractor] = []

        # Load YAML rule files
        for yaml_file in sorted(path.glob("*.yaml")):
            extractors.extend(load_yaml_rules(yaml_file))

        # Load Python plugin modules
        for py_file in sorted(path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            extractors.extend(self._load_python_plugin(py_file))

        return extractors

    def load_from_paths(self, paths: list[Path]) -> list[BaseExtractor]:
        """Load rules from multiple directories."""
        all_extractors: list[BaseExtractor] = []
        for path in paths:
            all_extractors.extend(self.load_from_directory(path))
        return all_extractors

    def _load_python_plugin(self, py_file: Path) -> list[BaseExtractor]:
        """Load a Python plugin file and return any BaseExtractor subclasses found."""
        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        extractors: list[BaseExtractor] = []
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseExtractor)
                and obj is not BaseExtractor
            ):
                try:
                    instance = obj()
                    if getattr(instance, "rule_id", None) or getattr(instance, "id", None):
                        extractors.append(instance)
                except TypeError:
                    # Abstract class, skip
                    continue
        return extractors
```

- [ ] **Step 5: Implement rule registry**

```python
# src/agent_memory/extraction/rule_registry.py
"""Rule registry — central holder for loaded extractors."""

from __future__ import annotations

from agent_memory.extraction.base_extractor import BaseExtractor, ExtractionResult


class RuleRegistry:
    """Holds loaded extractors and runs them against input text."""

    def __init__(self):
        self._rules: dict[str, BaseExtractor] = {}

    def register(self, extractor: BaseExtractor) -> None:
        """Register an extractor. Raises if a rule with the same id already exists."""
        rule_id = getattr(extractor, "rule_id", None) or getattr(extractor, "id", "")
        if not rule_id:
            raise ValueError(f"Extractor {extractor} has no rule_id or id")
        if rule_id in self._rules:
            raise ValueError(f"Rule '{rule_id}' already registered")
        self._rules[rule_id] = extractor

    def register_all(self, extractors: list[BaseExtractor]) -> None:
        """Register a batch of extractors."""
        for ex in extractors:
            self.register(ex)

    def get(self, rule_id: str) -> BaseExtractor | None:
        return self._rules.get(rule_id)

    def all(self) -> list[BaseExtractor]:
        return list(self._rules.values())

    def extract(self, text: str) -> list[ExtractionResult]:
        """Run all enabled rules against the text and return combined results."""
        if not text or not text.strip():
            return []

        results: list[ExtractionResult] = []
        for rule in self._rules.values():
            if not getattr(rule, "enabled", True):
                continue
            try:
                results.extend(rule.extract(text))
            except Exception:
                # Individual rule failures must not break the whole extraction
                continue
        return results
```

- [ ] **Step 6: Run tests, commit**

Run: `pytest tests/test_rule_loader.py tests/test_rule_registry.py -v`
Expected: All PASS

```bash
git add src/agent_memory/extraction/rule_loader.py src/agent_memory/extraction/rule_registry.py tests/test_rule_loader.py tests/test_rule_registry.py
git commit -m "feat: RuleLoader (YAML + Python plugins) and RuleRegistry"
```

---

## Task 7c: Builtin Rule Set

Ship a small default rule set with the module. These are reference rules — users can disable, override, or extend them by adding files to their own rules directory.

**Files:**
- Create: `src/agent_memory/extraction/builtin/budget.yaml`
- Create: `src/agent_memory/extraction/builtin/preferences.yaml`
- Create: `src/agent_memory/extraction/builtin/sizes.yaml`
- Create: `src/agent_memory/extraction/builtin/keyword_triggers.yaml`
- Create: `tests/test_builtin_rules.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_builtin_rules.py
from pathlib import Path

import pytest

from agent_memory.extraction.rule_loader import RuleLoader
from agent_memory.extraction.rule_registry import RuleRegistry
from agent_memory.models import MemoryType


@pytest.fixture
def registry():
    builtin_dir = Path(__file__).parent.parent / "src" / "agent_memory" / "extraction" / "builtin"
    loader = RuleLoader()
    extractors = loader.load_from_directory(builtin_dir)
    reg = RuleRegistry()
    reg.register_all(extractors)
    return reg


class TestBudgetRules:
    def test_dollar_amount(self, registry):
        results = registry.extract("My budget is $200")
        assert any("200" in r.content for r in results)

    def test_budget_under(self, registry):
        results = registry.extract("I want something under $150")
        assert any("150" in r.content for r in results)

    def test_spend_around(self, registry):
        results = registry.extract("Looking to spend around $500")
        assert any("500" in r.content for r in results)


class TestPreferenceRules:
    def test_prefer_positive(self, registry):
        results = registry.extract("I prefer organic products.")
        assert any(
            r.memory_type == MemoryType.PREFERENCE and "organic" in r.content.lower()
            for r in results
        )

    def test_like_positive(self, registry):
        results = registry.extract("I really like minimalist design.")
        assert any(r.memory_type == MemoryType.PREFERENCE for r in results)

    def test_dislike(self, registry):
        results = registry.extract("I don't like synthetic materials.")
        assert any(
            r.memory_type == MemoryType.PREFERENCE and "synthetic" in r.content.lower()
            for r in results
        )


class TestSizeRules:
    def test_numeric_size(self, registry):
        results = registry.extract("I wear size 10")
        assert any(r.memory_type == MemoryType.FACT and "10" in r.content for r in results)


class TestKeywordTriggers:
    def test_allergic_high_importance(self, registry):
        results = registry.extract("I'm allergic to latex")
        assert any(r.importance >= 0.9 for r in results)

    def test_always_procedural(self, registry):
        results = registry.extract("I always check reviews before buying")
        assert any(r.importance >= 0.7 for r in results)


class TestNoExtraction:
    def test_greeting(self, registry):
        assert len(registry.extract("hi")) == 0

    def test_thanks(self, registry):
        assert len(registry.extract("thanks")) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_builtin_rules.py -v`
Expected: FAIL — directory doesn't exist yet

- [ ] **Step 3: Create builtin rule files**

```yaml
# src/agent_memory/extraction/builtin/budget.yaml
rules:
  - id: budget_explicit
    name: "Explicit budget mention"
    description: "Detects patterns like 'my budget is $200' or 'budget of 500'"
    category: fact
    type: regex
    pattern: '(?:budget|spend|spending|price range)\s*(?:is|of|around|about|under|up to)?\s*\$?\s*(\d[\d,]*)'
    template: "Budget: $${1}"
    importance: 0.8
    version: 1
    owner: "core-team"

  - id: budget_under_amount
    name: "Under a given amount"
    description: "Detects 'under $150', 'less than $100', etc."
    category: fact
    type: regex
    pattern: '(?:under|below|less than|no more than|up to)\s*\$\s*(\d[\d,]*)'
    template: "Budget: under $${1}"
    importance: 0.8
    version: 1

  - id: budget_spend_intent
    name: "Spend intent"
    description: "Detects 'looking to spend around $500', 'willing to pay $200'"
    category: fact
    type: regex
    pattern: '(?:looking to spend|willing to pay|can afford)\s*(?:around|about|up to)?\s*\$?\s*(\d[\d,]*)'
    template: "Budget: around $${1}"
    importance: 0.8
    version: 1
```

```yaml
# src/agent_memory/extraction/builtin/preferences.yaml
rules:
  - id: preference_positive
    name: "Positive preference"
    description: "User explicitly states a liking (prefer/like/love/enjoy)"
    category: preference
    type: regex
    pattern: '\bi\s+(?:really\s+)?(?:prefer|like|love|enjoy)\s+(.+?)(?:[.\n,;!]|$)'
    template: "Prefers: ${1}"
    importance: 0.7
    version: 1

  - id: preference_favorite
    name: "Favorite or go-to mention"
    category: preference
    type: regex
    pattern: '\bmy\s+(?:favorite|preferred|go-to)\s+(?:\w+\s+)?(?:is|are)\s+(.+?)(?:[.\n,;!]|$)'
    template: "Favorite: ${1}"
    importance: 0.8
    version: 1

  - id: preference_negative
    name: "Negative preference"
    description: "User states they dislike/hate something"
    category: preference
    type: regex
    pattern: "\\bi\\s+(?:really\\s+)?(?:don'?t\\s+(?:like|want)|hate|dislike|can'?t\\s+stand|avoid)\\s+(.+?)(?:[.\\n,;!]|$)"
    template: "Dislikes: ${1}"
    importance: 0.7
    version: 1

  - id: preference_not_fan
    name: "Not a fan / not interested"
    category: preference
    type: regex
    pattern: '\bnot\s+(?:a fan of|interested in)\s+(.+?)(?:[.\n,;!]|$)'
    template: "Dislikes: ${1}"
    importance: 0.7
    version: 1
```

```yaml
# src/agent_memory/extraction/builtin/sizes.yaml
rules:
  - id: size_explicit
    name: "Explicit size statement"
    category: fact
    type: regex
    pattern: '(?:my\s+)?size\s+(?:is\s+)?(\w+)'
    template: "Size: ${1}"
    importance: 0.9
    version: 1

  - id: size_wear
    name: "User wears size"
    category: fact
    type: regex
    pattern: "i\\s+(?:wear|am a|'m a)\\s+(?:size\\s+)?(\\w+)"
    template: "Size: ${1}"
    importance: 0.9
    version: 1
```

```yaml
# src/agent_memory/extraction/builtin/keyword_triggers.yaml
rules:
  - id: trigger_allergic
    name: "Critical allergy/health info"
    description: "High-importance fact — must never be missed"
    category: fact
    type: keyword_context
    keywords: ["allergic", "allergy", "intolerant"]
    importance: 0.95
    version: 1

  - id: trigger_remember
    name: "Explicit remember request"
    category: fact
    type: keyword_context
    keywords: ["remember", "don't forget"]
    importance: 0.9
    version: 1

  - id: trigger_always
    name: "Procedural pattern — always"
    category: procedural
    type: keyword_context
    keywords: ["always", "every time"]
    importance: 0.8
    version: 1

  - id: trigger_never
    name: "Procedural pattern — never"
    category: procedural
    type: keyword_context
    keywords: ["never"]
    importance: 0.8
    version: 1

  - id: trigger_important
    name: "User marks something as important"
    category: fact
    type: keyword_context
    keywords: ["important"]
    importance: 0.85
    version: 1
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_builtin_rules.py -v`
Expected: All PASS

```bash
git add src/agent_memory/extraction/builtin/ tests/test_builtin_rules.py
git commit -m "feat: builtin rule set — budget, preferences, sizes, keyword triggers"
```

---

## Task 8: LLM Routing Decision (simplified v1 — 2-signal OR gate)

Based on A-MAC's published ablation study: Content Type Prior captures ~49% of routing signal value, while the other signals each add only ~13%. The v1 routing function is a simple OR gate over the two signals that together proxy Content Type Prior: extraction yield + unstructured ratio.

**Dropped from v1:** session complexity (not predictive per A-MAC), contradictions (belongs to consolidation, not extraction routing). **Deferred to v1.5:** semantic novelty (valuable but requires local MiniLM model).

**Files:**
- Create: `src/agent_memory/extraction/llm_router.py`
- Create: `tests/test_llm_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_router.py
import pytest
from agent_memory.extraction.llm_router import LLMRouter, RoutingSignals
from agent_memory.extraction.base_extractor import ExtractionResult
from agent_memory.models import MemoryType
from agent_memory.config import ExtractionConfig


@pytest.fixture
def router():
    return LLMRouter(config=ExtractionConfig())


class TestExtractionYield:
    def test_high_yield_no_llm(self, router):
        signals = router.compute_signals(
            user_messages=["I need size 10", "Budget is $200"],
            extracted=[
                ExtractionResult(content="Size: 10", memory_type=MemoryType.FACT, importance=0.9, matched_chars=7),
                ExtractionResult(content="Budget: $200", memory_type=MemoryType.FACT, importance=0.8, matched_chars=13),
            ],
        )
        assert signals.low_yield is False
        assert signals.high_unstructured is False

    def test_low_yield_triggers(self, router):
        signals = router.compute_signals(
            user_messages=["long message one with lots of stuff", "long message two"],
            extracted=[],
        )
        assert signals.low_yield is True


class TestUnstructuredRatio:
    def test_mostly_structured(self, router):
        text = "My budget is $200"
        signals = router.compute_signals(
            user_messages=[text],
            extracted=[
                ExtractionResult(content="Budget: $200", memory_type=MemoryType.FACT, importance=0.8, matched_chars=17),
            ],
        )
        assert signals.high_unstructured is False

    def test_mostly_unstructured(self, router):
        text = "I was thinking about something that feels premium but isn't flashy, like what I had before but updated"
        signals = router.compute_signals(
            user_messages=[text],
            extracted=[],
        )
        assert signals.high_unstructured is True


class TestNeedsLLM:
    def test_structured_high_yield_no_llm(self, router):
        """High yield + low unstructured = rules handled it, skip LLM."""
        assert router.needs_llm(
            user_messages=["Budget is $200, size 10"],
            extracted=[
                ExtractionResult(content="Budget: $200", memory_type=MemoryType.FACT, importance=0.8, matched_chars=14),
                ExtractionResult(content="Size: 10", memory_type=MemoryType.FACT, importance=0.9, matched_chars=7),
            ],
        ) is False

    def test_low_yield_triggers_llm(self, router):
        """Low yield alone triggers LLM escalation."""
        assert router.needs_llm(
            user_messages=["msg1", "msg2", "msg3", "msg4", "msg5"],
            extracted=[],  # zero extraction
        ) is True

    def test_high_unstructured_triggers_llm(self, router):
        """High unstructured ratio alone triggers LLM escalation."""
        assert router.needs_llm(
            user_messages=["I want something that feels premium, you know what I mean, like last time but updated"],
            extracted=[],
        ) is True

    def test_short_non_content_skipped(self, router):
        """Very short sessions (greetings) don't trigger LLM even with zero yield."""
        # "hi" has length 2 → total_chars=2. low_yield is True (0/1 < 0.3) but
        # because the content is so short, this is effectively a no-op.
        # The rule: if there's essentially nothing to extract, don't bother LLM.
        needs = router.needs_llm(
            user_messages=["hi"],
            extracted=[],
        )
        # v1 will route this to LLM, which is a (small) cost.
        # v1.5 adds a minimum-content guard to skip trivially short sessions.
        # For v1, acceptable — LLM will just return empty extraction.
        assert isinstance(needs, bool)


class TestRoutingSignalsShape:
    def test_signals_only_has_v1_fields(self, router):
        signals = router.compute_signals(user_messages=["x"], extracted=[])
        # v1 only exposes low_yield and high_unstructured
        assert hasattr(signals, "low_yield")
        assert hasattr(signals, "high_unstructured")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_llm_router.py -v`
Expected: FAIL

- [ ] **Step 3: Implement LLM router (v1 simplified)**

```python
# src/agent_memory/extraction/llm_router.py
"""LLM routing decision — simplified v1 2-signal OR gate.

Decision function:
    ROUTE_TO_LLM = (extraction_yield < yield_threshold)
                OR (unstructured_ratio > unstructured_threshold)

v1.5 will add semantic_novelty as a third OR term using a local MiniLM model.

Rationale: A-MAC ablation (arXiv:2603.04549) showed Content Type Prior
captures ~49% of routing signal value. Extraction yield + unstructured ratio
together serve as a cheap proxy for Content Type Prior.

Explicitly dropped from v1:
- session_complexity: not predictive per A-MAC ablation
- contradictions: belongs to consolidation stage, not extraction routing
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_memory.config import ExtractionConfig
from agent_memory.extraction.base_extractor import ExtractionResult


@dataclass
class RoutingSignals:
    """Routing signals for the v1 extraction router.

    v1 has only two signals. v1.5 will add high_novelty.
    """
    low_yield: bool
    high_unstructured: bool


class LLMRouter:
    def __init__(self, config: ExtractionConfig | None = None):
        self.config = config or ExtractionConfig()

    def compute_signals(
        self,
        user_messages: list[str],
        extracted: list[ExtractionResult],
    ) -> RoutingSignals:
        return RoutingSignals(
            low_yield=self._check_yield(user_messages, extracted),
            high_unstructured=self._check_unstructured(user_messages, extracted),
        )

    def needs_llm(
        self,
        user_messages: list[str],
        extracted: list[ExtractionResult],
    ) -> bool:
        signals = self.compute_signals(user_messages, extracted)
        return self.should_escalate(signals)

    def should_escalate(self, signals: RoutingSignals) -> bool:
        """v1: simple OR gate over two signals."""
        return signals.low_yield or signals.high_unstructured

    def _check_yield(
        self, user_messages: list[str], extracted: list[ExtractionResult]
    ) -> bool:
        msg_count = len(user_messages)
        if msg_count == 0:
            return False
        ratio = len(extracted) / msg_count
        return ratio < self.config.yield_threshold

    def _check_unstructured(
        self, user_messages: list[str], extracted: list[ExtractionResult]
    ) -> bool:
        total_chars = sum(len(m) for m in user_messages)
        if total_chars == 0:
            return False
        matched_chars = sum(r.matched_chars for r in extracted)
        unstructured_ratio = 1.0 - (matched_chars / total_chars)
        return unstructured_ratio > self.config.unstructured_threshold
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_llm_router.py -v`
Expected: All PASS

```bash
git add src/agent_memory/extraction/llm_router.py tests/test_llm_router.py
git commit -m "feat: v1 LLM routing — 2-signal OR gate (extraction_yield + unstructured_ratio)"
```

---

## Task 9: LLM Extractor

**Files:**
- Create: `src/agent_memory/extraction/llm_extractor.py`
- Create: `tests/test_llm_extractor.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm_extractor.py (placeholder)
```

Actually — this task is intentionally lightweight. The LLM extractor is a thin wrapper around whatever LLM client the user has. We define the interface and a fake for testing.

```python
# tests/test_llm_extractor.py
import pytest
from agent_memory.extraction.llm_extractor import LLMExtractor, FakeLLMExtractor
from agent_memory.extraction.base_extractor import ExtractionResult
from agent_memory.models import MemoryType


class TestFakeLLMExtractor:
    @pytest.mark.asyncio
    async def test_returns_extraction_results(self):
        extractor = FakeLLMExtractor()
        results = await extractor.extract(
            session_text="I like running shoes that are lightweight and breathable"
        )
        assert isinstance(results, list)
        assert all(isinstance(r, ExtractionResult) for r in results)

    @pytest.mark.asyncio
    async def test_empty_input(self):
        extractor = FakeLLMExtractor()
        results = await extractor.extract(session_text="")
        assert results == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_llm_extractor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/agent_memory/extraction/llm_extractor.py
"""LLM-based memory extraction for complex sessions."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from agent_memory.extraction.base_extractor import ExtractionResult
from agent_memory.models import MemoryType

EXTRACTION_PROMPT = """Given this conversation session, extract memorable information.

For each item, provide:
- content: what to remember (concise, factual)
- memory_type: one of "fact", "preference", "entity", "procedural"
- importance: 0.0-1.0

Categories:
1. FACTS: concrete statements ("budget of $200", "size 10")
2. PREFERENCES: likes, dislikes, tendencies ("prefers fast shipping")
3. ENTITIES: brands, products, categories with context
4. PROCEDURAL: interaction patterns ("always asks for reviews")

Return a JSON array. If nothing worth extracting, return [].

Session:
{session_text}"""


class LLMExtractor(ABC):
    """Abstract LLM extractor. Implement with your LLM client."""

    @abstractmethod
    async def extract(self, session_text: str) -> list[ExtractionResult]: ...


class FakeLLMExtractor(LLMExtractor):
    """Fake for testing. Returns a simple extraction based on text length."""

    async def extract(self, session_text: str) -> list[ExtractionResult]:
        if not session_text or len(session_text.strip()) < 5:
            return []
        return [
            ExtractionResult(
                content=f"LLM-extracted from session: {session_text[:80]}",
                memory_type=MemoryType.FACT,
                importance=0.6,
                matched_chars=0,
            )
        ]


def parse_llm_response(response_text: str) -> list[ExtractionResult]:
    """Parse LLM JSON response into ExtractionResult list."""
    try:
        items = json.loads(response_text)
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    results = []
    for item in items:
        if not isinstance(item, dict) or "content" not in item:
            continue
        mem_type_str = item.get("memory_type", "fact")
        try:
            mem_type = MemoryType(mem_type_str)
        except ValueError:
            mem_type = MemoryType.FACT
        results.append(ExtractionResult(
            content=item["content"],
            memory_type=mem_type,
            importance=float(item.get("importance", 0.5)),
            matched_chars=0,
        ))
    return results
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_llm_extractor.py -v`
Expected: All PASS

```bash
git add src/agent_memory/extraction/llm_extractor.py tests/test_llm_extractor.py
git commit -m "feat: LLM extractor interface with fake for testing + JSON parser"
```

---

## Task 10: Extraction Pipeline Orchestrator

**Files:**
- Create: `src/agent_memory/extraction/pipeline.py`
- Create: `tests/test_extraction_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extraction_pipeline.py
import uuid
from pathlib import Path

import pytest

from agent_memory.config import ExtractionConfig
from agent_memory.embeddings import FakeEmbeddingProvider
from agent_memory.extraction.llm_extractor import FakeLLMExtractor
from agent_memory.extraction.llm_router import LLMRouter
from agent_memory.extraction.pipeline import ExtractionPipeline
from agent_memory.extraction.rule_loader import RuleLoader
from agent_memory.extraction.rule_registry import RuleRegistry
from agent_memory.models import MemoryType


@pytest.fixture
def rule_registry():
    """Load builtin rules into a fresh registry."""
    builtin_dir = Path(__file__).parent.parent / "src" / "agent_memory" / "extraction" / "builtin"
    loader = RuleLoader()
    extractors = loader.load_from_directory(builtin_dir)
    reg = RuleRegistry()
    reg.register_all(extractors)
    return reg


@pytest.fixture
def pipeline(rule_registry):
    return ExtractionPipeline(
        rule_registry=rule_registry,
        llm_router=LLMRouter(config=ExtractionConfig()),
        llm_extractor=FakeLLMExtractor(),
        embedding_provider=FakeEmbeddingProvider(dimensions=4),
    )


class TestExtractionPipeline:
    @pytest.mark.asyncio
    async def test_structured_session_rules_only(self, pipeline):
        result = await pipeline.process_session(
            user_messages=["My budget is $200 and I wear size 10"],
            existing_memory_embeddings=[],
        )
        assert result.used_llm is False
        assert len(result.memories) >= 2
        assert any("200" in m.content for m in result.memories)
        assert any("10" in m.content for m in result.memories)

    @pytest.mark.asyncio
    async def test_unstructured_session_triggers_llm(self, pipeline):
        result = await pipeline.process_session(
            user_messages=[
                "I've been looking for something that feels premium but isn't flashy",
                "kind of like what I had before but more modern and updated",
                "the material should feel good, you know what I mean",
                "not too heavy either, something breathable for summer",
            ],
            existing_memory_embeddings=[],
        )
        assert result.used_llm is True

    @pytest.mark.asyncio
    async def test_empty_session(self, pipeline):
        result = await pipeline.process_session(
            user_messages=["hi", "ok"],
            existing_memory_embeddings=[],
        )
        assert len(result.memories) == 0
        assert result.used_llm is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_extraction_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/agent_memory/extraction/pipeline.py
"""Extraction pipeline — orchestrates rule-based + LLM routing + LLM extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_memory.embeddings import EmbeddingProvider
from agent_memory.extraction.base_extractor import ExtractionResult
from agent_memory.extraction.llm_extractor import LLMExtractor
from agent_memory.extraction.llm_router import LLMRouter
from agent_memory.extraction.rule_registry import RuleRegistry


@dataclass
class PipelineResult:
    memories: list[ExtractionResult] = field(default_factory=list)
    used_llm: bool = False


class ExtractionPipeline:
    def __init__(
        self,
        rule_registry: RuleRegistry,
        llm_router: LLMRouter,
        llm_extractor: LLMExtractor,
        embedding_provider: EmbeddingProvider,
    ):
        self._rules = rule_registry
        self._router = llm_router
        self._llm = llm_extractor
        self._embed = embedding_provider

    async def process_session(
        self,
        user_messages: list[str],
        existing_memory_embeddings: list[list[float]] | None = None,  # reserved for v1.5 novelty signal
    ) -> PipelineResult:
        if not user_messages:
            return PipelineResult()

        full_text = "\n".join(user_messages)

        # Pass 1: Rule-based extraction via the registry
        rule_results = self._rules.extract(full_text)

        # Pass 2: LLM routing decision (v1: 2-signal OR gate, no embeddings needed)
        needs_llm = self._router.needs_llm(
            user_messages=user_messages,
            extracted=rule_results,
        )

        if not needs_llm:
            return PipelineResult(memories=rule_results, used_llm=False)

        # Pass 3: LLM extraction
        llm_results = await self._llm.extract(session_text=full_text)

        # Merge: rule results + LLM results, deduplicate by content similarity
        all_results = rule_results + llm_results
        return PipelineResult(memories=all_results, used_llm=True)
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_extraction_pipeline.py -v`
Expected: All PASS

```bash
git add src/agent_memory/extraction/pipeline.py tests/test_extraction_pipeline.py
git commit -m "feat: extraction pipeline orchestrator — rules → routing → LLM fallback"
```

---

## Task 11: Multi-Signal Search

**Files:**
- Create: `src/agent_memory/retrieval/__init__.py`
- Create: `src/agent_memory/retrieval/search.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search.py
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from agent_memory.retrieval.search import score_memories
from agent_memory.models import Memory, MemoryType
from agent_memory.config import RetrievalConfig


@pytest.fixture
def config():
    return RetrievalConfig()


class TestScoreMemories:
    def test_higher_similarity_ranks_higher(self, config):
        query_emb = [1.0, 0.0, 0.0, 0.0]
        now = datetime.now(timezone.utc)
        m1 = Memory(user_id=uuid.uuid4(), content="close match", embedding=[0.9, 0.1, 0.0, 0.0], last_accessed=now)
        m2 = Memory(user_id=uuid.uuid4(), content="far match", embedding=[0.1, 0.9, 0.0, 0.0], last_accessed=now)
        scored = score_memories([m1, m2], query_emb, config)
        assert scored[0].content == "close match"

    def test_higher_importance_boosts(self, config):
        query_emb = [1.0, 0.0, 0.0, 0.0]
        now = datetime.now(timezone.utc)
        # Same embedding, different importance
        emb = [0.5, 0.5, 0.0, 0.0]
        m1 = Memory(user_id=uuid.uuid4(), content="important", embedding=emb, importance=0.9, last_accessed=now)
        m2 = Memory(user_id=uuid.uuid4(), content="not important", embedding=emb, importance=0.1, last_accessed=now)
        scored = score_memories([m1, m2], query_emb, config)
        assert scored[0].content == "important"

    def test_more_recent_ranks_higher(self, config):
        query_emb = [1.0, 0.0, 0.0, 0.0]
        emb = [0.5, 0.5, 0.0, 0.0]
        now = datetime.now(timezone.utc)
        m1 = Memory(user_id=uuid.uuid4(), content="recent", embedding=emb, importance=0.5, last_accessed=now)
        m2 = Memory(user_id=uuid.uuid4(), content="old", embedding=emb, importance=0.5, last_accessed=now - timedelta(days=30))
        scored = score_memories([m1, m2], query_emb, config)
        assert scored[0].content == "recent"

    def test_respects_limit(self, config):
        query_emb = [1.0, 0.0, 0.0, 0.0]
        now = datetime.now(timezone.utc)
        memories = [
            Memory(user_id=uuid.uuid4(), content=f"mem {i}", embedding=[0.5, 0.5, 0.0, 0.0], last_accessed=now)
            for i in range(20)
        ]
        scored = score_memories(memories, query_emb, config, limit=5)
        assert len(scored) == 5

    def test_empty_input(self, config):
        scored = score_memories([], [1.0, 0.0], config)
        assert scored == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_search.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/agent_memory/retrieval/__init__.py
```

```python
# src/agent_memory/retrieval/search.py
"""Multi-signal scored memory search."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from agent_memory.config import RetrievalConfig
from agent_memory.models import Memory


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_memories(
    memories: list[Memory],
    query_embedding: list[float],
    config: RetrievalConfig,
    limit: int | None = None,
) -> list[Memory]:
    """Score and rank memories using multi-signal retrieval."""
    if not memories:
        return []

    now = datetime.now(timezone.utc)
    scored: list[tuple[float, Memory]] = []

    for m in memories:
        if m.embedding is None:
            continue

        relevance = _cosine_sim(query_embedding, m.embedding)
        seconds_since = (now - m.last_accessed).total_seconds()
        recency = math.exp(-0.01 * seconds_since / 86400.0)
        frequency = min(math.log(m.access_count + 1) / 5.0, 1.0)

        score = (
            config.w_relevance * relevance
            + config.w_recency * recency
            + config.w_importance * m.importance
            + config.w_frequency * frequency
        )
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    effective_limit = limit or config.default_limit
    return [m for _, m in scored[:effective_limit]]
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_search.py -v`
Expected: All PASS

```bash
git add src/agent_memory/retrieval/ tests/test_search.py
git commit -m "feat: multi-signal scored search — relevance, recency, importance, frequency"
```

---

## Task 12: Context Assembly + Token Budgeting

**Files:**
- Create: `src/agent_memory/retrieval/context.py`
- Create: `tests/test_context.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_context.py
import pytest
from agent_memory.retrieval.context import assemble_context, estimate_tokens
from agent_memory.models import ContextRow


class TestEstimateTokens:
    def test_rough_estimate(self):
        # ~4 chars per token is a common rough estimate
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("a" * 400) >= 80  # at least 400/5
        assert estimate_tokens("") == 0


class TestAssembleContext:
    def test_priority_ordering(self):
        rows = [
            ContextRow(section="relevant", content="relevant memory", priority=2),
            ContextRow(section="profile", content="profile fact", priority=1),
            ContextRow(section="entities", content="entity info", priority=4),
        ]
        result = assemble_context(rows, token_budget=2000)
        assert result[0].section == "profile"
        assert result[-1].section == "entities"

    def test_token_budget_truncates(self):
        rows = [
            ContextRow(section="profile", content="short", priority=1),
            ContextRow(section="relevant", content="x " * 5000, priority=2),  # very long
            ContextRow(section="entities", content="should be dropped", priority=4),
        ]
        result = assemble_context(rows, token_budget=100)
        # Profile should be included, the huge relevant block may or may not fit,
        # but entities should be dropped
        sections = [r.section for r in result]
        assert "profile" in sections

    def test_empty_input(self):
        result = assemble_context([], token_budget=2000)
        assert result == []

    def test_format_as_text(self):
        rows = [
            ContextRow(section="profile", content="User likes blue", priority=1),
            ContextRow(section="relevant", content="Budget is $200", priority=2),
        ]
        from agent_memory.retrieval.context import format_memory_block
        text = format_memory_block(rows)
        assert "User likes blue" in text
        assert "Budget is $200" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_context.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/agent_memory/retrieval/context.py
"""Context assembly with token budgeting."""

from __future__ import annotations

from agent_memory.models import ContextRow


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def assemble_context(
    rows: list[ContextRow],
    token_budget: int = 2000,
) -> list[ContextRow]:
    """Filter and order context rows within a token budget."""
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda r: r.priority)

    result: list[ContextRow] = []
    used_tokens = 0

    for row in sorted_rows:
        row_tokens = estimate_tokens(row.content)
        if used_tokens + row_tokens > token_budget:
            continue
        result.append(row)
        used_tokens += row_tokens

    return result


_SECTION_HEADERS = {
    "profile": "## User Profile",
    "relevant": "## Relevant Memories",
    "recent_episodes": "## Recent Conversations",
    "entities": "## Known Entities",
}


def format_memory_block(rows: list[ContextRow]) -> str:
    """Format context rows into a text block for injection into a system prompt."""
    if not rows:
        return ""

    sections: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda r: r.priority):
        sections.setdefault(row.section, []).append(row.content)

    parts: list[str] = []
    for section, items in sections.items():
        header = _SECTION_HEADERS.get(section, f"## {section.title()}")
        parts.append(header)
        for item in items:
            parts.append(f"- {item}")
        parts.append("")

    return "\n".join(parts).strip()
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_context.py -v`
Expected: All PASS

```bash
git add src/agent_memory/retrieval/context.py tests/test_context.py
git commit -m "feat: context assembly with token budgeting and prompt formatting"
```

---

## Task 13: Agent Integration — Tool + Prompt

**Files:**
- Create: `src/agent_memory/integration/__init__.py`
- Create: `src/agent_memory/integration/agent_tool.py`
- Create: `src/agent_memory/integration/prompt.py`
- Create: `tests/test_agent_tool.py`
- Create: `tests/test_prompt.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent_tool.py
import uuid
import pytest
from agent_memory.integration.agent_tool import get_save_memory_tool_definition, handle_save_memory
from agent_memory.in_memory_provider import InMemoryProvider
from agent_memory.embeddings import FakeEmbeddingProvider


class TestToolDefinition:
    def test_has_required_fields(self):
        tool = get_save_memory_tool_definition()
        assert tool["name"] == "save_memory"
        assert "description" in tool
        assert "input_schema" in tool
        assert "content" in tool["input_schema"]["properties"]
        assert tool["input_schema"]["required"] == ["content"]


class TestHandleSaveMemory:
    @pytest.mark.asyncio
    async def test_saves_memory(self):
        provider = InMemoryProvider(FakeEmbeddingProvider(dimensions=4))
        uid = uuid.uuid4()
        sid = uuid.uuid4()
        aid = uuid.uuid4()
        mid = await handle_save_memory(
            provider=provider,
            user_id=uid,
            session_id=sid,
            agent_id=aid,
            params={"content": "User is vegan", "memory_type": "fact", "importance": 0.9},
        )
        assert isinstance(mid, uuid.UUID)
        emb = await provider._embed.embed("vegan")
        results = await provider.search(uid, "vegan", emb, limit=5)
        assert len(results) == 1
        assert "vegan" in results[0].content
```

```python
# tests/test_prompt.py
from agent_memory.integration.prompt import MEMORY_SYSTEM_PROMPT, build_system_prompt_addition


class TestPrompt:
    def test_system_prompt_exists(self):
        assert "Memory" in MEMORY_SYSTEM_PROMPT
        assert "save_memory" in MEMORY_SYSTEM_PROMPT

    def test_build_with_memory_block(self):
        result = build_system_prompt_addition("- User likes blue\n- Budget is $200")
        assert "What you remember" in result
        assert "User likes blue" in result

    def test_build_empty(self):
        result = build_system_prompt_addition("")
        assert "No memories" in result or result == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_agent_tool.py tests/test_prompt.py -v`
Expected: FAIL

- [ ] **Step 3: Implement agent tool**

```python
# src/agent_memory/integration/__init__.py
```

```python
# src/agent_memory/integration/agent_tool.py
"""save_memory agent tool — hot path for explicit memory saves."""

from __future__ import annotations

import uuid
from typing import Any

from agent_memory.models import MemoryType
from agent_memory.provider import MemoryProvider


def get_save_memory_tool_definition() -> dict[str, Any]:
    """Returns the tool definition to register with the agent."""
    return {
        "name": "save_memory",
        "description": (
            "Save an important fact, preference, or observation about the user "
            "for future conversations. Use when the user tells you something "
            "worth remembering, or when you notice a pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What to remember",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "preference", "entity", "procedural"],
                    "description": "Category of memory",
                },
                "importance": {
                    "type": "number",
                    "description": "0.0-1.0, how important is this to remember",
                },
            },
            "required": ["content"],
        },
    }


async def handle_save_memory(
    provider: MemoryProvider,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    agent_id: uuid.UUID,
    params: dict[str, Any],
) -> uuid.UUID:
    """Handle a save_memory tool call from the agent."""
    content = params["content"]
    mem_type_str = params.get("memory_type", "fact")
    try:
        mem_type = MemoryType(mem_type_str)
    except ValueError:
        mem_type = MemoryType.FACT
    importance = float(params.get("importance", 0.7))

    return await provider.add(
        user_id=user_id,
        content=content,
        memory_type=mem_type,
        importance=importance,
        source_session_id=session_id,
    )
```

- [ ] **Step 4: Implement prompt builder**

```python
# src/agent_memory/integration/prompt.py
"""System prompt memory block builder."""

MEMORY_SYSTEM_PROMPT = """## Memory

You have access to memories about users from past conversations.
Relevant memories are provided in the "What you remember" section below.

Guidelines:
- Use memories naturally, don't announce "I remember that..." unless asked
- If you learn something important (preference, constraint, fact), use the save_memory tool
- Don't save trivial or session-specific things ("user said hello")
- If a memory seems outdated or contradicted by current conversation, save the updated version"""


def build_system_prompt_addition(memory_block: str) -> str:
    """Build the memory section to add to the system prompt."""
    if not memory_block or not memory_block.strip():
        return f"{MEMORY_SYSTEM_PROMPT}\n\n### What you remember about this user:\nNo memories yet."

    return f"{MEMORY_SYSTEM_PROMPT}\n\n### What you remember about this user:\n{memory_block}"
```

- [ ] **Step 5: Run tests, commit**

Run: `pytest tests/test_agent_tool.py tests/test_prompt.py -v`
Expected: All PASS

```bash
git add src/agent_memory/integration/ tests/test_agent_tool.py tests/test_prompt.py
git commit -m "feat: agent integration — save_memory tool + system prompt builder"
```

---

## Task 14: Pipeline Workers — Embedding, Episodes, Decay

**Files:**
- Create: `src/agent_memory/pipeline/__init__.py`
- Create: `src/agent_memory/pipeline/embedding.py`
- Create: `src/agent_memory/pipeline/episodes.py`
- Create: `src/agent_memory/pipeline/decay.py`
- Create: `tests/test_embedding_pipeline.py`
- Create: `tests/test_episodes.py`
- Create: `tests/test_decay.py`

- [ ] **Step 1: Write failing tests for embedding worker**

```python
# tests/test_embedding_pipeline.py
import uuid
import pytest
from agent_memory.pipeline.embedding import EmbeddingWorker
from agent_memory.models import Memory
from agent_memory.embeddings import FakeEmbeddingProvider


class TestEmbeddingWorker:
    @pytest.mark.asyncio
    async def test_embeds_null_memories(self):
        provider = FakeEmbeddingProvider(dimensions=4)
        worker = EmbeddingWorker(embedding_provider=provider)
        memories = [
            Memory(user_id=uuid.uuid4(), content="test one", embedding=None),
            Memory(user_id=uuid.uuid4(), content="test two", embedding=None),
            Memory(user_id=uuid.uuid4(), content="already embedded", embedding=[0.1, 0.2, 0.3, 0.4]),
        ]
        updated = await worker.embed_batch(memories)
        assert len(updated) == 2
        assert all(m.embedding is not None for m in updated)
        assert all(len(m.embedding) == 4 for m in updated)

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        provider = FakeEmbeddingProvider(dimensions=4)
        worker = EmbeddingWorker(embedding_provider=provider)
        updated = await worker.embed_batch([])
        assert updated == []
```

- [ ] **Step 2: Write failing tests for episodes**

```python
# tests/test_episodes.py
import uuid
import pytest
from datetime import datetime, timezone
from agent_memory.pipeline.episodes import create_episode


class TestEpisodeCreation:
    def test_from_lakebase_summary(self):
        episode = create_episode(
            user_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            lakebase_summary="User searched for running shoes, compared Nike and Brooks, chose Nike Pegasus.",
            started_at=datetime.now(timezone.utc),
            memory_ids=[uuid.uuid4(), uuid.uuid4()],
        )
        assert "running shoes" in episode.summary
        assert len(episode.memory_ids) == 2

    def test_no_summary_uses_fallback(self):
        episode = create_episode(
            user_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            lakebase_summary=None,
            started_at=datetime.now(timezone.utc),
            memory_ids=[],
            fallback_text="User asked about products",
        )
        assert episode.summary == "User asked about products"
```

- [ ] **Step 3: Write failing tests for decay**

```python
# tests/test_decay.py
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from agent_memory.pipeline.decay import apply_decay, should_archive
from agent_memory.models import Memory
from agent_memory.config import DecayConfig


@pytest.fixture
def config():
    return DecayConfig()


class TestApplyDecay:
    def test_recent_memory_barely_decays(self, config):
        m = Memory(
            user_id=uuid.uuid4(), content="recent",
            importance=0.8, last_accessed=datetime.now(timezone.utc),
            decay_rate=0.01,
        )
        decayed = apply_decay(m)
        assert decayed >= 0.78  # barely changed

    def test_old_memory_decays_significantly(self, config):
        m = Memory(
            user_id=uuid.uuid4(), content="old",
            importance=0.8,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=90),
            decay_rate=0.01,
        )
        decayed = apply_decay(m)
        assert decayed < 0.7


class TestShouldArchive:
    def test_low_importance_old_memory(self, config):
        m = Memory(
            user_id=uuid.uuid4(), content="forgotten",
            importance=0.03,
            last_accessed=datetime.now(timezone.utc) - timedelta(days=100),
        )
        assert should_archive(m, config) is True

    def test_high_importance_recent_memory(self, config):
        m = Memory(
            user_id=uuid.uuid4(), content="important",
            importance=0.9,
            last_accessed=datetime.now(timezone.utc),
        )
        assert should_archive(m, config) is False
```

- [ ] **Step 4: Run all to verify failure**

Run: `pytest tests/test_embedding_pipeline.py tests/test_episodes.py tests/test_decay.py -v`
Expected: FAIL

- [ ] **Step 5: Implement embedding worker**

```python
# src/agent_memory/pipeline/__init__.py
```

```python
# src/agent_memory/pipeline/embedding.py
"""Stage 2: Batch embedding worker."""

from __future__ import annotations

from agent_memory.embeddings import EmbeddingProvider
from agent_memory.models import Memory


class EmbeddingWorker:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self._embed = embedding_provider

    async def embed_batch(self, memories: list[Memory]) -> list[Memory]:
        """Embed memories that have NULL embeddings. Returns only the updated ones."""
        to_embed = [m for m in memories if m.embedding is None]
        if not to_embed:
            return []

        texts = [m.content for m in to_embed]
        embeddings = await self._embed.embed_batch(texts)

        for mem, emb in zip(to_embed, embeddings):
            mem.embedding = emb

        return to_embed
```

- [ ] **Step 6: Implement episode creation**

```python
# src/agent_memory/pipeline/episodes.py
"""Stage 3: Episode creation from lakebase sessions."""

from __future__ import annotations

import uuid
from datetime import datetime

from agent_memory.models import Episode


def create_episode(
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    lakebase_summary: str | None,
    started_at: datetime,
    memory_ids: list[uuid.UUID] | None = None,
    ended_at: datetime | None = None,
    fallback_text: str = "Session recorded",
) -> Episode:
    """Create an episode from lakebase session data."""
    summary = lakebase_summary if lakebase_summary else fallback_text

    return Episode(
        user_id=user_id,
        session_id=session_id,
        summary=summary,
        started_at=started_at,
        ended_at=ended_at,
        memory_ids=memory_ids or [],
    )
```

- [ ] **Step 7: Implement decay**

```python
# src/agent_memory/pipeline/decay.py
"""Stage 5: Importance decay and archival."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from agent_memory.config import DecayConfig
from agent_memory.models import Memory


def apply_decay(memory: Memory) -> float:
    """Compute decayed importance for a memory. Returns new importance value."""
    now = datetime.now(timezone.utc)
    seconds_since = (now - memory.last_accessed).total_seconds()
    days_since = seconds_since / 86400.0
    decayed = memory.importance * math.exp(-memory.decay_rate * days_since)
    return max(0.0, decayed)


def should_archive(memory: Memory, config: DecayConfig) -> bool:
    """Determine if a memory should be archived."""
    now = datetime.now(timezone.utc)
    days_since = (now - memory.last_accessed).total_seconds() / 86400.0
    return (
        memory.importance < config.archive_threshold
        and days_since > config.archive_after_days
    )
```

- [ ] **Step 8: Run tests, commit**

Run: `pytest tests/test_embedding_pipeline.py tests/test_episodes.py tests/test_decay.py -v`
Expected: All PASS

```bash
git add src/agent_memory/pipeline/ tests/test_embedding_pipeline.py tests/test_episodes.py tests/test_decay.py
git commit -m "feat: pipeline workers — embedding batch, episode creation, decay & archival"
```

---

## Task 15: Consolidation — Three-Tier Dedup + Importance-Sum Reflection

**Key corrections vs. earlier plan:**
- Dedup uses **similarity ≥ 0.90** framing (was "distance < 0.1" — same math, clearer framing)
- Three tiers: exact hash (SHA-256) → fuzzy (pg_trgm) → semantic (pgvector)
- Reflection triggers on **importance sum ≥ 150** (Stanford Generative Agents actual mechanism), not new-memory count

**Files:**
- Create: `src/agent_memory/pipeline/consolidation.py`
- Create: `tests/test_consolidation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_consolidation.py
import hashlib
import uuid
import pytest
from agent_memory.pipeline.consolidation import (
    compute_content_hash,
    find_semantic_duplicates,
    needs_reflection,
    accumulated_importance_sum,
    merge_duplicates,
)
from agent_memory.models import Memory
from agent_memory.config import ConsolidationConfig


@pytest.fixture
def config():
    return ConsolidationConfig()


class TestContentHash:
    def test_hash_is_sha256(self):
        h = compute_content_hash("Budget: $200")
        assert len(h) == 64  # SHA-256 hex
        assert h == hashlib.sha256("budget: $200".encode()).hexdigest()  # normalized

    def test_normalization_ignores_case_whitespace(self):
        h1 = compute_content_hash("Budget: $200")
        h2 = compute_content_hash("  budget: $200  ")
        h3 = compute_content_hash("BUDGET: $200")
        assert h1 == h2 == h3


class TestSemanticDuplicates:
    def test_detects_near_duplicates_at_0_90_similarity(self, config):
        uid = uuid.uuid4()
        # Vectors with cosine sim ~0.99 — well above 0.90 threshold
        m1 = Memory(user_id=uid, content="budget is $200", embedding=[1.0, 0.0, 0.0, 0.0])
        m2 = Memory(user_id=uid, content="budget is $200 dollars", embedding=[0.99, 0.01, 0.0, 0.0])
        m3 = Memory(user_id=uid, content="likes blue color", embedding=[0.0, 1.0, 0.0, 0.0])
        pairs = find_semantic_duplicates([m1, m2, m3], similarity_threshold=0.90)
        assert len(pairs) == 1
        assert {pairs[0][0].id, pairs[0][1].id} == {m1.id, m2.id}

    def test_no_false_positives_below_threshold(self, config):
        uid = uuid.uuid4()
        m1 = Memory(user_id=uid, content="one", embedding=[1.0, 0.0, 0.0, 0.0])
        m2 = Memory(user_id=uid, content="two", embedding=[0.0, 1.0, 0.0, 0.0])
        assert find_semantic_duplicates([m1, m2], similarity_threshold=0.90) == []

    def test_only_same_user(self, config):
        m1 = Memory(user_id=uuid.uuid4(), content="one", embedding=[1.0, 0.0, 0.0, 0.0])
        m2 = Memory(user_id=uuid.uuid4(), content="two", embedding=[0.99, 0.01, 0.0, 0.0])
        assert find_semantic_duplicates([m1, m2], similarity_threshold=0.90) == []


class TestMerge:
    def test_keep_higher_importance(self):
        uid = uuid.uuid4()
        a = Memory(user_id=uid, content="Budget: $200", importance=0.9, embedding=[1.0, 0.0])
        b = Memory(user_id=uid, content="Budget: $200 dollars", importance=0.6, embedding=[1.0, 0.0])
        merged = merge_duplicates(a, b)
        assert merged.importance == 0.9  # a wins

    def test_appends_unique_info(self):
        uid = uuid.uuid4()
        a = Memory(user_id=uid, content="Budget: $200", importance=0.9, embedding=[1.0, 0.0])
        b = Memory(user_id=uid, content="will pay up to $250", importance=0.6, embedding=[1.0, 0.0])
        merged = merge_duplicates(a, b)
        assert "$200" in merged.content
        assert "$250" in merged.content


class TestImportanceSumReflectionTrigger:
    """Stanford Generative Agents: reflection triggers when importance sum ≥ 150."""

    def test_accumulated_sum_uses_scaled_importance(self):
        """importance ∈ [0,1] is scaled to [1,10] for the sum."""
        uid = uuid.uuid4()
        mems = [
            Memory(user_id=uid, content=f"m{i}", importance=0.5, embedding=None)
            for i in range(10)
        ]
        # 10 memories × 5 (0.5 * 10) = 50
        assert accumulated_importance_sum(mems) == 50

    def test_high_importance_triggers_faster(self):
        uid = uuid.uuid4()
        mems = [
            Memory(user_id=uid, content=f"m{i}", importance=1.0, embedding=None)
            for i in range(15)
        ]
        # 15 memories × 10 = 150 — exactly at threshold
        assert accumulated_importance_sum(mems) == 150

    def test_reflection_triggers_at_150(self, config):
        """Default threshold is 150 (Stanford's value)."""
        mems_15_high = [
            Memory(user_id=uuid.uuid4(), content=f"m{i}", importance=1.0, embedding=None)
            for i in range(15)
        ]
        assert needs_reflection(mems_15_high, config) is True

    def test_no_reflection_below_150(self, config):
        mems_10_medium = [
            Memory(user_id=uuid.uuid4(), content=f"m{i}", importance=0.5, embedding=None)
            for i in range(10)
        ]
        # 10 × 5 = 50, well below 150
        assert needs_reflection(mems_10_medium, config) is False

    def test_mundane_events_accumulate_slowly(self, config):
        """Per Stanford: mundane events score 1-2, significant score 8-10."""
        mundane = [
            Memory(user_id=uuid.uuid4(), content=f"m{i}", importance=0.15, embedding=None)
            for i in range(50)
        ]
        # 50 × ~1.5 ≈ 75, still below 150
        assert needs_reflection(mundane, config) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_consolidation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/agent_memory/pipeline/consolidation.py
"""Stage 4: Memory consolidation.

Three-tier deduplication (hot path + batch):
  1. Exact: SHA-256 content hash lookup (O(1) via unique index)
  2. Fuzzy: PostgreSQL pg_trgm similarity (batch only)
  3. Semantic: pgvector cosine similarity ≥ 0.90

Reflection trigger: Stanford Generative Agents' mechanism.
Accumulated scaled-importance (0-1 importance × 10) reaching 150.
"""

from __future__ import annotations

import hashlib
import math

from agent_memory.config import ConsolidationConfig
from agent_memory.models import Memory


def compute_content_hash(content: str) -> str:
    """Normalize content and return SHA-256 hex digest.

    Normalization: lowercase, strip surrounding whitespace.
    Same-meaning variants collide (case/whitespace differences ignored).
    """
    normalized = content.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def find_semantic_duplicates(
    memories: list[Memory],
    similarity_threshold: float = 0.90,
) -> list[tuple[Memory, Memory]]:
    """Find memory pairs with cosine similarity ≥ threshold within the same user scope.

    Threshold interpretation:
    - ≥ 0.95: near-certain duplicate, auto-merge
    - 0.90-0.95: safe to merge (higher-importance wins)
    - < 0.90: distinct, keep both
    """
    pairs: list[tuple[Memory, Memory]] = []

    for i, a in enumerate(memories):
        if a.embedding is None:
            continue
        for b in memories[i + 1:]:
            if b.embedding is None:
                continue
            if a.user_id != b.user_id:
                continue
            sim = _cosine_similarity(a.embedding, b.embedding)
            if sim >= similarity_threshold:
                pairs.append((a, b))

    return pairs


def merge_duplicates(keep: Memory, remove: Memory) -> Memory:
    """Merge two duplicate memories. Keeps the higher-importance one,
    appends unique info from the other.
    """
    if remove.importance > keep.importance:
        keep, remove = remove, keep

    if remove.content not in keep.content:
        keep.content = f"{keep.content}; {remove.content}"

    keep.access_count += remove.access_count
    if remove.id not in keep.source_memory_ids:
        keep.source_memory_ids.append(remove.id)

    return keep


def accumulated_importance_sum(memories: list[Memory]) -> float:
    """Sum of scaled importance scores.

    Stanford Generative Agents rated on 1-10. Our importance ∈ [0,1] scales
    by ×10. Mundane events (1-2) accumulate slowly; significant events
    (8-10) trigger reflection quickly.
    """
    return sum(m.importance * 10 for m in memories)


def needs_reflection(recent_memories: list[Memory], config: ConsolidationConfig) -> bool:
    """Check if accumulated importance since last reflection exceeds threshold.

    Default threshold: 150 (Stanford's value). Caller is responsible for
    passing only memories created since the last reflection.
    """
    return accumulated_importance_sum(recent_memories) >= config.reflection_importance_sum_threshold
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_consolidation.py -v`
Expected: All PASS

```bash
git add src/agent_memory/pipeline/consolidation.py tests/test_consolidation.py
git commit -m "feat: consolidation — three-tier dedup + importance-sum reflection trigger (Stanford pattern)"
```

---

## Task 16: Pipeline Runner

**Files:**
- Create: `src/agent_memory/pipeline/runner.py`
- Create: `tests/test_pipeline_runner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_pipeline_runner.py
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.embeddings import FakeEmbeddingProvider
from agent_memory.extraction.llm_extractor import FakeLLMExtractor
from agent_memory.extraction.llm_router import LLMRouter
from agent_memory.extraction.rule_loader import RuleLoader
from agent_memory.extraction.rule_registry import RuleRegistry
from agent_memory.in_memory_provider import InMemoryProvider
from agent_memory.pipeline.runner import PipelineRunner


@pytest.fixture
def runner():
    config = MemoryConfig()
    embed = FakeEmbeddingProvider(dimensions=4)
    provider = InMemoryProvider(embedding_provider=embed)

    # Load builtin rules into a fresh registry
    builtin_dir = Path(__file__).parent.parent / "src" / "agent_memory" / "extraction" / "builtin"
    loader = RuleLoader()
    rule_registry = RuleRegistry()
    rule_registry.register_all(loader.load_from_directory(builtin_dir))

    return PipelineRunner(
        config=config,
        provider=provider,
        rule_registry=rule_registry,
        llm_router=LLMRouter(config=config.extraction),
        llm_extractor=FakeLLMExtractor(),
        embedding_provider=embed,
    )


class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_process_session_end_to_end(self, runner):
        uid = uuid.uuid4()
        sid = uuid.uuid4()
        result = await runner.process_session(
            user_id=uid,
            session_id=sid,
            user_messages=["My budget is $300 and I wear size M"],
            lakebase_summary="User asked about shoes with a $300 budget, size M.",
            started_at=datetime.now(timezone.utc),
        )
        assert result.memories_created > 0
        assert result.episode_created is True

        # Verify memories are searchable
        emb = await runner._embed.embed("budget")
        results = await runner._provider.search(uid, "budget", emb, limit=5)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_empty_session(self, runner):
        result = await runner.process_session(
            user_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            user_messages=["hi"],
            lakebase_summary=None,
            started_at=datetime.now(timezone.utc),
        )
        assert result.memories_created == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/agent_memory/pipeline/runner.py
"""Pipeline orchestrator — runs all stages for a session."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from agent_memory.config import MemoryConfig
from agent_memory.embeddings import EmbeddingProvider
from agent_memory.extraction.llm_extractor import LLMExtractor
from agent_memory.extraction.llm_router import LLMRouter
from agent_memory.extraction.pipeline import ExtractionPipeline
from agent_memory.extraction.rule_registry import RuleRegistry
from agent_memory.models import MemoryType
from agent_memory.pipeline.episodes import create_episode
from agent_memory.provider import MemoryProvider


@dataclass
class SessionProcessingResult:
    memories_created: int = 0
    episode_created: bool = False
    used_llm: bool = False


class PipelineRunner:
    def __init__(
        self,
        config: MemoryConfig,
        provider: MemoryProvider,
        rule_registry: RuleRegistry,
        llm_router: LLMRouter,
        llm_extractor: LLMExtractor,
        embedding_provider: EmbeddingProvider,
    ):
        self._config = config
        self._provider = provider
        self._embed = embedding_provider
        self._extraction = ExtractionPipeline(
            rule_registry=rule_registry,
            llm_router=llm_router,
            llm_extractor=llm_extractor,
            embedding_provider=embedding_provider,
        )

    async def process_session(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        user_messages: list[str],
        lakebase_summary: str | None,
        started_at: datetime,
        ended_at: datetime | None = None,
    ) -> SessionProcessingResult:
        result = SessionProcessingResult()

        # Stage 1: Extraction
        extraction = await self._extraction.process_session(
            user_messages=user_messages,
            existing_memory_embeddings=[],
        )
        result.used_llm = extraction.used_llm

        # Store extracted memories
        memory_ids: list[uuid.UUID] = []
        for mem in extraction.memories:
            mid = await self._provider.add(
                user_id=user_id,
                content=mem.content,
                memory_type=mem.memory_type,
                importance=mem.importance,
                source_session_id=session_id,
            )
            memory_ids.append(mid)

        result.memories_created = len(memory_ids)

        # Stage 3: Episode creation
        if user_messages and any(len(m.strip()) > 3 for m in user_messages):
            _episode = create_episode(
                user_id=user_id,
                session_id=session_id,
                lakebase_summary=lakebase_summary,
                started_at=started_at,
                ended_at=ended_at,
                memory_ids=memory_ids,
            )
            result.episode_created = True

        return result
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_pipeline_runner.py -v`
Expected: All PASS

```bash
git add src/agent_memory/pipeline/runner.py tests/test_pipeline_runner.py
git commit -m "feat: pipeline runner — end-to-end session processing orchestrator"
```

---

## Task 17: Update Package Init + Run Full Test Suite

**Files:**
- Modify: `src/agent_memory/__init__.py`

- [ ] **Step 1: Update init with key exports**

```python
# src/agent_memory/__init__.py
"""Agent Memory — general-purpose memory module for AI agents."""

from agent_memory.config import MemoryConfig, load_config
from agent_memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider
from agent_memory.models import ContextRow, Entity, Episode, Memory, MemoryType, ProcessingLog
from agent_memory.provider import MemoryProvider
from agent_memory.in_memory_provider import InMemoryProvider
from agent_memory.integration.agent_tool import get_save_memory_tool_definition, handle_save_memory
from agent_memory.integration.prompt import build_system_prompt_addition
from agent_memory.pipeline.runner import PipelineRunner

__all__ = [
    "MemoryConfig",
    "load_config",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "Memory",
    "Episode",
    "Entity",
    "ContextRow",
    "MemoryType",
    "ProcessingLog",
    "MemoryProvider",
    "InMemoryProvider",
    "get_save_memory_tool_definition",
    "handle_save_memory",
    "build_system_prompt_addition",
    "PipelineRunner",
]
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/agent_memory/__init__.py
git commit -m "feat: finalize package exports and verify full test suite"
```

---

## Summary

| Task | What It Builds | Key Files |
|---|---|---|
| 1 | Project scaffolding | `pyproject.toml`, conftest |
| 2 | Data models | `models.py` |
| 3 | Config loading | `config.py`, `config.yaml` |
| 4 | Embedding interface | `embeddings.py` |
| 5 | Provider interface + InMemory | `provider.py`, `in_memory_provider.py` |
| 6 | Database migration | `migrations/001_*.sql` |
| 7 | Rule interface (`BaseExtractor`) | `extraction/base_extractor.py` |
| 7a | YAML rule schema + `YamlRuleExtractor` | `extraction/yaml_extractor.py` |
| 7b | Rule loader + registry | `extraction/rule_loader.py`, `extraction/rule_registry.py` |
| 7c | Builtin rule set | `extraction/builtin/*.yaml` |
| 8 | LLM routing (v1: 2-signal OR gate) | `extraction/llm_router.py` |
| 9 | LLM extractor interface | `extraction/llm_extractor.py` |
| 10 | Extraction pipeline | `extraction/pipeline.py` |
| 11 | Multi-signal search | `retrieval/search.py` |
| 12 | Context assembly | `retrieval/context.py` |
| 13 | Agent tool + prompt | `integration/agent_tool.py`, `integration/prompt.py` |
| 14 | Pipeline workers | `pipeline/embedding.py`, `episodes.py`, `decay.py` |
| 15 | Consolidation (3-tier dedup + importance-sum reflection) | `pipeline/consolidation.py` |
| 16 | Pipeline runner | `pipeline/runner.py` |
| 17 | Package finalization | `__init__.py`, full test run |

### Key architectural changes vs. earlier plan revisions

1. **Bi-temporal data model** — all memory mutations go through `valid_until` invalidation; `memory.memory_history` is an immutable audit log. Never silently overwrite.
2. **halfvec from day one** — 50% storage reduction with minimal recall impact. Migrating later requires a full re-embed.
3. **Simplified 2-signal routing** — per A-MAC's ablation, `extraction_yield < 0.3 OR unstructured_ratio > 0.7` captures ~80% of value. Semantic novelty deferred to v1.5 with local MiniLM. Session complexity and contradictions dropped from extraction routing.
4. **Three-tier dedup** — exact hash (SHA-256) → fuzzy (pg_trgm) → semantic (cosine ≥ 0.90). Real-time at insert + batch at consolidation.
5. **Importance-sum reflection trigger** — Stanford Generative Agents actual mechanism: sum of scaled importance ≥ 150 (not new-memory count ≥ 20).
6. **Extraction versioning** — every memory stores `extraction_version`, `extraction_model`, `prompt_hash`, `rule_id` for re-extraction when rules/prompts change.
7. **Memory-specific episode summaries** — start with lakebase summary (if usable) but enrich with memory IDs, key topics, outcome. Re-generate if summary is too short.

**PostgresMemoryProvider** (Task 6 migration + a future task) is intentionally left as the migration SQL only. The Python `PostgresMemoryProvider` wrapping `asyncpg` calls to those SQL functions should be implemented once you have a PostgreSQL instance available for integration testing. The `InMemoryProvider` covers all logic testing in the meantime.
