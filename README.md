# Mnemosyne

Persistent memory for AI agents. Drop it into your agent server and your agent remembers users across sessions — preferences, facts, past decisions, context. Mnemosyne extracts memories from conversations (rule-based + LLM), automatically identifies entities (people, brands, products, locations), stores everything with bi-temporal versioning, and retrieves it via multi-signal scored search (relevance, recency, importance, frequency). It generates high-level insights through reflection when memories accumulate and resolves contradictions when new information conflicts with existing memories. It runs as an embedded Python module, not a separate service. Swap between in-memory (dev) and PostgreSQL (production) without changing your code.

## Features

- **Hybrid extraction** — YAML/Python rules try first; an LLM picks up whatever rules can't handle.
- **Multi-signal retrieval** — relevance + recency + importance + frequency, fused with vector + full-text search.
- **Entity-aware search** — finds memories by mentioned entity (person, brand, location), not just text similarity. Reciprocal-rank fusion with vector results.
- **Bi-temporal data model** — nothing is ever deleted; old facts are invalidated and remain queryable for audit.
- **Three-tier dedup** — exact content hash, fuzzy trigram match, and cosine-similarity collapse.
- **Reflection** — when a user's accumulated importance crosses a threshold, the system synthesizes higher-level insights and stores them as first-class memories.
- **Contradiction detection** — cross-encoder NLI flags conflicts; resolution invalidates older memories.
- **Privacy controls** — list / get / delete individual memories, export everything for a user, GDPR-grade physical delete with audit trail, per-user extraction toggle.
- **Versioned extraction** — when rules or prompts change, mark old memories and re-run extraction on demand.
- **Observability** — Prometheus exporter at `/metrics`, startup health checks for PG / pgvector / embedding dim / LLM availability.
- **Pluggable backends** — `InMemoryProvider` for dev, `PostgresMemoryProvider` (pgvector + pg_trgm) for production. Same API.
- **Pluggable models** — Ollama, OpenAI, Anthropic, Google, FastEmbed, or any OpenAI-compatible endpoint.

## Install

```bash
pip install "git+https://github.com/hr231/mnemosyne@master"
```

Optional extras:

```bash
pip install "mnemosyne[ner]"          # spaCy + GLiNER for entity extraction
pip install "mnemosyne[nli]"          # cross-encoder for contradiction detection
pip install "mnemosyne[monitoring]"   # Prometheus exporter
pip install "mnemosyne[fastembed]"    # FastEmbed local embeddings
pip install "mnemosyne[anthropic]"    # Anthropic SDK
pip install "mnemosyne[google]"       # Google GenAI SDK
pip install "mnemosyne[all]"          # everything
```

## Quick Start

```python
import asyncio
from uuid import uuid4
from mnemosyne import (
    Settings, InMemoryProvider, ExtractionPipeline,
    EmbeddingClient, assemble_context,
)

async def main():
    settings = Settings.from_env()
    provider = InMemoryProvider()
    embedder = EmbeddingClient.from_config(settings.embedding_config)
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)

    user_id = uuid4()
    await pipeline.process(
        user_id=user_id,
        text="I like Nike shoes, size 10, budget under $150"
    )

    query_vec = await embedder.embed("shoes")
    context = await assemble_context(
        provider, user_id, query_vec, embedder, token_budget=500
    )
    print(context.text)

asyncio.run(main())
```

## Configuration

All settings come from environment variables via `Settings.from_env()`. Every field has a sensible default.

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_LLM_PROVIDER` | `ollama` | `ollama`, `openai_compatible`, `anthropic`, `google`, or `fake` |
| `MNEMOSYNE_LLM_MODEL` | `gemma3:4b` | Model name passed to the API |
| `MNEMOSYNE_LLM_BASE_URL` | `http://localhost:11434/v1` | API base URL |
| `MNEMOSYNE_LLM_API_KEY` | — | Bearer token (optional for local Ollama) |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_EMBEDDING_PROVIDER` | `ollama` | `ollama`, `openai_compatible`, `fastembed`, `google`, or `fake` |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | Model name |
| `MNEMOSYNE_EMBEDDING_BASE_URL` | `http://localhost:11434` | API base URL |
| `MNEMOSYNE_EMBEDDING_API_KEY` | — | Bearer token (optional for local Ollama) |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | Expected vector dimensions |

### Retrieval & Pipeline

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_TOKEN_BUDGET` | `2000` | Max tokens for context assembly |
| `MNEMOSYNE_ROUTER_UNSTRUCTURED_THRESHOLD` | `0.7` | Unstructured ratio above this triggers LLM extraction |
| `MNEMOSYNE_EXTRACTION_VERSION` | `0.1.0` | Version stamped on extracted memories |
| `MNEMOSYNE_RULES_DIR` | `rules/core` | Path to YAML rule directory |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_PG_DSN` | — | PostgreSQL connection string. When set, enables `PostgresMemoryProvider` |

### Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_METRICS_ENABLED` | `1` | Set `0` to disable the Prometheus exporter |
| `MNEMOSYNE_METRICS_PORT` | `9090` | Port for the `/metrics` HTTP endpoint |
| `MNEMOSYNE_STARTUP_WARN_ONLY` | `0` | Set `1` to log failed startup checks instead of raising |

## Provider Setup

### InMemoryProvider (dev/test)

Zero setup. Data lives in process memory, lost on restart.

```python
from mnemosyne import InMemoryProvider

provider = InMemoryProvider()
```

### PostgresMemoryProvider (production)

Requires PostgreSQL 16+ with pgvector and pg_trgm.

```bash
# Start Postgres with pgvector
docker run -d --name mnemosyne-pg \
  -e POSTGRES_USER=mnemosyne \
  -e POSTGRES_PASSWORD=mnemosyne \
  -e POSTGRES_DB=mnemosyne \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Set the DSN
export MNEMOSYNE_PG_DSN=postgresql://mnemosyne:mnemosyne@localhost:5432/mnemosyne

# Run migrations
pip install alembic asyncpg pgvector
alembic upgrade head
```

```python
from mnemosyne.providers.postgres import PostgresMemoryProvider

provider = await PostgresMemoryProvider.connect(os.environ["MNEMOSYNE_PG_DSN"])

# On shutdown:
await provider.close()
```

Both providers implement the same `MemoryProvider` interface. Application code does not change when switching backends.

## Embedding Providers

```python
from mnemosyne import Settings, EmbeddingClient

settings = Settings.from_env()
embedder = EmbeddingClient.from_config(settings.embedding_config)
```

### Ollama (local)

```bash
ollama pull nomic-embed-text
```

```
MNEMOSYNE_EMBEDDING_PROVIDER=ollama
MNEMOSYNE_EMBEDDING_BASE_URL=http://localhost:11434
MNEMOSYNE_EMBEDDING_MODEL=nomic-embed-text
MNEMOSYNE_EMBEDDING_DIM=768
```

### OpenAI

```
MNEMOSYNE_EMBEDDING_PROVIDER=openai_compatible
MNEMOSYNE_EMBEDDING_BASE_URL=https://api.openai.com
MNEMOSYNE_EMBEDDING_MODEL=text-embedding-3-small
MNEMOSYNE_EMBEDDING_API_KEY=sk-...
MNEMOSYNE_EMBEDDING_DIM=1536
```

### Any OpenAI-Compatible Endpoint

Works with Databricks, vLLM, Azure OpenAI, Together, Anyscale, and similar — anything that speaks the `/v1/embeddings` format.

```
MNEMOSYNE_EMBEDDING_PROVIDER=openai_compatible
MNEMOSYNE_EMBEDDING_BASE_URL=https://your-endpoint.com
MNEMOSYNE_EMBEDDING_MODEL=your-model-name
MNEMOSYNE_EMBEDDING_API_KEY=your-key
MNEMOSYNE_EMBEDDING_DIM=768
```

### FastEmbed (local, no daemon)

```bash
pip install "mnemosyne[fastembed]"
```

```
MNEMOSYNE_EMBEDDING_PROVIDER=fastembed
MNEMOSYNE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
MNEMOSYNE_EMBEDDING_DIM=384
```

### Fake (testing)

Deterministic vectors from blake2b hashing. No network calls.

```
MNEMOSYNE_EMBEDDING_PROVIDER=fake
MNEMOSYNE_EMBEDDING_DIM=768
```

## LLM Providers

Used for extraction escalation (when rules can't parse the input), episode summarization, reflection generation, and contradiction adjudication.

```python
from mnemosyne.llm.base import LLMClient

settings = Settings.from_env()
llm = LLMClient.from_config(settings.llm_config)
```

### Ollama (local)

```
MNEMOSYNE_LLM_PROVIDER=openai_compatible
MNEMOSYNE_LLM_BASE_URL=http://localhost:11434/v1
MNEMOSYNE_LLM_MODEL=gemma3:4b
```

### OpenAI

```
MNEMOSYNE_LLM_PROVIDER=openai_compatible
MNEMOSYNE_LLM_BASE_URL=https://api.openai.com/v1
MNEMOSYNE_LLM_MODEL=gpt-4o-mini
MNEMOSYNE_LLM_API_KEY=sk-...
```

### Anthropic

```bash
pip install "mnemosyne[anthropic]"
```

```
MNEMOSYNE_LLM_PROVIDER=anthropic
MNEMOSYNE_LLM_MODEL=claude-3-5-sonnet-latest
MNEMOSYNE_LLM_API_KEY=sk-ant-...
```

### Google GenAI

```bash
pip install "mnemosyne[google]"
```

```
MNEMOSYNE_LLM_PROVIDER=google
MNEMOSYNE_LLM_MODEL=gemini-1.5-flash
MNEMOSYNE_LLM_API_KEY=...
```

### Any OpenAI-Compatible Endpoint

```
MNEMOSYNE_LLM_PROVIDER=openai_compatible
MNEMOSYNE_LLM_BASE_URL=https://your-endpoint.com/v1
MNEMOSYNE_LLM_MODEL=your-model
MNEMOSYNE_LLM_API_KEY=your-key
```

## Bootstrap

A single helper wires the whole subsystem at startup: it runs health checks (PostgreSQL reachable, pgvector loaded, schema version current, embedding dimension matches, LLM endpoint responsive) and starts the Prometheus exporter.

```python
from mnemosyne import Settings
from mnemosyne.integration import bootstrap_memory_subsystem
from mnemosyne.config.startup_checks import (
    check_postgres, check_pgvector, check_embedding_dim, check_llm_endpoint,
)

async def startup(provider, embedder, llm, settings):
    ctx = await bootstrap_memory_subsystem(
        checks=[
            check_postgres(settings),
            check_pgvector(provider),
            check_embedding_dim(embedder, expected=settings.embedding_dim),
            check_llm_endpoint(llm),
        ],
        start_exporter=True,    # serves /metrics on MNEMOSYNE_METRICS_PORT
        fail_fast=True,         # raises StartupCheckFailed on any failure
    )
    print(ctx.report.summary())
```

Set `MNEMOSYNE_STARTUP_WARN_ONLY=1` to keep the process alive on a failed check (logs warnings instead of raising).

## Agent Integration

Mnemosyne plugs into your agent server at three points.

### 1. Inject Memory Before LLM Call

Before each LLM call, build a context block from the user's memories and prepend it to the system prompt.

```python
from mnemosyne import assemble_context, build_system_prompt_memory_block

async def get_system_prompt(user_id, user_message, embedder, provider):
    query_vec = await embedder.embed(user_message)
    context = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=2000,
    )
    memory_block = build_system_prompt_memory_block(context)
    return f"You are a helpful assistant.\n\n{memory_block}"
```

### 2. Register the save_memory Tool

Give the agent the ability to explicitly save important information.

```python
from mnemosyne import save_memory_tool_spec, handle_save_memory

# OpenAI function-calling schema
tool = save_memory_tool_spec()
# → {"name": "save_memory", "parameters": {"properties": {"content": ..., "memory_type": ..., "importance": ...}}}

async def on_tool_call(user_id, tool_name, tool_args):
    if tool_name == "save_memory":
        return await handle_save_memory(provider, embedder, user_id, tool_args)
```

### 3. Session Close Hook

When a conversation ends, queue it for background processing (embedding, episode creation, dedup, decay, reflection).

```python
from mnemosyne.integration.hooks import on_session_close
from mnemosyne.pipeline.runner import process_session

async def end_session(session_id, user_id):
    await on_session_close(session_id=session_id, user_id=user_id)

    # Process inline, or let a background scheduler call this:
    await process_session(
        session_id=session_id,
        user_id=user_id,
        provider=provider,
        embedder=embedder,
        settings=settings,
    )
```

## Memory Management API

A typed service for user-facing memory operations: list, view, delete, export, and the GDPR physical-delete path.

```python
from mnemosyne.integration.memory_management import MemoryManagementService
from mnemosyne.integration.memory_management_models import (
    ListMemoriesRequest, GetMemoryRequest, DeleteMemoryRequest,
    DeleteUserRequest, ToggleExtractionRequest,
)

svc = MemoryManagementService(provider=provider, entity_store=entity_store)
```

### List, view, delete a single memory

```python
page = await svc.list_memories(ListMemoriesRequest(
    user_id=user_id, limit=50, offset=0, include_invalidated=False
))
# → ListMemoriesResponse(total=243, items=[Memory, ...])

mem = await svc.get_memory(GetMemoryRequest(memory_id=some_uuid))

await svc.delete_memory(DeleteMemoryRequest(
    memory_id=some_uuid, requestor="user@example.com"
))
# Soft delete — sets valid_until, keeps the row for audit
```

### Export everything for a user

```python
snapshot = await svc.export_user(user_id)
# → ExportUserResponse(memory_count=243, entity_count=58, memories=[...], entities=[...])
```

JSON-serializable. Hand to your "download my data" endpoint.

### GDPR physical delete

Hard removal with pre-delete audit. The audit row is written first, inside the same transaction as the cascade, so the deletion is provable.

```python
result = await svc.delete_user(DeleteUserRequest(
    user_id=user_id,
    requestor="dpo@example.com",
    dry_run=False,    # set True to compute counts without deleting
))
# → DeleteUserResponse(rows_deleted=312, dry_run=False)
```

What gets removed: the user's memories, memory_history rows, entities, entity_mentions, episodes. Reflections that referenced any deleted memory across users are soft-invalidated with reason `gdpr_source_deleted` (kept as rows so other users' reflections aren't silently corrupted).

### Per-user extraction toggle

```python
await svc.toggle_extraction(ToggleExtractionRequest(user_id=user_id, enabled=False))

if svc.is_extraction_enabled(user_id):
    await pipeline.process(user_id=user_id, text=message)
```

Useful for an "incognito mode" or paused-agent UX.

## Custom Rules

Add domain-specific extraction rules by dropping YAML files into your rules directory.

```yaml
# rules/custom/product_feedback.yaml
rules:
  - id: feedback_positive
    category: preference
    type: regex
    pattern: '\bi\s+(?:really\s+)?(?:like|love|enjoy)\s+the\s+(.+?)(?:[.\n,;!]|$)'
    template: "Likes: ${1}"
    importance: 0.7

  - id: feedback_allergy
    category: fact
    type: keyword_context
    keywords: ["allergic", "allergy", "intolerant"]
    importance: 0.95
```

Point Mnemosyne at the rules directory:

```bash
MNEMOSYNE_RULES_DIR=rules/custom
```

Three rule types:

| Type | What it does | Required fields |
|------|-------------|-----------------|
| `regex` | Regex match with capture-group templating (`${1}`, `${2}`) | `pattern`, `template` |
| `keyword` | Fires when any keyword is present in the text | `keywords` |
| `keyword_context` | Fires on keyword match, extracts the containing sentence | `keywords` |

Custom Python extractors are also supported — subclass `BaseExtractor` and drop the `.py` file in the rules directory. See `src/mnemosyne/rules/base_extractor.py` for the interface.

## Entity Extraction

Mnemosyne extracts entities (people, brands, products, locations) from conversations using spaCy + GLiNER + LLM fallback. Entities are resolved across mentions, deduplicated per user, and linked to the memories that mention them.

```bash
pip install "mnemosyne[ner]"
python -m spacy download en_core_web_sm
```

Entity-aware search finds memories by entity relationship, not just text similarity. A query for "Nike" returns every memory that mentions the Nike entity, even if the memory text is "those running shoes felt narrow."

```python
from mnemosyne.retrieval.entity_search import entity_aware_search

scored = await entity_aware_search(
    provider=provider,
    entity_store=entity_store,
    embedder=embedder,
    user_id=user_id,
    query="comfortable shoes",
    limit=10,
)
```

## Reflection & Contradiction

**Reflection** — when a user's accumulated memory importance crosses a threshold, Mnemosyne synthesizes higher-level insights and stores them as searchable, scorable memories. Reflections are wired into the pipeline runner; they fire automatically as part of background processing.

**Contradiction** — when new information conflicts with existing memories, Mnemosyne detects the conflict (via local NLI model or cosine similarity) and resolves it: supersede, keep both, merge, or keep old. Old memories are invalidated, never deleted.

```bash
# Optional: local NLI for fast contradiction detection
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "mnemosyne[nli]"
```

## Versioned Extraction

When extraction rules or LLM prompts change, existing memories carry the old extractor's output. The version registry tracks rule + prompt fingerprints; the re-extraction driver reprocesses memories below a target version.

```python
from mnemosyne.pipeline.extraction.version_registry import VersionRegistry
from mnemosyne.pipeline.extraction.reextraction_driver import ReextractionDriver

registry = VersionRegistry(provider)
await registry.register_current_version()    # detects rule/prompt changeset hash

driver = ReextractionDriver(provider=provider, pipeline=pipeline)
result = await driver.reextract_user(user_id, target_version="0.4.0")
# → ReextractionResult(processed=243, changed=58, kept=185)
```

Idempotent: rerunning with the same target version is a no-op.

## Monitoring

The Prometheus exporter is started by `bootstrap_memory_subsystem` (override with `start_exporter=False`). Scrape `http://localhost:9090/metrics`.

Exported metrics:

| Metric | Type | What |
|--------|------|------|
| `mnemosyne_extraction_total` | counter | extraction attempts since process start |
| `mnemosyne_extraction_failed_total` | counter | extraction failures since process start |
| `mnemosyne_retrieval_latency_p50_ms` | gauge | retrieval p50 |
| `mnemosyne_retrieval_latency_p95_ms` | gauge | retrieval p95 |
| `mnemosyne_retrieval_latency_p99_ms` | gauge | retrieval p99 |
| `mnemosyne_pipeline_lag_seconds` | gauge | seconds since last processed session |
| `mnemosyne_dedup_rate` | gauge | fraction of memories deduplicated |
| `mnemosyne_decay_archive_total` | counter | memories archived by decay |
| `mnemosyne_queue_depth` | gauge | pending sessions awaiting processing |

A `config/prometheus.example.yml` and `src/mnemosyne/monitoring/grafana_panel_example.json` are bundled to get you scraping and graphing in one paste.

## License

See `LICENSE`.
