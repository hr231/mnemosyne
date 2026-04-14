# Mnemosyne

Persistent memory for AI agents. Drop it into your agent server and your agent remembers users across sessions — preferences, facts, past decisions, context. Mnemosyne extracts memories from conversations (rule-based + LLM), automatically identifies entities (people, brands, products, locations), stores everything with bi-temporal versioning, and retrieves it via multi-signal scored search (relevance, recency, importance, frequency). It generates high-level insights through reflection when memories accumulate, and resolves contradictions when new information conflicts with existing memories. It runs as an embedded Python module, not a separate service. Swap between in-memory (dev) and PostgreSQL (production) without changing your code.

## Install

```bash
pip install "git+https://github.com/hr231/mnemosyne@v0.2.0"
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
    await pipeline.process(user_id=user_id, text="I like Nike shoes, size 10, budget under $150")

    query_vec = await embedder.embed("shoes")
    context = await assemble_context(provider, user_id, query_vec, embedder, token_budget=500)
    print(context.text)

asyncio.run(main())
```

## Configuration

All settings are read from environment variables via `Settings.from_env()`. Every field has a sensible default.

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_LLM_PROVIDER` | `ollama` | `ollama`, `openai_compatible`, or `fake` |
| `MNEMOSYNE_LLM_MODEL` | `gemma3:4b` | Model name passed to the API |
| `MNEMOSYNE_LLM_BASE_URL` | `http://localhost:11434/v1` | API base URL |
| `MNEMOSYNE_LLM_API_KEY` | — | Bearer token (optional for local Ollama) |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_EMBEDDING_PROVIDER` | `ollama` | `ollama`, `openai_compatible`, or `fake` |
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

Both providers implement the same `MemoryProvider` interface. Your application code doesn't change when you switch.

## Embedding Providers

Use `EmbeddingClient.from_config()` with a config dict or `settings.embedding_config`:

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

Works with Databricks, vLLM, Azure OpenAI, Together, Anyscale — anything that speaks the `/v1/embeddings` format.

```
MNEMOSYNE_EMBEDDING_PROVIDER=openai_compatible
MNEMOSYNE_EMBEDDING_BASE_URL=https://your-endpoint.com
MNEMOSYNE_EMBEDDING_MODEL=your-model-name
MNEMOSYNE_EMBEDDING_API_KEY=your-key
MNEMOSYNE_EMBEDDING_DIM=768
```

### Fake (testing)

Deterministic vectors from blake2b hashing. No network calls.

```
MNEMOSYNE_EMBEDDING_PROVIDER=fake
MNEMOSYNE_EMBEDDING_DIM=768
```

## LLM Providers

Used for extraction escalation (when rules can't parse the input) and episode summarization.

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

### Any OpenAI-Compatible Endpoint

```
MNEMOSYNE_LLM_PROVIDER=openai_compatible
MNEMOSYNE_LLM_BASE_URL=https://your-endpoint.com/v1
MNEMOSYNE_LLM_MODEL=your-model
MNEMOSYNE_LLM_API_KEY=your-key
```

## Agent Integration

Mnemosyne connects to your agent server at three points.

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

Give your agent the ability to explicitly save important information.

```python
from mnemosyne import save_memory_tool_spec, handle_save_memory

# Get the OpenAI function-calling schema
tool = save_memory_tool_spec()
# → {"name": "save_memory", "parameters": {"properties": {"content": ..., "memory_type": ..., "importance": ...}}}

# When the agent calls the tool:
async def on_tool_call(user_id, tool_name, tool_args):
    if tool_name == "save_memory":
        return await handle_save_memory(provider, embedder, user_id, tool_args)
```

### 3. Session Close Hook

When a conversation ends, queue it for background processing (embedding, episode creation, dedup, decay).

```python
from mnemosyne.integration.hooks import on_session_close
from mnemosyne.pipeline.runner import process_session

async def end_session(session_id, user_id):
    await on_session_close(session_id=session_id, user_id=user_id)

    # Process immediately, or let a background scheduler handle it:
    await process_session(
        session_id=session_id,
        user_id=user_id,
        provider=provider,
        embedder=embedder,
        settings=settings,
    )
```

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

Point Mnemosyne at your rules directory:

```bash
MNEMOSYNE_RULES_DIR=rules/custom
```

Three rule types are supported:

| Type | What it does | Required fields |
|------|-------------|-----------------|
| `regex` | Regex match with capture-group templating (`${1}`, `${2}`) | `pattern`, `template` |
| `keyword` | Fires when any keyword is present in the text | `keywords` |
| `keyword_context` | Fires on keyword match, extracts the containing sentence | `keywords` |

You can also write Python extractors by subclassing `BaseExtractor` and placing the `.py` file in the rules directory. See `src/mnemosyne/rules/base_extractor.py` for the interface.

## Entity Extraction

Mnemosyne extracts entities (people, brands, products, locations) from conversations using spaCy + GLiNER + LLM fallback. Entities are resolved, deduplicated, and linked to memories. Entity-aware search finds memories by entity relationship, not just vector similarity.

```bash
pip install "mnemosyne[ner]"
python -m spacy download en_core_web_sm
```

## Reflection & Contradiction

**Reflection:** When enough memories accumulate (importance sum >= 150), Mnemosyne generates high-level insights stored as first-class searchable memories.

**Contradiction:** When new information conflicts with existing memories, Mnemosyne detects the conflict (via NLI model or cosine similarity) and resolves it: supersede, keep both, merge, or keep old. Old memories are invalidated, never deleted.

```bash
# Optional: local NLI for fast contradiction detection
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "mnemosyne[nli]"
```