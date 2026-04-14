# Integrating Mnemosyne into Your Agent Server

Mnemosyne provides three integration points for adding persistent memory
to your AI agent. This guide shows how to wire them into your server.

## Prerequisites

```bash
pip install "mnemosyne[dev] @ git+https://github.com/hr231/mnemosyne@v0.2.0"
```

## Setup

```python
from mnemosyne import Settings, InMemoryProvider, FakeEmbeddingClient
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
# For production, use PostgresMemoryProvider and OllamaEmbeddingClient

settings = Settings.from_env()
provider = InMemoryProvider()  # or: await PostgresMemoryProvider.connect(dsn)
embedder = FakeEmbeddingClient(dim=768)  # or: OllamaEmbeddingClient(...)
pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)
```

## Touch Point 1: Inject Memory Before LLM Call

Before each LLM call, assemble the user's relevant memories into the
system prompt.

```python
from mnemosyne import assemble_context, build_system_prompt_memory_block

async def build_prompt(user_id, user_message, embedder, provider):
    query_vec = await embedder.embed(user_message)
    context = await assemble_context(
        provider=provider,
        user_id=user_id,
        query_embedding=query_vec,
        embedder=embedder,
        token_budget=2000,
    )
    memory_block = build_system_prompt_memory_block(context)

    system_prompt = f"""You are a helpful assistant.

{memory_block}"""
    return system_prompt
```

### FastAPI Example

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    system_prompt = await build_prompt(
        request.user_id, request.message, embedder, provider
    )
    # Pass system_prompt to your LLM call
    response = await llm.complete(system_prompt + "\n\nUser: " + request.message)
    return {"response": response}
```

### Flask Example

```python
# Flask requires running async calls through an event loop
import asyncio

@app.post("/chat")
def chat():
    data = request.get_json()
    system_prompt = asyncio.run(
        build_prompt(data["user_id"], data["message"], embedder, provider)
    )
    response = llm.complete(system_prompt + "\n\nUser: " + data["message"])
    return {"response": response}
```

`build_system_prompt_memory_block` returns an empty string when the user
has no stored memories, so the system prompt is unaffected for new users.

## Touch Point 2: Register the save_memory Tool

Register `save_memory` as a tool the LLM can call during conversation.

```python
from mnemosyne import save_memory_tool_spec, handle_save_memory

# Get the OpenAI function-calling schema
tool_schema = save_memory_tool_spec()

# When the LLM calls the tool:
async def handle_tool_call(user_id, tool_name, tool_args):
    if tool_name == "save_memory":
        result = await handle_save_memory(
            provider=provider,
            embedder=embedder,
            user_id=user_id,
            args=tool_args,
        )
        return result
```

`handle_save_memory` validates all inputs and returns a dict:

- `{"status": "saved", "memory_id": "<uuid>"}` on success
- `{"status": "error", "error": "<reason>"}` on validation failure

It is safe to call multiple times with the same content — content-hash
deduplication at the provider level makes repeated saves a no-op.

### Tool Schema

The `save_memory_tool_spec()` schema conforms to the OpenAI
function-calling format and can be passed directly to any OpenAI-
compatible API's `tools` parameter.

| Field | Type | Required | Description |
|---|---|---|---|
| content | string | yes | The text to remember |
| memory_type | enum | no | `fact`, `preference`, `entity`, `procedural` (default: `fact`) |
| importance | float | no | 0.0–1.0 importance weight (default: 0.5) |
| source_session_id | string | no | UUID of the originating session |

## Touch Point 3: Session Close Hook

When a session ends, queue it for background processing.

```python
from mnemosyne.integration.hooks import on_session_close

async def end_session(session_id, user_id):
    entry = await on_session_close(
        session_id=session_id,
        user_id=user_id,
    )
    # entry is a ProcessingLog(status="pending", pipeline_step="extraction")
    # Persist it via your database session, or pass it to the pipeline runner.
    # The call itself is non-blocking — it does no I/O.
```

`on_session_close` returns a `ProcessingLog` dataclass immediately.
Persisting the entry to the database and triggering the pipeline are left
to the caller so that session close latency is never affected by I/O.

### FastAPI Lifecycle Example

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # startup
    yield
    # shutdown — flush any pending sessions
    for sid, uid in active_sessions.items():
        await on_session_close(session_id=sid, user_id=uid)
```

## Background Pipeline

Run the pipeline periodically to process queued sessions:

```python
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline

# In a background task or cron job:
async def run_pipeline_for_session(session_id, user_id, transcript):
    results = await pipeline.process(user_id=user_id, text=transcript)
    print(f"Extracted {len(results)} memories for session {session_id}")
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| MNEMOSYNE_LLM_PROVIDER | ollama | LLM provider name |
| MNEMOSYNE_LLM_MODEL | gemma3:4b | Model name |
| MNEMOSYNE_LLM_BASE_URL | http://localhost:11434/v1 | API base URL |
| MNEMOSYNE_EMBEDDING_DIM | 768 | Embedding dimensions |
| MNEMOSYNE_PG_DSN | — | PostgreSQL connection string |
| MNEMOSYNE_TOKEN_BUDGET | 2000 | Token budget for memory context injection |
| MNEMOSYNE_ROUTER_UNSTRUCTURED_THRESHOLD | 0.7 | Fraction of unstructured text that triggers LLM extraction |
| MNEMOSYNE_EXTRACTION_VERSION | 0.1.0 | Extraction pipeline version |

## Entity Extraction

Mnemosyne automatically extracts entities (people, organizations, products, locations) from conversations using a three-tier NER pipeline:

1. **spaCy** — standard entities (PERSON, ORG, GPE)
2. **GLiNER** — domain-specific zero-shot NER (brands, product categories)
3. **LLM fallback** — for ambiguous or implicit entities

### Setup

```bash
# Install NER dependencies
pip install "mnemosyne[ner]"
python -m spacy download en_core_web_sm
```

### Entity-Aware Search

When entities are available, search finds memories linked to entities even when the query text doesn't match:

```python
from mnemosyne.retrieval.entity_search import entity_aware_search
from mnemosyne.db.repositories.entity import PostgresEntityStore  # or InMemoryEntityStore

entity_store = PostgresEntityStore(provider._pool)  # shares the provider's pool

results = await entity_aware_search(
    provider=provider,
    entity_store=entity_store,
    query_text="Nike shoes",
    query_embedding=await embedder.embed("Nike shoes"),
    user_id=user_id,
    embedder=embedder,
    limit=10,
)
```

### Context Assembly with Entities

```python
context = await assemble_context(
    provider=provider,
    user_id=user_id,
    query_embedding=query_vec,
    embedder=embedder,
    token_budget=2000,
    entity_store=entity_store,
    query_text="Nike shoes",
)
```

## Reflection Generation

Mnemosyne automatically generates high-level insights when enough memories accumulate. Reflections trigger when the scaled importance sum exceeds 150 (based on the Stanford Generative Agents paper).

```python
from mnemosyne.pipeline.reflection import should_generate_reflection, generate_reflections

if await should_generate_reflection(provider, user_id):
    reflections = await generate_reflections(
        provider=provider,
        user_id=user_id,
        llm_client=llm,
        embedder=embedder,
    )
    # reflections are stored as memory_type="reflection"
    # and appear in future search results
```

Reflections are capped at depth 2 (reflection → meta-reflection, no further).

## Contradiction Resolution

When new information contradicts existing memories, Mnemosyne detects and resolves the conflict:

```python
from mnemosyne.pipeline.contradiction import run_contradiction_check

resolved = await run_contradiction_check(
    provider=provider,
    user_id=user_id,
    llm_client=llm,
    embedder=embedder,
    use_nli=True,  # requires torch + transformers
)
```

Four possible actions:
- **SUPERSEDE** — new memory replaces old (old is invalidated)
- **KEEP_BOTH** — both are valid in different contexts
- **MERGE** — combined into a single updated memory
- **KEEP_OLD** — old memory is correct, new is rejected

### NLI Setup (optional, for fast local detection)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "mnemosyne[nli]"
```

Without NLI, contradiction detection falls back to cosine similarity + LLM adjudication.

## Embedding Providers

### FastEmbed (local, zero API)

```bash
pip install "mnemosyne[fastembed]"
```

```
MNEMOSYNE_EMBEDDING_PROVIDER=fastembed
MNEMOSYNE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```
