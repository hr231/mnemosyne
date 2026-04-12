# Mnemosyne

Agent memory module for AI agents. Processes conversations into structured long-term memory with hybrid extraction (rule-based + LLM), multi-signal retrieval (relevance, recency, importance, frequency), bi-temporal data model, and token-budgeted context assembly. Designed as an embedded Python package with a clean `MemoryProvider` abstraction — swap backends without changing application code.

## Install

```bash
pip install "git+https://github.com/hr231/mnemosyne@v0.1.0"
```

## Quickstart

```python
import asyncio
from uuid import uuid4
from mnemosyne import (
    InMemoryProvider, FakeEmbeddingClient, ExtractionPipeline,
    Settings, assemble_context,
)

async def main():
    settings = Settings.from_env()
    provider = InMemoryProvider()
    embedder = FakeEmbeddingClient(dim=1536)
    pipeline = ExtractionPipeline.from_settings(settings, provider, embedder)

    user_id = uuid4()
    await pipeline.process(user_id=user_id, text="I like Nike running shoes, size 10")

    query_vec = await embedder.embed("shoes")
    block = await assemble_context(provider, user_id, query_vec, embedder, token_budget=500)
    print(block.text)

asyncio.run(main())
```

# With real LLM (currently need openAI compatible endpoints)

MNEMOSYNE_LLM_INTEGRATION=1 pytest tests/integration/test_real_llm.py