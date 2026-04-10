# Agent Memory System — Design Document

**Status:** Draft
**Scope:** General-purpose agent memory platform

---

## 1. Problem Statement

The agent currently operates with session-scoped memory only — it forgets everything between conversations. This design introduces a persistent, intelligent memory system that processes session data from an existing lakebase (PostgreSQL) into structured long-term memory, enabling cross-session continuity.

### Goals

- **General-purpose memory platform** — not tied to any specific domain; personalization and e-commerce-specific features are future layers that plug in on top
- **Embedded module** — lives inside the agent server as a library, not a separate service. Designed with a clean interface so it can be extracted into a standalone service later
- **Model-agnostic** — works with any LLM and embedding provider; swappable via configuration
- **Cost-efficient** — hybrid extraction pipeline (rule-based first, LLM only when needed), informed by A-MAC research showing 31% latency reduction with this approach ([arXiv:2603.04549](https://arxiv.org/abs/2603.04549))
- **Multi-scope** — scoped by `user_id`, `agent_id`, `session_id`, `org_id` for future multi-agent/multi-tenant expansion, following the 4-scope model validated by Mem0 in production ([State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026))

### Non-Goals (for now)

- Personalization / taste profiling
- Behavioral pattern detection
- Product-preference scoring
- Recommendation integration
- Multi-agent shared memory
- Standalone memory service deployment
- Graph-based memory (entity relationships via knowledge graph)

---

## 2. Design Decisions & Rationale

### 2.1 Embedded Module vs Standalone Service

**Decision:** Start as an embedded module inside the agent server.

**Why:** At early stage (internal use and qa_testing), a separate service adds operational complexity with no benefit. The module exposes an abstract `MemoryProvider` interface — when scale demands extraction, a `ServiceMemoryProvider` implementation makes HTTP calls to a standalone service. The agent code doesn't change.

**Precedent:** Letta's architecture started as an embedded library before evolving into a server-based deployment ([Letta Docs](https://docs.letta.com/concepts/memgpt/)). LangMem similarly ships as a library that runs inside LangGraph ([LangMem SDK](https://blog.langchain.com/langmem-sdk-launch/)).

### 2.2 Same Database, Separate Schema

**Decision:** The `memory` schema lives in the same PostgreSQL instance as the existing `lakebase` schema.

**Why:** Processing is cross-schema queries — no network hop, no sync issues, no eventual consistency to worry about. At early stage, one database is simpler to operate. When scaling demands separation, the `MemoryProvider` abstraction allows pointing to a different database without changing application code.

### 2.3 Hybrid Extraction (Rule-Based + LLM)

**Decision:** Rule-based extraction handles the majority of sessions. LLM is invoked only when a routing function determines the session is too complex for rules alone.

**Why:** Every major memory framework (Mem0, LangMem, CrewAI, Letta) uses LLM calls for all extraction ([Mem0 paper](https://arxiv.org/abs/2504.19413), [CrewAI Cognitive Memory](https://blog.crewai.com/how-we-built-cognitive-memory-for-agentic-systems/)). This works but is expensive at scale. Alternatives are:

- **MemMachine** (MemVerge, 2025) achieves approximately **80% fewer input tokens** than Mem0 with comparable benchmark scores (0.9169 on LoCoMo, 93.0% on LongMemEvalS with gpt-4.1-mini) by storing raw episodes and using NLTK's Punkt tokenizer for sentence-level segmentation, reserving LLM for overflow summarization only ([arXiv:2604.04853](https://arxiv.org/html/2604.04853))
- **A-MAC** (Workday AI, ICLR 2026 Workshop MemAgent) formalizes the hybrid approach — five interpretable admission factors (future utility, factual confidence, semantic novelty, temporal recency, content type prior) — achieving F1 of 0.583 on LoCoMo with **31% latency reduction** versus LLM-native systems ([arXiv:2603.04549](https://arxiv.org/abs/2603.04549))
- **SeCom** (Tsinghua + Microsoft, ICLR 2025) tested three segmenters for conversation structuring: GPT-4 (default), Mistral-7B-Instruct, and a RoBERTa model fine-tuned on SuperDialSeg as a lightweight alternative. The full SeCom pipeline with a RoBERTa segmenter retains competitive end-to-end QA performance — *not* that RoBERTa matches GPT-4 at segmentation quality itself ([arXiv:2502.05589](https://arxiv.org/html/2502.05589v3))

#### Alternative to hybrid routing: "always extract with a cheap model"

A striking finding from production systems (Mem0, Claude Code, OpenAI Codex): most **skip explicit routing entirely**. They run LLM extraction on every session using a cheap model (gpt-4o-mini, ~$0.00014/session), letting classification happen inside the extraction prompt. At this cost, routing engineering may not justify its complexity.

**We keep the hybrid approach** because:
1. The rule-based layer has independent value beyond cost — it's deterministic, debuggable, and gives product/domain teams direct control over what the system captures
2. Latency matters for our hot path (explicit agent tool calls) — rule-based extraction is 1-5ms vs 200-500ms for LLM calls
3. The routing decision itself is nearly free once built

But the "always-extract" path is a **legitimate fallback** if rule maintenance burden becomes unsustainable.


### 2.3.1 Rules as a Plugin System

**Decision:** The rule-based extraction layer is a **plugin architecture**. Core memory code defines the rule interface and loader; it knows nothing about specific rules. Rules live in YAML files and/or Python plugin modules loaded from a configurable path at startup.

**Why:** With a projected scale of 50-90 rules at maturity (core + e-commerce-specific) — not thousands — a full rule engine (Drools, DSLs, database-backed rule stores) would be overkill. But hardcoding rules in Python has real costs:

- Every rule change is a PR, review, deploy cycle
- Non-engineers (product, domain experts) can't contribute
- Generic rules and domain-specific rules mix in the same file
- No rule governance (id, owner, version, enabled flag)
- Testing rules requires testing extraction pipeline internals

A plugin architecture solves all of this without adopting a heavyweight rule engine:

| Component | Responsibility |
|---|---|
| **`BaseExtractor`** (interface) | Core defines the contract: each rule has an `id`, `category`, `importance`, `extract()` method |
| **YAML rule schema** | 90% of rules expressed declaratively: regex, keyword, keyword_context types |
| **Python plugin interface** | 10% of rules needing complex logic (NER, dependency parsing) as classes subclassing `BaseExtractor` |
| **`RuleLoader`** | Scans configured directories for `.yaml` files and `.py` plugin modules |
| **`RuleRegistry`** | Holds loaded extractors, runs them on input text, handles errors per-rule |

**What this enables:**

1. **Contributors write rules without touching core code** — product/domain experts add YAML files
2. **Rule governance** — every rule has `id`, `version`, `owner`, `enabled`, `test_cases` in metadata
3. **Domain separation** — `rules/core/` and `rules/ecommerce/` live in different directories (or different repos)
4. **Testing in isolation** — rules come with their own test cases inline with the YAML definition
5. **Rule-level observability** — which rules fired, how often, from what content (future analytics)
6. **Extensibility without forks** — teams ship new rule packs without modifying the core module
7. **Future: runtime rule updates** — YAML files can be hot-reloaded without restarting; database-backed rule store is a future upgrade path

**What this does NOT require:**

- A custom DSL or rule language (YAML + regex is enough)
- A rule dependency/priority graph (rules run independently)
- A rule compiler or optimizer (~50-90 rules runs in microseconds)
- A rule versioning system (YAML files in git provide this)

**Deployment flexibility:** The rule directory is a config value. Options:
- **Same repo, `rules/` directory** — simplest, early stage
- **Separate repo, imported as Python package** — when a rules team becomes a separate concern
- **External config path** — multi-tenant or per-environment deployments

All three work with zero changes to the core module.

**Precedent:** spaCy's `Matcher`/`EntityRuler` uses a similar pattern — rules as data plus Python plugins for complex logic. Haptik's `chatbot_ner` (production chatbot NER library) is explicitly pattern-based and pluggable. LangChain's tool registry follows the same philosophy for tools.


### 2.4 LLM Routing: How to Decide When a Session Needs LLM

**Decision:** A **simplified two-signal routing function** (v1) determines whether to escalate to LLM. Both signals are computable without an LLM call.

**Why two signals instead of five:** A-MAC's published ablation study tested removing each of five admission signals individually. The results are stark:

| Signal removed | F1 drop | % of total signal value |
|---|---|---|
| **Content Type Prior** | **−0.107** | **~49%** |
| Semantic Novelty | −0.028 | ~13% |
| Future Utility | −0.028 | ~13% |
| Factual Confidence | −0.028 | ~13% |
| Temporal Recency | −0.028 | ~13% |

Content Type Prior alone captures nearly half the routing value. It's a classifier asking "does this content look like something worth remembering?" — user preferences and identity statements score high, greetings and transient emotions score low. That's exactly what the combination of **extraction yield** + **unstructured ratio** measures in our rule-based setup:

- If fast-path patterns **did** match a lot (high yield, low unstructured) → the content type is extractable → rules handled it → skip LLM
- If fast-path patterns **did not** match (low yield, high unstructured) → the content type contains implicit/nuanced information → escalate to LLM

**The v1 routing function:**

```
ROUTE_TO_LLM = (extraction_yield < yield_threshold) OR (unstructured_ratio > unstructured_threshold)
```

| Signal | What it Measures | Threshold (default) |
|---|---|---|
| **Extraction yield** | `extracted_count / user_message_count` | < 0.3 triggers LLM |
| **Unstructured ratio** | `1 - (chars_matched_by_rules / total_user_chars)` | > 0.7 triggers LLM |

**Signals explicitly dropped from v1 routing (with reasoning):**

| Signal | Why dropped |
|---|---|
| **Session complexity** | Message count and conditional/comparison language don't predict memory value per A-MAC's ablation. Complex sessions often contain nothing worth remembering; simple sessions often contain critical facts ("I'm allergic to peanuts"). |
| **Contradictions** | Contradictions matter for memory **update/consolidation** (post-extraction), not for extraction routing. Both Mem0 and A-MAC handle contradictions during the consolidation stage. We'll do the same. |
| **Semantic novelty** | Deferred to v1.5 — adds real value but requires embedding the session text. With local MiniLM (all-MiniLM-L6-v2, 80 MB on disk, ~15ms on CPU, zero API cost), this becomes cheap enough to include once we have a real extraction baseline to beat. |

**v1.5 planned enhancement:**

```
ROUTE_TO_LLM = yield_low OR unstructured_high OR novelty_high
```

where `novelty_high = 1 - max_cosine_sim(session_embedding, user_existing_embeddings) > 0.7`, computed with a local MiniLM model (not an API embedding).

**Practical examples (v1):**

| Session | Rules Extract | LLM? | Why |
|---|---|---|---|
| "I need running shoes, size 10, under $150" | size, budget, category | No | High yield, structured content |
| "hi" / "thanks" / "ok bye" | nothing | No | Low unstructured content (short messages), nothing to route to LLM anyway |
| "Last time I got those Nikes but they felt too narrow, maybe something wider..." | brand:Nike | Yes | Low yield + high unstructured ratio — implicit fit preference the rules missed |
| "I was thinking something that feels premium but isn't flashy, like what I had before but updated" | nothing | Yes | Zero yield + fully unstructured |

**Path to RL-based routing (v2):** MemFactory (arXiv:2603.29493, MemTensor 2026) demonstrates that routing, extraction, and updating can be jointly optimized with Group Relative Policy Optimization (GRPO) using labeled production data. Our v1 hand-tuned thresholds become the collection mechanism — log every routing decision and its downstream memory value, then train a policy on the accumulated data.

### 2.5 Multi-Signal Retrieval

**Decision:** Retrieval scores memories on four signals, not just vector similarity.

**Why:** The Stanford Generative Agents paper demonstrated that combining recency + importance + relevance significantly outperforms single-signal retrieval for agent memory ([arXiv:2304.03442](https://arxiv.org/abs/2304.03442)). Mem0's production results confirm this with 26% accuracy improvement over baseline ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)).

**The four retrieval signals:**

| Signal | Weight (default) | Purpose |
|---|---|---|
| **Relevance** | 0.5 | Vector cosine similarity + full-text search boost (hybrid search via pgvector + tsvector) |
| **Recency** | 0.2 | Exponential decay from last access time — recent memories score higher |
| **Importance** | 0.2 | Assigned at extraction time (0.0-1.0), decays over time if not accessed |
| **Access frequency** | 0.1 | Log-scaled access count — frequently retrieved memories are more valuable |

Weights are configurable per deployment. The hybrid vector + full-text approach follows Reciprocal Rank Fusion patterns documented in pgvector best practices and production implementations at Zep ([arXiv:2501.13956](https://arxiv.org/html/2501.13956v1)).

### 2.6 Context Assembly with Token Budgeting

**Decision:** A dedicated context assembly function builds the working memory block for each LLM call, prioritized by section type and constrained by a token budget.

**Why:** Letta's MemGPT research established that structured working memory (always-present profile + retrieved context) outperforms naive "load last N messages" approaches ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)). The OpenAI Agents SDK cookbook validates the layered pattern: profile (structured, authoritative) > global memory (long-term notes) > session memory (current context) ([OpenAI Context Personalization](https://developers.openai.com/cookbook/examples/agents_sdk/context_personalization)).

**Priority ordering (highest first):**

1. **User profile** — reflections and high-importance facts (always included)
2. **Query-relevant memories** — multi-signal scored search results
3. **Recent episodes** — last 3 session summaries for conversational continuity
4. **Relevant entities** — structured entity knowledge matching the current query

Application layer reads rows in priority order, estimates tokens per row, and stops when the budget is reached. Lower-priority sections get truncated first.

### 2.7 Two Write Paths: Hot + Cold

**Decision:** The agent can write memories explicitly during conversation (hot path) AND background pipeline processes sessions asynchronously (cold path).

**Why:** Most of the papers shows that pure agent-managed memory (Letta approach) misses things — the LLM doesn't always recognize what's worth saving. Pure auto-extraction (Mem0 approach) lacks precision on critical facts. The hybrid approach catches both: the agent explicitly saves "I'm allergic to latex" immediately (hot path), while the background pipeline catches implicit behavioral patterns (cold path).

**Precedent:** LangMem implements exactly this two-track pattern — in-conversation memory via tool calls + post-conversation background extraction ([LangMem SDK](https://blog.langchain.com/langmem-sdk-launch/)).

---

## 3. Architecture

### System Context

The memory module is embedded inside the agent server. It exposes three interfaces:

- **MemoryStore** — write path: `add()`, `update()`, `delete()`
- **MemoryQuery** — read path: `search()`, `hybrid()`, `by_scope()`
- **MemoryCtx** — context assembly: `assemble()`, `budget()`

All three delegate to a `MemoryProvider` abstract interface with swappable implementations:
- `PostgresMemoryProvider` — production (PostgreSQL + pgvector(or any other alternative))
- `InMemoryProvider` — dev/test
- `ServiceMemoryProvider` — future standalone service client

### Storage Layout

Same PostgreSQL instance, two schemas:
- **`lakebase` Product schema** (existing) — sessions, user inputs, agent summaries, metadata
- **`memory` schema** (new) — processed long-term memories, episodes, entities, processing log

Processing is cross-schema queries. No network hop, no sync issues.

### Data Flow

```
                    ┌─────────────┐
                    │  Lakebase   │
                    │  (sessions, │
                    │  inputs,    │
                    │  summaries) │
                    └──────┬──────┘
                           │ reads (cold path)
                           ▼
┌──────────────────────────────────────────────────┐
│              Processing Pipeline                  │
│                                                  │
│  Extraction → Embedding → Episodes →             │
│  Consolidation → Decay                           │
└──────────────────────┬───────────────────────────┘
                       │ writes
                       ▼
                ┌─────────────┐
                │   memory    │
                │   schema    │ ◀── writes (hot path, from agent tool)
                │             │
                │  memories   │ ──▶ reads (query engine, context assembly)
                │  episodes   │
                │  entities   │         ┌──────────────┐
                │  proc_log   │ ──────▶ │ Agent Server │
                └─────────────┘         └──────────────┘
```

---

## 4. Data Model

Four tables in the `memory` schema:

### 4.1 `memory.memories` — Core Memory Table

Stores all extracted knowledge: facts, preferences, entities, learned rules, reflections.

**Key columns:**
- **Scoping:** `user_id`, `agent_id`, `org_id` — 4-scope model
- **Content:** `content` (text), `content_hash` (SHA-256 for exact-dup detection), `memory_type` (enum: fact, preference, entity, procedural, reflection)
- **Vector search:** `embedding` (pgvector **`halfvec(1536)`** — half-precision floats, 50% storage reduction from day one with minimal recall impact), `content_tsv` (generated tsvector for full-text)
- **Scoring:** `importance` (0-1, set at extraction, decays over time), `access_count`, `last_accessed`, `decay_rate`
- **Bi-temporal (Zep/Graphiti pattern):** `valid_from`, `valid_until` — when the fact is true. Invalidated facts retain `valid_until = now()` but are not deleted
- **Extraction versioning:** `extraction_version` (semver), `extraction_model`, `prompt_hash` — enables re-extraction when rules/prompts change (query `WHERE extraction_version < 'current'`)
- **Provenance:** `source_session_id`, `source_memory_ids` (for reflections pointing to evidence), `rule_id` (which rule extracted this, for rule-level observability)
- **Flexible:** `metadata` (JSONB for domain-specific extensions)

### 4.2 `memory.episodes` — Session Summaries

Compressed records of past conversations. Bridge between raw lakebase data and long-term memory.

**Key columns:**
- 1:1 with lakebase sessions via `session_id` (unique)
- `summary` + `summary_embedding` for temporal and semantic retrieval
- `key_topics` (text array for quick filtering)
- `memory_ids` (links to memories extracted from this episode)
- `outcome` (what was the result of this session)

### 4.3 `memory.entities` — Structured Entity Knowledge

Deduplicated knowledge about specific things (users, brands, categories, products). Future personalization hooks into this table.

**Key columns:**
- `entity_name` + `entity_type` — unique per user+agent
- `facts` (JSONB) — structured key-value facts about the entity
- `confidence` — decreases on contradictory evidence
- `source_memory_ids` — provenance chain

### 4.4 `memory.processing_log` — Pipeline Audit Trail

Tracks what has been processed from lakebase. Ensures idempotency and enables debugging.

**Key columns:**
- `session_id` + `pipeline_step` — tracks which sessions have been through which pipeline stages
- `status` (pending/completed/failed)
- `memories_created` — links to output for auditability

### 4.5 `memory.memory_history` — Immutable Audit Log

Append-only log of every memory mutation. Enables "show me what this memory used to say" queries, debugging, and rollback. Inspired by Letta's MemFS (git-backed memory versioning) and Mem0's old_memory → new_memory audit trail.

**Key columns:**
- `id` (UUID primary key)
- `memory_id` — the memory being mutated (FK to `memory.memories.id`)
- `operation` — enum: `create`, `update`, `delete`, `merge`, `invalidate`
- `old_content`, `new_content` — text snapshots
- `old_importance`, `new_importance`
- `actor` — what caused the change: `agent_tool`, `pipeline_extraction`, `pipeline_consolidation`, `pipeline_decay`, `manual`
- `actor_details` (JSONB) — rule_id, extraction_version, session_id, etc.
- `occurred_at` (timestamp)

Never updated, never deleted. Retention is pruned by a separate background job (e.g., keep 90 days of history per memory).

### 4.6 `memory.extraction_versions` — Extraction Config Registry

Tracks extraction logic versions so we can re-extract when rules or prompts change.

**Key columns:**
- `version` (semver, primary key)
- `rule_pack_commit` — git SHA of the rules directory at this version
- `llm_model` — which LLM the extractor used
- `prompt_hash` — SHA-256 of the LLM extraction prompt
- `config_snapshot` (JSONB) — thresholds, weights at this version
- `deployed_at`

Memories reference this via `memories.extraction_version`. Re-extraction batch job queries `WHERE extraction_version < 'current'` and re-runs extraction on the source session.

### Indexing Strategy

- **HNSW indexes** on all embedding columns (pgvector on halfvec, m=16, ef_construction=64) — approximate nearest neighbor for vector search
- **Composite indexes** on scoping columns (`user_id` + `memory_type`, `user_id` + `importance DESC`)
- **Bi-temporal index**: `(user_id, valid_from DESC, valid_until)` and a partial index `WHERE valid_until IS NULL` for "currently valid" queries
- **Exact-dup index**: `UNIQUE (user_id, content_hash) WHERE valid_until IS NULL` for hot-path dedup
- **Fuzzy-dup index**: `GIN (content gin_trgm_ops)` for pg_trgm-based fuzzy matching (requires `CREATE EXTENSION pg_trgm`)
- **GIN indexes** on tsvector column for full-text search and on JSONB metadata
- **Partial indexes** for pipeline processing (`WHERE status = 'pending'`, `WHERE embedding IS NULL`)
- **Memory history index**: `(memory_id, occurred_at DESC)` for "show me the history of this memory"

---

## 5. Processing Pipeline

Five stages, all idempotent. Runs as scheduled background jobs within the agent server. Reads from lakebase, writes to `memory` schema.

```
Lakebase → [1. Extraction] → [2. Embedding] → [3. Episodes]
                                                     ↓
           [5. Decay & Cleanup] ← [4. Consolidation]
```

### Stage 1: Hybrid Extraction

**Pass 1 — Rule-Based (handles ~60-70% of sessions):**
- Regex patterns for structured data: budgets, sizes, quantities, explicit preferences ("I prefer X", "I don't like Y")
- Keyword triggers for high-importance signals: "remember", "always", "never", "favorite"
- NER (spaCy or equivalent) for entities: brands, products, categories, people
- Pattern matching for comparison language ("X vs Y") and conditional preferences

**Pass 2 — LLM Routing Decision (no LLM needed for the decision itself):**
Five signals scored to determine if LLM extraction is needed (see Section 2.4).

**Pass 3 — LLM Extraction (when triggered, ~20-25% of sessions):**
Batched, async, sent to configurable LLM. Extracts facts, preferences, entities, and procedural patterns as structured JSON.

### Stage 2: Embedding (Batch)

Batch-embeds all memories with NULL embeddings. Decoupled from extraction so embedding model can be swapped independently. Changing models triggers a one-time full re-embed batch job.

### Stage 3: Episode Creation

Compresses each session into an episode summary with **memory-specific extraction**, not reused analytics summaries.

**Why not just reuse the lakebase summary:** Mem0 explicitly separates "memory formation" from "summarization" — summarization compresses conversations into shorter text, losing important details, while memory formation selectively identifies specific facts and patterns worth remembering. MemMachine's strong LoCoMo results (0.9169) come from storing raw episodes and extracting memory-specific signals, minimizing reliance on pre-made summaries.

**Our lakebase summaries are agent output summaries** (what the agent did) rather than analytics, so they're a valid **starting point** but not sufficient as the episode summary. The pipeline:

1. **Start with the lakebase summary** if present (cheap baseline — no LLM call)
2. **Enrich with extracted memory context** — append key topics, extracted memory IDs, session outcome
3. **Optionally invoke a memory-specific summary prompt** when the lakebase summary is missing or too short (<50 words)

The memory-specific prompt targets: user preferences stated, facts learned, entity relationships, decisions made, outcomes — *not* "what was discussed" or "what the agent did."

### Stage 4: Consolidation (daily/weekly)

Three sub-operations:

**Deduplication — three-tier approach:**

| Tier | Method | Threshold | Latency |
|---|---|---|---|
| **Exact** | SHA-256 hash of normalized content | unique index lookup | O(1), ~0.1 ms |
| **Fuzzy** | PostgreSQL `pg_trgm` extension | `similarity(content, new) > 0.8` | ~1-5 ms with GIN index |
| **Semantic** | pgvector cosine similarity | **≥ 0.90** (cosine distance ≤ 0.10) | ~5-20 ms with HNSW |

Threshold framing: **≥ 0.95 cosine similarity = near-certain duplicate** (auto-merge), **0.90-0.95 = safe to merge/skip** with the higher-importance memory preserved, **< 0.85 = distinct enough to keep both**.

Hybrid schedule:
- **Real-time** (insert-time): exact hash lookup + high-threshold (≥ 0.95) semantic check — adds 5-20 ms per write to catch obvious duplicates
- **Batch** (nightly): fuzzy and lower-threshold semantic dedup across all new memories from the day

**Reflection generation (LLM, based on importance sum):**

Stanford Generative Agents' actual mechanism: reflections trigger when the **sum of importance scores for recent events exceeds a threshold of 150**. Every observation is rated on a 1-10 importance scale at creation time; mundane events score 1-2, significant events score 8-10. The system accumulates importance scores since the last reflection. In the Stanford simulation, agents reflected roughly **2-3 times per simulated day**.

This is **importance-sum-based, not count-based**. Translating to our system:
- Every memory at creation gets `importance` in [0.0, 1.0] (we're already doing this)
- Scale to [1, 10] for the trigger check: `scaled_importance = round(importance * 10)`
- When `SUM(scaled_importance) since last reflection ≥ 150` for a user, trigger reflection
- Reflection queries the ~100 most recent memories, generates candidate reflection questions, retrieves relevant evidence, synthesizes higher-level insights stored as `memory_type = 'reflection'`
- Reflections can recursively generate further reflections

**Contradiction resolution (LLM, via bi-temporal model):**

When consolidation detects contradictory memories (high semantic similarity + different content), use LLM to determine which is current. **Never silently overwrite** — the old memory is marked as invalidated (`valid_until` set to now) but retained for audit. This is Zep/Graphiti's bi-temporal pattern: every fact has `valid_from` and `valid_until` timestamps; invalidated facts remain queryable but default retrieval excludes them.

### Stage 5: Decay & Cleanup (daily)

- **Importance decay** — exponential decay based on time since last access, scaled by per-memory `decay_rate`
- **Archival** — memories below importance threshold (0.05) untouched for 90+ days are soft-archived via metadata flag
- Inspired by Ebbinghaus forgetting curve, adapted for agent memory as described in the Generative Agents paper

### Pipeline Configuration

All thresholds, schedules, batch sizes, model references, and weights are externalized as YAML configuration. Nothing is hardcoded.

---

## 6. Retrieval & Query Engine

### Multi-Signal Scored Search

Four-signal scoring function (see Section 2.5) with configurable weights. Uses a two-stage approach:
1. **Pre-filter** — HNSW index fetches top `5 * limit` vector-similar candidates (fast)
2. **Re-rank** — applies full-text boost, recency, importance, and frequency scoring

Side effect: returned memories get their access stats bumped, feeding back into future decay and frequency scoring.

### Context Assembly

Builds the working memory block injected into the LLM system prompt before each call. Four sections in priority order (see Section 2.6). Token budgeting happens in the application layer.

### Hybrid Search

Combines pgvector cosine similarity with PostgreSQL full-text search (tsvector/tsquery). Pure vector similarity misses exact keyword matches (names, IDs, technical terms). Pure full-text misses semantic similarity. The combination with text-rank boosting yields better retrieval, consistent with production experience at Zep and documented pgvector patterns.

---

## 7. Agent Integration

Three touch points in the existing agent server. Minimal changes to existing code.

### Touch Point 1: Before LLM Call — Inject Memory

Before each LLM call, the agent server embeds the user query, calls `assemble_context`, formats the result into a memory block, and injects it into the system prompt under a "What you remember about this user" section.

### Touch Point 2: Agent Tool — Explicit Memory Save

A `save_memory` tool is registered alongside existing tools. The agent can call it during conversation to immediately persist critical information (hot path). The tool accepts content, memory_type, and importance.

### Touch Point 3: After Session Ends — Trigger Pipeline

A hook on session close inserts a pending record into `processing_log`. The background pipeline picks it up for cold-path extraction.


## 8. Module Interface

The memory module exposes an abstract `MemoryProvider` interface with five core methods:

- **add** — write a new memory (sync embed for hot path)
- **search** — multi-signal scored search with configurable weights and filters
- **assemble_context** — build prioritized working memory block within a token budget
- **update** — modify existing memory content, importance, or metadata
- **delete** — remove a memory

Three implementations:
- **PostgresMemoryProvider** — production, backed by PostgreSQL + pgvector
- **InMemoryProvider** — dev/test, in-memory dictionaries
- **ServiceMemoryProvider** (future) — HTTP client to standalone memory service

The agent server codes against the interface, never the implementation. Swapping backends is a config change.

---

## 9. Capacity Planning & Cost Projections

### Storage math

A 1536-dim `halfvec` embedding (half-precision) requires ~3,076 bytes (~3 KB) per vector — half the ~6 KB of a full-precision `vector`. With metadata JSON (~500 bytes), content text (~1 KB), content_tsv, tuple overhead, and bi-temporal columns, each memory record totals **~5 KB with halfvec** (~7.7 KB with full-precision vector). The HNSW index adds approximately **1.5-2× the raw vector data size**.

| Scale | Table size (halfvec) | HNSW index | Total | RAM needed |
|---|---|---|---|---|
| **10K records** | ~50 MB | ~60-90 MB | **~110-140 MB** | 256 MB+ |
| **100K records** | ~500 MB | ~600 MB-1.2 GB | **~1.1-1.7 GB** | 2-4 GB |
| **1M records** | ~5 GB | ~6-12 GB | **~11-17 GB** | 16-32 GB |
| **10M records** | ~50 GB | ~60-120 GB | **~110-170 GB** | 128 GB+ |

**The HNSW index must fit in RAM** (`shared_buffers` or OS page cache) for optimal performance. If evicted, sub-millisecond queries degrade to multi-second queries — this is the #1 performance trap with pgvector.

### Query latency by scale (pgvector HNSW)

| Scale | p50 latency | p95 latency | Recall |
|---|---|---|---|
| 10K vectors | 1-3 ms | 3-5 ms | 98-100% |
| 100K vectors | 3-8 ms | 8-15 ms | 95-99% |
| 1M vectors | 5-12 ms | 15-25 ms | 95-98% |
| 10M vectors | 15-50 ms | 50-120 ms | 90-95% |

**Tuning:** increase `ef_search` (100-200) before increasing `m`. The combination `m=16, ef_search=100-200` is the recommended sweet spot.

### Scaling thresholds

At 15 memories per user per week (3 sessions × 5 memories), users accumulate ~**780 memories per year**. Critical thresholds:

| Users | Total memories | Action required |
|---|---|---|
| ~1,300 | 1M | Ensure HNSW fits in `shared_buffers` |
| ~6,400 | 5M | Increase `maintenance_work_mem` to 4-8 GB |
| ~12,800 | 10M | **Evaluate pgvectorscale** (DiskANN, 9× index compression) |
| ~64,000 | 50M | **Move to dedicated vector DB or shard with Citus** |

**Use `halfvec` from day one** — 50% storage savings, minimal recall impact. Migrating full-precision `vector` to `halfvec` later requires a full re-embed.

### API cost math

**Embedding cost** per session (~2,000 tokens, 5 embeddings): **$0.0002** with text-embedding-3-small. **LLM extraction cost** at 22.5% escalation rate using GPT-4o-mini: **~$0.000135 blended per session**. Total marginal API cost per session: **~$0.0004-0.0015**.

At scale:
- **10K users** (~30K sessions/week) ≈ **$40-50/month** in API costs
- **100K users** (~300K sessions/week) ≈ **$400-500/month** in API costs

**The system is infrastructure-bound, not API-cost-bound** at all relevant scales. Engineering effort should prioritize storage, index maintenance, and retrieval latency over reducing LLM calls.

### Per-user memory budget

Start with **1,000-5,000 memories per user** as a soft limit. Eviction policy (importance-weighted):

```
eviction_score = 0.4 * importance + 0.3 * recency + 0.3 * access_frequency
```

When a user exceeds the budget, the lowest-scoring memories are soft-archived (marked in metadata, not deleted). Heavy users can request profile compaction — a reflection-generation pass that merges many low-level memories into higher-level insights.

For reference: **ChatGPT stores ~1,200-1,400 words total** across all saved memories (the tightest limit among production systems). **Letta evicts ~70% of messages** when the context window reaches capacity, recursively summarizing the evicted content.

---

## 10. Evaluation Strategy

A three-tier evaluation stack — useful even without labeled ground-truth training data.

### Tier 1: Offline benchmarks (regression testing)

Run **LoCoMo** (ACL 2024, Snap Research) and **LongMemEvalS** (ICLR 2025) as regression tests:

- **LoCoMo**: 10 conversations of ~300 turns, ~1,986 questions across single-hop, multi-hop, temporal, open-domain categories. Primary metrics: partial-match F1, BLEU-1, LLM-as-Judge binary score. Note: designed for 32K context era; modern long-context models can score competitively by brute-force context stuffing — use with this caveat.
- **LongMemEvalS**: 500 questions at ~115K tokens, tests five memory abilities (extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention).
- **BEAM** (ICLR 2026, newer): 2,000 validated questions scaling from 100K to 10M tokens — use this once pgvector grows past ~1M records.

Report F1, BLEU-1, and LLM-Judge per category alongside latency and token consumption. **Prefer LLM-as-Judge over token F1** — token F1 is largely insensitive to meaningful accuracy differences (0.221 vs 0.168 for a 20-point accuracy gap per arXiv:2602.11243).

### Tier 2: Component diagnostics (three-probe framework)

From arXiv:2603.02473, diagnose every wrong answer:

1. **Retrieval failure** (was the right memory fetched? Precision@k) — this is the dominant failure mode at **11-46% of errors**
2. **Utilization failure** (was the right memory fetched but not used?) — 4-8% of errors
3. **Hallucination** (was the answer fabricated from corrupted memory?) — 0.4-1.4% of errors

This breakdown tells you where to invest. Retrieval failures dominate at every scale — improving retrieval is always the highest-leverage intervention.

### Tier 3: Production monitoring

Sample 1-5% of production queries for LLM-as-Judge scoring on relevance and correctness. Track continuously:

| Metric | Target | Alerting |
|---|---|---|
| Retrieval latency p50 | <100 ms | >200 ms |
| Retrieval latency p95 | <300 ms | >500 ms |
| Memory hit rate (retrieval returns ≥1 useful memory) | >70% | <50% |
| Memory utilization rate (retrieved memory influences response) | >50% | <30% |
| Duplicate rate (memories flagged by dedup) | <5% | >15% |
| Memory growth rate per user | ~10-30/week | >100/week |
| Extraction version lag (% memories below current version) | <10% | >30% |

Calibrate the LLM judge against **100-200 human-annotated examples monthly** to catch judge drift.

---

## 11. Failure Modes & Safety

### The five most dangerous failure modes in production

**1. Context poisoning** (most insidious)
Incorrect information enters the memory store and silently corrupts all future retrievals for that user. Mitigation: (a) every memory has a provenance chain back to its source session, (b) `memory.memory_history` provides rollback capability, (c) contradiction detection at consolidation time flags potentially wrong memories for LLM review.

**2. Silent behavioral drift**
Unlike hard errors, memory quality can degrade gradually — no alert fires, but user experience worsens over weeks. Mitigation: Tier 3 production monitoring with memory hit rate and utilization rate tracked per week. An alert fires when either drops >10% week-over-week.

**3. Memory staleness**
Vector similarity has no concept of time. A memory that was true six months ago still matches semantically even when contradicted by newer memories. Mitigation: the bi-temporal model (`valid_from`, `valid_until`) — retrieval filters to `WHERE valid_until IS NULL OR valid_until > now()`. The recency signal in multi-signal scoring reinforces this by decaying old memories.

**4. Retrieval failures (the dominant failure mode)**
11-46% of all memory system errors per diagnostic probes. Mitigation: hybrid search (vector + full-text + entity structured lookup), multi-signal scoring, and retrieval-specific regression tests in Tier 1 evaluation.

**5. Memory wipe / corruption incidents (real-world precedent)**
ChatGPT's **February 2025 memory wipe crisis** caused ~83% failure rate during a backend update. The **October 2025 deletion bug** caused deleted memories to reappear when users approached memory capacity. Mitigation: (a) daily logical backups of the `memory` schema, (b) `memory.memory_history` as immutable audit trail, (c) any batch mutation (decay, consolidation, bulk re-extraction) runs in a transaction with dry-run mode first.

### Safety principles

- **Never silently overwrite** — every change goes through `memory.memory_history` (Zep/Graphiti pattern, validated by production incidents)
- **Every memory has provenance** — you can always trace a memory back to the session that produced it and the rule/LLM that extracted it
- **Hot-path writes are idempotent** — the agent can call `save_memory` twice without creating duplicates (exact-hash dedup at insert time)
- **Dry-run before destructive operations** — decay, archival, and re-extraction all support a dry-run mode that reports what would change without applying changes
- **Per-rule observability** — rule-level metrics (how often each rule fired, what it produced) surface broken extraction rules quickly

### User-facing memory tools (future)

When we expose memory management to users (post-v1), the features to implement — based on ChatGPT, Mem0, and Letta:

- **View all memories** — chronological list with content, importance, source session
- **Delete individual memories** — soft delete via `valid_until`, not hard delete
- **Delete all memories** — "forget me" — invalidates all memories for the user
- **Memory history** — per-memory audit trail showing old_content → new_content transitions
- **Toggle memory on/off** — per-user and per-session flags for privacy-sensitive contexts
- **Export memories** — JSON dump of all memories for data portability (Letta's `.af` format is a good reference)

---

## 12. Versioning & Migration

### Embedding model migration — dual-write pattern

The production consensus for changing embedding models:

1. **Add a new embedding column** alongside the old one, using `CREATE INDEX CONCURRENTLY` to avoid table locks
2. **Batch re-embed** with rate limiting (OpenAI Batch API offers 50% cost savings — use it)
3. **Validate via dry-run**: compare top-10 search results between old and new embeddings. A published migration study found ~82% overlap is typical — significantly lower indicates model drift.
4. **Feature-flag the switch** — application reads from the new column behind a flag
5. **Clean up** the old column after a stability period (~2 weeks)

**Cost estimates:** ~$1 for 100K records, ~$10 for 1M records using text-embedding-3-small standard pricing. At our projected scale, re-embedding is cheap.

**Principle:** treat embeddings as reproducible artifacts tied to their generation method. **Never overwrite existing embeddings** — always dual-write.

### Schema migration patterns

PostgreSQL rules for safe schema migrations on a live system:

- `ADD COLUMN` with nullable default is fast (PG 11+, no table rewrite)
- Always use `CREATE INDEX CONCURRENTLY` to avoid blocking writes
- Use `NOT VALID` then `VALIDATE CONSTRAINT` separately for check constraints
- Set `lock_timeout` on DDL statements to avoid blocking the entire database

We'll use a migration tool (Knex.js for Node.js, Alembic for Python — whichever matches the host agent server) rather than running SQL by hand.

### Extraction logic versioning

When rules change or LLM prompts are updated, old memories become stale relative to what the extractor *would* produce today. The `memory.extraction_versions` table tracks this:

- Every memory stores the `extraction_version` it was produced under
- When a new extraction version is deployed, a **re-extraction batch job** queries `WHERE extraction_version < 'current' ORDER BY last_accessed DESC` and re-extracts from the source session
- Re-extraction is rate-limited and runs in the background — it's not urgent

This prevents silent quality drift when the extraction layer improves.

### Rule versioning

Rules are git-tracked YAML files + Python plugins. Each rule has a `version` field in its YAML definition. The rule loader records `rule_pack_commit` (git SHA of the rules directory) at load time. When rules change significantly, `extraction_version` is bumped and the re-extraction job kicks in.

---

## 13. Future Extension Points

The general-purpose platform is designed with these future layers in mind:

| Future Layer | Hooks Into |
|---|---|
| **Taste profiling** | New `memory_type` values, domain-specific extractors in Stage 1 |
| **Behavioral patterns** | New procedural memory extractors, Stage 4 pattern detection |
| **Product preferences** | Entity table (`entity_type = 'product'`), metadata fields |
| **Recommendation integration** | Query engine filters, context assembly sections |
| **Event ingestion** (clickstream, purchases) | New pipeline source alongside lakebase, CDC patterns (Debezium) |
| **Multi-agent shared memory** | `agent_id` scoping already in schema, add sharing policies, add semantic consistency detection (open problem per arXiv:2603.10062) |
| **Standalone service** | Implement `ServiceMemoryProvider`, extract pipeline to workers |
| **Graph memory extension** | Semantic + temporal graphs first (implementable with pgvector + timestamp indexes), add causal/entity graphs incrementally per MAGMA's blueprint ([arXiv:2601.03236](https://arxiv.org/abs/2601.03236)) |
| **RL-based routing (v2)** | Log routing decisions and downstream memory value from v1, train a GRPO policy using MemFactory infrastructure ([arXiv:2603.29493](https://arxiv.org/abs/2603.29493)) |
| **Small-model extraction (middle tier)** | Insert MemReader-0.6B between rules and GPT-class LLM — 95.66% extraction accuracy at a fraction of the cost ([arXiv:2604.07877](https://arxiv.org/abs/2604.07877)) |
| **Context distillation** | Bake frequently-used memory patterns into model weights for common personalization (TSUBASA pattern, arXiv:2604.07894) |
| **User-facing memory tools** | `memory.memory_history` already supports audit trail; add read API and CRUD UI |

---

## 14. Research Foundations & References

### Core Architecture Influences

| Influence | What I Took | Citation |
|---|---|---|
| **MemGPT / Letta** | Virtual memory hierarchy (core/recall/archival), agent-managed memory via tool calls, PostgreSQL as primary backend | Packer et al., 2023 ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)); [Letta GitHub](https://github.com/letta-ai/letta) |
| **Stanford Generative Agents** | Multi-signal retrieval scoring (recency + importance + relevance), reflection mechanism for memory consolidation, importance scoring at creation time | Park et al., UIST 2023 ([arXiv:2304.03442](https://arxiv.org/abs/2304.03442)) |
| **Mem0** | 4-scope model (user/agent/session/org), async-first memory writes, hybrid vector + graph retrieval | [arXiv:2504.19413](https://arxiv.org/abs/2504.19413); [State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026) |
| **A-MAC** (Workday AI) | 5-signal admission ablation proved Content Type Prior captures ~49% of routing value — drove our simplification to 2-signal routing | ICLR 2026 Workshop MemAgent ([arXiv:2603.04549](https://arxiv.org/abs/2603.04549)) |
| **MemMachine** (MemVerge) | ~80% token savings via NLTK segmentation and ground-truth episode preservation; informed our raw-session + memory-specific-extraction split | 2025 ([arXiv:2604.04853](https://arxiv.org/html/2604.04853)); [GitHub](https://github.com/MemMachine/MemMachine) |
| **Zep / Graphiti** | Bi-temporal model (`valid_from`, `valid_until`) for conflict resolution — **never silently overwrite** | [arXiv:2501.13956](https://arxiv.org/html/2501.13956v1); [Website](https://www.getzep.com/) |
| **CoALA Framework** | Cognitive architecture taxonomy: working, episodic, semantic, procedural memory | 2024 ([arXiv:2309.02427](https://arxiv.org/abs/2309.02427)) |
| **OpenAI Agents SDK** | Profile + global memory + session memory layered pattern, memory distillation rules | [Context Personalization Cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/context_personalization) |
| **LangGraph / LangMem** | Thread-scoped (short-term) + cross-thread (long-term) memory, PostgreSQL store, background memory formation | [LangMem SDK](https://blog.langchain.com/langmem-sdk-launch/); [Memory Docs](https://docs.langchain.com/oss/python/langgraph/memory) |

### Additional Research (2025-2026)

**Extraction & routing:**
- **SeCom** (Tsinghua + Microsoft, ICLR 2025) — Tested GPT-4, Mistral-7B, and RoBERTa segmenters; full pipeline with RoBERTa segmenter retains competitive end-to-end QA performance ([arXiv:2502.05589](https://arxiv.org/html/2502.05589v3))
- **MemReader** (MemTensor, 2026) — Active extraction with 0.6B model achieving 95.66% extraction accuracy; introduces "buffer/defer" action for multi-turn extraction ([arXiv:2604.07877](https://arxiv.org/abs/2604.07877))
- **SimpleMem** (2026) — Semantic density gating as an implicit memory-worthiness signal ([arXiv:2601.02553](https://arxiv.org/abs/2601.02553))

**RL-optimized memory:**
- **MemFactory** (MemTensor, 2026) — Unified framework decomposing memory into Extractors/Updaters/Retrievers trained with GRPO; up to 14.8% gains over hand-tuned systems. Path to v2 RL-based routing ([arXiv:2603.29493](https://arxiv.org/abs/2603.29493))
- **TSUBASA** (2026) — Evolutionary memory manager + context distillation; 17-49% F1 improvements over Mem0 on LoCoMo ([arXiv:2604.07894](https://arxiv.org/abs/2604.07894))
- **AgeMem** (2026) — Unified LTM/STM via tool operations (ADD/RETRIEVE/UPDATE/DELETE/SUMMARY/FILTER) with three-stage progressive RL ([arXiv:2601.01885](https://arxiv.org/abs/2601.01885))

**Graph & knowledge structures:**
- **MAGMA** (2026) — Four orthogonal graphs (semantic/temporal/causal/entity) with dual-stream write path; blueprint for future graph extension on pgvector ([arXiv:2601.03236](https://arxiv.org/abs/2601.03236))
- **MIA** (2026) — Bidirectional parametric ↔ non-parametric memory conversion, 9% boost on LiveVQA ([arXiv:2604.04503](https://arxiv.org/abs/2604.04503))

**Evaluation & failure analysis:**
- **HaluMem** (Nov 2025) — First operation-level hallucination benchmark — extraction / update / QA stages with cumulative error amplification ([arXiv:2511.03506](https://arxiv.org/abs/2511.03506))
- **LongMemEval** (ICLR 2025) — Tests extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention
- **BEAM** (ICLR 2026) — 2,000 questions scaling 100K-10M tokens, nugget-based evaluation
- **MemoryAgentBench** (July 2025) — 17 datasets across four competencies

**Multi-agent:**
- **Multi-Agent Memory: A Computer Architecture Perspective** (UCSD/Georgia Tech, 2026) — I/O → Cache → Memory hierarchy for multi-agent systems; identifies open problem of semantic consistency detection ([arXiv:2603.10062](https://arxiv.org/abs/2603.10062))
- **Hindsight** (Dec 2025) — Four-network epistemic split (world facts / experiences / opinions / observations), 39% → 83.6% on LongMemEval ([arXiv:2512.12818](https://arxiv.org/abs/2512.12818))

**Surveys:**
- **Memory in the Age of AI Agents** (Dec 2025, 47 authors) — Forms-Functions-Dynamics taxonomy ([arXiv:2512.13564](https://arxiv.org/abs/2512.13564))
- **RAPTOR** — Recursive abstractive processing for hierarchical memory retrieval at multiple granularities ([arXiv:2401.18059](https://arxiv.org/abs/2401.18059))
- **CrewAI Memory** — Four memory types with LLM-powered cognitive extraction ([Docs](https://docs.crewai.com/concepts/memory))

### Industry Case Studies

- **Amazon Rufus** — Account-level memory, cross-ecosystem personalization, 250M+ users, agentic auto-buy capabilities ([About Amazon](https://www.aboutamazon.com/news/retail/amazon-rufus-ai-assistant-personalized-shopping-features))
- **Klarna AI** — LangGraph orchestration, knowledge graph + memory layer, 85M+ active users, 70% task automation ([LangChain Case Study](https://blog.langchain.com/customers-klarna/))
- **Shopify Sidekick** — JIT instructions architecture, contextual learning across conversations, agentic execution ([Shopify Engineering](https://shopify.engineering/building-production-ready-agentic-systems))
- **PostgreSQL as Agent OS** — PostgreSQL positioned as unified data plane for agent infrastructure: relational, vector, full-text, JSON, time-series in one database ([Oreate AI](https://www.oreateai.com/blog/postgresql-as-the-cornerstone-of-the-ai-agent-operating-system-memory-storage-and-the-future-of-agent-infrastructure/318c62d30a90b6cd22df70af44eaf44d))
- **pgvector** — v0.8+ provides up to 9x faster queries, HNSW + IVFFlat indexes, halfvec for 2x storage reduction ([GitHub](https://github.com/pgvector/pgvector))
