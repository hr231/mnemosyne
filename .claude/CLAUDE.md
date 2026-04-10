# Mnemosyne — Agent Memory System

## Project Overview

Mnemosyne is a general-purpose agent memory platform. It processes session data from an existing lakebase (PostgreSQL) into structured long-term memory, enabling cross-session continuity. It is an **embedded module** inside the agent server, not a standalone service.

## Architecture Summary

- **Database**: PostgreSQL with pgvector, `memory` schema alongside existing `lakebase` schema
- **Pipeline**: 5-stage background processing (Extraction → Embedding → Episodes → Consolidation → Decay)
- **Retrieval**: Multi-signal scored search (relevance + recency + importance + frequency) with hybrid vector + full-text
- **Interface**: `MemoryProvider` abstract with `PostgresMemoryProvider` and `InMemoryProvider`
- **Agent Integration**: 3 touch points (inject memory before LLM call, save_memory tool, session close hook)
- **Rule Engine**: Plugin architecture with YAML rules + Python extractors

## Reference Documents

1. **Design Document** — `docs/agent-memory-design.md` — the authoritative specification. All design decisions must reference it.
2. **Implementation Plan** — `docs/agent-memory-implementation-plan.md` — the reference implementation with code for every task (17 tasks, TDD-style). Contains working code for models, config, extractors, pipeline, retrieval, context assembly, and integration. **Use as a blueprint, not copy-paste** — adapt to the directory structure and patterns defined in this CLAUDE.md.

If you disagree with a design or implementation decision, flag it — don't silently deviate.

## Directory Structure

```
mnemosyne/
├── .claude/agents/          — Agent definitions (this team)
├── docs/                    — Design doc, PRD, sprint plans, evaluation results
├── src/
│   ├── db/                  — Database models, repositories, connection management
│   │   ├── models/          — SQLAlchemy models for memory schema
│   │   └── repositories/    — Repository pattern (all SQL lives here)
│   ├── pipeline/            — 5-stage processing pipeline
│   │   ├── extraction/      — Hybrid extraction (rules + LLM)
│   │   ├── embedding/       — Batch embedding stage
│   │   ├── episodes/        — Episode creation
│   │   ├── consolidation/   — Dedup, reflection, contradiction resolution
│   │   └── decay/           — Importance decay, archival
│   ├── rules/               — Rule engine: BaseExtractor, RuleLoader, RuleRegistry
│   ├── retrieval/           — Query engine, multi-signal scoring, hybrid search
│   ├── context/             — Context assembly, token budgeting
│   ├── providers/           — MemoryProvider interface + implementations
│   ├── embedding/           — Embedding client abstraction
│   ├── llm/                 — LLM client abstraction
│   ├── integration/         — Agent server touch points, tools, hooks
│   ├── config/              — Configuration loader, settings, validation
│   └── monitoring/          — Metrics, alerting thresholds
├── rules/
│   └── core/                — Core YAML rule definitions
├── prompts/                 — LLM prompt templates
├── migrations/              — Alembic migration scripts
├── tests/
│   ├── unit/                — Mirrors src/ structure
│   ├── integration/         — Real PostgreSQL tests
│   │   ├── db/
│   │   └── e2e/
│   ├── benchmarks/          — LoCoMo, LongMemEvalS harnesses
│   ├── diagnostics/         — Three-probe framework
│   └── fixtures/            — Shared test data
├── sql/                     — Raw SQL for complex queries
└── config/                  — Default YAML configuration
```

## Agent Team Domains

Each agent owns specific directories. Do NOT modify files outside your domain without lead approval.

| Agent | Owns | Does NOT touch |
|---|---|---|
| product-manager | `docs/` | `src/`, `tests/`, `migrations/`, `rules/` |
| database-engineer | `src/db/`, `migrations/`, `sql/`, `tests/*/db/` | `src/pipeline/`, `src/retrieval/`, `src/integration/` |
| pipeline-engineer | `src/pipeline/`, `src/rules/`, `rules/`, `src/llm/`, `prompts/`, `tests/*/pipeline/`, `tests/*/rules/` | `src/db/models/`, `migrations/`, `src/retrieval/`, `src/integration/` |
| retrieval-engineer | `src/retrieval/`, `src/context/`, `src/providers/`, `src/embedding/`, `tests/*/retrieval/`, `tests/*/context/`, `tests/*/providers/` | `src/db/models/`, `migrations/`, `src/pipeline/`, `src/integration/` |
| integration-engineer | `src/integration/`, `src/config/`, `config/`, `tests/*/integration/`, `tests/integration/e2e/` | `src/db/models/`, `migrations/`, `src/pipeline/`, `src/retrieval/` |
| qa-engineer | `tests/`, `src/monitoring/`, `docs/TESTING.md`, `docs/EVALUATION.md` | `src/` (except monitoring) — writes tests, never fixes production code |

## Dependency Order

```
database-engineer → pipeline-engineer → retrieval-engineer → integration-engineer
                                    ↘                    ↗
                                      qa-engineer (parallel)
```

The database schema must exist before pipeline can write to it. The MemoryProvider interface must be defined before integration can wire it. QA runs in parallel from Sprint 1.

## Git Workflow

- `main` — protected, human-reviewed merges only
- `feature/*` — one branch per agent per sprint task
- Each agent works in its own git worktree
- PRs require passing tests + human approval

## Tech Stack

- Python 3.12+
- PostgreSQL 16+ with pgvector 0.8+, pg_trgm
- SQLAlchemy 2.0 (async, asyncpg)
- Alembic for migrations
- pydantic for config validation
- pytest + pytest-asyncio for testing
- tiktoken for token estimation
