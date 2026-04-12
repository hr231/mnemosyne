"""Create memory schema with all tables and indexes.

Revision ID: 001
Revises: None
Create Date: 2026-04-12
"""
from typing import Sequence, Union
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE SCHEMA IF NOT EXISTS memory")

    # memories table
    op.execute("""
        CREATE TABLE memory.memories (
            memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            agent_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
            org_id UUID,
            memory_type TEXT NOT NULL DEFAULT 'fact',
            content TEXT NOT NULL,
            content_hash TEXT,
            embedding halfvec(768),
            importance FLOAT NOT NULL DEFAULT 0.5,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed TIMESTAMPTZ NOT NULL DEFAULT now(),
            decay_rate FLOAT NOT NULL DEFAULT 0.01,
            valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
            valid_until TIMESTAMPTZ,
            extraction_version TEXT NOT NULL DEFAULT '0.1.0',
            extraction_model TEXT,
            prompt_hash TEXT,
            rule_id TEXT,
            source_session_id UUID,
            source_memory_ids UUID[] DEFAULT '{}',
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        )
    """)

    # episodes table
    op.execute("""
        CREATE TABLE memory.episodes (
            episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            agent_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
            session_id UUID NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            summary_embedding halfvec(768),
            key_topics TEXT[] DEFAULT '{}',
            memory_ids UUID[] DEFAULT '{}',
            outcome TEXT,
            started_at TIMESTAMPTZ,
            ended_at TIMESTAMPTZ,
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # memory_history table (immutable audit log)
    op.execute("""
        CREATE TABLE memory.memory_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            memory_id UUID NOT NULL REFERENCES memory.memories(memory_id),
            operation TEXT NOT NULL,
            old_content TEXT,
            new_content TEXT,
            old_importance FLOAT,
            new_importance FLOAT,
            actor TEXT NOT NULL,
            actor_details JSONB NOT NULL DEFAULT '{}',
            occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # processing_log table
    op.execute("""
        CREATE TABLE memory.processing_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL,
            pipeline_step TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            memories_created UUID[] DEFAULT '{}',
            processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # --- Indexes ---

    # HNSW vector index for similarity search
    op.execute("""
        CREATE INDEX idx_mem_embedding_hnsw ON memory.memories
        USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # Composite indexes for common queries
    op.execute("CREATE INDEX idx_mem_user_type ON memory.memories (user_id, memory_type)")
    op.execute("CREATE INDEX idx_mem_user_importance ON memory.memories (user_id, importance DESC)")

    # Bi-temporal: currently-valid memories
    op.execute("CREATE INDEX idx_mem_valid ON memory.memories (user_id, valid_from DESC, valid_until)")
    op.execute("CREATE INDEX idx_mem_active ON memory.memories (user_id) WHERE valid_until IS NULL")

    # Content-hash dedup (unique partial index)
    op.execute("""
        CREATE UNIQUE INDEX idx_mem_content_hash
        ON memory.memories (user_id, content_hash)
        WHERE valid_until IS NULL
    """)

    # Full-text search
    op.execute("CREATE INDEX idx_mem_tsv ON memory.memories USING gin (content_tsv)")

    # JSONB metadata
    op.execute("CREATE INDEX idx_mem_metadata ON memory.memories USING gin (metadata)")

    # pg_trgm fuzzy matching
    op.execute("CREATE INDEX idx_mem_trgm ON memory.memories USING gin (content gin_trgm_ops)")

    # Memory history lookup
    op.execute("CREATE INDEX idx_history_memory ON memory.memory_history (memory_id, occurred_at DESC)")

    # Processing log: pending records
    op.execute("CREATE INDEX idx_proclog_pending ON memory.processing_log (status) WHERE status = 'pending'")

    # Embedding NULL (for batch embedding worker)
    op.execute("CREATE INDEX idx_mem_null_embed ON memory.memories (memory_id) WHERE embedding IS NULL")


def downgrade() -> None:
    op.execute("DROP SCHEMA IF EXISTS memory CASCADE")
