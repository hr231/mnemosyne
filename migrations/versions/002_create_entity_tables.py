"""Create entity tables and indexes.

Revision ID: 002
Revises: 001
Create Date: 2026-04-13
"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # entities table
    op.execute("""
        CREATE TABLE IF NOT EXISTS memory.entities (
            entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            agent_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            embedding halfvec(768),
            facts JSONB NOT NULL DEFAULT '{}',
            confidence FLOAT NOT NULL DEFAULT 1.0,
            mention_count INTEGER NOT NULL DEFAULT 0,
            source_memory_ids UUID[] DEFAULT '{}',
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # entity_mentions join table
    op.execute("""
        CREATE TABLE IF NOT EXISTS memory.entity_mentions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_id UUID NOT NULL REFERENCES memory.entities(entity_id),
            memory_id UUID NOT NULL REFERENCES memory.memories(memory_id),
            mention_text TEXT NOT NULL DEFAULT '',
            context TEXT NOT NULL DEFAULT '',
            occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # Indexes
    op.execute("""
        CREATE INDEX idx_entity_embedding_hnsw ON memory.entities
        USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute("""
        CREATE UNIQUE INDEX idx_entity_unique_name
        ON memory.entities (user_id, normalized_name, entity_type)
    """)
    op.execute("CREATE INDEX idx_entity_user ON memory.entities (user_id)")
    op.execute("CREATE INDEX idx_entity_facts ON memory.entities USING gin (facts)")
    op.execute("CREATE INDEX idx_mention_entity ON memory.entity_mentions (entity_id, occurred_at DESC)")
    op.execute("CREATE INDEX idx_mention_memory ON memory.entity_mentions (memory_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memory.entity_mentions CASCADE")
    op.execute("DROP TABLE IF EXISTS memory.entities CASCADE")
