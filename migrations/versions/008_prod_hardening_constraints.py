"""processing_log dedup, domain CHECK constraints, episodes index, reflection watermark

Revision ID: 008
Revises: 007
Create Date: 2026-07-03
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


UP_SQL = """
CREATE SCHEMA IF NOT EXISTS memory;

-- A freshly enqueued row carries no processing timestamp; only a terminal
-- transition (completed/failed) stamps processed_at.
ALTER TABLE memory.processing_log
    ALTER COLUMN processed_at DROP DEFAULT;
ALTER TABLE memory.processing_log
    ALTER COLUMN processed_at DROP NOT NULL;

-- Collapse any existing in-flight duplicates for the same (session, step)
-- so the partial unique index below can be created, keeping the oldest row.
DELETE FROM memory.processing_log a
USING memory.processing_log b
WHERE a.status IN ('pending', 'processing')
  AND b.status IN ('pending', 'processing')
  AND a.session_id = b.session_id
  AND a.pipeline_step = b.pipeline_step
  AND (a.created_at > b.created_at
       OR (a.created_at = b.created_at AND a.ctid > b.ctid));

-- At most one in-flight row per (session, step) prevents duplicate enqueues
-- from driving redundant extraction work.
CREATE UNIQUE INDEX IF NOT EXISTS idx_processing_log_dedup
    ON memory.processing_log (session_id, pipeline_step)
    WHERE status IN ('pending', 'processing');

CREATE INDEX IF NOT EXISTS idx_episodes_user_created
    ON memory.episodes (user_id, created_at DESC);

ALTER TABLE memory.user_settings
    ADD COLUMN IF NOT EXISTS last_reflected_at TIMESTAMPTZ;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_memories_memory_type') THEN
        ALTER TABLE memory.memories
            ADD CONSTRAINT ck_memories_memory_type
            CHECK (memory_type IN ('fact', 'preference', 'entity', 'procedural', 'reflection'))
            NOT VALID;
    END IF;
END $$;
ALTER TABLE memory.memories VALIDATE CONSTRAINT ck_memories_memory_type;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_memories_importance') THEN
        ALTER TABLE memory.memories
            ADD CONSTRAINT ck_memories_importance
            CHECK (importance BETWEEN 0 AND 1)
            NOT VALID;
    END IF;
END $$;
ALTER TABLE memory.memories VALIDATE CONSTRAINT ck_memories_importance;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_memories_decay_rate') THEN
        ALTER TABLE memory.memories
            ADD CONSTRAINT ck_memories_decay_rate
            CHECK (decay_rate BETWEEN 0 AND 1)
            NOT VALID;
    END IF;
END $$;
ALTER TABLE memory.memories VALIDATE CONSTRAINT ck_memories_decay_rate;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_entities_confidence') THEN
        ALTER TABLE memory.entities
            ADD CONSTRAINT ck_entities_confidence
            CHECK (confidence BETWEEN 0 AND 1)
            NOT VALID;
    END IF;
END $$;
ALTER TABLE memory.entities VALIDATE CONSTRAINT ck_entities_confidence;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_memory_history_operation') THEN
        ALTER TABLE memory.memory_history
            ADD CONSTRAINT ck_memory_history_operation
            CHECK (operation IN ('create', 'update', 'invalidate', 'merge'))
            NOT VALID;
    END IF;
END $$;
ALTER TABLE memory.memory_history VALIDATE CONSTRAINT ck_memory_history_operation;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_memory_history_actor') THEN
        ALTER TABLE memory.memory_history
            ADD CONSTRAINT ck_memory_history_actor
            CHECK (actor IN (
                'agent_tool', 'pipeline_extraction', 'pipeline_consolidation',
                'pipeline_decay', 'manual'))
            NOT VALID;
    END IF;
END $$;
ALTER TABLE memory.memory_history VALIDATE CONSTRAINT ck_memory_history_actor;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_processing_log_status') THEN
        ALTER TABLE memory.processing_log
            ADD CONSTRAINT ck_processing_log_status
            CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
            NOT VALID;
    END IF;
END $$;
ALTER TABLE memory.processing_log VALIDATE CONSTRAINT ck_processing_log_status;
"""

DOWN_SQL = """
ALTER TABLE memory.processing_log DROP CONSTRAINT IF EXISTS ck_processing_log_status;
ALTER TABLE memory.memory_history DROP CONSTRAINT IF EXISTS ck_memory_history_actor;
ALTER TABLE memory.memory_history DROP CONSTRAINT IF EXISTS ck_memory_history_operation;
ALTER TABLE memory.entities DROP CONSTRAINT IF EXISTS ck_entities_confidence;
ALTER TABLE memory.memories DROP CONSTRAINT IF EXISTS ck_memories_decay_rate;
ALTER TABLE memory.memories DROP CONSTRAINT IF EXISTS ck_memories_importance;
ALTER TABLE memory.memories DROP CONSTRAINT IF EXISTS ck_memories_memory_type;

ALTER TABLE memory.user_settings DROP COLUMN IF EXISTS last_reflected_at;

DROP INDEX IF EXISTS memory.idx_episodes_user_created;
DROP INDEX IF EXISTS memory.idx_processing_log_dedup;

UPDATE memory.processing_log SET processed_at = now() WHERE processed_at IS NULL;
ALTER TABLE memory.processing_log
    ALTER COLUMN processed_at SET DEFAULT now();
ALTER TABLE memory.processing_log
    ALTER COLUMN processed_at SET NOT NULL;
"""


def upgrade() -> None:
    op.execute(UP_SQL)


def downgrade() -> None:
    op.execute(DOWN_SQL)
