"""processing_log scoping, decay watermark, and user_settings

Revision ID: 007
Revises: 006
Create Date: 2026-06-12
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


UP_SQL = """
CREATE SCHEMA IF NOT EXISTS memory;

ALTER TABLE memory.processing_log
    ADD COLUMN IF NOT EXISTS user_id UUID;

ALTER TABLE memory.processing_log
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;
ALTER TABLE memory.processing_log
    ALTER COLUMN created_at SET DEFAULT now();
UPDATE memory.processing_log SET created_at = now() WHERE created_at IS NULL;
ALTER TABLE memory.processing_log
    ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE memory.processing_log
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
ALTER TABLE memory.processing_log
    ALTER COLUMN updated_at SET DEFAULT now();
UPDATE memory.processing_log SET updated_at = now() WHERE updated_at IS NULL;
ALTER TABLE memory.processing_log
    ALTER COLUMN updated_at SET NOT NULL;

ALTER TABLE memory.processing_log
    ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_processing_log_status_created
    ON memory.processing_log (status, created_at);

ALTER TABLE memory.memories
    ADD COLUMN IF NOT EXISTS last_decayed_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS memory.user_settings (
    user_id            UUID        PRIMARY KEY,
    extraction_enabled BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

DOWN_SQL = """
DROP TABLE IF EXISTS memory.user_settings;

ALTER TABLE memory.memories
    DROP COLUMN IF EXISTS last_decayed_at;

DROP INDEX IF EXISTS memory.idx_processing_log_status_created;

ALTER TABLE memory.processing_log
    DROP COLUMN IF EXISTS retry_count;

ALTER TABLE memory.processing_log
    DROP COLUMN IF EXISTS updated_at;

ALTER TABLE memory.processing_log
    DROP COLUMN IF EXISTS created_at;

ALTER TABLE memory.processing_log
    DROP COLUMN IF EXISTS user_id;
"""


def upgrade() -> None:
    op.execute(UP_SQL)


def downgrade() -> None:
    op.execute(DOWN_SQL)
