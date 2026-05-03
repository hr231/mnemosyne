"""gdpr_deletions audit table

Revision ID: 004
Revises: 003
Create Date: 2026-04-23
"""
from typing import Sequence, Union

from alembic import op


revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS memory")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory.gdpr_deletions (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL,
            requestor TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT 'user_request',
            rows_memories INTEGER NOT NULL DEFAULT 0,
            rows_entities INTEGER NOT NULL DEFAULT 0,
            rows_mentions INTEGER NOT NULL DEFAULT 0,
            rows_episodes INTEGER NOT NULL DEFAULT 0,
            rows_history INTEGER NOT NULL DEFAULT 0,
            occurred_at TIMESTAMPTZ NOT NULL,
            dry_run BOOLEAN NOT NULL DEFAULT FALSE
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_gdpr_deletions_user_occurred
        ON memory.gdpr_deletions (user_id, occurred_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS memory.ix_gdpr_deletions_user_occurred")
    op.execute("DROP TABLE IF EXISTS memory.gdpr_deletions")
