"""extraction_versions + reextraction_jobs

Revision ID: 003
Revises: 002
Create Date: 2026-04-23
"""
from typing import Sequence, Union

from alembic import op


revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS memory")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory.extraction_versions (
            version TEXT PRIMARY KEY,
            changeset_hash TEXT NOT NULL UNIQUE,
            changed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            changed_by TEXT NOT NULL,
            summary TEXT NOT NULL,
            rule_files JSONB NOT NULL DEFAULT '[]',
            prompt_files JSONB NOT NULL DEFAULT '[]'
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory.reextraction_jobs (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL,
            target_version TEXT NOT NULL,
            status TEXT NOT NULL,
            count_processed INTEGER NOT NULL DEFAULT 0,
            count_changed INTEGER NOT NULL DEFAULT 0,
            started_at TIMESTAMPTZ NOT NULL,
            finished_at TIMESTAMPTZ,
            error TEXT
        )
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_reextraction_jobs_user_status
        ON memory.reextraction_jobs (user_id, status)
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS memory.ix_reextraction_jobs_user_status"
    )
    op.execute("DROP TABLE IF EXISTS memory.reextraction_jobs")
    op.execute("DROP TABLE IF EXISTS memory.extraction_versions")
