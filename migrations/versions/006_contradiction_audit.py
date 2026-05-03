"""contradiction_audit table

Revision ID: 006
Revises: 004
Create Date: 2026-04-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS memory")
    op.create_table(
        "contradiction_audit",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("detected_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("new_memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("existing_memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "nli_scores",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("llm_adjudication", sa.Text(), nullable=True),
        sa.Column("resolution", sa.Text(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column(
            "applied",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.CheckConstraint(
            "resolution IN ('supersede', 'keep_both', 'merge', 'keep_old')",
            name="contradiction_audit_resolution_check",
        ),
        schema="memory",
    )
    op.create_index(
        "ix_contradiction_audit_user_id",
        "contradiction_audit",
        ["user_id"],
        schema="memory",
    )
    op.create_index(
        "ix_contradiction_audit_detected_at",
        "contradiction_audit",
        [sa.text("detected_at DESC")],
        schema="memory",
    )
    op.create_index(
        "ix_contradiction_audit_resolution",
        "contradiction_audit",
        ["resolution"],
        schema="memory",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_contradiction_audit_resolution",
        table_name="contradiction_audit",
        schema="memory",
    )
    op.drop_index(
        "ix_contradiction_audit_detected_at",
        table_name="contradiction_audit",
        schema="memory",
    )
    op.drop_index(
        "ix_contradiction_audit_user_id",
        table_name="contradiction_audit",
        schema="memory",
    )
    op.drop_table("contradiction_audit", schema="memory")
