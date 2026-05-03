"""session_buffer table for persisted multi-turn message assembly

Revision ID: 005
Revises: 004
Create Date: 2026-04-23
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


UP_SQL = """
CREATE SCHEMA IF NOT EXISTS memory;

CREATE TABLE IF NOT EXISTS memory.session_buffer (
    session_id   UUID        NOT NULL,
    ordinal      INTEGER     NOT NULL,
    message_id   UUID        NOT NULL,
    user_id      UUID        NOT NULL,
    role         TEXT        NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content      TEXT        NOT NULL,
    sent_at      TIMESTAMPTZ NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, ordinal)
);

CREATE INDEX IF NOT EXISTS ix_session_buffer_user
    ON memory.session_buffer (user_id);
"""

DOWN_SQL = """
DROP TABLE IF EXISTS memory.session_buffer;
"""


def upgrade() -> None:
    op.execute(UP_SQL)


def downgrade() -> None:
    op.execute(DOWN_SQL)
