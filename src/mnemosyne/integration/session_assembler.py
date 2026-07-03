from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from asyncpg.exceptions import UniqueViolationError as _UNIQUE_VIOLATION

from mnemosyne.integration.session_models import ConversationMessage, SessionBatch


class SessionAssembler:
    """In-process buffer that accumulates ConversationMessages into SessionBatches."""

    def __init__(self) -> None:
        self._buffers: dict[uuid.UUID, list[ConversationMessage]] = {}
        self._users: dict[uuid.UUID, uuid.UUID] = {}
        self._started_at: dict[uuid.UUID, datetime] = {}
        self._lock = asyncio.Lock()

    async def add_message(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        message: ConversationMessage,
    ) -> None:
        """Append a message to the session's buffer."""
        async with self._lock:
            buf = self._buffers.setdefault(session_id, [])
            buf.append(message)
            self._users[session_id] = user_id
            self._started_at.setdefault(session_id, message.sent_at)

    async def flush(self, session_id: uuid.UUID) -> SessionBatch | None:
        """Return the assembled batch and clear the buffer. None if unknown."""
        async with self._lock:
            messages = self._buffers.pop(session_id, None)
            user_id = self._users.pop(session_id, None)
            started_at = self._started_at.pop(session_id, None)
            if not messages or user_id is None or started_at is None:
                return None
            return SessionBatch(
                session_id=session_id,
                user_id=user_id,
                messages=messages,
                started_at=started_at,
                closed_at=datetime.now(timezone.utc),
            )


_INSERT_MESSAGE_SQL = """
    INSERT INTO memory.session_buffer
        (session_id, ordinal, message_id, user_id, role, content, sent_at)
    SELECT $1, COALESCE(MAX(ordinal), -1) + 1, $2, $3, $4, $5, $6
    FROM memory.session_buffer
    WHERE session_id = $1
"""

_SELECT_MESSAGES_SQL = """
    SELECT message_id, user_id, role, content, sent_at
    FROM memory.session_buffer
    WHERE session_id = $1
    ORDER BY ordinal ASC
"""

_DELETE_SESSION_SQL = "DELETE FROM memory.session_buffer WHERE session_id = $1"


class PersistentSessionAssembler:
    """Postgres-backed assembler. Use when MNEMOSYNE_SESSION_PERSIST=1."""

    def __init__(self, pool) -> None:
        self._pool = pool

    async def add_message(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        message: ConversationMessage,
    ) -> None:
        """Append a message using a single ordinal-computing INSERT statement.

        The ordinal is derived inside the statement via ``COALESCE(MAX(ordinal),
        -1) + 1`` so no separate read round-trip is needed. Concurrent writers
        can collide on the ``(session_id, ordinal)`` unique constraint; on a
        unique violation the insert is retried exactly once (the retry re-reads
        the now-higher MAX inside the same statement).
        """
        args = (
            session_id,
            message.message_id,
            user_id,
            message.role,
            message.content,
            message.sent_at,
        )
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(_INSERT_MESSAGE_SQL, *args)
        except _UNIQUE_VIOLATION:
            async with self._pool.acquire() as conn:
                await conn.execute(_INSERT_MESSAGE_SQL, *args)

    def _rows_to_batch(
        self, session_id: uuid.UUID, rows
    ) -> SessionBatch | None:
        if not rows:
            return None
        user_id = rows[0]["user_id"]
        messages = [
            ConversationMessage(
                message_id=r["message_id"],
                role=r["role"],
                content=r["content"],
                sent_at=r["sent_at"],
            )
            for r in rows
        ]
        return SessionBatch(
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            started_at=messages[0].sent_at,
            closed_at=datetime.now(timezone.utc),
        )

    async def peek(self, session_id: uuid.UUID) -> SessionBatch | None:
        """Return the assembled batch without deleting the buffered rows.

        Used by the cold path so messages are only removed after the pipeline
        has successfully processed them (via ``delete_session``).
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SELECT_MESSAGES_SQL, session_id)
        return self._rows_to_batch(session_id, rows)

    async def delete_session(self, session_id: uuid.UUID) -> None:
        """Remove all buffered rows for a session."""
        async with self._pool.acquire() as conn:
            await conn.execute(_DELETE_SESSION_SQL, session_id)

    async def flush(self, session_id: uuid.UUID) -> SessionBatch | None:
        """Read and clear the buffer atomically in a single transaction."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_MESSAGES_SQL, session_id)
                if not rows:
                    return None
                await conn.execute(_DELETE_SESSION_SQL, session_id)
        return self._rows_to_batch(session_id, rows)
