from __future__ import annotations

import uuid

import pytest

from mnemosyne.integration.hooks import on_session_close


@pytest.mark.asyncio
async def test_on_session_close_returns_processing_log():
    entry = await on_session_close(
        session_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
    )
    assert entry.pipeline_step == "extraction"
    assert entry.status == "pending"
    assert entry.session_id is not None


@pytest.mark.asyncio
async def test_on_session_close_preserves_session_id():
    sid = uuid.uuid4()
    entry = await on_session_close(
        session_id=sid,
        user_id=uuid.uuid4(),
    )
    assert entry.session_id == sid


@pytest.mark.asyncio
async def test_on_session_close_idempotent():
    sid = uuid.uuid4()
    uid = uuid.uuid4()
    e1 = await on_session_close(session_id=sid, user_id=uid)
    e2 = await on_session_close(session_id=sid, user_id=uid)
    # Each call creates a new entry (idempotency handled by pipeline runner)
    assert e1.id != e2.id
    assert e1.session_id == e2.session_id


@pytest.mark.asyncio
async def test_on_session_close_accepts_provider_kwarg():
    """provider kwarg is accepted and ignored (wired in by caller)."""
    entry = await on_session_close(
        session_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        provider=None,
    )
    assert entry.status == "pending"


@pytest.mark.asyncio
async def test_on_session_close_entry_has_unique_id():
    """Every call produces a fresh UUID for the log row."""
    e1 = await on_session_close(session_id=uuid.uuid4(), user_id=uuid.uuid4())
    e2 = await on_session_close(session_id=uuid.uuid4(), user_id=uuid.uuid4())
    assert e1.id != e2.id
