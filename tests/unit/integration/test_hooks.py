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


# ---------------------------------------------------------------------------
# Durable queued-mode persistence
# ---------------------------------------------------------------------------


class _FakeConn:
    """Connection stub for the extraction-gate lookup.

    ``fetchval`` returns ``extraction_enabled`` (``None`` ⇒ enabled, matching a
    missing user_settings row).
    """

    def __init__(self, extraction_enabled=None):
        self._extraction_enabled = extraction_enabled

    async def fetchval(self, *args, **kwargs):
        return self._extraction_enabled


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    """Minimal stand-in so insert_pending can be exercised without a DB.

    Also answers the extraction-gate lookup via ``acquire``; by default the
    user is treated as enabled.
    """

    def __init__(self, extraction_enabled=None):
        self.calls: list[tuple] = []
        self._conn = _FakeConn(extraction_enabled)

    def acquire(self):
        return _FakeAcquire(self._conn)


@pytest.mark.asyncio
async def test_queued_mode_persists_pending_row(monkeypatch):
    """When a pool is supplied, a durable pending row is inserted and its id
    is reflected on the returned ProcessingLog.
    """
    import mnemosyne.integration.hooks as hooks_mod

    captured = {}
    persisted_id = uuid.uuid4()

    async def fake_insert_pending(conn_or_pool, session_id, user_id=None, **kw):
        captured["session_id"] = session_id
        captured["user_id"] = user_id
        captured["pool"] = conn_or_pool
        return persisted_id

    monkeypatch.setattr(hooks_mod, "insert_pending", fake_insert_pending)

    pool = _FakePool()
    sid = uuid.uuid4()
    uid = uuid.uuid4()
    entry = await on_session_close(session_id=sid, user_id=uid, pool=pool)

    assert captured["session_id"] == sid
    assert captured["user_id"] == uid
    assert captured["pool"] is pool
    assert entry.id == persisted_id
    assert entry.status == "pending"
    assert entry.user_id == uid


@pytest.mark.asyncio
async def test_queued_mode_without_pool_returns_in_memory_log(monkeypatch):
    """No pool → no durable insert, but a pending log is still returned."""
    import mnemosyne.integration.hooks as hooks_mod

    called = False

    async def fake_insert_pending(*a, **k):
        nonlocal called
        called = True
        return uuid.uuid4()

    monkeypatch.setattr(hooks_mod, "insert_pending", fake_insert_pending)

    entry = await on_session_close(session_id=uuid.uuid4(), user_id=uuid.uuid4())
    assert called is False
    assert entry.status == "pending"


@pytest.mark.asyncio
async def test_queued_mode_persist_failure_fail_open(monkeypatch):
    """A durable-insert failure under fail_open returns a pending log, no raise."""
    import mnemosyne.integration.hooks as hooks_mod

    async def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(hooks_mod, "insert_pending", boom)

    entry = await on_session_close(
        session_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        pool=_FakePool(),
        fail_open=True,
    )
    assert entry.status == "pending"


@pytest.mark.asyncio
async def test_queued_mode_persist_failure_fail_closed_raises(monkeypatch):
    import mnemosyne.integration.hooks as hooks_mod

    async def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(hooks_mod, "insert_pending", boom)

    with pytest.raises(RuntimeError):
        await on_session_close(
            session_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            pool=_FakePool(),
            fail_open=False,
        )
