from __future__ import annotations

import os
import uuid

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()  # loads .env from repo root


@pytest.fixture
def user_id():
    return uuid.uuid4()


@pytest.fixture
def agent_id():
    return uuid.UUID("00000000-0000-0000-0000-000000000000")


@pytest.fixture
def session_id():
    return uuid.uuid4()


@pytest.fixture
def real_llm_or_skip():
    val = os.environ.get("MNEMOSYNE_LLM_INTEGRATION", "")
    if val not in ("1", "true", "yes"):
        pytest.skip("MNEMOSYNE_LLM_INTEGRATION not enabled")

    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip("Ollama not reachable at http://localhost:11434")


# ---------------------------------------------------------------------------
# Parameterized provider fixture — runs provider tests against both backends
# ---------------------------------------------------------------------------

from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.providers.in_memory import InMemoryProvider


@pytest.fixture(params=["in_memory", "postgres"])
async def provider(request):
    """Yield a MemoryProvider. Each test that uses this fixture runs twice:
    once against InMemoryProvider and once against PostgresMemoryProvider.

    The postgres variant is skipped when MNEMOSYNE_PG_DSN is not set or when
    the memory schema tables have not been created (run: alembic upgrade head).
    """
    if request.param == "in_memory":
        yield InMemoryProvider()
        return

    # --- postgres branch ---
    dsn = os.environ.get("MNEMOSYNE_PG_DSN")
    if not dsn:
        pytest.skip("MNEMOSYNE_PG_DSN not set — skipping postgres variant")

    from mnemosyne.providers.postgres import PostgresMemoryProvider

    pg_provider = await PostgresMemoryProvider.connect(dsn)

    # Verify the schema exists; if not, skip rather than fail.
    async with pg_provider._pool.acquire() as conn:
        try:
            await conn.execute("SELECT 1 FROM memory.memories LIMIT 0")
        except Exception:
            await pg_provider.close()
            pytest.skip(
                "memory schema not found — run: alembic upgrade head"
            )

        # Truncate for test isolation before the test runs.
        await conn.execute("TRUNCATE memory.memory_history CASCADE")
        await conn.execute("TRUNCATE memory.memories CASCADE")
        await conn.execute("TRUNCATE memory.episodes CASCADE")
        await conn.execute("TRUNCATE memory.processing_log CASCADE")

    yield pg_provider

    # Cleanup after the test.
    try:
        async with pg_provider._pool.acquire() as conn:
            await conn.execute("TRUNCATE memory.memory_history CASCADE")
            await conn.execute("TRUNCATE memory.memories CASCADE")
            await conn.execute("TRUNCATE memory.episodes CASCADE")
            await conn.execute("TRUNCATE memory.processing_log CASCADE")
    finally:
        await pg_provider.close()


@pytest.fixture
def embedder():
    return FakeEmbeddingClient(dim=768)


# ---------------------------------------------------------------------------
# Parameterized entity_store fixture — runs entity tests against both backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=["in_memory", "postgres"])
async def entity_store(request):
    """Yield an EntityStore. Each test that uses this fixture runs twice:
    once against InMemoryEntityStore and once against PostgresEntityStore.

    The postgres variant is skipped when MNEMOSYNE_PG_DSN is not set or when
    the entity tables have not been created (run: alembic upgrade head).
    """
    if request.param == "in_memory":
        from mnemosyne.providers.in_memory_entity_store import InMemoryEntityStore
        yield InMemoryEntityStore()
        return

    # --- postgres branch ---
    dsn = os.environ.get("MNEMOSYNE_PG_DSN")
    if not dsn:
        pytest.skip("MNEMOSYNE_PG_DSN not set — skipping postgres variant")

    import asyncpg
    from pgvector.asyncpg import register_vector
    from mnemosyne.db.repositories.entity import PostgresEntityStore

    pool = await asyncpg.create_pool(
        dsn,
        min_size=1,
        max_size=5,
        init=lambda conn: register_vector(conn),
    )

    async with pool.acquire() as conn:
        try:
            await conn.execute("SELECT 1 FROM memory.entities LIMIT 0")
        except Exception:
            await pool.close()
            pytest.skip("Entity tables not found — run: alembic upgrade head")

        await conn.execute("TRUNCATE memory.entity_mentions CASCADE")
        await conn.execute("TRUNCATE memory.entities CASCADE")
        # memories is truncated last because entity_mentions has FK to memories
        await conn.execute("TRUNCATE memory.memories CASCADE")

    store = PostgresEntityStore(pool)
    # Attach pool so tests that need to seed memories can access it.
    store._pool = pool  # already set by __init__; this is for clarity
    yield store

    try:
        async with pool.acquire() as conn:
            await conn.execute("TRUNCATE memory.entity_mentions CASCADE")
            await conn.execute("TRUNCATE memory.entities CASCADE")
    finally:
        await pool.close()


async def _insert_stub_memory(pool, memory_id: uuid.UUID) -> None:
    """Insert a minimal stub row into memory.memories for FK satisfaction."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory.memories (memory_id, user_id, content)
            VALUES ($1, $2, 'stub memory for entity mention test')
            ON CONFLICT DO NOTHING
            """,
            memory_id,
            uuid.UUID("00000000-0000-0000-0000-000000000001"),
        )


@pytest.fixture
async def stub_memory_id(entity_store) -> uuid.UUID:
    """Yield a memory_id that satisfies the FK constraint for entity_mentions.

    For InMemoryEntityStore: returns a random UUID (no FK enforcement).
    For PostgresEntityStore: inserts a stub row into memory.memories first.
    """
    from mnemosyne.providers.in_memory_entity_store import InMemoryEntityStore

    memory_id = uuid.uuid4()
    if not isinstance(entity_store, InMemoryEntityStore):
        # PostgresEntityStore — must have a real memories row
        await _insert_stub_memory(entity_store._pool, memory_id)
    return memory_id
