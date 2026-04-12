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
