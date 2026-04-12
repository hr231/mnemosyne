from __future__ import annotations

import os
import uuid

import httpx
import pytest


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
    if not os.environ.get("MNEMOSYNE_LLM_INTEGRATION"):
        pytest.skip("MNEMOSYNE_LLM_INTEGRATION not set")

    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip("Ollama not reachable at http://localhost:11434")
