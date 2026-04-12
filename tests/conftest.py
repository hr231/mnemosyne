from __future__ import annotations

import uuid
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
