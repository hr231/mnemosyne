from __future__ import annotations

import pytest

from mnemosyne.llm.base import LLMClient
from mnemosyne.llm.fake import FakeLLMClient
from mnemosyne.llm.openai_compatible import OpenAICompatibleClient


def test_factory_fake():
    """from_config with provider='fake' returns a FakeLLMClient."""
    client = LLMClient.from_config({"provider": "fake"})
    assert isinstance(client, FakeLLMClient)


def test_factory_openai_compatible():
    """from_config with provider='openai_compatible' returns an OpenAICompatibleClient."""
    client = LLMClient.from_config({
        "provider": "openai_compatible",
        "base_url": "http://localhost:11434/v1",
        "model": "gemma3:4b",
    })
    assert isinstance(client, OpenAICompatibleClient)


def test_factory_unknown_raises():
    """from_config with an unrecognised provider raises ValueError."""
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMClient.from_config({"provider": "unknown"})


def test_factory_default_is_fake():
    """from_config with an empty dict defaults to FakeLLMClient."""
    client = LLMClient.from_config({})
    assert isinstance(client, FakeLLMClient)
