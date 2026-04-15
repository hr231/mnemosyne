from __future__ import annotations

import pytest

from mnemosyne.llm.base import LLMClient
from mnemosyne.llm.openai_compatible import OpenAICompatibleClient


def test_factory_openai_compatible():
    client = LLMClient.from_config({
        "provider": "openai_compatible",
        "base_url": "http://localhost:11434/v1",
        "model": "gemma3:4b",
    })
    assert isinstance(client, OpenAICompatibleClient)


def test_factory_openai_sdk():
    from mnemosyne.llm.openai_sdk import OpenAILLMClient
    client = LLMClient.from_config({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
    })
    assert isinstance(client, OpenAILLMClient)


def test_factory_azure_sdk():
    from mnemosyne.llm.openai_sdk import OpenAILLMClient
    client = LLMClient.from_config({
        "provider": "azure",
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "azure_endpoint": "https://myorg.openai.azure.com",
    })
    assert isinstance(client, OpenAILLMClient)


def test_factory_anthropic_sdk():
    from mnemosyne.llm.anthropic_sdk import AnthropicLLMClient
    client = LLMClient.from_config({
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "api_key": "sk-ant-test",
    })
    assert isinstance(client, AnthropicLLMClient)


def test_factory_google_sdk():
    from mnemosyne.llm.google_sdk import GoogleLLMClient
    client = LLMClient.from_config({
        "provider": "google",
        "model": "gemini-2.0-flash",
        "api_key": "goog-key",
    })
    assert isinstance(client, GoogleLLMClient)


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMClient.from_config({"provider": "unknown"})


def test_factory_missing_provider_raises():
    with pytest.raises(ValueError, match="provider"):
        LLMClient.from_config({})
