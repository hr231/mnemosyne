from __future__ import annotations

import pytest

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.embedding.fake import FakeEmbeddingClient
from mnemosyne.embedding.ollama import OllamaEmbeddingClient
from mnemosyne.embedding.openai_compatible import OpenAICompatibleEmbeddingClient


def test_factory_fake():
    """from_config with provider='fake' returns a FakeEmbeddingClient."""
    client = EmbeddingClient.from_config({"provider": "fake", "dimensions": 768})
    assert isinstance(client, FakeEmbeddingClient)


def test_factory_fake_default_dim():
    """from_config with provider='fake' and no dimensions uses dim=768."""
    client = EmbeddingClient.from_config({"provider": "fake"})
    assert isinstance(client, FakeEmbeddingClient)
    assert client.dim == 768


def test_factory_ollama():
    """from_config with provider='ollama' returns an OllamaEmbeddingClient."""
    client = EmbeddingClient.from_config({
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text",
    })
    assert isinstance(client, OllamaEmbeddingClient)


def test_factory_openai_compatible():
    """from_config with provider='openai_compatible' returns an OpenAICompatibleEmbeddingClient."""
    client = EmbeddingClient.from_config({
        "provider": "openai_compatible",
        "base_url": "http://api.openai.com",
        "model": "text-embedding-3-small",
        "api_key": "sk-test",
    })
    assert isinstance(client, OpenAICompatibleEmbeddingClient)


def test_factory_unknown_raises():
    """from_config with an unrecognised provider raises ValueError."""
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingClient.from_config({"provider": "unknown"})


def test_factory_openai_missing_base_url():
    """from_config for openai_compatible without base_url raises KeyError."""
    with pytest.raises(KeyError):
        EmbeddingClient.from_config({"provider": "openai_compatible", "model": "x"})
