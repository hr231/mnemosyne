from __future__ import annotations

import pytest

from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.embedding.ollama import OllamaEmbeddingClient
from mnemosyne.embedding.openai_compatible import OpenAICompatibleEmbeddingClient


def test_factory_ollama():
    client = EmbeddingClient.from_config({
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text",
    })
    assert isinstance(client, OllamaEmbeddingClient)


def test_factory_openai_compatible():
    client = EmbeddingClient.from_config({
        "provider": "openai_compatible",
        "base_url": "http://api.openai.com",
        "model": "text-embedding-3-small",
        "api_key": "sk-test",
    })
    assert isinstance(client, OpenAICompatibleEmbeddingClient)


def test_factory_openai_sdk():
    from mnemosyne.embedding.openai_sdk import OpenAIEmbeddingClient
    client = EmbeddingClient.from_config({
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "sk-test",
    })
    assert isinstance(client, OpenAIEmbeddingClient)


def test_factory_azure_sdk():
    from mnemosyne.embedding.openai_sdk import OpenAIEmbeddingClient
    client = EmbeddingClient.from_config({
        "provider": "azure",
        "model": "text-embedding-3-small",
        "api_key": "sk-test",
        "azure_endpoint": "https://myorg.openai.azure.com",
    })
    assert isinstance(client, OpenAIEmbeddingClient)


def test_factory_google_sdk():
    from mnemosyne.embedding.google_sdk import GoogleEmbeddingClient
    client = EmbeddingClient.from_config({
        "provider": "google",
        "model": "text-embedding-004",
        "api_key": "goog-key",
    })
    assert isinstance(client, GoogleEmbeddingClient)


def test_factory_fastembed():
    from mnemosyne.embedding.fastembed import FastEmbedClient
    client = EmbeddingClient.from_config({
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
    })
    assert isinstance(client, FastEmbedClient)


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingClient.from_config({"provider": "unknown"})


def test_factory_missing_provider_raises():
    with pytest.raises(ValueError, match="provider"):
        EmbeddingClient.from_config({})


def test_factory_openai_missing_base_url():
    with pytest.raises(KeyError):
        EmbeddingClient.from_config({"provider": "openai_compatible", "model": "x"})
