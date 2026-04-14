from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestFastEmbedClient:
    def test_default_model_name(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient()
        assert client._model_name == "BAAI/bge-small-en-v1.5"

    def test_custom_model_name(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient(model_name="test-model")
        assert client._model_name == "test-model"

    def test_model_not_loaded_on_init(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient(model_name="test-model")
        assert client._model is None

    def test_get_model_raises_import_error_when_fastembed_missing(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient(model_name="test-model")
        with patch.dict("sys.modules", {"fastembed": None}):
            with pytest.raises(ImportError, match="FastEmbed not installed"):
                client._get_model()

    def test_get_model_returns_cached_instance(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient()
        mock_model = MagicMock()
        client._model = mock_model
        result = client._get_model()
        assert result is mock_model

    @pytest.mark.asyncio
    async def test_embed_single(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient()

        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.array([0.1, 0.2, 0.3])])
        client._model = mock_model

        result = await client.embed("hello")
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
        mock_model.embed.assert_called_once_with(["hello"])

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_float(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient()

        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.array([0.5, -0.3, 1.0, 0.0])])
        client._model = mock_model

        result = await client.embed("test text")
        assert isinstance(result, list)
        assert result == pytest.approx([0.5, -0.3, 1.0, 0.0])

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient()

        mock_model = MagicMock()
        mock_model.embed.return_value = iter([
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ])
        client._model = mock_model

        result = await client.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        assert all(isinstance(v, float) for v in result[0])
        assert all(isinstance(v, float) for v in result[1])
        mock_model.embed.assert_called_once_with(["hello", "world"])

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient()
        result = await client.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_empty_skips_model_load(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient()
        # _model is None — but empty list must not trigger model load
        result = await client.embed_batch([])
        assert result == []
        assert client._model is None

    @pytest.mark.asyncio
    async def test_embed_batch_single_item(self):
        from mnemosyne.embedding.fastembed import FastEmbedClient
        client = FastEmbedClient()

        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.array([0.7, 0.8])])
        client._model = mock_model

        result = await client.embed_batch(["only one"])
        assert len(result) == 1
        assert result[0] == pytest.approx([0.7, 0.8])

    def test_factory_creates_fastembed(self):
        from mnemosyne.embedding.base import EmbeddingClient
        from mnemosyne.embedding.fastembed import FastEmbedClient
        config = {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"}
        client = EmbeddingClient.from_config(config)
        assert isinstance(client, FastEmbedClient)
        assert client._model_name == "BAAI/bge-small-en-v1.5"

    def test_factory_fastembed_default_model(self):
        from mnemosyne.embedding.base import EmbeddingClient
        from mnemosyne.embedding.fastembed import FastEmbedClient
        config = {"provider": "fastembed"}
        client = EmbeddingClient.from_config(config)
        assert isinstance(client, FastEmbedClient)
        assert client._model_name == "BAAI/bge-small-en-v1.5"

    def test_factory_fastembed_custom_model(self):
        from mnemosyne.embedding.base import EmbeddingClient
        from mnemosyne.embedding.fastembed import FastEmbedClient
        config = {"provider": "fastembed", "model": "sentence-transformers/all-MiniLM-L6-v2"}
        client = EmbeddingClient.from_config(config)
        assert isinstance(client, FastEmbedClient)
        assert client._model_name == "sentence-transformers/all-MiniLM-L6-v2"
