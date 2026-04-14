from __future__ import annotations

import logging

from mnemosyne.embedding.base import EmbeddingClient

logger = logging.getLogger(__name__)


class FastEmbedClient(EmbeddingClient):
    """Local embedding client using FastEmbed (Qdrant). Zero API dependency."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs):
        self._model_name = model_name
        self._kwargs = kwargs
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self._model_name, **self._kwargs)
                logger.info("FastEmbed model loaded: %s", self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "FastEmbed not installed. Install: pip install fastembed"
                ) from exc
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = self._get_model()
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]
