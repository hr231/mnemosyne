from __future__ import annotations

import asyncio
import logging
import threading

from mnemosyne.embedding.base import EmbeddingClient

logger = logging.getLogger(__name__)


class FastEmbedClient(EmbeddingClient):
    """Local embedding client using FastEmbed (Qdrant). Zero API dependency.

    The model is loaded lazily on first call to ``embed`` / ``embed_batch``.
    First load downloads the ONNX weights to the FastEmbed cache (default
    ``~/.cache/fastembed``); subsequent loads are local-only.

    Setup
    -----
    Install the optional extra::

        pip install 'mnemosyne[fastembed]'

    Default model is ``BAAI/bge-small-en-v1.5`` (384-dim). To pick a
    different model, pass ``model_name`` explicitly::

        FastEmbedClient(model_name="BAAI/bge-base-en-v1.5")

    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs):
        self._model_name = model_name
        self._kwargs = kwargs
        self._model = None
        self._load_lock = threading.Lock()

    def _get_model(self):
        # Model load and inference are CPU-bound (ONNX) and must run in a
        # worker thread; the lock keeps concurrent first calls from racing
        # the lazy load.
        with self._load_lock:
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

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        return [e.tolist() for e in model.embed(texts)]

    async def embed(self, text: str) -> list[float]:
        embeddings = await asyncio.to_thread(self._embed_sync, [text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)
