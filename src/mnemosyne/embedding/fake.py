from __future__ import annotations

import hashlib

import numpy as np

from mnemosyne.embedding.base import EmbeddingClient


class FakeEmbeddingClient(EmbeddingClient):
    """Deterministic embedding client for tests.

    Uses blake2b hashing (not Python's built-in hash(), which is
    randomised per-process) to seed a PRNG, then unit-normalises the
    output so that cosine similarity equals the dot product.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        seed = int.from_bytes(
            hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
        )
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]
