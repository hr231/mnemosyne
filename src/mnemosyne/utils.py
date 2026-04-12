from __future__ import annotations

import hashlib


def content_hash(content: str) -> str:
    """Canonical sha256 hash of normalised content (strip + lower)."""
    return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
