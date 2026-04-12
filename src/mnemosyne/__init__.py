"""Mnemosyne — general-purpose agent memory module."""

# Lazy imports to allow building modules incrementally.
# Final __init__.py will re-export all public types directly.


def __getattr__(name):
    if name == "Memory":
        from mnemosyne.db.models.memory import Memory
        return Memory
    if name == "ScoredMemory":
        from mnemosyne.db.models.memory import ScoredMemory
        return ScoredMemory
    if name == "ExtractionResult":
        from mnemosyne.db.models.memory import ExtractionResult
        return ExtractionResult
    if name == "MemoryProvider":
        from mnemosyne.providers.base import MemoryProvider
        return MemoryProvider
    if name == "InMemoryProvider":
        from mnemosyne.providers.in_memory import InMemoryProvider
        return InMemoryProvider
    raise AttributeError(f"module 'mnemosyne' has no attribute {name!r}")


__all__ = [
    "Memory",
    "ScoredMemory",
    "ExtractionResult",
    "MemoryProvider",
    "InMemoryProvider",
]
