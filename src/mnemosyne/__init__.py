"""Mnemosyne — general-purpose agent memory module."""

from mnemosyne.config.settings import Settings
from mnemosyne.context.assembly import ContextBlock, assemble_context
from mnemosyne.db.models.memory import ExtractionResult, Memory, ScoredMemory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.integration.prompt_builder import build_system_prompt_memory_block
from mnemosyne.integration.save_memory_tool import handle_save_memory, save_memory_tool_spec
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.providers.in_memory import InMemoryProvider
from mnemosyne.retrieval.scoring import ScoringWeights

__all__ = [
    "assemble_context",
    "build_system_prompt_memory_block",
    "ContextBlock",
    "EmbeddingClient",
    "ExtractionPipeline",
    "ExtractionResult",
    "handle_save_memory",
    "InMemoryProvider",
    "Memory",
    "MemoryProvider",
    "save_memory_tool_spec",
    "ScoredMemory",
    "ScoringWeights",
    "Settings",
]
