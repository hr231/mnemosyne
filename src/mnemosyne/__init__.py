"""Mnemosyne — general-purpose agent memory module."""

from importlib.metadata import PackageNotFoundError, version

from mnemosyne.config.settings import Settings
from mnemosyne.context.assembly import ContextBlock, assemble_context
from mnemosyne.db.models.memory import ExtractionResult, Memory, ScoredMemory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.integration.bootstrap import bootstrap_memory_subsystem
from mnemosyne.integration.hook_queue import HookQueue
from mnemosyne.integration.hooks import on_session_close
from mnemosyne.integration.memory_context import assemble_context_safe
from mnemosyne.integration.memory_management import MemoryManagementService
from mnemosyne.integration.prompt_builder import build_system_prompt_memory_block
from mnemosyne.integration.save_memory_tool import handle_save_memory, save_memory_tool_spec
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.pipeline.runner import PipelineWorker
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.providers.in_memory import InMemoryProvider
from mnemosyne.providers.postgres import PostgresMemoryProvider
from mnemosyne.retrieval.scoring import ScoringWeights

try:
    __version__ = version("mnemosyne")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "assemble_context",
    "assemble_context_safe",
    "bootstrap_memory_subsystem",
    "build_system_prompt_memory_block",
    "ContextBlock",
    "EmbeddingClient",
    "ExtractionPipeline",
    "ExtractionResult",
    "handle_save_memory",
    "HookQueue",
    "InMemoryProvider",
    "Memory",
    "MemoryManagementService",
    "MemoryProvider",
    "on_session_close",
    "PipelineWorker",
    "PostgresMemoryProvider",
    "save_memory_tool_spec",
    "ScoredMemory",
    "ScoringWeights",
    "Settings",
]
