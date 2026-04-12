from __future__ import annotations

import logging
import uuid

from mnemosyne.config.settings import Settings
from mnemosyne.db.models.memory import ExtractionResult, Memory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.rules.base_extractor import BaseExtractor
from mnemosyne.rules.stub import StubRegexExtractor

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Orchestrates the memory extraction pipeline for a single text input.

    Stages:
    1. Run all enabled ``BaseExtractor`` instances against *text*.
    2. Routing decision (currently stubbed; LLM escalation path pending).
    3. Embed each ``ExtractionResult``, construct a ``Memory``, persist
       via ``provider.add``, and stamp the returned UUID on the result.

    Per-extractor errors are isolated — one bad rule never aborts the pipeline.
    """

    def __init__(
        self,
        settings: Settings,
        provider: MemoryProvider,
        embedder: EmbeddingClient,
        extractors: list[BaseExtractor] | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._embedder = embedder
        self._extractors: list[BaseExtractor] = extractors or [
            StubRegexExtractor(extraction_version=settings.extraction_version)
        ]

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        provider: MemoryProvider,
        embedder: EmbeddingClient,
    ) -> ExtractionPipeline:
        """Factory method — loads rules from the configured rules directory.

        Falls back to ``StubRegexExtractor`` when no rules are found (e.g.
        the rules directory does not exist).
        """
        from mnemosyne.rules.rule_loader import RuleLoader
        from mnemosyne.rules.rule_registry import RuleRegistry

        loader = RuleLoader()
        raw_extractors = loader.load_from_directory(settings.rules_dir)

        if not raw_extractors:
            extractors = [StubRegexExtractor(extraction_version=settings.extraction_version)]
        else:
            registry = RuleRegistry()
            registry.register_all(raw_extractors)
            extractors = registry.all()

        return cls(settings=settings, provider=provider, embedder=embedder, extractors=extractors)

    async def process(
        self,
        user_id: uuid.UUID,
        text: str,
    ) -> list[ExtractionResult]:
        """Extract memories from *text* for *user_id* and persist them.

        Returns the list of ``ExtractionResult`` objects with
        ``memory_id`` set to the UUID assigned by the provider.
        """
        # --- Stage 1: run rule extractors ---
        all_results: list[ExtractionResult] = []
        for extractor in self._extractors:
            if not extractor.enabled:
                continue
            try:
                results = extractor.extract(text)
                all_results.extend(results)
            except Exception as exc:  # noqa: BLE001 — per-rule isolation
                logger.warning(
                    "extractor %r raised an exception — skipping",
                    extractor.id,
                    exc_info=exc,
                )

        # --- Stage 2: routing decision ---
        route_to_llm = False  # stub: LLM escalation not yet wired
        if route_to_llm:
            pass

        # --- Stage 3: embed + persist each result ---
        final_results: list[ExtractionResult] = []
        for result in all_results:
            # Stamp extraction_version from settings so YAML rules (which do
            # not set it themselves) always carry the correct pipeline version.
            result = result.model_copy(
                update={"extraction_version": self._settings.extraction_version}
            )
            embedding = await self._embedder.embed(result.content)
            memory = Memory(
                user_id=user_id,
                content=result.content,
                memory_type=result.memory_type,
                importance=result.importance,
                embedding=embedding,
                extraction_version=result.extraction_version,
                rule_id=result.rule_id,
                metadata=result.metadata,
            )
            mem_id = await self._provider.add(memory)
            result = result.model_copy(update={"memory_id": mem_id})
            final_results.append(result)

        logger.debug(
            "extraction complete: %d result(s) for user %s",
            len(final_results),
            user_id,
        )
        return final_results
