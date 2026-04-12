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

    Day 1 morning shape (v0.1.0 walking skeleton):
    -----------------------------------------------
    1. Run all enabled ``BaseExtractor`` instances against *text*.
    2. Router stub: ``route_to_llm = False`` — LLM path wired in Day 2
       (Task 8 creates ``router.py`` and replaces the inline stub).
    3. For each ``ExtractionResult``, embed the content, construct a
       ``Memory``, call ``provider.add``, and stamp the returned UUID
       on the result.

    The ``extractors`` list defaults to a single ``StubRegexExtractor``
    seeded with ``settings.extraction_version``.  Day 1 afternoon replaces
    the default with ``RuleRegistry.load(settings.rules_dir).all()`` —
    one line changes in ``from_settings``; the rest of this class is
    untouched.

    Per-extractor errors are swallowed and logged so one bad rule never
    aborts the pipeline.  This mirrors the production rule isolation
    contract described in Section 2.3.1 of the design doc.
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
        """Factory method.  Day 1 afternoon upgrades the extractor list here."""
        return cls(settings=settings, provider=provider, embedder=embedder)

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

        # --- Stage 2: routing decision (stub: always False for Day 1) ---
        # Task 8 (Day 2 morning) creates router.py and replaces this line.
        route_to_llm = False  # stub: route_to_llm = False
        if route_to_llm:
            pass  # LLM extraction path — Day 2

        # --- Stage 3: embed + persist each result ---
        final_results: list[ExtractionResult] = []
        for result in all_results:
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
