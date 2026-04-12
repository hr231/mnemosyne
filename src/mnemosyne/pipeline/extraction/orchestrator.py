from __future__ import annotations

import logging
import uuid

from mnemosyne.config.settings import Settings
from mnemosyne.db.models.memory import ExtractionResult, Memory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.llm.base import LLMClient
from mnemosyne.pipeline.extraction.llm_extractor import LLMExtractor
from mnemosyne.pipeline.extraction.router import ExtractionStats, should_route_to_llm
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.rules.base_extractor import BaseExtractor
from mnemosyne.rules.stub import StubRegexExtractor
from mnemosyne.utils import content_hash

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Orchestrates memory extraction: rules -> routing -> optional LLM -> persist."""

    def __init__(
        self,
        settings: Settings,
        provider: MemoryProvider,
        embedder: EmbeddingClient,
        extractors: list[BaseExtractor] | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._embedder = embedder
        self._extractors: list[BaseExtractor] = extractors or [
            StubRegexExtractor(extraction_version=settings.extraction_version)
        ]
        self._llm_extractor: LLMExtractor | None = None
        if llm_client is not None:
            self._llm_extractor = LLMExtractor(
                llm_client, extraction_version=settings.extraction_version
            )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        provider: MemoryProvider,
        embedder: EmbeddingClient,
        llm_client: LLMClient | None = None,
    ) -> ExtractionPipeline:
        """Factory method — loads rules from the configured directory."""
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

        return cls(
            settings=settings,
            provider=provider,
            embedder=embedder,
            extractors=extractors,
            llm_client=llm_client,
        )

    async def process(
        self,
        user_id: uuid.UUID,
        text: str,
    ) -> list[ExtractionResult]:
        """Extract memories from *text* and persist them."""
        # --- Stage 1: run rule extractors ---
        rule_results: list[ExtractionResult] = []
        total_matched_chars = 0
        for extractor in self._extractors:
            if not extractor.enabled:
                continue
            try:
                results = extractor.extract(text)
                rule_results.extend(results)
                total_matched_chars += sum(r.matched_chars for r in results)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "extractor %r raised — skipping", extractor.id, exc_info=exc
                )

        # --- Stage 2: routing decision ---
        all_results = list(rule_results)
        stats = ExtractionStats(
            extracted_count=len(rule_results),
            total_chars=len(text),
            chars_matched_by_rules=total_matched_chars,
        )
        if self._llm_extractor and should_route_to_llm(
            stats, self._settings.router_unstructured_threshold
        ):
            try:
                llm_results = await self._llm_extractor.extract(text)
                seen_hashes = {content_hash(r.content) for r in rule_results}
                for lr in llm_results:
                    if content_hash(lr.content) not in seen_hashes:
                        all_results.append(lr)
                        seen_hashes.add(content_hash(lr.content))
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM extraction failed — using rule results only", exc_info=exc)

        # --- Stage 3: embed + persist ---
        final_results: list[ExtractionResult] = []
        for result in all_results:
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
