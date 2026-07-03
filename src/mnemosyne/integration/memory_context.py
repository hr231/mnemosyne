from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from mnemosyne.context.assembly import ContextBlock, assemble_context
from mnemosyne.monitoring.metrics import global_registry

logger = logging.getLogger(__name__)


async def assemble_context_safe(
    provider: Any,
    user_id: UUID,
    query_embedding: list[float],
    embedder: Any,
    token_budget: int = 2000,
    entity_store: Any = None,
    query_text: str = "",
    *,
    fail_open: bool = True,
) -> ContextBlock:
    """Assemble the memory context block with host-protecting failure isolation.

    Delegates to :func:`mnemosyne.context.assembly.assemble_context`. When
    ``fail_open`` is True (the default) any exception is swallowed: an empty
    :class:`ContextBlock` is returned, a warning is logged, and a context
    failure metric is recorded so the host LLM call proceeds without memory
    rather than breaking. With ``fail_open=False`` the original exception
    propagates (used by tests).
    """
    try:
        return await assemble_context(
            provider=provider,
            user_id=user_id,
            query_embedding=query_embedding,
            embedder=embedder,
            token_budget=token_budget,
            entity_store=entity_store,
            query_text=query_text,
        )
    except Exception as exc:
        if not fail_open:
            raise
        logger.warning(
            "assemble_context failed for user %s (isolated): %s",
            user_id,
            exc,
            exc_info=True,
        )
        try:
            global_registry().record_context_truncate()
        except Exception:  # pragma: no cover - metrics never break the host
            pass
        return ContextBlock(text="", token_count=0, sections=[])
