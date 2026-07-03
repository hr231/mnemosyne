from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from mnemosyne.db.models.memory import Memory
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)

# Default thresholds — all overridable via function parameters.
DEFAULT_ARCHIVE_THRESHOLD: float = 0.05
DEFAULT_ARCHIVE_AFTER_DAYS: int = 90
DEFAULT_HALF_LIFE_DAYS: float = 60.0

# Minimum elapsed time before a memory is decayed again. Decay is applied in
# discrete intervals so re-running within the same interval is a no-op.
DECAY_MIN_INTERVAL_DAYS: float = 1.0


class DecayConfig(BaseModel):
    """Tunable parameters for the decay worker."""

    importance_half_life_days: float = Field(
        default=DEFAULT_HALF_LIFE_DAYS, gt=0.0
    )
    archival_threshold: float = Field(
        default=DEFAULT_ARCHIVE_THRESHOLD, ge=0.0, le=1.0
    )
    archival_age_days_min: int = Field(
        default=DEFAULT_ARCHIVE_AFTER_DAYS, ge=0
    )

    @classmethod
    def from_settings(cls, settings: object) -> "DecayConfig":
        """Build a config from any object with optional matching attributes."""
        kwargs: dict = {}
        if hasattr(settings, "decay_archival_threshold"):
            kwargs["archival_threshold"] = settings.decay_archival_threshold
        if hasattr(settings, "decay_archival_age_days_min"):
            kwargs["archival_age_days_min"] = settings.decay_archival_age_days_min
        if hasattr(settings, "decay_importance_half_life_days"):
            kwargs["importance_half_life_days"] = (
                settings.decay_importance_half_life_days
            )
        return cls(**kwargs)


def _decay_anchor(memory: Memory) -> datetime:
    """Return the timestamp decay should be measured from.

    Anchored at ``last_decayed_at`` when the memory has been decayed before,
    otherwise ``last_accessed``. This makes repeated decay passes idempotent:
    once a memory is decayed and stamped, a re-run measures zero elapsed time
    and leaves the importance unchanged.
    """
    anchor = memory.last_decayed_at or memory.last_accessed
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    return anchor


def compute_decayed_importance(memory: Memory, now: datetime | None = None) -> float:
    """Return the exponentially decayed importance for *memory*.

    Formula: ``importance * exp(-decay_rate * days_since_anchor)`` where the
    anchor is ``COALESCE(last_decayed_at, last_accessed)``.

    The result is clamped to [0.0, 1.0].
    """
    if now is None:
        now = datetime.now(timezone.utc)

    anchor = _decay_anchor(memory)
    days_since = max(0.0, (now - anchor).total_seconds() / 86400.0)
    decayed = memory.importance * math.exp(-memory.decay_rate * days_since)
    return max(0.0, min(1.0, decayed))


def should_archive(
    memory: Memory,
    archive_threshold: float = DEFAULT_ARCHIVE_THRESHOLD,
    archive_after_days: int = DEFAULT_ARCHIVE_AFTER_DAYS,
    now: datetime | None = None,
) -> bool:
    """Return True if *memory* should be soft-archived.

    A memory is archived when ALL of the following hold:
    - Its current importance is below *archive_threshold*.
    - It has not been accessed in the last *archive_after_days* days.
    - It is still active (``valid_until`` is ``None``).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if memory.valid_until is not None:
        return False

    last = memory.last_accessed
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    days_since = (now - last).total_seconds() / 86400.0

    return memory.importance < archive_threshold and days_since > archive_after_days


async def apply_decay(
    provider: MemoryProvider,
    user_id: uuid.UUID | None = None,
    archive_threshold: float = DEFAULT_ARCHIVE_THRESHOLD,
    archive_after_days: int = DEFAULT_ARCHIVE_AFTER_DAYS,
    dry_run: bool = False,
    metrics: object | None = None,
) -> dict:
    """Apply exponential importance decay to active memories and archive stale ones.

    For each active memory:
    1. Compute the decayed importance using ``compute_decayed_importance``.
    2. Write the updated importance back via ``provider.update()``.
    3. If the post-decay importance is below *archive_threshold* AND the memory
       has not been accessed in the last *archive_after_days* days, soft-archive
       it by setting a metadata flag ``"archived": true`` (NOT invalidating — it
       stays queryable for history purposes).

    Parameters
    ----------
    provider:
        The memory provider to scan and update.
    user_id:
        When given, only memories belonging to this user are processed.
        When ``None``, all users are processed (InMemoryProvider only).
    archive_threshold:
        Importance below which a stale memory is archived.
    archive_after_days:
        Minimum days since last access before archival.
    dry_run:
        When ``True``, compute and log what *would* change but make no writes.

    Returns
    -------
    dict
        ``{"decayed": int, "archived": int}`` — counts of updated / archived
        memories.
    """
    stats: dict[str, int] = {"decayed": 0, "archived": 0}
    now = datetime.now(timezone.utc)

    if hasattr(provider, "_memories"):
        await _apply_decay_in_memory(
            provider,
            user_id,
            archive_threshold,
            archive_after_days,
            dry_run,
            now,
            stats,
            metrics,
        )
    elif hasattr(provider, "_pool"):
        await _apply_decay_postgres(
            provider,
            user_id,
            archive_threshold,
            archive_after_days,
            dry_run,
            now,
            stats,
            metrics,
        )
    else:
        logger.warning(
            "apply_decay: unknown provider type %s — skipping",
            type(provider).__name__,
        )

    if not dry_run:
        logger.info(
            "apply_decay: decayed=%d archived=%d user=%s",
            stats["decayed"],
            stats["archived"],
            user_id,
        )
    else:
        logger.info(
            "apply_decay (dry_run): would decay=%d would archive=%d user=%s",
            stats["decayed"],
            stats["archived"],
            user_id,
        )

    return stats


async def _apply_decay_in_memory(
    provider: MemoryProvider,
    user_id: uuid.UUID | None,
    archive_threshold: float,
    archive_after_days: int,
    dry_run: bool,
    now: datetime,
    stats: dict,
    metrics: object | None,
) -> None:
    """Decay + archive logic for InMemoryProvider."""
    memories: list[Memory] = [
        m
        for m in provider._memories.values()  # type: ignore[attr-defined]
        if m.valid_until is None
        and (user_id is None or m.user_id == user_id)
    ]

    for mem in memories:
        anchor = _decay_anchor(mem)
        if (now - anchor).total_seconds() / 86400.0 < DECAY_MIN_INTERVAL_DAYS:
            continue

        new_importance = compute_decayed_importance(mem, now)
        if new_importance == mem.importance:
            continue

        if not dry_run:
            await provider.update(
                mem.memory_id, importance=new_importance, last_decayed_at=now
            )
        stats["decayed"] += 1

        if new_importance < archive_threshold and _days_since(mem.last_accessed, now) > archive_after_days:
            if not dry_run:
                current = provider._memories.get(mem.memory_id)  # type: ignore[attr-defined]
                if current:
                    current.metadata["archived"] = True
            stats["archived"] += 1
            _record_archive(metrics)


async def _apply_decay_postgres(
    provider: MemoryProvider,
    user_id: uuid.UUID | None,
    archive_threshold: float,
    archive_after_days: int,
    dry_run: bool,
    now: datetime,
    stats: dict,
    metrics: object | None,
) -> None:
    """Decay + archive logic for PostgresMemoryProvider."""
    pool = provider._pool  # type: ignore[attr-defined]

    params: list = []
    user_filter = ""
    if user_id is not None:
        params.append(user_id)
        user_filter = "AND user_id = $1"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT memory_id, importance, last_accessed, last_decayed_at,
                   decay_rate, metadata
            FROM memory.memories
            WHERE valid_until IS NULL
            {user_filter}
            """,
            *params,
        )

    cutoff = now - timedelta(days=archive_after_days)

    updates: list[tuple[float, str, uuid.UUID]] = []
    for row in rows:
        anchor = row["last_decayed_at"] or row["last_accessed"]
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        last_accessed = row["last_accessed"]
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)

        days_since = max(0.0, (now - anchor).total_seconds() / 86400.0)
        if days_since < DECAY_MIN_INTERVAL_DAYS:
            continue
        new_importance = max(
            0.0, min(1.0, row["importance"] * math.exp(-row["decay_rate"] * days_since))
        )
        if new_importance == row["importance"]:
            continue

        meta = _decode_jsonb(row["metadata"])
        do_archive = new_importance < archive_threshold and last_accessed < cutoff
        if do_archive:
            meta["archived"] = True
            stats["archived"] += 1
            _record_archive(metrics)
        stats["decayed"] += 1
        updates.append((new_importance, json.dumps(meta), row["memory_id"]))

    if not dry_run and updates:
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                UPDATE memory.memories
                SET importance = $1,
                    metadata = $2::jsonb,
                    last_decayed_at = now(),
                    updated_at = now()
                WHERE memory_id = $3
                """,
                updates,
            )


def _decode_jsonb(value: object) -> dict:
    """Return a mutable dict from an asyncpg jsonb column.

    asyncpg returns jsonb as a JSON string unless a codec is registered, so
    decode strings explicitly and copy dicts to avoid mutating shared state.
    """
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _record_archive(metrics: object | None) -> None:
    if metrics is not None and hasattr(metrics, "record_decay_archive"):
        try:
            metrics.record_decay_archive()
        except Exception:  # noqa: BLE001
            logger.debug("metrics.record_decay_archive failed", exc_info=True)


def _days_since(last_accessed: datetime, now: datetime) -> float:
    last = last_accessed
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last).total_seconds() / 86400.0


async def apply_decay_with_config(
    provider: MemoryProvider,
    config: DecayConfig,
    user_id: uuid.UUID | None = None,
    dry_run: bool = False,
    metrics: object | None = None,
) -> dict:
    """Run decay using a DecayConfig object. Thin wrapper over apply_decay."""
    return await apply_decay(
        provider=provider,
        user_id=user_id,
        archive_threshold=config.archival_threshold,
        archive_after_days=config.archival_age_days_min,
        dry_run=dry_run,
        metrics=metrics,
    )
