from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timedelta, timezone

from mnemosyne.db.models.memory import Memory
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)

# Default thresholds — all overridable via function parameters.
DEFAULT_ARCHIVE_THRESHOLD: float = 0.05
DEFAULT_ARCHIVE_AFTER_DAYS: int = 90


def compute_decayed_importance(memory: Memory, now: datetime | None = None) -> float:
    """Return the exponentially decayed importance for *memory*.

    Formula: ``importance * exp(-decay_rate * days_since_last_access)``

    The result is clamped to [0.0, 1.0].
    """
    if now is None:
        now = datetime.now(timezone.utc)

    last = memory.last_accessed
    # Ensure both datetimes are timezone-aware for comparison
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    days_since = max(0.0, (now - last).total_seconds() / 86400.0)
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
) -> None:
    """Decay + archive logic for InMemoryProvider."""
    memories: list[Memory] = [
        m
        for m in provider._memories.values()  # type: ignore[attr-defined]
        if m.valid_until is None
        and (user_id is None or m.user_id == user_id)
    ]

    for mem in memories:
        new_importance = compute_decayed_importance(mem, now)
        if new_importance == mem.importance:
            continue

        if not dry_run:
            await provider.update(mem.memory_id, importance=new_importance)
        stats["decayed"] += 1

        if should_archive(mem, archive_threshold, archive_after_days, now) or (
            new_importance < archive_threshold
            and _days_since(mem.last_accessed, now) > archive_after_days
        ):
            if not dry_run:
                current = provider._memories.get(mem.memory_id)  # type: ignore[attr-defined]
                if current:
                    current.metadata["archived"] = True
            stats["archived"] += 1


async def _apply_decay_postgres(
    provider: MemoryProvider,
    user_id: uuid.UUID | None,
    archive_threshold: float,
    archive_after_days: int,
    dry_run: bool,
    now: datetime,
    stats: dict,
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
            SELECT memory_id, importance, last_accessed, decay_rate, metadata
            FROM memory.memories
            WHERE valid_until IS NULL
            {user_filter}
            """,
            *params,
        )

    cutoff = now - timedelta(days=archive_after_days)

    updates: list[tuple[float, dict, uuid.UUID]] = []
    for row in rows:
        last = row["last_accessed"]
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days_since = max(0.0, (now - last).total_seconds() / 86400.0)
        new_importance = max(0.0, min(1.0, row["importance"] * math.exp(-row["decay_rate"] * days_since)))
        if new_importance == row["importance"]:
            continue

        meta = dict(row["metadata"] or {})
        do_archive = new_importance < archive_threshold and last < cutoff
        if do_archive:
            meta["archived"] = True
            stats["archived"] += 1
        stats["decayed"] += 1
        updates.append((new_importance, meta, row["memory_id"]))

    if not dry_run and updates:
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                UPDATE memory.memories
                SET importance = $1, metadata = $2::jsonb, updated_at = now()
                WHERE memory_id = $3
                """,
                updates,
            )


def _days_since(last_accessed: datetime, now: datetime) -> float:
    last = last_accessed
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last).total_seconds() / 86400.0
