"""Registry for extraction-logic versions.

Computes a deterministic changeset hash over rule YAMLs, prompt files, and
(optionally) Python extractor modules. When the hash differs from the most
recent row in ``memory.extraction_versions``, a new version row is inserted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

_VERSIONED_SUFFIXES = {".yaml", ".yml", ".txt", ".md", ".jinja", ".j2"}


def discover_versioned_files(
    rules_dir: Path,
    prompts_dir: Path,
    extractor_dir: Path | None = None,
) -> list[Path]:
    """Return every rule, prompt, and (optional) Python extractor file whose
    content should be hashed into the extraction-logic version.

    Args:
        rules_dir: Directory containing rule YAMLs.
        prompts_dir: Directory containing prompt templates.
        extractor_dir: Optional directory containing Python extractor modules.

    Returns:
        A list of absolute paths. Missing directories are ignored silently so
        this is safe to call during startup even before any rules exist.
    """
    found: list[Path] = []
    for base in (rules_dir, prompts_dir):
        if base is None or not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in _VERSIONED_SUFFIXES:
                found.append(path)
    if extractor_dir and extractor_dir.exists():
        for path in sorted(extractor_dir.rglob("*.py")):
            if path.is_file():
                found.append(path)
    return found


def compute_changeset_hash(files: list[Path]) -> str:
    """Compute a stable SHA-256 over ``(file_name, sha256(contents))`` pairs.

    The result is order-independent (files are sorted by name before hashing)
    and content-sensitive (any change to any file changes the digest).
    """
    items: list[tuple[str, str]] = []
    for p in files:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        items.append((p.name, digest))
    items.sort()
    h = hashlib.sha256()
    for name, digest in items:
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class ExtractionVersionRecord(BaseModel):
    """A single row in ``memory.extraction_versions``."""

    version: str
    changeset_hash: str
    changed_at: datetime
    changed_by: str
    summary: str
    rule_files: list[str]
    prompt_files: list[str]


@dataclass
class VersionRegistry:
    """Reads and writes extraction-version rows.

    When ``pool`` is ``None`` the registry operates in-memory only — useful
    for unit tests and offline hash computation.
    """

    pool: object | None = None

    async def latest(self) -> ExtractionVersionRecord | None:
        """Return the most recently registered version, or None if the table is empty."""
        if self.pool is None:
            return None
        async with self.pool.acquire() as conn:  # type: ignore[attr-defined]
            row = await conn.fetchrow(
                """
                SELECT version, changeset_hash, changed_at, changed_by, summary,
                       rule_files, prompt_files
                FROM memory.extraction_versions
                ORDER BY changed_at DESC
                LIMIT 1
                """
            )
        if row is None:
            return None
        return _row_to_record(row)

    async def get_by_hash(self, changeset_hash: str) -> ExtractionVersionRecord | None:
        if self.pool is None:
            return None
        async with self.pool.acquire() as conn:  # type: ignore[attr-defined]
            row = await conn.fetchrow(
                """
                SELECT version, changeset_hash, changed_at, changed_by, summary,
                       rule_files, prompt_files
                FROM memory.extraction_versions
                WHERE changeset_hash = $1
                """,
                changeset_hash,
            )
        if row is None:
            return None
        return _row_to_record(row)

    async def register_if_new(
        self,
        rules_dir: Path,
        prompts_dir: Path,
        changed_by: str = "auto",
        summary: str = "automatic version bump",
    ) -> ExtractionVersionRecord | None:
        """Register a new row iff the changeset hash differs from the latest one.

        Returns the existing record when the hash is unchanged so callers can
        treat the return value as "the version currently in effect".
        """
        files = discover_versioned_files(rules_dir=rules_dir, prompts_dir=prompts_dir)
        digest = compute_changeset_hash(files)

        existing_same_hash = await self.get_by_hash(digest)
        if existing_same_hash is not None:
            return existing_same_hash

        latest = await self.latest()
        now = datetime.now(timezone.utc)
        next_version = _next_version(latest.version if latest else None)

        rule_files = [
            str(p.relative_to(rules_dir))
            for p in files
            if rules_dir in p.parents or p == rules_dir
        ]
        prompt_files = [
            str(p.relative_to(prompts_dir))
            for p in files
            if prompts_dir in p.parents or p == prompts_dir
        ]

        record = ExtractionVersionRecord(
            version=next_version,
            changeset_hash=digest,
            changed_at=now,
            changed_by=changed_by,
            summary=summary,
            rule_files=rule_files,
            prompt_files=prompt_files,
        )

        if self.pool is None:
            return record

        async with self.pool.acquire() as conn:  # type: ignore[attr-defined]
            await conn.execute(
                """
                INSERT INTO memory.extraction_versions
                    (version, changeset_hash, changed_at, changed_by, summary,
                     rule_files, prompt_files)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
                """,
                record.version,
                record.changeset_hash,
                record.changed_at,
                record.changed_by,
                record.summary,
                json.dumps(record.rule_files),
                json.dumps(record.prompt_files),
            )
        return record


def _row_to_record(row) -> ExtractionVersionRecord:
    raw_rules = row["rule_files"]
    raw_prompts = row["prompt_files"]
    if isinstance(raw_rules, str):
        raw_rules = json.loads(raw_rules)
    if isinstance(raw_prompts, str):
        raw_prompts = json.loads(raw_prompts)
    return ExtractionVersionRecord(
        version=row["version"],
        changeset_hash=row["changeset_hash"],
        changed_at=row["changed_at"],
        changed_by=row["changed_by"],
        summary=row["summary"],
        rule_files=list(raw_rules or []),
        prompt_files=list(raw_prompts or []),
    )


def _next_version(previous: str | None) -> str:
    """Bump the patch segment of ``previous`` (semver) or start at 1.0.0."""
    if not previous:
        return "1.0.0"
    try:
        major, minor, patch = (int(x) for x in previous.split("."))
    except (ValueError, TypeError):
        return "1.0.0"
    return f"{major}.{minor}.{patch + 1}"
