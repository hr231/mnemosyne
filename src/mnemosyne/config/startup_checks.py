from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class StartupCheckFailed(RuntimeError):
    """Raised when a required startup check fails and fail_fast is enabled."""


@dataclass
class CheckResult:
    """Outcome of a single startup probe."""

    name: str
    passed: bool
    message: str


@dataclass
class StartupCheckReport:
    """Aggregated outcome of a startup-check run."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def any_failed(self) -> bool:
        return any(not r.passed for r in self.results)

    def format(self) -> str:
        lines = []
        for r in self.results:
            status = "OK" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name}: {r.message}")
        return "\n".join(lines)


async def check_env_vars(required: list[str]) -> CheckResult:
    """Verify that every required environment variable is set and non-empty."""
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        return CheckResult(
            name="env_vars",
            passed=False,
            message=f"missing: {', '.join(missing)}",
        )
    return CheckResult(
        name="env_vars", passed=True, message="all required vars present"
    )


async def check_pg_reachable(dsn: str, timeout_seconds: float = 5.0) -> CheckResult:
    """Connect to PostgreSQL and verify pgvector + memory schema exist."""
    try:
        import asyncpg

        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout_seconds)
        try:
            ext = await conn.fetchval(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
            schema = await conn.fetchval(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'memory'"
            )
        finally:
            await conn.close()
        if not ext:
            return CheckResult(
                name="pg_reachable",
                passed=False,
                message="pgvector extension not installed",
            )
        if not schema:
            return CheckResult(
                name="pg_reachable",
                passed=False,
                message="memory schema missing (run migrations)",
            )
        return CheckResult(
            name="pg_reachable", passed=True, message="OK, pgvector loaded"
        )
    except Exception as e:
        return CheckResult(name="pg_reachable", passed=False, message=str(e))


async def check_embedding_dim(embedder: Any, expected_dim: int) -> CheckResult:
    """Embed a probe string and verify vector dimensionality matches expectation."""
    try:
        vec = await embedder.embed("startup-probe")
        dim = len(vec)
        if dim != expected_dim:
            return CheckResult(
                name="embedding_dim",
                passed=False,
                message=f"embedder returned dim={dim}, expected {expected_dim}",
            )
        return CheckResult(name="embedding_dim", passed=True, message=f"dim={dim}")
    except Exception as e:
        return CheckResult(name="embedding_dim", passed=False, message=str(e))


async def check_llm_reachable(llm: Any, timeout_seconds: float = 10.0) -> CheckResult:
    """Send a small prompt to the LLM client and verify it returns a string."""
    try:
        resp = await asyncio.wait_for(llm.complete("ping"), timeout=timeout_seconds)
        if not isinstance(resp, str):
            return CheckResult(
                name="llm_reachable",
                passed=False,
                message=f"non-string response: {type(resp).__name__}",
            )
        return CheckResult(name="llm_reachable", passed=True, message="LLM responded")
    except Exception as e:
        return CheckResult(name="llm_reachable", passed=False, message=str(e))


Check = Callable[[], Awaitable[CheckResult]]


async def run_startup_checks(
    checks: list[Check], fail_fast: bool = True
) -> StartupCheckReport:
    """Run each zero-arg async check, aggregate results, optionally raise on failure."""
    report = StartupCheckReport()
    for check in checks:
        result = await check()
        report.results.append(result)
        if not result.passed:
            logger.warning(
                "Startup check failed: %s — %s", result.name, result.message
            )

    if report.any_failed and fail_fast:
        raise StartupCheckFailed(report.format())
    return report
