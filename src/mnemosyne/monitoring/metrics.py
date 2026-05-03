from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass(frozen=True)
class MetricsSnapshot:
    """Point-in-time view of runtime metric counters and gauges."""

    extraction_total: int
    extraction_failed: int
    extraction_success_rate: float
    retrieval_p50_ms: float
    retrieval_p95_ms: float
    retrieval_p99_ms: float
    pipeline_lag_seconds: float
    queue_depth: int
    dedup_merges: int
    decay_archived: int
    session_queue_depth: int
    session_dlq_total: int
    context_assembly_inject_total: int
    context_assembly_truncate_total: int
    context_assembly_token_utilization: float


@dataclass
class MetricsRegistry:
    """Thread-safe in-process metrics registry.

    Records extraction outcomes, retrieval latencies, pipeline lag,
    queue depth, dedup merges, decay archival, and session hook queue
    state. Exposes a frozen snapshot for scrape endpoints to read
    without racing.
    """

    _lock: Lock = field(default_factory=Lock)
    _extraction_total: int = 0
    _extraction_failed: int = 0
    _retrieval_latencies: list[float] = field(default_factory=list)
    _pipeline_lag: float = 0.0
    _queue_depth: int = 0
    _dedup: int = 0
    _decay_archived: int = 0
    _session_queue_depth: int = 0
    _session_dlq_total: int = 0
    _context_inject: int = 0
    _context_truncate: int = 0
    _context_utilization: float = 0.0

    def record_extraction(self, success: bool, latency_ms: float) -> None:
        with self._lock:
            self._extraction_total += 1
            if not success:
                self._extraction_failed += 1

    def record_retrieval_latency_ms(self, ms: float) -> None:
        with self._lock:
            self._retrieval_latencies.append(ms)
            if len(self._retrieval_latencies) > 10_000:
                self._retrieval_latencies = self._retrieval_latencies[-5_000:]

    def set_pipeline_lag_seconds(self, lag: float) -> None:
        with self._lock:
            self._pipeline_lag = lag

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = depth

    def record_dedup_merge(self) -> None:
        with self._lock:
            self._dedup += 1

    def record_decay_archive(self) -> None:
        with self._lock:
            self._decay_archived += 1

    def set_session_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._session_queue_depth = depth

    def record_session_dlq(self) -> None:
        with self._lock:
            self._session_dlq_total += 1

    def record_context_inject(self) -> None:
        with self._lock:
            self._context_inject += 1

    def record_context_truncate(self) -> None:
        with self._lock:
            self._context_truncate += 1

    def set_context_token_utilization(self, ratio: float) -> None:
        with self._lock:
            self._context_utilization = max(0.0, min(1.0, float(ratio)))

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            rate = (
                1.0 - (self._extraction_failed / self._extraction_total)
                if self._extraction_total
                else 1.0
            )
            p50 = p95 = p99 = 0.0
            if self._retrieval_latencies:
                sorted_l = sorted(self._retrieval_latencies)
                n = len(sorted_l)
                p50 = sorted_l[n // 2]
                p95 = sorted_l[min(n - 1, int(n * 0.95))]
                p99 = sorted_l[min(n - 1, int(n * 0.99))]
            return MetricsSnapshot(
                extraction_total=self._extraction_total,
                extraction_failed=self._extraction_failed,
                extraction_success_rate=rate,
                retrieval_p50_ms=p50,
                retrieval_p95_ms=p95,
                retrieval_p99_ms=p99,
                pipeline_lag_seconds=self._pipeline_lag,
                queue_depth=self._queue_depth,
                dedup_merges=self._dedup,
                decay_archived=self._decay_archived,
                session_queue_depth=self._session_queue_depth,
                session_dlq_total=self._session_dlq_total,
                context_assembly_inject_total=self._context_inject,
                context_assembly_truncate_total=self._context_truncate,
                context_assembly_token_utilization=self._context_utilization,
            )


_GLOBAL = MetricsRegistry()


def global_registry() -> MetricsRegistry:
    """Return the process-wide shared MetricsRegistry instance."""
    return _GLOBAL
