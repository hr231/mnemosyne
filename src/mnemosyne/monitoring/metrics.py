from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock

DEFAULT_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
)


@dataclass(frozen=True)
class HistogramSnapshot:
    """Cumulative histogram view suitable for Prometheus exposition.

    ``cumulative_counts[i]`` is the number of observations less than or equal
    to ``buckets[i]``; the counts are already cumulative because the buckets
    are sorted ascending. ``total_count`` is the ``+Inf`` bucket value.
    """

    buckets: tuple[float, ...]
    cumulative_counts: tuple[int, ...]
    total_count: int
    total_sum: float


class _HistogramState:
    """Thread-agnostic cumulative histogram over a latency series in ms.

    Callers must hold an external lock; the registry serializes access.
    """

    __slots__ = ("_buckets", "_counts", "_sum", "_count")

    def __init__(self, buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_MS) -> None:
        self._buckets = buckets
        self._counts = [0] * len(buckets)
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float) -> None:
        self._count += 1
        self._sum += value
        for i, upper in enumerate(self._buckets):
            if value <= upper:
                self._counts[i] += 1

    def snapshot(self) -> HistogramSnapshot:
        return HistogramSnapshot(
            buckets=self._buckets,
            cumulative_counts=tuple(self._counts),
            total_count=self._count,
            total_sum=self._sum,
        )


@dataclass(frozen=True)
class MetricsSnapshot:
    """Point-in-time view of runtime metric counters, gauges, and histograms."""

    extraction_total: int
    extraction_failed: int
    extraction_success_rate: float
    extraction_latency: HistogramSnapshot
    retrieval_latency: HistogramSnapshot
    processing_failed_total: int
    processing_failed_backlog: int
    pipeline_lag_seconds: float
    queue_depth: int
    dedup_merges: int
    decay_archived: int
    session_queue_depth: int
    session_dlq_total: int
    hook_failures_total: int
    llm_calls_total: int
    llm_tokens_total: int
    last_poll_timestamp_seconds: float
    last_maintenance_timestamp_seconds: float
    stage_success: dict[str, int]
    stage_failed: dict[str, int]
    context_assembly_inject_total: int
    context_assembly_truncate_total: int
    context_assembly_token_utilization: float


@dataclass
class MetricsRegistry:
    """Thread-safe in-process metrics registry.

    Records extraction and processing outcomes, retrieval/extraction latency
    histograms, pipeline lag, queue depth, dedup merges, decay archival,
    session hook queue state, hook and LLM activity, worker liveness
    timestamps, per-stage outcomes, and context-assembly state. Exposes a
    frozen snapshot for scrape endpoints to read without racing.
    """

    _lock: Lock = field(default_factory=Lock)
    _extraction_total: int = 0
    _extraction_failed: int = 0
    _extraction_latency: _HistogramState = field(default_factory=_HistogramState)
    _retrieval_latency: _HistogramState = field(default_factory=_HistogramState)
    _processing_failed_total: int = 0
    _processing_failed_backlog: int = 0
    _pipeline_lag: float = 0.0
    _queue_depth: int = 0
    _dedup: int = 0
    _decay_archived: int = 0
    _session_queue_depth: int = 0
    _session_dlq_total: int = 0
    _hook_failures_total: int = 0
    _llm_calls_total: int = 0
    _llm_tokens_total: int = 0
    _last_poll_ts: float = 0.0
    _last_maintenance_ts: float = 0.0
    _stage_success: dict[str, int] = field(default_factory=dict)
    _stage_failed: dict[str, int] = field(default_factory=dict)
    _context_inject: int = 0
    _context_truncate: int = 0
    _context_utilization: float = 0.0

    def record_extraction(self, success: bool, latency_ms: float) -> None:
        with self._lock:
            self._extraction_total += 1
            if not success:
                self._extraction_failed += 1
            self._extraction_latency.observe(latency_ms)

    def record_retrieval_latency_ms(self, ms: float) -> None:
        with self._lock:
            self._retrieval_latency.observe(ms)

    def record_processing_failed(self, count: int = 1) -> None:
        """Count processing_log rows that have permanently failed."""
        with self._lock:
            self._processing_failed_total += int(count)

    def set_processing_failed_backlog(self, count: int) -> None:
        """Set the current number of failed processing_log rows awaiting action."""
        with self._lock:
            self._processing_failed_backlog = int(count)

    def record_pipeline_lag_seconds(self, seconds: float) -> None:
        """Set the age of the oldest pending processing_log row, in seconds."""
        with self._lock:
            self._pipeline_lag = float(seconds)

    # Backward-compatible alias for callers that predate the semantics change.
    def set_pipeline_lag_seconds(self, seconds: float) -> None:
        self.record_pipeline_lag_seconds(seconds)

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

    def record_hook_failure(self) -> None:
        """Count a session-close hook that raised before enqueue/dispatch."""
        with self._lock:
            self._hook_failures_total += 1

    def record_llm_call(self, count: int = 1) -> None:
        """Count outbound LLM API calls."""
        with self._lock:
            self._llm_calls_total += int(count)

    def record_llm_tokens(self, tokens: int) -> None:
        """Count tokens consumed by outbound LLM API calls."""
        with self._lock:
            self._llm_tokens_total += int(tokens)

    def set_last_poll_timestamp(self, ts: float | None = None) -> None:
        """Record the wall-clock time of the most recent worker poll."""
        with self._lock:
            self._last_poll_ts = time.time() if ts is None else float(ts)

    def set_last_maintenance_timestamp(self, ts: float | None = None) -> None:
        """Record the wall-clock time of the most recent maintenance sweep."""
        with self._lock:
            self._last_maintenance_ts = time.time() if ts is None else float(ts)

    def record_stage(self, stage: str, success: bool) -> None:
        """Count a per-stage outcome (embedding/episode/contradiction/reflection)."""
        with self._lock:
            target = self._stage_success if success else self._stage_failed
            target[stage] = target.get(stage, 0) + 1

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
            return MetricsSnapshot(
                extraction_total=self._extraction_total,
                extraction_failed=self._extraction_failed,
                extraction_success_rate=rate,
                extraction_latency=self._extraction_latency.snapshot(),
                retrieval_latency=self._retrieval_latency.snapshot(),
                processing_failed_total=self._processing_failed_total,
                processing_failed_backlog=self._processing_failed_backlog,
                pipeline_lag_seconds=self._pipeline_lag,
                queue_depth=self._queue_depth,
                dedup_merges=self._dedup,
                decay_archived=self._decay_archived,
                session_queue_depth=self._session_queue_depth,
                session_dlq_total=self._session_dlq_total,
                hook_failures_total=self._hook_failures_total,
                llm_calls_total=self._llm_calls_total,
                llm_tokens_total=self._llm_tokens_total,
                last_poll_timestamp_seconds=self._last_poll_ts,
                last_maintenance_timestamp_seconds=self._last_maintenance_ts,
                stage_success=dict(self._stage_success),
                stage_failed=dict(self._stage_failed),
                context_assembly_inject_total=self._context_inject,
                context_assembly_truncate_total=self._context_truncate,
                context_assembly_token_utilization=self._context_utilization,
            )


_GLOBAL = MetricsRegistry()


def global_registry() -> MetricsRegistry:
    """Return the process-wide shared MetricsRegistry instance."""
    return _GLOBAL
