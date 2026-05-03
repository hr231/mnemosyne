from __future__ import annotations

import logging
import os
from threading import Thread
from typing import Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest, start_http_server

from mnemosyne.monitoring.metrics import MetricsRegistry

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Snapshot-based exporter. Each render() call reads a MetricsRegistry snapshot."""

    def __init__(self, registry: MetricsRegistry, port: Optional[int] = None) -> None:
        self._registry = registry
        self.port = port if port is not None else int(
            os.environ.get("MNEMOSYNE_METRICS_PORT", "9090")
        )
        self._prom = CollectorRegistry()

        self._extraction_total = Counter(
            "mnemosyne_extraction_total",
            "Number of extraction attempts since process start",
            registry=self._prom,
        )
        self._extraction_failed = Counter(
            "mnemosyne_extraction_failed_total",
            "Number of extraction failures since process start",
            registry=self._prom,
        )
        self._retrieval_p50 = Gauge(
            "mnemosyne_retrieval_latency_p50_ms",
            "Retrieval latency p50 in milliseconds",
            registry=self._prom,
        )
        self._retrieval_p95 = Gauge(
            "mnemosyne_retrieval_latency_p95_ms",
            "Retrieval latency p95 in milliseconds",
            registry=self._prom,
        )
        self._retrieval_p99 = Gauge(
            "mnemosyne_retrieval_latency_p99_ms",
            "Retrieval latency p99 in milliseconds",
            registry=self._prom,
        )
        self._pipeline_lag = Gauge(
            "mnemosyne_pipeline_lag_seconds",
            "Seconds since last completed pipeline tick",
            registry=self._prom,
        )
        self._queue_depth = Gauge(
            "mnemosyne_queue_depth",
            "Number of pending sessions in processing_log",
            registry=self._prom,
        )
        self._dedup = Counter(
            "mnemosyne_dedup_merges_total",
            "Total dedup merges since process start",
            registry=self._prom,
        )
        self._decay = Counter(
            "mnemosyne_decay_archived_total",
            "Total memories archived by decay since process start",
            registry=self._prom,
        )
        self._server_thread: Thread | None = None

    def _sync_from_snapshot(self) -> None:
        snap = self._registry.snapshot()

        delta_total = snap.extraction_total - int(self._extraction_total._value.get())
        if delta_total > 0:
            self._extraction_total.inc(delta_total)

        delta_failed = snap.extraction_failed - int(self._extraction_failed._value.get())
        if delta_failed > 0:
            self._extraction_failed.inc(delta_failed)

        self._retrieval_p50.set(snap.retrieval_p50_ms)
        self._retrieval_p95.set(snap.retrieval_p95_ms)
        self._retrieval_p99.set(snap.retrieval_p99_ms)
        self._pipeline_lag.set(snap.pipeline_lag_seconds)
        self._queue_depth.set(snap.queue_depth)

        delta_dedup = snap.dedup_merges - int(self._dedup._value.get())
        if delta_dedup > 0:
            self._dedup.inc(delta_dedup)

        delta_decay = snap.decay_archived - int(self._decay._value.get())
        if delta_decay > 0:
            self._decay.inc(delta_decay)

    def render(self) -> str:
        """Render a Prometheus text-format exposition from the current snapshot."""
        self._sync_from_snapshot()
        return generate_latest(self._prom).decode("utf-8")

    def serve_in_background(self) -> None:
        """Start the Prometheus HTTP server on a background thread."""
        if self._server_thread is not None:
            logger.warning("Prometheus exporter already running on port %s", self.port)
            return
        start_http_server(self.port, registry=self._prom)
        logger.info("Prometheus metrics exposed at :%s/metrics", self.port)
