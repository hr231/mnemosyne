from __future__ import annotations

import logging
import os
from threading import Thread
from typing import Iterable, Optional

from prometheus_client import CollectorRegistry, generate_latest, start_http_server
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)
from prometheus_client.registry import Collector

from mnemosyne.monitoring.metrics import HistogramSnapshot, MetricsRegistry

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PORT = 9464
DEFAULT_METRICS_ADDR = "127.0.0.1"


def _histogram_family(
    name: str, documentation: str, hist: HistogramSnapshot
) -> HistogramMetricFamily:
    buckets: list[tuple[str, float]] = [
        (repr(float(upper)), float(count))
        for upper, count in zip(hist.buckets, hist.cumulative_counts)
    ]
    buckets.append(("+Inf", float(hist.total_count)))
    return HistogramMetricFamily(
        name, documentation, buckets=buckets, sum_value=hist.total_sum
    )


class _SnapshotCollector(Collector):
    """prometheus_client Collector that reads a MetricsRegistry snapshot.

    Reading happens lazily inside collect(), so every scrape reflects the
    registry state at scrape time rather than at exporter-construction time.
    """

    def __init__(self, registry: MetricsRegistry) -> None:
        self._registry = registry

    def collect(self) -> Iterable:
        snap = self._registry.snapshot()

        yield CounterMetricFamily(
            "mnemosyne_extraction",
            "Number of extraction attempts since process start",
            value=snap.extraction_total,
        )
        yield CounterMetricFamily(
            "mnemosyne_extraction_failed",
            "Number of extraction failures since process start",
            value=snap.extraction_failed,
        )
        yield CounterMetricFamily(
            "mnemosyne_processing_failed",
            "Number of processing_log rows that permanently failed since process start",
            value=snap.processing_failed_total,
        )
        yield GaugeMetricFamily(
            "mnemosyne_processing_failed_backlog",
            "Current number of failed processing_log rows awaiting operator action",
            value=snap.processing_failed_backlog,
        )
        yield _histogram_family(
            "mnemosyne_extraction_latency_ms",
            "Extraction latency distribution in milliseconds",
            snap.extraction_latency,
        )
        yield _histogram_family(
            "mnemosyne_retrieval_latency_ms",
            "Retrieval latency distribution in milliseconds",
            snap.retrieval_latency,
        )
        yield GaugeMetricFamily(
            "mnemosyne_pipeline_lag_seconds",
            "Age of the oldest pending processing_log row, in seconds",
            value=snap.pipeline_lag_seconds,
        )
        yield GaugeMetricFamily(
            "mnemosyne_queue_depth",
            "Number of pending sessions in processing_log",
            value=snap.queue_depth,
        )
        yield CounterMetricFamily(
            "mnemosyne_dedup_merges",
            "Total dedup merges since process start",
            value=snap.dedup_merges,
        )
        yield CounterMetricFamily(
            "mnemosyne_decay_archived",
            "Total memories archived by decay since process start",
            value=snap.decay_archived,
        )
        yield GaugeMetricFamily(
            "mnemosyne_session_queue_depth",
            "Number of session-close events buffered in the in-process hook queue",
            value=snap.session_queue_depth,
        )
        yield CounterMetricFamily(
            "mnemosyne_session_dlq",
            "Total session-close events that exceeded max_retries and were dead-lettered",
            value=snap.session_dlq_total,
        )
        yield CounterMetricFamily(
            "mnemosyne_hook_failures",
            "Total session-close hook invocations that failed before enqueue/dispatch",
            value=snap.hook_failures_total,
        )
        yield CounterMetricFamily(
            "mnemosyne_llm_calls",
            "Total outbound LLM API calls since process start",
            value=snap.llm_calls_total,
        )
        yield CounterMetricFamily(
            "mnemosyne_llm_tokens",
            "Total tokens consumed by outbound LLM API calls since process start",
            value=snap.llm_tokens_total,
        )
        yield GaugeMetricFamily(
            "mnemosyne_last_poll_timestamp_seconds",
            "Unix time of the most recent worker poll (0 if never polled)",
            value=snap.last_poll_timestamp_seconds,
        )
        yield GaugeMetricFamily(
            "mnemosyne_last_maintenance_timestamp_seconds",
            "Unix time of the most recent maintenance sweep (0 if never run)",
            value=snap.last_maintenance_timestamp_seconds,
        )

        stage_success = CounterMetricFamily(
            "mnemosyne_stage_success",
            "Per-stage successful outcomes since process start",
            labels=["stage"],
        )
        for stage, count in sorted(snap.stage_success.items()):
            stage_success.add_metric([stage], count)
        yield stage_success

        stage_failed = CounterMetricFamily(
            "mnemosyne_stage_failed",
            "Per-stage failed outcomes since process start",
            labels=["stage"],
        )
        for stage, count in sorted(snap.stage_failed.items()):
            stage_failed.add_metric([stage], count)
        yield stage_failed

        yield CounterMetricFamily(
            "mnemosyne_context_assembly_inject",
            "Total context-assembly calls that fit within the token budget",
            value=snap.context_assembly_inject_total,
        )
        yield CounterMetricFamily(
            "mnemosyne_context_assembly_truncate",
            "Total context-assembly calls that hit the token budget cap",
            value=snap.context_assembly_truncate_total,
        )
        yield GaugeMetricFamily(
            "mnemosyne_context_assembly_token_utilization",
            "Last context-assembly token-budget utilization ratio (0..1)",
            value=snap.context_assembly_token_utilization,
        )


class PrometheusExporter:
    """Snapshot-based exporter backed by a custom Collector.

    The collector reads the MetricsRegistry at scrape time, so render() and the
    background HTTP server always reflect current registry state.
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        port: Optional[int] = None,
        addr: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self.port = (
            port
            if port is not None
            else int(os.environ.get("MNEMOSYNE_METRICS_PORT", str(DEFAULT_METRICS_PORT)))
        )
        self.addr = (
            addr
            if addr is not None
            else os.environ.get("MNEMOSYNE_METRICS_ADDR", DEFAULT_METRICS_ADDR)
        )
        self._prom = CollectorRegistry()
        self._collector = _SnapshotCollector(registry)
        self._prom.register(self._collector)

        self._server = None
        self._server_thread: Thread | None = None

    def render(self) -> str:
        """Render a Prometheus text-format exposition from the current snapshot."""
        return generate_latest(self._prom).decode("utf-8")

    def serve_in_background(self) -> None:
        """Start the Prometheus HTTP server on a background thread (idempotent)."""
        if self._server_thread is not None:
            logger.warning(
                "Prometheus exporter already running on %s:%s", self.addr, self.port
            )
            return
        server, thread = start_http_server(
            self.port, addr=self.addr, registry=self._prom
        )
        self._server = server
        self._server_thread = thread
        logger.info(
            "Prometheus metrics exposed at %s:%s/metrics", self.addr, self.port
        )

    def stop(self) -> None:
        """Shut down the background HTTP server if it is running (idempotent)."""
        server = self._server
        if server is None:
            return
        try:
            server.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("Prometheus exporter shutdown failed", exc_info=True)
        finally:
            self._server = None
            self._server_thread = None
            logger.info("Prometheus exporter stopped")
