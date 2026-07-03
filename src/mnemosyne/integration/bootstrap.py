from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mnemosyne.config.loader import load_config
from mnemosyne.config.startup_checks import (
    Check,
    StartupCheckReport,
    run_startup_checks,
)

if TYPE_CHECKING:
    from mnemosyne.config.settings import MnemosyneConfig
    from mnemosyne.monitoring.prometheus_exporter import PrometheusExporter

logger = logging.getLogger(__name__)


@dataclass
class BootstrapContext:
    """Result of bootstrapping the memory subsystem.

    ``worker`` is the running background pipeline worker when one was passed
    to :func:`bootstrap_memory_subsystem`. The host owns its lifecycle and
    MUST call ``await worker.stop(timeout=...)`` during its own shutdown to
    drain the hook queue and finish in-flight processing before exit. The
    exporter, when present, should be stopped by the host as well.
    """

    report: StartupCheckReport
    exporter: "PrometheusExporter | None"
    worker: Any = None


async def bootstrap_memory_subsystem(
    checks: list[Check],
    start_exporter: bool = True,
    fail_fast: bool | None = None,
    worker: Any = None,
    config: "MnemosyneConfig | None" = None,
) -> BootstrapContext:
    """Run startup checks, optionally start the exporter and the worker.

    Configuration is layered: the YAML ``MnemosyneConfig`` (``config/default.yaml``
    unless a ``config`` is injected) supplies base values, and environment
    variables always override them. Specifically:

    - ``fail_fast`` resolution: the explicit argument wins; otherwise the
      ``MNEMOSYNE_STARTUP_WARN_ONLY`` env var wins ("1" ⇒ warn only); otherwise
      ``config.startup.fail_fast`` is used.
    - The exporter is started when ``start_exporter`` is True and metrics are
      enabled. ``MNEMOSYNE_METRICS_ENABLED`` overrides ``config.monitoring.enabled``.
      The listen port comes from ``config.monitoring.port`` unless
      ``MNEMOSYNE_METRICS_PORT`` is set. prometheus_client is imported lazily so
      the integration package can be loaded without it installed.

    When a ``worker`` (a PipelineWorker) is provided, it is started only after
    the startup checks pass, and returned on the context for the host to own.

    Shutdown contract
    -----------------
    The host is responsible for an orderly shutdown of everything returned on
    the :class:`BootstrapContext`. On host shutdown it should, in order:

    1. ``await ctx.worker.stop(timeout=...)`` if a worker is present — this
       cancels the poll/maintenance loops, drains the hook queue into durable
       pending rows, and lets in-flight processing finish.
    2. Stop the exporter if one is present.

    A failed startup check raises before the worker is started, so there is
    nothing to tear down in that case.
    """
    if config is None:
        config = load_config()

    if fail_fast is None:
        warn_only = os.environ.get("MNEMOSYNE_STARTUP_WARN_ONLY")
        if warn_only is not None:
            fail_fast = warn_only != "1"
        else:
            fail_fast = config.startup.fail_fast

    report = await run_startup_checks(checks=checks, fail_fast=fail_fast)

    concurrency_env = os.environ.get("MNEMOSYNE_LLM_MAX_CONCURRENCY")
    if concurrency_env is not None:
        from mnemosyne.llm.base import configure_llm_concurrency

        configure_llm_concurrency(int(concurrency_env))

    exporter: "PrometheusExporter | None" = None
    metrics_env = os.environ.get("MNEMOSYNE_METRICS_ENABLED")
    metrics_enabled = (
        metrics_env == "1" if metrics_env is not None else config.monitoring.enabled
    )
    if start_exporter and metrics_enabled:
        from mnemosyne.monitoring.metrics import global_registry
        from mnemosyne.monitoring.prometheus_exporter import PrometheusExporter

        port = (
            None
            if os.environ.get("MNEMOSYNE_METRICS_PORT") is not None
            else config.monitoring.port
        )
        exporter = PrometheusExporter(global_registry(), port=port)
        exporter.serve_in_background()

    if worker is not None:
        await worker.start()

    return BootstrapContext(report=report, exporter=exporter, worker=worker)
