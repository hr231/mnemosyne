from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from mnemosyne.config.startup_checks import (
    Check,
    StartupCheckReport,
    run_startup_checks,
)
from mnemosyne.monitoring.metrics import global_registry
from mnemosyne.monitoring.prometheus_exporter import PrometheusExporter

logger = logging.getLogger(__name__)


@dataclass
class BootstrapContext:
    """Result of bootstrapping the memory subsystem."""

    report: StartupCheckReport
    exporter: PrometheusExporter | None


async def bootstrap_memory_subsystem(
    checks: list[Check],
    start_exporter: bool = True,
    fail_fast: bool | None = None,
) -> BootstrapContext:
    """Run startup checks and optionally start the Prometheus exporter.

    The fail_fast default is derived from MNEMOSYNE_STARTUP_WARN_ONLY when
    the argument is None: if the env var is "1", checks warn only and the
    report is returned even on failure; otherwise a failed check raises
    StartupCheckFailed.

    The exporter is only started when start_exporter=True and
    MNEMOSYNE_METRICS_ENABLED is not "0".
    """
    if fail_fast is None:
        fail_fast = os.environ.get("MNEMOSYNE_STARTUP_WARN_ONLY", "0") != "1"

    report = await run_startup_checks(checks=checks, fail_fast=fail_fast)

    exporter: PrometheusExporter | None = None
    if start_exporter and os.environ.get("MNEMOSYNE_METRICS_ENABLED", "1") == "1":
        exporter = PrometheusExporter(global_registry())
        exporter.serve_in_background()

    return BootstrapContext(report=report, exporter=exporter)
