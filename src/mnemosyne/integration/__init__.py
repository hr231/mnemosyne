from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mnemosyne.integration.bootstrap import (
        BootstrapContext,
        bootstrap_memory_subsystem,
    )
    from mnemosyne.monitoring.prometheus_exporter import PrometheusExporter


__all__ = ["BootstrapContext", "bootstrap_memory_subsystem", "PrometheusExporter"]


def __getattr__(name: str) -> Any:
    if name in ("BootstrapContext", "bootstrap_memory_subsystem"):
        from mnemosyne.integration.bootstrap import (
            BootstrapContext,
            bootstrap_memory_subsystem,
        )
        return {
            "BootstrapContext": BootstrapContext,
            "bootstrap_memory_subsystem": bootstrap_memory_subsystem,
        }[name]
    if name == "PrometheusExporter":
        from mnemosyne.monitoring.prometheus_exporter import PrometheusExporter
        return PrometheusExporter
    raise AttributeError(f"module 'mnemosyne.integration' has no attribute {name!r}")
