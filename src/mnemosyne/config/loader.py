from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mnemosyne.config.settings import MnemosyneConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | str | None = None) -> MnemosyneConfig:
    """Load a MnemosyneConfig from YAML.

    If `path` is None, the repo-root `config/default.yaml` is used. If that
    file is missing, defaults from MnemosyneConfig are returned.
    """
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if source.exists():
        with source.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"config file {source} must contain a top-level mapping"
            )
        data = loaded

    defaults = MnemosyneConfig().model_dump()
    merged = _deep_merge(defaults, data)
    return MnemosyneConfig.model_validate(merged)
