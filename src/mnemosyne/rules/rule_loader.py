from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path

from mnemosyne.rules.base_extractor import BaseExtractor
from mnemosyne.rules.yaml_extractor import load_yaml_rules

logger = logging.getLogger(__name__)


class RuleLoader:
    """Discovers and loads rule extractors from a directory.

    Supports two source formats:

    - **YAML** (``.yaml`` files): declarative rules; each file may contain
      multiple rule definitions.  Loaded via ``load_yaml_rules``.
    - **Python plugins** (``.py`` files): any class that subclasses
      ``BaseExtractor`` and has a non-empty ``id`` attribute is discovered
      and instantiated automatically.

    Both types of malformed files are logged as warnings and skipped — they
    never cause the loader to raise.  An empty or non-existent directory
    returns ``[]``.
    """

    def load_from_directory(self, path: Path) -> list[BaseExtractor]:
        """Load all YAML rules and Python plugins from *path*.

        Parameters
        ----------
        path:
            Directory to scan.  Returns ``[]`` if it does not exist or is not
            a directory.

        Returns
        -------
        list[BaseExtractor]
            All extractors found in ``.yaml`` and ``.py`` files directly
            inside *path* (non-recursive).  YAML extractors come first,
            followed by Python plugin extractors (both sorted by filename).
        """
        if not path.exists() or not path.is_dir():
            return []

        extractors: list[BaseExtractor] = []

        for yaml_file in sorted(path.glob("*.yaml")):
            try:
                loaded = load_yaml_rules(yaml_file)
                extractors.extend(loaded)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping malformed YAML rule file %s: %s",
                    yaml_file,
                    exc,
                )

        extractors.extend(self._load_python_plugins(path))
        return extractors

    def load_from_paths(self, paths: list[Path]) -> list[BaseExtractor]:
        """Load rules from multiple directories and merge results."""
        all_extractors: list[BaseExtractor] = []
        for path in paths:
            all_extractors.extend(self.load_from_directory(path))
        return all_extractors

    def _load_python_plugins(self, path: Path) -> list[BaseExtractor]:
        """Discover ``BaseExtractor`` subclasses in ``.py`` files under *path*.

        Files whose names start with ``_`` (e.g. ``__init__.py``) are skipped.
        Classes that cannot be instantiated with zero arguments are skipped
        with a warning.  Any import error is also caught and warned.

        Returns
        -------
        list[BaseExtractor]
            Instantiated extractors, in filename-sorted order.
        """
        extractors: list[BaseExtractor] = []

        for py_file in sorted(path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec is None or spec.loader is None:
                    logger.warning("Could not create module spec for %s — skipping", py_file)
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]

                for _name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj is BaseExtractor:
                        continue
                    if not issubclass(obj, BaseExtractor):
                        continue
                    # Only instantiate classes defined in this module
                    if obj.__module__ != module.__name__:
                        continue
                    try:
                        instance = obj()
                        if instance.id:
                            extractors.append(instance)
                        else:
                            logger.warning(
                                "Python plugin class %s in %s has no id — skipping",
                                obj.__name__,
                                py_file,
                            )
                    except TypeError as exc:
                        logger.warning(
                            "Cannot instantiate plugin %s from %s (requires args?): %s",
                            obj.__name__,
                            py_file,
                            exc,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping Python plugin %s: %s", py_file, exc)

        return extractors
