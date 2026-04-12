from __future__ import annotations

import logging
from pathlib import Path

from mnemosyne.rules.base_extractor import BaseExtractor
from mnemosyne.rules.yaml_extractor import load_yaml_rules

logger = logging.getLogger(__name__)


class RuleLoader:
    """Discovers and loads rule extractors from a directory of YAML files.

    Malformed YAML files are logged as warnings and skipped — they never cause
    the loader to raise.  An empty or non-existent directory returns ``[]``.
    """

    def load_from_directory(self, path: Path) -> list[BaseExtractor]:
        """Load all YAML rules from *path*.

        Parameters
        ----------
        path:
            Directory to scan.  Returns ``[]`` if it does not exist or is not
            a directory.

        Returns
        -------
        list[BaseExtractor]
            All ``YamlRuleExtractor`` instances loaded from ``.yaml`` files
            found directly inside *path* (non-recursive).
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

        return extractors

    def load_from_paths(self, paths: list[Path]) -> list[BaseExtractor]:
        """Load rules from multiple directories and merge results."""
        all_extractors: list[BaseExtractor] = []
        for path in paths:
            all_extractors.extend(self.load_from_directory(path))
        return all_extractors
