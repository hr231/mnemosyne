from __future__ import annotations

from abc import ABC, abstractmethod

from mnemosyne.db.models.memory import ExtractionResult, MemoryType


class BaseExtractor(ABC):
    """Base class for all memory extractors.

    Every extractor — whether backed by a YAML rule file or a Python
    plugin class — must subclass ``BaseExtractor`` and implement
    ``extract()``.

    Class-level attributes
    ----------------------
    id : str
        Unique rule identifier.  Set by each concrete subclass.
    category : MemoryType
        The ``MemoryType`` tag applied to every result this extractor
        produces.
    importance : float
        Default importance score (0.0–1.0) stamped on results.
    enabled : bool
        When ``False``, ``RuleRegistry`` (and ``ExtractionPipeline``)
        skip this extractor entirely without raising an error.
    version : int
        Schema version of the rule definition.  Incremented when the
        rule logic changes in a backward-incompatible way.

    Subclasses that do not implement ``extract`` will raise
    ``TypeError`` at instantiation time (enforced by ``ABC``).
    Subclasses may override any class-level attribute on the class body
    or in ``__init__``.
    """

    id: str = ""
    category: MemoryType = MemoryType.FACT
    importance: float = 0.5
    enabled: bool = True
    version: int = 1

    @abstractmethod
    def extract(self, text: str) -> list[ExtractionResult]:
        """Run this rule on *text* and return zero or more results."""
        ...
