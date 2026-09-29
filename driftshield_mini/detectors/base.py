"""Base class for all drift detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from driftshield.models import BaselineStats, DriftEvent, TraceEvent
from driftshield.storage import TraceStore


class BaseDetector(ABC):
    """Abstract base class for drift detectors."""

    def __init__(self, store: TraceStore, enabled: bool = True):
        self.store = store
        self.enabled = enabled

    @abstractmethod
    def check(
        self,
        event: TraceEvent,
        baseline: BaselineStats | None,
    ) -> DriftEvent | None:
        """
        Evaluate a new trace event against the baseline.
        Returns a DriftEvent if drift is detected, None otherwise.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...
