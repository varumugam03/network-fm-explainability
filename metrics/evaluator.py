from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class Evaluator(ABC):
    """
    Base evaluator interface.

    Concrete evaluators should extend this and implement reset, update and compute.
    This allows passing evaluator objects around with a common type for typing and
    easier testing.
    """

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal accumulators."""

    @abstractmethod
    def update(self, predictions, targets) -> None:
        """Consume a batch of predictions and targets to update metrics."""

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Return a dictionary of computed metrics for the accumulated data."""
