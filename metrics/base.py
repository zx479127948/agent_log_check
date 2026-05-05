"""Base class for metric detectors."""

from abc import ABC, abstractmethod
from parser import LogSession


class BaseMetric(ABC):
    name: str = ""
    category: str = ""  # resource / tool / thinking / content

    @abstractmethod
    def compute(self, session: LogSession) -> dict:
        """Compute metrics for a session. Returns {metric_key: value}."""
