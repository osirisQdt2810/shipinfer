"""The metrics-exporter contract."""

from __future__ import annotations

import abc
from typing import ClassVar

from shipinfer.core.metrics.registry import MetricsRegistry

__all__ = ["MetricsExporter"]


class MetricsExporter(abc.ABC):
    """Renders a :class:`MetricsRegistry` into some wire format."""

    name: ClassVar[str] = "abstract"
    content_type: ClassVar[str] = "text/plain"

    @abc.abstractmethod
    def render(self, registry: MetricsRegistry) -> str: ...
