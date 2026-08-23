"""Registry of metrics exporters."""

from __future__ import annotations

from shipinfer.core.metrics.exporters.base import MetricsExporter
from shipinfer.core.registry import Registry

__all__ = ["EXPORTERS"]

EXPORTERS: Registry[MetricsExporter] = Registry("metrics exporter", MetricsExporter)
