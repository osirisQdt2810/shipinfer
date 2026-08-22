"""Prometheus text exposition format."""

from __future__ import annotations

from shipinfer.core.metrics.base import render_labels
from shipinfer.core.metrics.exporters.base import MetricsExporter
from shipinfer.core.metrics.exporters.registry import EXPORTERS
from shipinfer.core.metrics.registry import MetricsRegistry

__all__ = ["PrometheusExporter"]


@EXPORTERS.register("prometheus", "prom")
class PrometheusExporter(MetricsExporter):
    """Text exposition format v0.0.4 — what a Prometheus scrape expects."""

    name = "prometheus"
    content_type = "text/plain; version=0.0.4; charset=utf-8"

    def render(self, registry: MetricsRegistry) -> str:
        lines: list[str] = []
        for metric in registry.collect():
            lines.append(f"# HELP {metric.name} {metric.help}")
            lines.append(f"# TYPE {metric.name} {metric.kind}")
            for sample_name, labels, value in metric.samples():
                lines.append(f"{sample_name}{render_labels(labels)} {value:g}")
        return "\n".join(lines) + "\n"
