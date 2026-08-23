"""One JSON object per metric sample — for log-based collection."""

from __future__ import annotations

import json

from shipinfer.core.metrics.exporters.base import MetricsExporter
from shipinfer.core.metrics.exporters.registry import EXPORTERS
from shipinfer.core.metrics.registry import MetricsRegistry

__all__ = ["JsonLinesExporter"]


@EXPORTERS.register("jsonl", "json_lines")
class JsonLinesExporter(MetricsExporter):
    """Line-delimited JSON.

    Useful where there is no Prometheus to scrape — an edge node that only ships logs can
    still get the same numbers out.
    """

    name = "jsonl"
    content_type = "application/x-ndjson"

    def render(self, registry: MetricsRegistry) -> str:
        out: list[str] = []
        for metric in registry.collect():
            for sample_name, labels, value in metric.samples():
                out.append(
                    json.dumps(
                        {
                            "metric": sample_name,
                            "type": metric.kind,
                            "labels": dict(labels),
                            "value": value,
                        },
                        separators=(",", ":"),
                    )
                )
        return "\n".join(out) + "\n"
