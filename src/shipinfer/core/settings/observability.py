"""Logging and metrics wiring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ObservabilitySettings"]


class ObservabilitySettings(BaseModel):
    """What gets logged and how metrics leave the process."""

    model_config = ConfigDict(extra="forbid")

    log_level: str = "INFO"
    #: A name registered in :data:`shipinfer.core.logging.SINKS`. ``async`` is the right
    #: choice in production: it keeps a slow terminal or log volume off the GPU-feeding
    #: threads.
    log_sink: str = "async"
    log_sink_options: dict[str, Any] = Field(default_factory=dict)

    metrics_enabled: bool = True
    #: A name registered in :data:`shipinfer.core.metrics.EXPORTERS`.
    metrics_exporter: str = "prometheus"
    #: Emit a per-instance queue-depth / latency line every N seconds; 0 disables it.
    stats_interval_s: float = Field(default=0.0, ge=0.0)
