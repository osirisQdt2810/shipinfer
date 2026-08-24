"""The ingest plane's metric handles, resolved once at construction.

Same shape as :class:`shipinfer.core.metrics.ServerMetrics` and for the same reason: at
1000 frames per second a string lookup per frame is a hash and a dict probe the process
does not need to pay for. Every handle is labelled by ``camera``, because the number that
matters here is never the total — it is *which* camera stopped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shipinfer.core.metrics import Counter, Gauge, Histogram, MetricsRegistry

__all__ = ["IngestMetrics"]

#: Frame intervals in microseconds. 50 ms is 20 fps, so the interesting range is 10 ms to
#: 1 s; the default latency buckets are shaped for a request, not a camera period.
_INTERVAL_BUCKETS_US = (
    10_000.0,
    20_000.0,
    33_000.0,
    50_000.0,
    66_000.0,
    100_000.0,
    200_000.0,
    500_000.0,
    1_000_000.0,
    5_000_000.0,
)


@dataclass(slots=True)
class IngestMetrics:
    """Counters, gauges and histograms for the camera actors."""

    registry: MetricsRegistry = field(default_factory=MetricsRegistry)

    frames_total: Counter = field(init=False)
    frames_published: Counter = field(init=False)
    frames_dropped: Counter = field(init=False)
    empty_reads_total: Counter = field(init=False)
    reconnects_total: Counter = field(init=False)
    connect_failures_total: Counter = field(init=False)

    frame_interval_us: Histogram = field(init=False)

    camera_fps: Gauge = field(init=False)
    cameras_total: Gauge = field(init=False)
    cameras_streaming: Gauge = field(init=False)
    cameras_unhealthy: Gauge = field(init=False)

    def __post_init__(self) -> None:
        r = self.registry
        self.frames_total = r.counter(
            "shipinfer_ingest_frames_total", "Frames decoded, per camera."
        )
        self.frames_published = r.counter(
            "shipinfer_ingest_frames_published_total",
            "Frames accepted by the inference queue, per camera.",
        )
        self.frames_dropped = r.counter(
            "shipinfer_ingest_frames_dropped_total",
            "Frames the pipeline refused, per camera and reason. The single most "
            "important ingest metric: it is the number the previous system could not "
            "report at all.",
        )
        self.empty_reads_total = r.counter(
            "shipinfer_ingest_empty_reads_total",
            "Reads that produced no frame (stream timeout or end of file), per camera.",
        )
        self.reconnects_total = r.counter(
            "shipinfer_ingest_reconnects_total", "Successful (re)connections, per camera."
        )
        self.connect_failures_total = r.counter(
            "shipinfer_ingest_connect_failures_total",
            "Failed connection attempts, per camera.",
        )
        self.frame_interval_us = r.histogram(
            "shipinfer_ingest_frame_interval_us",
            "Time between consecutive frames, per camera. A camera at 20 fps that "
            "spreads across three buckets is dropping frames upstream of us.",
            _INTERVAL_BUCKETS_US,
        )
        self.camera_fps = r.gauge(
            "shipinfer_ingest_camera_fps", "Measured frame rate over the last window."
        )
        self.cameras_total = r.gauge(
            "shipinfer_ingest_cameras_total", "Cameras the manager is running."
        )
        self.cameras_streaming = r.gauge(
            "shipinfer_ingest_cameras_streaming", "Cameras currently delivering frames."
        )
        self.cameras_unhealthy = r.gauge(
            "shipinfer_ingest_cameras_unhealthy",
            "Cameras that have failed to connect repeatedly.",
        )
