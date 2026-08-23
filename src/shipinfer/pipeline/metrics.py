"""The pipeline's metric handles, resolved once at construction.

Same shape and the same reason as :class:`shipinfer.core.metrics.ServerMetrics` and
:class:`shipinfer.ingest.IngestMetrics`: at 1000 frames a second a string lookup per frame
is a hash and a dict probe nobody needs to pay for.

Every handle that can be is labelled by ``camera``, because the number that matters in this
system is never the fleet total — it is *which* camera is being starved. That is the whole
lesson of the previous generation, whose shared buffer could report "1000 entries" and
never "camera 7 lost 40% of its frames".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shipinfer.core.metrics import Counter, Gauge, Histogram, MetricsRegistry

__all__ = ["PipelineMetrics"]

#: End-to-end frame latency in microseconds. The interesting range for this pipeline is
#: 10 ms (detect only, warm) to 2 s (the reassembly timeout), so the default request
#: buckets — shaped for a single inference — bottom out too early to be readable.
_E2E_BUCKETS_US = (
    5_000.0,
    10_000.0,
    25_000.0,
    50_000.0,
    100_000.0,
    200_000.0,
    500_000.0,
    1_000_000.0,
    2_000_000.0,
    5_000_000.0,
)

#: Detections (and therefore crops) per frame. The reference deployment saw 2 on a corridor
#: camera and 30 on a square, and that spread *is* the load-imbalance problem.
_FANOUT_BUCKETS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


@dataclass(slots=True)
class PipelineMetrics:
    """Counters, gauges and histograms for the perception DAG."""

    registry: MetricsRegistry = field(default_factory=MetricsRegistry)

    frames_accepted: Counter = field(init=False)
    frames_emitted: Counter = field(init=False)
    frames_partial: Counter = field(init=False)
    frames_evicted: Counter = field(init=False)
    frames_failed: Counter = field(init=False)
    frames_expired: Counter = field(init=False)
    late_arrivals: Counter = field(init=False)
    stages_run: Counter = field(init=False)
    stages_skipped: Counter = field(init=False)
    stages_failed: Counter = field(init=False)
    objects_total: Counter = field(init=False)
    crops_total: Counter = field(init=False)
    sink_failures: Counter = field(init=False)

    frame_latency_us: Histogram = field(init=False)
    stage_latency_us: Histogram = field(init=False)
    objects_per_frame: Histogram = field(init=False)

    pending_frames: Gauge = field(init=False)
    queue_depth: Gauge = field(init=False)
    workers_busy: Gauge = field(init=False)

    def __post_init__(self) -> None:
        r = self.registry
        self.frames_accepted = r.counter(
            "shipinfer_pipeline_frames_accepted_total",
            "Frames the pipeline took off the ingest queue, per camera.",
        )
        self.frames_emitted = r.counter(
            "shipinfer_pipeline_frames_emitted_total",
            "Events published to the result sink, per camera and completeness.",
        )
        self.frames_partial = r.counter(
            "shipinfer_pipeline_frames_partial_total",
            "Events published with stages missing, per camera and reason. A frame that "
            "times out is emitted partial rather than dropped: a counted loss is an "
            "operational fact, an uncounted one is a mystery.",
        )
        self.frames_evicted = r.counter(
            "shipinfer_pipeline_frames_evicted_total",
            "Incomplete frames dropped because reassembly was full, per camera. The "
            "single most important number here: it names the camera that flooded, not "
            "the one that suffered.",
        )
        self.frames_failed = r.counter(
            "shipinfer_pipeline_frames_failed_total",
            "Frames whose graph raised, per camera.",
        )
        self.frames_expired = r.counter(
            "shipinfer_pipeline_frames_expired_total",
            "Frames dropped past their budget before the graph ran, per camera.",
        )
        self.late_arrivals = r.counter(
            "shipinfer_pipeline_late_arrivals_total",
            "Stage results that arrived after their frame was already emitted or evicted.",
        )
        self.stages_run = r.counter(
            "shipinfer_pipeline_stages_run_total", "Stage executions, per stage."
        )
        self.stages_skipped = r.counter(
            "shipinfer_pipeline_stages_skipped_total",
            "Stage executions skipped because their branch was empty, per stage. The "
            "conditional segmenter is what makes the heaviest model in the DAG affordable.",
        )
        self.stages_failed = r.counter(
            "shipinfer_pipeline_stages_failed_total", "Stage executions that raised, per stage."
        )
        self.objects_total = r.counter(
            "shipinfer_pipeline_objects_total", "Detections kept, per camera and class."
        )
        self.crops_total = r.counter(
            "shipinfer_pipeline_crops_total", "Crops produced, per crop set."
        )
        self.sink_failures = r.counter(
            "shipinfer_pipeline_sink_failures_total",
            "Emissions the result sink refused, per sink.",
        )

        self.frame_latency_us = r.histogram(
            "shipinfer_pipeline_frame_latency_us",
            "Capture to emission, microseconds. The number the deployment is judged on.",
            _E2E_BUCKETS_US,
        )
        self.stage_latency_us = r.histogram(
            "shipinfer_pipeline_stage_latency_us",
            "Submit to result for one stage, microseconds.",
        )
        self.objects_per_frame = r.histogram(
            "shipinfer_pipeline_objects_per_frame",
            "Detections kept per frame, per camera. A camera whose distribution sits two "
            "buckets above the fleet is the one that will flood reassembly.",
            _FANOUT_BUCKETS,
        )

        self.pending_frames = r.gauge(
            "shipinfer_pipeline_pending_frames",
            "Frames waiting for stage results, per camera.",
        )
        self.queue_depth = r.gauge(
            "shipinfer_pipeline_queue_depth", "Frames waiting for a pipeline worker."
        )
        self.workers_busy = r.gauge(
            "shipinfer_pipeline_workers_busy", "Workers currently inside the graph."
        )
