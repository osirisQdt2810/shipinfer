"""The perception pipeline: how a frame becomes an event, and where that event goes.

Two models in one file because they are one concern. :class:`PipelineSettings` describes the
DAG's *deployment* knobs — how big a crop is, how long a frame may wait for its stages, how
many frames may be in flight — and :class:`ReassemblySettings` describes the one part of it
with an operational failure mode of its own, which is why it is separate enough to name and
close enough to keep here.

Nothing per-*model* lives here. The detector's batch size, its instance count and its
preferred batch sizes are in ``model_repository/ship_detector/config.yaml``, because those
are properties of the model and must travel with it; the crop size the pipeline asks for is
a property of *this* deployment's wiring and must not (ADR-006 keeps that split sharp).

Like every other module under ``core``, this one knows nothing about torch, Kafka or a
camera: it is vocabulary, so a pipeline can be configured and validated on a host with no
GPU and no broker (ADR-001).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shipinfer.core.settings.enums import OverflowPolicy

__all__ = ["PipelineSettings", "ReassemblySettings"]


class ReassemblySettings(BaseModel):
    """Joining a frame's stage results back together — bounded, fair, and never silent.

    This is the section that fixes the inherited bug. The previous system joined per-person
    results in one shared 1000-slot buffer that evicted the **globally oldest** entry when
    full, so a camera producing 30 detections a frame pushed out the frames of a camera
    producing 2, and nothing logged it (``references/bitbucket-subfaceid/docs/flow.md``).
    Three fields are the fix: a bound that is honest, a timeout that *emits* rather than
    drops, and an eviction policy that charges the overflow to the camera that caused it.
    """

    model_config = ConfigDict(extra="forbid")

    #: Frames that may be waiting for their stages at once, across the whole fleet. At 1000
    #: frames/s and a ~200 ms end-to-end budget roughly 200 are legitimately in flight, so
    #: this is 5x headroom: large enough that a burst does not evict, small enough that a
    #: wedged model stage cannot consume the host's memory.
    capacity: int = Field(default=1024, ge=1)
    #: How long a frame waits for its missing stages before it is emitted **partial**. 1500
    #: ms is the reference system's ``aLiveTimeInMS``, kept deliberately: it is a tuned
    #: number from a running deployment, and the shape of the mechanism is unchanged.
    timeout_ms: int = Field(default=1500, ge=1)
    #: How often the sweeper looks for timed-out frames. The timeout is therefore accurate
    #: to within one interval, which is the trade for not arming a timer per frame.
    sweep_interval_ms: int = Field(default=100, ge=1)
    #: A name registered in :data:`shipinfer.pipeline.reassembly.EVICTION_POLICIES`. The
    #: default penalises the camera holding the most incomplete frames. Do not set this to
    #: ``oldest_frame`` outside a benchmark: that is the inherited bug, kept only as its own
    #: regression foil (ADR-005).
    eviction_policy: str = "greediest_camera"

    @field_validator("eviction_policy")
    @classmethod
    def _policy_is_named(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("eviction_policy must name a registered policy")
        return value.strip()


class PipelineSettings(BaseModel):
    """The perception DAG as an operator configures it.

    The stage *graph* is code, not configuration: which models run in which order, and what
    is conditional on what, is the application (:mod:`shipinfer.pipeline.graph`). What is
    configurable is everything a second deployment would legitimately differ on — tensor
    sizes, thresholds, in-flight limits, and where results are published.
    """

    model_config = ConfigDict(extra="forbid")

    # -- the graph's shape -------------------------------------------------------------

    #: Letterbox target for the detector, ``(height, width)``. Must match the detector's
    #: declared input dims; the graph checks that at start-up rather than on frame one.
    detector_input: tuple[int, int] = (640, 640)
    #: Detections below this score are discarded before any crop is made. Filtering here
    #: rather than after cropping is the whole point: a crop that will be thrown away still
    #: costs a resize and an embedding.
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Hard cap on detections carried per frame. A detector that returns 25 000 candidate
    #: boxes on a noisy frame must not turn into 25 000 crops; bounding the fan-out is what
    #: keeps one bad frame from becoming a fleet-wide latency spike.
    max_detections: int = Field(default=100, ge=1)
    #: Detector class id -> label. The graph branches on the *label*, so a fleet that adds a
    #: third class configures it here and the branch declarations keep reading in English.
    class_labels: dict[int, str] = Field(default_factory=lambda: {0: "ship", 1: "person"})
    #: Crop size fed to the ship segmenter, ``(height, width)``.
    ship_mask_crop: tuple[int, int] = (512, 512)
    #: Crop size fed to the ship embedder.
    ship_reid_crop: tuple[int, int] = (256, 128)
    #: Crop size fed to the person embedder — the highest-volume tensor in the system at
    #: ~15 000 crops/s.
    person_reid_crop: tuple[int, int] = (256, 128)
    #: Rename a stage's model without touching code: ``{ship_detector: yolov8_ship_v3}``.
    #: The graph resolves every stage's model through this map, so a deployment can A/B a
    #: model by editing settings instead of the DAG.
    model_overrides: dict[str, str] = Field(default_factory=dict)

    # -- execution ---------------------------------------------------------------------

    #: Threads running the graph. Each one drives a frame through the DAG and blocks on
    #: every stage, so this is the *frame* concurrency; GPU batching happens inside each
    #: model, across all frames in flight.
    workers: int = Field(default=4, ge=1)
    #: Frames one worker takes per queue wake-up. 1 is the lowest-latency choice and the
    #: default; a larger value amortises the queue lock at the cost of serialising those
    #: frames behind one worker.
    frames_per_wakeup: int = Field(default=1, ge=1)
    #: A name registered in :data:`shipinfer.scheduling.queues.QUEUES`. ``fair`` buckets by
    #: camera so one busy camera cannot occupy consecutive slots (ADR-005).
    queue_type: str = "fair"
    #: Frames accepted from ingest but not yet picked up by a worker. Small on purpose: a
    #: deep queue in front of the pipeline converts a throughput problem into a latency one
    #: and then hides it.
    queue_capacity: int = Field(default=256, ge=1)
    #: What a full ingest queue does. ``REJECT`` raises ``QueueFullError`` at the camera
    #: actor, which is the only place that can charge the drop to the right camera.
    overflow_policy: OverflowPolicy = OverflowPolicy.REJECT
    #: How long a worker waits for one stage's model before giving up on it. Bounded so a
    #: wedged instance costs one frame and one worker for this long, not forever.
    stage_timeout_ms: int = Field(default=5_000, ge=1)
    #: Drop a frame this many milliseconds after capture. 0 disables it. Distinct from
    #: ``ingest.frame_deadline_ms``, which is applied when the frame is *enqueued*: this one
    #: is the pipeline's own budget for work it has already accepted.
    frame_budget_ms: int = Field(default=0, ge=0)

    # -- output ------------------------------------------------------------------------

    #: A name registered in :data:`shipinfer.pipeline.sinks.RESULT_SINKS`.
    result_sink: str = "null"
    #: Constructor keyword arguments for that sink (a path, a broker list, a topic).
    #: Validated by the sink's own ``__init__``, so a typo fails at start-up.
    result_sink_options: dict[str, Any] = Field(default_factory=dict)
    #: Identifies this process in every emitted event — the reference system's ``sub_id``.
    #: With several perception processes feeding one MOT service, this is how a consumer
    #: knows which one produced a message.
    source_id: str = "shipinfer"

    reassembly: ReassemblySettings = Field(default_factory=ReassemblySettings)

    @field_validator("detector_input", "ship_mask_crop", "ship_reid_crop", "person_reid_crop")
    @classmethod
    def _extents_are_positive(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] < 1 or value[1] < 1:
            raise ValueError(f"tensor extents must be >= 1, got {value}")
        return value

    @model_validator(mode="after")
    def _labels_are_distinct(self) -> PipelineSettings:
        labels = list(self.class_labels.values())
        if len(set(labels)) != len(labels):
            raise ValueError(f"class_labels maps two ids to one label: {labels}")
        return self
