"""The ship + person application: cameras in, perception events out.

This is the top layer. :mod:`shipinfer.server` knows how to run *a model* — batching it,
placing it on a GPU, keeping its queue fair. This package knows what the models are *for*:
that a ship needs segmenting and a person does not, that one frame becomes a variable number
of crops, that a frame's results have to be joined back together before anyone downstream can
use them, and that the tracking tier is already listening on a particular Kafka contract.

It is PLANE 2's application half in
``references/bitbucket-subfaceid/docs/new-system-architecture.md``::

    frame ──► DETECT ─┬─ ship   ──► SEGMENT ──► EMBED ──► RECOGNISE
                      └─ person ──► EMBED
                                        │  per-frame results, tagged (camera, frame)
                                        ▼
                                  reassembly ──► PerceptionEvent ──► Kafka ──► MOT / MTMC

Layout, one reason each::

    sink.py        the ingest -> scheduler adapter: a frame becomes a queued request
    graph/         the DAG: stages, the crop fan-out, planning, liveness, validation
    reassembly/    joining one frame's stage results — bounded, camera-fair, timed out
    sinks/         where an event goes: null (default) / jsonlines / kafka
    schema.py      the emitted event: the old Det2MOT contract, extended for ships
    metrics.py     the handles an operator watches
    runner.py      the wiring, and the one file to read to understand the flow

Three properties are worth knowing before changing anything here.

**This package owns both halves of the frame-to-request mapping.** ``ingest`` publishes into
a :class:`~shipinfer.ingest.sink.FrameSink` protocol it defines; the implementation is
:class:`~shipinfer.pipeline.sink.QueueFrameSink`, here, next to the reassembly that undoes
the mapping. One decision split across two packages is one that drifts (ADR-011).

**Nothing here re-implements the scheduler or the ensemble.** The queue in front of the
pipeline is :class:`~shipinfer.scheduling.queues.FairPriorityQueue`; placement, batching and
spillover belong to the models; tensor-level DAG validation reuses the same
:class:`~shipinfer.core.types.TensorSpec` checks the ensemble uses. What is new is the layer
above: cardinality, conditional branching, reassembly and emission.

**No frame ever disappears silently.** A frame is completed, emitted partial with the missing
stages named, or evicted and counted against the camera that caused the overflow. That is the
whole difference from the system this replaces, whose shared buffer dropped the globally
oldest entry and logged nothing (ADR-005).
"""

from shipinfer.pipeline.graph import (
    Cardinality,
    CropSpec,
    CropStage,
    DecodeParams,
    Detection,
    Detections,
    DetectStage,
    FrameState,
    ObjectBatch,
    ObjectStage,
    PipelineGraph,
    PipelineStage,
    StageOutcome,
    StageStatus,
    TrackerShard,
    TrackStage,
    build_perception_graph,
    tracking_available,
)
from shipinfer.pipeline.metrics import PipelineMetrics
from shipinfer.pipeline.reassembly import (
    EVICTION_POLICIES,
    EvictionPolicy,
    FrameCollector,
    FrameResult,
)
from shipinfer.pipeline.runner import FrameProducer, PipelineRunner
from shipinfer.pipeline.schema import (
    MESSAGE_TYPE,
    SCHEMA_VERSION,
    ObjectRecord,
    PerceptionEvent,
)
from shipinfer.pipeline.sink import QueueFrameSink, TaggedFrame
from shipinfer.pipeline.sinks import (
    RESULT_SINKS,
    JsonLinesResultSink,
    KafkaResultSink,
    NullResultSink,
    ResultSink,
)

__all__ = [
    "EVICTION_POLICIES",
    "MESSAGE_TYPE",
    "RESULT_SINKS",
    "SCHEMA_VERSION",
    "Cardinality",
    "CropSpec",
    "CropStage",
    "DecodeParams",
    "DetectStage",
    "Detection",
    "Detections",
    "EvictionPolicy",
    "FrameCollector",
    "FrameProducer",
    "FrameResult",
    "FrameState",
    "JsonLinesResultSink",
    "KafkaResultSink",
    "NullResultSink",
    "ObjectBatch",
    "ObjectRecord",
    "ObjectStage",
    "PerceptionEvent",
    "PipelineGraph",
    "PipelineMetrics",
    "PipelineRunner",
    "PipelineStage",
    "QueueFrameSink",
    "ResultSink",
    "StageOutcome",
    "StageStatus",
    "TaggedFrame",
    "TrackStage",
    "TrackerShard",
    "build_perception_graph",
    "tracking_available",
]
