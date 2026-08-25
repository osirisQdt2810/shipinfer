"""The perception DAG: the stages, the fan-out, and what runs on which frame.

The application layer above the inference server. :mod:`shipinfer.server` knows how to run
*a model*; this package knows that a ship needs segmenting and a person does not, that one
frame becomes a variable number of crops, and that a frame with no ships must not reach the
segmenter at all.

Layout, one reason each::

    stage.py        the contract: cardinality, outcomes, and a stage that drives a model
    detections.py   what a detector said, in source-frame pixels
    detect.py       stage 1: letterbox, infer, decode
    objects.py      the per-object tensor, and the model stage that consumes one
    masks.py        a segmentation engine's two outputs -> one area per object
    crop.py         the fan-out: one frame -> N crops, the first cardinality change
    tracking.py     plane 3: the one stateful stage, one tracker per camera
    state.py        one frame's working state, and how it becomes event records
    graph.py        the DAG itself: planning, liveness, execution, validation
    ops.py          one ImageOps per worker thread, spread across the visible devices
"""

from shipinfer.pipeline.graph.crop import CropSpec, CropStage
from shipinfer.pipeline.graph.detect import DetectStage
from shipinfer.pipeline.graph.detections import (
    DecodeParams,
    Detection,
    Detections,
    decode_detections,
)
from shipinfer.pipeline.graph.graph import (
    DEFAULT_RECORD_FIELDS,
    TRACKING_RECORD_FIELDS,
    PipelineGraph,
    StageObserver,
    build_perception_graph,
)
from shipinfer.pipeline.graph.masks import InstanceMaskArea
from shipinfer.pipeline.graph.objects import ObjectBatch, ObjectStage
from shipinfer.pipeline.graph.ops import ThreadLocalImageOps
from shipinfer.pipeline.graph.stage import (
    Cardinality,
    ModelStage,
    PipelineStage,
    Servable,
    StageOutcome,
    StageStatus,
)
from shipinfer.pipeline.graph.state import DETECTIONS, FRAME_INPUT, FrameState
from shipinfer.pipeline.graph.tracking import (
    TRACK_IDS,
    TRACK_STATES,
    TrackerShard,
    TrackStage,
    build_tracking_stage,
    tracking_available,
)

__all__ = [
    "DEFAULT_RECORD_FIELDS",
    "DETECTIONS",
    "FRAME_INPUT",
    "TRACKING_RECORD_FIELDS",
    "TRACK_IDS",
    "TRACK_STATES",
    "Cardinality",
    "CropSpec",
    "CropStage",
    "DecodeParams",
    "DetectStage",
    "Detection",
    "Detections",
    "FrameState",
    "InstanceMaskArea",
    "ModelStage",
    "ObjectBatch",
    "ObjectStage",
    "PipelineGraph",
    "PipelineStage",
    "Servable",
    "StageObserver",
    "StageOutcome",
    "StageStatus",
    "ThreadLocalImageOps",
    "TrackStage",
    "TrackerShard",
    "build_perception_graph",
    "build_tracking_stage",
    "decode_detections",
    "tracking_available",
]
