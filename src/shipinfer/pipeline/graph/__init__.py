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
    crop.py         the fan-out: one frame -> N crops, the only cardinality change
    state.py        one frame's working state, and how it becomes event records
    graph.py        the DAG itself: planning, liveness, execution, validation
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
    PipelineGraph,
    StageObserver,
    build_perception_graph,
)
from shipinfer.pipeline.graph.objects import ObjectBatch, ObjectStage, mask_area
from shipinfer.pipeline.graph.stage import (
    Cardinality,
    ModelStage,
    PipelineStage,
    Servable,
    StageOutcome,
    StageStatus,
)
from shipinfer.pipeline.graph.state import DETECTIONS, FRAME_INPUT, FrameState

__all__ = [
    "DEFAULT_RECORD_FIELDS",
    "DETECTIONS",
    "FRAME_INPUT",
    "Cardinality",
    "CropSpec",
    "CropStage",
    "DecodeParams",
    "DetectStage",
    "Detection",
    "Detections",
    "FrameState",
    "ModelStage",
    "ObjectBatch",
    "ObjectStage",
    "PipelineGraph",
    "PipelineStage",
    "Servable",
    "StageObserver",
    "StageOutcome",
    "StageStatus",
    "build_perception_graph",
    "decode_detections",
    "mask_area",
]
