"""Fixtures for the pipeline tier — no GPU, no camera, no broker, no build.

Everything here is deliberate about *what* it fakes. The models are faked, because a stage's
job is to submit a request and place the answer, and a real model would only add latency and
a GPU. Nothing else is: the queue, the collector, the graph, the schema and the sinks under
test are the production classes.

:class:`StubModel` is the one double that carries weight. It records every request it was
given, which is how "the ship segmenter was never called on a frame with only people" becomes
an assertion about a *call* rather than about output that happens to look right — output can
look right for the wrong reason, an empty call list cannot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from shipinfer.core.request import (
    InferenceRequest,
    InferenceResponse,
    Priority,
    RequestContext,
    ResponseFuture,
)
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.ingest import IngestSettings
from shipinfer.core.settings.pipeline import PipelineSettings
from shipinfer.core.types import DataType, Tensor, TensorSpec
from shipinfer.pipeline.graph import (
    CropSpec,
    CropStage,
    DecodeParams,
    DetectStage,
    FrameState,
    ObjectStage,
    PipelineGraph,
)
from shipinfer.runtime.ops.numpy_ops import NumpyImageOps

#: Small enough that a whole DAG runs in microseconds, large enough that a crop is a crop.
DETECTOR_INPUT = (8, 8)
CROP_SIZE = (4, 4)
EMBEDDING_DIM = 8


# -- doubles -------------------------------------------------------------------------------


@dataclass
class FakeConfig:
    """The attributes a stage reads off a model's config.

    ``max_batch_size`` is here because `ObjectStage` splits a frame's crops to fit it: a
    TensorRT plan is built at a fixed batch, and a frame with more objects than that used to
    arrive as one request the engine could never accept. ``0`` means "the artefact does not
    constrain us", which is what a model with no declared limit reports.
    """

    input_specs: tuple[TensorSpec, ...] = ()
    output_specs: tuple[TensorSpec, ...] = ()
    max_batch_size: int = 0


@dataclass
class FakeArtifact:
    config: FakeConfig


class StubModel:
    """A ``Servable`` that answers from a function and remembers every call.

    Args:
        respond: ``(request) -> {output_name: Tensor}``. Given the request so a per-object
            model can size its answer to the batch it was handed — which is the property the
            cardinality check exists to enforce.
        error: raise this instead of answering, to exercise a failed stage.
        artifact: declared specs, when a test is about start-up validation.
    """

    def __init__(
        self,
        name: str,
        respond: Callable[[InferenceRequest], dict[str, Tensor]] | None = None,
        *,
        error: BaseException | None = None,
        hang: bool = False,
        artifact: FakeArtifact | None = None,
    ) -> None:
        self.name = name
        self._respond = respond or (lambda _request: {})
        self._error = error
        self._hang = hang
        self.artifact = artifact
        self.calls: list[InferenceRequest] = []

    @property
    def is_ready(self) -> bool:
        return True

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        self.calls.append(request)
        future = ResponseFuture(request)
        if self._hang:
            # Never resolved: the stage's own timeout is what has to notice.
            return future
        future.set_running_or_notify_cancel()
        if self._error is not None:
            future.set_exception(self._error)
            return future
        future.set_result(
            InferenceResponse(
                request_id=request.request_id,
                model_name=self.name,
                model_version=1,
                outputs=self._respond(request),
                context=request.context,
            )
        )
        return future

    @property
    def batch_sizes(self) -> list[int]:
        """Rows per call — how a test asserts the fan-out actually batched."""
        return [next(iter(r.inputs.values())).batch_size for r in self.calls]


@dataclass
class FakeServer:
    """Just enough server for :class:`PipelineRunner`: a model table and a started flag."""

    models: dict[str, StubModel] = field(default_factory=dict)
    settings: ServerSettings = field(default_factory=ServerSettings)
    is_started: bool = True

    def model(self, name: str) -> StubModel:
        return self.models[name]


# -- responses -----------------------------------------------------------------------------


def detector_response(
    rows: np.ndarray | list[list[float]],
) -> Callable[[Any], dict[str, Tensor]]:
    """A detector that always returns the same ``[x1, y1, x2, y2, score, class]`` rows."""
    array = np.asarray(rows, dtype=np.float32).reshape(1, -1, 6)

    def respond(_request: InferenceRequest) -> dict[str, Tensor]:
        return {
            "boxes": Tensor.from_numpy(array),
            "num_detections": Tensor.from_numpy(np.array([[array.shape[1]]], dtype=np.int32)),
        }

    return respond


def embedder_response(dim: int = EMBEDDING_DIM) -> Callable[[Any], dict[str, Tensor]]:
    """One embedding per input row, valued by row index so a mix-up is visible."""

    def respond(request: InferenceRequest) -> dict[str, Tensor]:
        rows = next(iter(request.inputs.values())).batch_size
        data = np.arange(rows, dtype=np.float32).reshape(rows, 1) * np.ones(
            (1, dim), dtype=np.float32
        )
        return {"embedding": Tensor.from_numpy(data)}

    return respond


#: Prototype planes per segmentation row, and the extent of one plane in these tests. The
#: engine's real bank is 32 planes at a quarter of the input; 32 is kept because
#: :class:`~shipinfer.pipeline.graph.masks.InstanceMaskArea` refuses a coefficient count that
#: disagrees with the plane count, and that refusal is worth exercising.
SEG_COEFFICIENTS = 32
SEG_PROTO_HW = (2, 2)


def segmenter_response(
    coefficients: int = SEG_COEFFICIENTS,
) -> Callable[[Any], dict[str, Tensor]]:
    """The two outputs a YOLO segmentation engine actually emits.

    Not a mask — that is the point. ``output0`` carries ``(N, rows, 6 + M)`` detections whose
    last M columns are mask coefficients, and ``output1`` carries the ``(N, M, h, w)``
    prototype bank; a mask is the two multiplied through a sigmoid. Emitting a ready-made
    ``masks`` tensor here would have let the pipeline pass tests against a contract no engine
    honours, which is the mismatch this fixture now cannot hide.

    One coefficient is set against a saturated first plane, so every cell is foreground and
    the folded area is the crop's full pixel count. The arithmetic in ``InstanceMaskArea``
    runs for real; only the expected answer is closed-form.
    """

    def respond(request: InferenceRequest) -> dict[str, Tensor]:
        rows = next(iter(request.inputs.values())).batch_size
        detections = np.zeros((rows, 1, 6 + coefficients), dtype=np.float32)
        detections[:, 0, 4] = 0.9  # score, above the stage's threshold
        detections[:, 0, 5] = 8.0  # the detector's boat class
        detections[:, 0, 6] = 1.0  # all the weight on prototype plane 0
        protos = np.zeros((rows, coefficients, *SEG_PROTO_HW), dtype=np.float32)
        protos[:, 0] = 10.0  # sigmoid(10) ~ 1, so every cell lands inside the instance
        return {
            "output0": Tensor.from_numpy(detections),
            "output1": Tensor.from_numpy(protos),
        }

    return respond


# -- fixtures ------------------------------------------------------------------------------


@pytest.fixture()
def ops() -> NumpyImageOps:
    """The numpy image ops.

    Chosen explicitly rather than through ``get_image_ops`` so this tier behaves identically
    on a laptop and on the 8-GPU dev box. A test that quietly takes the CUDA path on one
    machine and the numpy path on another is a test whose failures are not reproducible.
    """
    return NumpyImageOps()


@pytest.fixture()
def pipeline_settings() -> PipelineSettings:
    """Tiny tensors, one worker, deterministic thresholds."""
    return PipelineSettings(
        detector_input=DETECTOR_INPUT,
        ship_mask_crop=CROP_SIZE,
        ship_reid_crop=CROP_SIZE,
        person_reid_crop=CROP_SIZE,
        score_threshold=0.5,
        workers=1,
    )


@pytest.fixture()
def frame_image() -> np.ndarray:
    """A 16x16 BGR frame whose content is a gradient, so a wrong crop is visible."""
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(16, dtype=np.uint8)[:, None]
    return image


@pytest.fixture()
def make_state(frame_image: np.ndarray):
    """Build a :class:`FrameState` the way the runner would."""

    def _make(camera: str = "cam0", frame: int = 0, **kwargs) -> FrameState:
        return FrameState(
            context=RequestContext(
                camera_id=camera, frame_id=frame, captured_ns=1_000, captured_unix_ns=2_000
            ),
            image=frame_image.copy(),
            **kwargs,
        )

    return _make


@pytest.fixture()
def make_request(frame_image: np.ndarray):
    """A pipeline *entry* request, exactly as ``QueueFrameSink`` builds one."""

    def _make(camera: str = "cam0", frame: int = 0, **kwargs) -> InferenceRequest:
        return InferenceRequest(
            model_name="ship_detector",
            inputs={"images": Tensor.from_numpy(frame_image[None, ...].copy())},
            context=RequestContext(
                camera_id=camera, frame_id=frame, captured_ns=1_000, captured_unix_ns=2_000
            ),
            priority=Priority.NORMAL,
            **kwargs,
        )

    return _make


@pytest.fixture()
def models() -> dict[str, StubModel]:
    """The five models the perception DAG drives, each answering plausibly."""
    return {
        "ship_detector": StubModel("ship_detector", detector_response([[0, 0, 8, 8, 0.9, 8]])),
        "ship_segmenter": StubModel("ship_segmenter", segmenter_response()),
        "ship_embedder": StubModel("ship_embedder", embedder_response()),
        "person_embedder": StubModel("person_embedder", embedder_response()),
    }


@pytest.fixture()
def build_graph(pipeline_settings: PipelineSettings, ops: NumpyImageOps, models):
    """The production graph over stub models, with the detector's answer scripted."""

    def _build(detections: list[list[float]] | None = None, **overrides) -> PipelineGraph:
        if detections is not None:
            models["ship_detector"] = StubModel("ship_detector", detector_response(detections))
        models.update(overrides)
        from shipinfer.pipeline.graph import build_perception_graph

        return build_perception_graph(pipeline_settings, resolve=models.__getitem__, ops=ops)

    return _build


@pytest.fixture()
def tiny_graph(ops: NumpyImageOps, models):
    """A two-stage graph — detect then crop — for tests about the fan-out alone."""

    def _build(detections: list[list[float]]) -> PipelineGraph:
        models["ship_detector"] = StubModel("ship_detector", detector_response(detections))
        return PipelineGraph(
            [
                DetectStage(
                    "detect",
                    "ship_detector",
                    resolve=models.__getitem__,
                    ops=ops,
                    dst_size=DETECTOR_INPUT,
                    decode=DecodeParams(score_threshold=0.5),
                ),
                CropStage(
                    "crop",
                    ops=ops,
                    crops=[
                        CropSpec("ship_reid_crops", "ship", CROP_SIZE),
                        CropSpec("person_crops", "person", CROP_SIZE),
                    ],
                ),
                ObjectStage(
                    "person_embedder",
                    "person_embedder",
                    resolve=models.__getitem__,
                    source="person_crops",
                    input_name="crops",
                    outputs={"embedding": "person_embedding"},
                    row_shape=(3, *CROP_SIZE),
                ),
            ],
            field_map={"embedding": ("person_embedding",)},
            name="tiny",
        )

    return _build


@pytest.fixture()
def ingest_settings() -> IngestSettings:
    return IngestSettings(target_model="ship_detector", input_name="images")


@pytest.fixture()
def spec():
    """Build a :class:`TensorSpec` without repeating the dtype enum in every test."""

    def _make(name: str, shape: tuple[int, ...], dtype: DataType = DataType.FP32) -> TensorSpec:
        return TensorSpec(name=name, dtype=dtype, shape=shape)

    return _make
