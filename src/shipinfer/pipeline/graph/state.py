"""One frame's working state as it travels the DAG, and how it becomes an event.

Two things in this module are load-bearing and neither is obvious.

**The frame is megabytes and the results are kilobytes.** A 1080p BGR frame is 6 MB and a
batch of 15 ship crops at 512x512 float32 is 47 MB, while the results worth publishing are
15 embeddings — about 30 KB. Reassembly holds a frame's state until every stage has
answered, so a state that keeps its image and its crops alive turns a 1024-frame bound into
tens of gigabytes of host memory. So a name is **dropped as soon as its last consumer has
run** (:meth:`drop`), and the image is released the moment the last stage that reads pixels
is done (:meth:`release_image`). The dimensions the emitted event needs are captured up
front, precisely so the pixels can go.

**Nothing is scattered onto per-object records until emission.** The tempting design is for
each stage to write its result into a per-object dict as it lands; the problem is that a
row of a crop tensor is a *view*, so those dicts would keep the 47 MB alive after the batch
was dropped. Building the records once, at emission, from the batches that are still
present is what makes :meth:`drop` actually free anything.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shipinfer.core.errors import ValidationError
from shipinfer.core.request import InferenceRequest, Priority, RequestContext
from shipinfer.pipeline.graph.detections import Detections
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.schema import ObjectRecord

__all__ = ["DETECTIONS", "FRAME_INPUT", "RECORD_CONVERTERS", "FrameState", "field_map_names"]

#: The name the frame's pixels are known by inside the graph. A stage declares
#: ``consumes=(FRAME_INPUT,)`` and the graph then knows when the pixels may be released.
FRAME_INPUT = "image"

#: The name the detector's output is known by. Not an
#: :class:`~shipinfer.pipeline.graph.objects.ObjectBatch`: detections carry boxes, scores and
#: labels as parallel arrays, and forcing them into a single per-object array would mean
#: three of them and a convention for which is which.
DETECTIONS = "detections"

#: Freed image placeholder. A zero-size array rather than ``None`` so :attr:`FrameState.image`
#: keeps one type and a stage that reads it after release fails on shape, loudly, instead of
#: on ``NoneType``.
_RELEASED = np.zeros((0, 0, 3), dtype=np.uint8)


def _as_embedding(row: np.ndarray) -> tuple[float, ...]:
    """A model row as a tuple of floats — a **copy**, so the source batch can be freed."""
    return tuple(float(v) for v in np.asarray(row).reshape(-1))


def _as_int(row: np.ndarray) -> int:
    return int(np.asarray(row).reshape(-1)[0])


def _as_float(row: np.ndarray) -> float:
    return float(np.asarray(row).reshape(-1)[0])


#: How a per-object model output becomes a field of
#: :class:`~shipinfer.pipeline.schema.ObjectRecord`. A table rather than a chain of
#: conditionals, so adding a field is an entry here and a field there — and so a typo in a
#: graph's field map is caught at construction against this table's keys.
RECORD_CONVERTERS: Mapping[str, Callable[[np.ndarray], Any]] = {
    "embedding": _as_embedding,
    "ship_id": _as_int,
    "similarity": _as_float,
    "mask_area_px": _as_float,
}


@dataclass(slots=True)
class FrameState:
    """Everything known about one frame, mutated in place as stages run.

    Mutable and *not* frozen, unlike most values in this codebase: it is a scratchpad owned
    by exactly one worker thread for the frame's whole life, and the alternative — a new
    immutable state per stage — would copy the crop tensors six times per frame.

    The one thing that never changes is :attr:`context`. The ``(camera_id, frame_id)`` tag
    rides through untouched, which is what makes every reorder between here and the response
    safe (ADR-002).
    """

    context: RequestContext
    image: np.ndarray
    #: Source frame dimensions, captured at construction so the pixels can be released
    #: while the emitted event still reports the frame it came from.
    height: int = 0
    width: int = 0
    #: The camera's configured frame rate, for the event's ``img_fps``. Ingest knows it;
    #: the frame does not carry it, because it is per-camera configuration.
    fps: float = 0.0
    priority: Priority = Priority.NORMAL
    deadline_ns: int = 0
    detections: Detections = field(default_factory=Detections.empty)
    #: Whether the detector has answered. Distinct from ``len(detections) == 0``, which is a
    #: quiet camera: "no objects" and "no answer yet" must drive different decisions, and
    #: conflating them is how a failed detector looks like an empty scene.
    detections_ready: bool = False
    #: Letterbox scale and pad from the detector's pre-processing, kept so the decode uses
    #: the numbers pre-processing *reported* rather than recomputed ones.
    scale: float = 1.0
    pad: tuple[float, float] = (0.0, 0.0)
    batches: dict[str, ObjectBatch] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.height == 0 or self.width == 0:
            if self.image.ndim != 3:
                raise ValidationError(
                    f"a frame image must be (H, W, 3), got shape {self.image.shape}"
                )
            self.height, self.width = int(self.image.shape[0]), int(self.image.shape[1])

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_request(
        cls, request: InferenceRequest, input_name: str, *, fps: float = 0.0
    ) -> FrameState:
        """Unpack the request :class:`~shipinfer.pipeline.sink.QueueFrameSink` produced.

        The two halves of the frame-to-request mapping live in this package precisely so
        that this function and the sink can be read together (ADR-011).

        Raises:
            ValidationError: the request does not carry a decoded frame under
                ``input_name``. Raised rather than coerced: an FP32 NCHW tensor here means
                somebody submitted an already-preprocessed batch to the *pipeline's* entry
                point, and silently accepting it would letterbox it a second time.
        """
        tensor = request.inputs.get(input_name)
        if tensor is None:
            raise ValidationError(
                f"pipeline entry request carries no input {input_name!r} "
                f"(has: {sorted(request.inputs)})"
            )
        array = tensor.numpy()
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[3] != 3:
            raise ValidationError(
                f"pipeline entry input {input_name!r} must be (1, H, W, 3) HWC BGR, "
                f"got {array.shape}"
            )
        if array.dtype != np.uint8:
            raise ValidationError(
                f"pipeline entry input {input_name!r} must be uint8 as decoded, got "
                f"{array.dtype}; pre-processing happens in the graph, not before it"
            )
        return cls(
            context=request.context,
            image=array[0],
            fps=fps,
            priority=request.priority,
            deadline_ns=request.deadline_ns,
        )

    # -- identity ----------------------------------------------------------------------

    @property
    def camera_id(self) -> str:
        return self.context.camera_id

    @property
    def frame_id(self) -> int:
        return self.context.frame_id

    @property
    def key(self) -> tuple[str, int]:
        """``(camera_id, frame_id)`` — the reassembly key."""
        return self.context.key

    # -- the graph's namespace ---------------------------------------------------------

    def batch(self, name: str) -> ObjectBatch:
        """One per-object batch.

        Raises:
            ValidationError: no such batch. A stage whose source is missing is a graph
                wiring error, and :meth:`PipelineGraph.validate` should have caught it at
                start-up — so if it happens here, saying which name is missing is the whole
                value of the message.
        """
        try:
            return self.batches[name]
        except KeyError:
            raise ValidationError(
                f"frame {self.key} has no object batch {name!r} "
                f"(has: {sorted(self.batches)})"
            ) from None

    def set_detections(self, detections: Detections) -> None:
        """Record what the detector said, and mark the name available to the planner."""
        self.detections = detections
        self.detections_ready = True

    def attach(self, batch: ObjectBatch) -> None:
        """Add a per-object batch under its own name."""
        self.batches[batch.name] = batch

    def drop(self, name: str) -> None:
        """Forget a batch whose last consumer has run. Idempotent."""
        self.batches.pop(name, None)

    def release_image(self) -> None:
        """Release the frame's pixels. Idempotent.

        Called by the graph as soon as the last stage declaring ``consumes=("image",)`` has
        finished. At 50 cameras x 20 fps this is the difference between reassembly holding
        6 GB and holding 30 MB, and it is the same argument ADR-004 makes for placement:
        the frame stays put, only the small things travel.
        """
        self.image = _RELEASED

    @property
    def image_released(self) -> bool:
        """Whether the pixels are gone. A frame is never legitimately zero-size."""
        return self.image.size == 0

    def available(self) -> set[str]:
        """Names that exist for this frame — what the planner checks ``consumes`` against."""
        names = set(self.batches)
        if self.image.size:
            names.add(FRAME_INPUT)
        if self.detections_ready:
            names.add(DETECTIONS)
        return names

    def non_empty(self) -> set[str]:
        """Names that exist **and** have rows — what ``requires`` checks.

        This is conditional execution in one line: a frame with only people has no
        ``ship_reid_crops`` row, so no stage requiring it is planned, so the ship segmenter
        is never called.
        """
        names = {name for name, batch in self.batches.items() if not batch.is_empty}
        if self.image.size:
            names.add(FRAME_INPUT)
        if self.detections_ready and not self.detections.is_empty:
            names.add(DETECTIONS)
        return names

    # -- emission ----------------------------------------------------------------------

    def objects(self, field_map: Mapping[str, Sequence[str]]) -> tuple[ObjectRecord, ...]:
        """Build one :class:`ObjectRecord` per detection, from whatever landed.

        Args:
            field_map: ``ObjectRecord`` field -> the batch names that can fill it, in
                priority order. Two names per field is normal: ``embedding`` comes from
                ``ship_embedding`` for a ship and ``person_embedding`` for a person, and a
                batch only ever holds the rows of its own class, so they cannot collide.

        A field left unset means the stage that fills it did not run — which is exactly what
        the emitted event should say, and is distinguishable from a zero.
        """
        # Bound once. The generator below used to re-read `self.detections` while `fields`
        # had been sized from an earlier read, so a concurrent `set_detections` made
        # `detection.index` run past the end — and unlike the scatter loop below, that line
        # had no bounds guard. The snapshot in the collector closes the race that caused it;
        # this makes the function internally consistent whatever happens around it.
        detections = self.detections
        fields: list[dict[str, Any]] = [{} for _ in range(len(detections))]
        for name, candidates in field_map.items():
            convert = RECORD_CONVERTERS.get(name)
            if convert is None:
                raise ValidationError(
                    f"no converter for ObjectRecord field {name!r}; "
                    f"known: {sorted(RECORD_CONVERTERS)}"
                )
            for candidate in candidates:
                batch = self.batches.get(candidate)
                if batch is None or batch.is_empty:
                    continue
                for index, row in batch.scatter():
                    if 0 <= index < len(fields):
                        fields[index][name] = convert(row)
        return tuple(
            ObjectRecord(
                det_id=f"{self.camera_id}_{self.frame_id}_{detection.index}",
                class_name=detection.class_name,
                score=detection.score,
                bbox=detection.box,
                **(fields[detection.index] if 0 <= detection.index < len(fields) else {}),
            )
            for detection in detections
        )

    def footprint_bytes(self) -> int:
        """Host bytes this state is holding — what a memory assertion reads."""
        return int(self.image.nbytes) + sum(b.data.nbytes for b in self.batches.values())

    def __repr__(self) -> str:
        return (
            f"<FrameState cam={self.camera_id} frame={self.frame_id} "
            f"objects={len(self.detections)} batches={sorted(self.batches)}>"
        )


def field_map_names(field_map: Mapping[str, Iterable[str]]) -> set[str]:
    """Every batch name an event could read — the names the graph must never drop."""
    return {name for candidates in field_map.values() for name in candidates}
