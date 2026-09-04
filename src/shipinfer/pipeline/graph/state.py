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

# Re-exported, not redefined: the row-to-floats conversion now lives beside the event it
# feeds (`core/events/convert.py`) because the chain's `output` element needs the same one.
# Imported here under its own name so existing callers of
# `pipeline.graph.state.as_embedding` keep resolving to that single helper.
from shipinfer.core.events import as_embedding
from shipinfer.core.request import InferenceRequest, Priority, RequestContext
from shipinfer.pipeline.graph.detections import Detections
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.schema import ObjectRecord

__all__ = [
    "DETECTIONS",
    "FRAME_INPUT",
    "RECORD_CONVERTERS",
    "EmissionInputs",
    "FrameState",
    "build_records",
    "field_map_names",
]

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


def _as_int(row: np.ndarray) -> int:
    return int(np.asarray(row).reshape(-1)[0])


def _as_float(row: np.ndarray) -> float:
    return float(np.asarray(row).reshape(-1)[0])


def _as_str(row: np.ndarray) -> str:
    """A one-element row of a unicode batch as a Python ``str``.

    ``str(...)`` and not ``.item()``: numpy's ``str_`` *is* a ``str`` subclass, so it
    compares and serialises correctly and the cast looks redundant — until the record is
    pickled between processes or a consumer type-checks it, where the subclass is a
    surprise nobody signed up for. One call per tracked object is not a cost worth having
    that conversation about.
    """
    return str(np.asarray(row).reshape(-1)[0])


#: How a per-object model output becomes a field of
#: :class:`~shipinfer.pipeline.schema.ObjectRecord`. A table rather than a chain of
#: conditionals, so adding a field is an entry here and a field there — and so a typo in a
#: graph's field map is caught at construction against this table's keys.
RECORD_CONVERTERS: Mapping[str, Callable[[np.ndarray], Any]] = {
    "embedding": as_embedding,
    "ship_id": _as_int,
    "similarity": _as_float,
    "mask_area_px": _as_float,
    "track_id": _as_int,
    "track_state": _as_str,
}


@dataclass(frozen=True, slots=True)
class EmissionInputs:
    """A finished frame's emission inputs, captured under the collector's lock.

    Four references, not a copy of the data and not the records themselves. The split is the
    point: capturing has to happen while the lock is held, because the owning worker may
    still be inside `graph.execute`; *building* must not, because it is the most expensive
    per-object work in the pipeline.
    """

    camera_id: str
    frame_id: int
    detections: Detections
    batches: Mapping[str, ObjectBatch]

    def records(self, field_map: Mapping[str, Sequence[str]]) -> tuple[ObjectRecord, ...]:
        """Build one :class:`ObjectRecord` per detection. Call this outside the lock."""
        return build_records(
            self.camera_id, self.frame_id, self.detections, self.batches, field_map
        )


def build_records(
    camera_id: str,
    frame_id: int,
    detections: Detections,
    batches: Mapping[str, ObjectBatch],
    field_map: Mapping[str, Sequence[str]],
) -> tuple[ObjectRecord, ...]:
    """One :class:`ObjectRecord` per detection, from whatever landed.

    Args:
        field_map: ``ObjectRecord`` field -> the batch names that can fill it, in priority
            order -- and the priority is real: two candidates CAN cover one detection, once a
            crop slot declares no ``classes:`` and therefore every row, so the first one that
            mentions a row wins. Two names per field is the ordinary case (``embedding`` from
            the ship embedder for a ship and the person embedder for a person).

    A field left unset means the stage that fills it did not run — which is exactly what the
    emitted event should say, and is distinguishable from a zero.

    A free function rather than a method because both callers hold different things: a live
    `FrameState` during a synchronous emit, and an :class:`EmissionInputs` capture after the
    collector has already let go of the frame. One implementation for both.
    """
    fields: list[dict[str, Any]] = [{} for _ in range(len(detections))]
    for name, candidates in field_map.items():
        convert = RECORD_CONVERTERS.get(name)
        if convert is None:
            raise ValidationError(
                f"no converter for ObjectRecord field {name!r}; "
                f"known: {sorted(RECORD_CONVERTERS)}"
            )
        for candidate in candidates:
            batch = batches.get(candidate)
            if batch is None or batch.is_empty:
                continue
            for index, row in batch.scatter():
                # doc: long the rule was undocumented and the docstring said the opposite
                # FIRST candidate wins -- what "in priority order" says, and what this loop
                # did NOT do: it overwrote, so the last batch to mention a row set the field.
                # This is the ONLY guard: nothing refuses an ambiguous chain at load, and
                # `RECORDS-CLASS-PREMISE` records why that refusal was deferred. So this
                # clause is not defensive code on an impossible state.
                if 0 <= index < len(fields) and name not in fields[index]:
                    fields[index][name] = convert(row)
    return tuple(
        ObjectRecord(
            det_id=f"{camera_id}_{frame_id}_{detection.index}",
            class_name=detection.class_name,
            score=detection.score,
            bbox=detection.box,
            # Guarded, because `detections` and `fields` are sized from the same capture but
            # a detection's `index` is data: a batch scattered against a shorter list would
            # otherwise raise here rather than simply have nothing to fill.
            **(fields[detection.index] if 0 <= detection.index < len(fields) else {}),
        )
        for detection in detections
    )


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
    #: The resized extent ``(out_h, out_w)`` pre-processing actually wrote, before padding —
    #: what the resize divided by. Kept so nothing downstream re-derives it from ``scale``,
    #: which can disagree by a pixel while scale and pad both still match.
    extents: tuple[int, int] = (0, 0)
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

    def emission_inputs(self) -> EmissionInputs:
        """Everything :func:`build_records` needs, captured in four reference copies.

        **What makes this safe is not the collector's lock.** Review corrected an earlier
        version of this docstring that said it was, and the distinction is load-bearing: the
        owning worker mutates the state *inside* `graph.execute` via `attach`, and only takes
        the collector's mutex afterwards in `deliver`. So that mutex does not exclude the
        writer at all.

        What makes it safe is that every field here is a **single reference read**. There is
        no window in which one field has been read and another has not yet been, so the
        capture cannot be internally torn. The bug this replaced could be torn: it read
        `len(detections)`, sized a `fields` list from that, and then re-iterated the
        attribute — so a concurrent `set_detections` made the index run past the end.

        A maintainer who believes the state is mutex-protected will add the next multi-step
        read "under the lock" and reintroduce exactly that. It is not; keep reads single.

        The lock does buy one thing, and only this: the frame is removed from `_pending`
        and captured without another thread finishing it in between. Building the records
        under it bought nothing and cost ~770 us of hold per frame on the one mutex every
        worker takes in `open`/`expect`/`deliver`/`seal`.

        `dict(self.batches)` is a shallow copy of a handful of entries; the batches
        themselves are not copied and are not mutated after they are attached — and it is
        therefore a **second reference** to those arrays, which is why a worker's `drop()`
        does not free them until the event has been built. Bounded by the synchronous emit,
        so not a leak, but this module's docstring makes "`drop` frees something"
        load-bearing and this is the exception to it.
        """
        return EmissionInputs(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            detections=self.detections,
            batches=dict(self.batches),
        )

    def objects(self, field_map: Mapping[str, Sequence[str]]) -> tuple[ObjectRecord, ...]:
        """Build one :class:`ObjectRecord` per detection, from whatever landed."""
        return self.emission_inputs().records(field_map)

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
