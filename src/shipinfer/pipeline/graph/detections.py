"""What a detector said about one frame, in source-frame pixels.

Kept as parallel arrays rather than a list of objects because everything downstream is
batched: the crop step wants one ``(N, 4)`` array to hand a kernel, not N four-tuples it has
to re-pack. :class:`Detection` exists for the places that genuinely read one object at a
time — a log line, an assertion, building an event record — and is materialised there rather
than stored.

The decode is the one piece of arithmetic in this package that must agree exactly with the
pre-processing that produced it. Letterboxing scaled and padded the frame; undoing it with
recomputed numbers instead of the ones the letterbox reported is where off-by-one box drift
comes from, which is why :class:`~shipinfer.runtime.ops.LetterboxResult` carries the scale
and the pad and this module consumes them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from shipinfer.core.errors import ValidationError

__all__ = ["UNKNOWN_LABEL", "DecodeParams", "Detection", "Detections", "decode_detections"]

#: Label for a class id the deployment did not configure. Kept rather than dropped: a
#: detector that starts emitting class 4 is a model/config mismatch an operator must see,
#: and silently discarding those rows makes it invisible.
UNKNOWN_LABEL = "unknown"

#: Columns of one detector row: ``[x1, y1, x2, y2, score, class]``. The demo repository's
#: ``ship_detector`` declares exactly this, and so does every YOLO export in the fleet.
_ROW_WIDTH = 6


@dataclass(frozen=True, slots=True)
class Detection:
    """One object, materialised for a caller that reads objects one at a time."""

    index: int
    class_id: int
    class_name: str
    score: float
    #: ``(x1, y1, x2, y2)`` in the **source** frame's pixels.
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detections:
    """A frame's detections as parallel arrays, in descending score order.

    ``boxes`` is ``(N, 4)`` float32 xyxy in source-frame pixels, ``scores`` is ``(N,)`` and
    ``class_ids`` is ``(N,)`` int32. ``labels`` is the resolved name per row, so no
    downstream stage has to carry the class-id table around.
    """

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    labels: tuple[str, ...] = ()
    #: Rows the detector produced that were dropped by the score threshold or the per-frame
    #: cap. Carried so "we saw 40 objects and kept 20" is a number rather than a guess.
    discarded: int = 0

    def __post_init__(self) -> None:
        n = self.boxes.shape[0]
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 4:
            raise ValidationError(f"detection boxes must be (N, 4), got {self.boxes.shape}")
        if self.scores.shape != (n,) or self.class_ids.shape != (n,):
            raise ValidationError(
                f"detection arrays disagree: boxes={self.boxes.shape} "
                f"scores={self.scores.shape} class_ids={self.class_ids.shape}"
            )
        if len(self.labels) != n:
            raise ValidationError(f"got {len(self.labels)} labels for {n} detections")

    @classmethod
    def empty(cls) -> Detections:
        """No objects — the common case on a quiet camera, and not an error."""
        return cls(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int32),
            labels=(),
        )

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def indices_of(self, class_name: str) -> tuple[int, ...]:
        """Row indices of one class, in score order. The branch's membership list."""
        return tuple(i for i, label in enumerate(self.labels) if label == class_name)

    def boxes_of(self, class_name: str) -> tuple[np.ndarray, tuple[int, ...]]:
        """``(boxes, indices)`` for one class: what the crop step needs, in one call.

        The indices come back with the boxes because a crop tensor's rows have to be
        traceable to the detections they came from. Losing that mapping is how an embedding
        ends up attached to the wrong object — a failure with no visible symptom until a
        tracker starts swapping identities.
        """
        indices = self.indices_of(class_name)
        if not indices:
            return np.zeros((0, 4), dtype=np.float32), ()
        return np.ascontiguousarray(self.boxes[list(indices)]), indices

    def counts(self) -> dict[str, int]:
        """Objects per class — what the fan-out metric and a log line read."""
        counts: dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def __iter__(self) -> Iterator[Detection]:
        for i in range(len(self)):
            box = self.boxes[i]
            yield Detection(
                index=i,
                class_id=int(self.class_ids[i]),
                class_name=self.labels[i],
                score=float(self.scores[i]),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )

    def __repr__(self) -> str:
        return f"<Detections n={len(self)} {self.counts()} discarded={self.discarded}>"


@dataclass(frozen=True, slots=True)
class DecodeParams:
    """How a detector's raw rows become :class:`Detections`.

    Grouped into a value rather than passed as five arguments because the same five travel
    together from settings to the stage to this function, and because a test that changes
    one of them should not have to restate the rest.
    """

    #: Kept in step with :attr:`~shipinfer.core.settings.pipeline.PipelineSettings.class_labels`
    #: — the shipped detector's COCO numbering, where 0 is a person and 8 is a boat.
    class_labels: Mapping[int, str] = field(default_factory=lambda: {0: "person", 8: "ship"})
    score_threshold: float = 0.25
    max_detections: int = 100


def decode_detections(
    rows: np.ndarray,
    *,
    params: DecodeParams,
    scale: float = 1.0,
    pad: tuple[float, float] = (0.0, 0.0),
    frame_hw: tuple[int, int] | None = None,
    count: int | None = None,
) -> Detections:
    """Turn one frame's detector output into source-frame detections.

    Four things happen here, in this order, and the order matters:

    1. **Truncate** to ``count`` rows when the model reports how many of its fixed-size
       output it filled. A padded output's trailing rows are undefined, not zero, so reading
       them produces plausible boxes out of nothing.
    2. **Threshold** on score. Before cropping, because a crop that will be discarded still
       costs a resize and an embedding — at 15 000 crops a second that is the difference
       between fitting on the GPUs and not.
    3. **Cap** at ``max_detections``, keeping the highest scores. A detector that returns
       25 000 candidates on a noisy frame must not become 25 000 crops; bounding the fan-out
       here is what stops one bad frame becoming a fleet-wide latency spike.
    4. **Un-letterbox**, using the scale and pad the pre-processing *reported*, then clamp to
       the frame.

    Args:
        rows: ``(R, 6)`` — ``[x1, y1, x2, y2, score, class]`` in letterboxed input pixels.
        scale: the letterbox scale for this frame.
        pad: the letterbox ``(pad_x, pad_y)`` in destination pixels.
        frame_hw: the source frame's ``(height, width)``, for clamping. ``None`` skips it.
        count: how many rows the detector filled, if it says.

    Raises:
        ValidationError: the output is not ``(R, 6)``. Raised rather than reshaped: a
            detector whose output layout changed is a deployment error, and guessing at the
            new layout would attach scores to coordinates.
    """
    array = np.asarray(rows)
    if array.ndim != 2 or array.shape[1] != _ROW_WIDTH:
        raise ValidationError(
            f"detector output must be (rows, {_ROW_WIDTH}) "
            f"[x1,y1,x2,y2,score,class], got {array.shape}"
        )
    if count is not None:
        array = array[: max(0, min(int(count), array.shape[0]))]

    total = array.shape[0]
    if total == 0:
        return Detections.empty()

    scores = array[:, 4].astype(np.float32, copy=False)
    keep = np.flatnonzero(scores >= params.score_threshold)
    if keep.size > params.max_detections:
        # argsort on the *kept* scores only, so the cap costs O(k log k) rather than a sort
        # of every candidate row.
        strongest = np.argsort(scores[keep], kind="stable")[::-1][: params.max_detections]
        keep = keep[np.sort(strongest)]
    if keep.size == 0:
        return Detections(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int32),
            labels=(),
            discarded=total,
        )

    order = keep[np.argsort(scores[keep], kind="stable")[::-1]]
    boxes = array[order, :4].astype(np.float32, copy=True)
    pad_x, pad_y = pad
    inv = 1.0 / scale if scale else 1.0
    # `boxes[:, 0::2]` is a basic slice and therefore a *view*; `boxes[:, [0, 2]]` is a fancy
    # index and therefore a copy, which `out=` would write into and then discard. Both
    # spellings read identically and one of them silently does nothing.
    xs = boxes[:, 0::2]
    ys = boxes[:, 1::2]
    xs -= pad_x
    ys -= pad_y
    xs *= inv
    ys *= inv
    if frame_hw is not None:
        height, width = frame_hw
        np.clip(xs, 0.0, float(width), out=xs)
        np.clip(ys, 0.0, float(height), out=ys)

    class_ids = array[order, 5].astype(np.int32, copy=False)
    labels = tuple(params.class_labels.get(int(c), UNKNOWN_LABEL) for c in class_ids)
    return Detections(
        boxes=boxes,
        scores=scores[order].copy(),
        class_ids=class_ids.copy(),
        labels=labels,
        discarded=total - int(order.size),
    )
