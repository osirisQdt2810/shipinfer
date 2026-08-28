"""What a detector said about one frame, in source-frame pixels.

Pure: numpy and :mod:`shipinfer.core.errors`, nothing else. That is what lets it sit in
``topology`` beside the element that fills it in
(:class:`~shipinfer.topology.elements.pool.PoolDetect`) while the counting-simulation
pipeline keeps reading it through the one-line re-export at
``pipeline/graph/detections.py`` -- the coexistence arch.md section 9 describes.

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

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shipinfer.core.errors import ConfigurationError, ValidationError

__all__ = [
    "UNKNOWN_LABEL",
    "DecodeParams",
    "Detection",
    "Detections",
    "Normalization",
    "decode_detections",
    "parse_classes",
]

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

    def indices_of_any(self, class_names: Collection[str]) -> tuple[int, ...]:
        """Row indices of any of several classes, in score order.

        What a slot's ``params: classes:`` selects, and the reason the rule lives on the value
        rather than in the elements: an exact label match, in the detector's own descending
        score order, is one rule, and the ``track`` element and every crop element all ask for
        it (:func:`parse_classes` is the other half). Two copies would be two places for "a
        case difference is not a match" to drift, and the drift has no symptom — the element
        simply covers no rows.

        ``class_names`` is the tuple the caller resolved once at ``open``, so the membership
        test is a scan of two or three strings and this allocates nothing per frame beyond the
        answer.
        """
        return tuple(i for i, label in enumerate(self.labels) if label in class_names)

    def boxes_at(self, indices: Collection[int]) -> np.ndarray:
        """The ``(K, 4)`` boxes of ``indices``, contiguous, ready for a crop kernel.

        Contiguous because the fancy index that gathers them is handed straight to an ops
        implementation; a non-contiguous view would be copied inside it instead, once per
        frame and out of sight. Selecting *every* row hands the array through without a copy,
        which is the common case on a chain whose crop element declares no ``classes:``.

        The fast path asks ``indices == range(len(self))`` and not ``len(indices) == len(self)``
        — the values, not the count. A length test would answer a full-length *reordered*
        selection with the boxes unpermuted, and a box that belongs to another row is exactly
        the corruption this module exists to prevent: it has no symptom until a tracker starts
        swapping identities. Range-to-range equality is O(1), and ``range`` is what the
        no-``classes:`` path returns, so the check is exact and the fast path is unchanged.

        Args:
            indices: what :meth:`indices_of`, :meth:`indices_of_any` and
                ``range(len(detections))`` return. Any order is answered correctly; the rows
                come back in the order asked for.
        """
        if isinstance(indices, range) and indices == range(len(self)):
            return self.boxes
        if not indices:
            return np.zeros((0, 4), dtype=np.float32)
        return np.ascontiguousarray(self.boxes[list(indices)])

    def boxes_of(self, class_name: str) -> tuple[np.ndarray, tuple[int, ...]]:
        """``(boxes, indices)`` for one class: what the crop step needs, in one call.

        The indices come back with the boxes because a crop tensor's rows have to be
        traceable to the detections they came from. Losing that mapping is how an embedding
        ends up attached to the wrong object — a failure with no visible symptom until a
        tracker starts swapping identities.

        The boxes are always a fresh contiguous gather, never :attr:`boxes` itself: the
        indices are a tuple and :meth:`boxes_at`'s pass-through wants a ``range``. Worth
        saying because the caller on the proven path (``pipeline/graph/crop.py``) stores what
        it gets on an ``ObjectBatch`` that outlives the frame, and a single-class frame
        silently handing that batch the live detection array is a difference no shape
        assertion would show.
        """
        indices = self.indices_of(class_name)
        return self.boxes_at(indices), indices

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


@dataclass(frozen=True, slots=True)
class Normalization:
    """Mean/std normalisation and channel order, in the source pixel scale (0-255).

    The structural twin of :class:`shipinfer.runtime.ops.base.NormalizeParams`, declared here
    because ``topology`` may not import ``runtime`` -- the same inversion
    :class:`~shipinfer.topology.base.ImageOpsLike` is, one member down: an element is *told*
    what the model wants and hands it to an ops implementation that reads three attributes off
    it (``mean``, ``std``, ``swap_rb``) and never asks what class it is.

    Two values that agree by construction rather than by comment would be better, and there is
    no place to put one: ``core`` has no word for pre-processing and ``runtime`` is the layer a
    pure element must not name. So the parity is pinned by a test instead
    (``tests/topology/test_pool_detect_decode.py``), which asserts the defaults here are the
    defaults there -- a drift would silently feed a detector unnormalised pixels.

    Raises:
        ValidationError: a zero in ``std``. Refused rather than divided by, and at
            construction, because the symptom otherwise is a frame of infinities reaching a
            model that reports no error at all.
    """

    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (255.0, 255.0, 255.0)
    #: Swap channel order (BGR from OpenCV -> RGB for most checkpoints).
    swap_rb: bool = True

    def __post_init__(self) -> None:
        if any(value == 0 for value in self.std):
            raise ValidationError(f"normalisation std must be non-zero, got {self.std}")


def parse_classes(declared: Any, what: str) -> tuple[str, ...] | None:
    """A slot's ``params: classes:`` as a tuple of labels, or ``None`` for "every row".

    One function rather than one copy per element, for the reason :func:`_normalization` in
    ``elements/pool.py`` is one: ``track`` selecting the rows it tracks and a crop element
    selecting the rows it embeds are the same key, the same vocabulary and the same refusal,
    and the only thing that differs is which element the message names — which is what
    ``what`` carries (``"embed element 'embed_ship'"``). It sits here because
    :meth:`Detections.indices_of_any` is what consumes the answer, and this module already
    owns the label vocabulary both ends are talking about.

    ``None`` and not ``()`` for an absent key: an empty list means "select nothing", which is
    a strange thing to ask for but an unambiguous one, and conflating the two would make a
    typo silently select *everything* — at ``track`` a wrong answer, at an embedder a doubled
    GPU bill.

    Raises:
        ConfigurationError: anything that is not a list of labels. A bare ``classes: ship``
            is refused by name rather than iterated, which would otherwise select the rows
            labelled ``s``, ``h``, ``i`` and ``p`` — that is, none, silently.
    """
    if declared is None:
        return None
    if isinstance(declared, str) or not isinstance(declared, Sequence):
        raise ConfigurationError(
            f"{what}: `params: classes:` must be a list of detection labels, got "
            f"{type(declared).__name__}"
        )
    return tuple(str(entry) for entry in declared)


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
