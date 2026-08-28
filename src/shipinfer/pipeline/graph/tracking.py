"""Plane 3: stateful, single-camera tracking, sharded by camera.

``references/bitbucket-subfaceid/docs/new-system-architecture.md`` divides the system into
three planes — ingest, a **stateless** GPU inference pool, and a **stateful** CPU tracking
tier — and the division is the reason everything below the pipeline can be pooled across
sixteen GPUs without a thought. This module is the stateful tier, and it is the only place in
the perception DAG where a stage remembers anything between frames.

Three things follow from that, and each one is a defect this module is built to prevent.

**A tracker is per camera, and sharing one is silent corruption.** Kalman state, track ids
and ageing all belong to one camera's view, so two cameras on one instance associate one
camera's objects with the other's — and the output is not degraded tracking, it is a real
identity reported somewhere nothing happened. :class:`TrackerShard` keys an instance on
``camera_id``; ``shipvision``'s :meth:`BaseTracker.begin` refuses a camera change as a second
line of defence, and that belt-and-braces is deliberate because the failure has no symptom.

**The pipeline is multi-threaded and a tracker is not re-entrant.** Frames for one camera are
handled by whichever worker takes them off the fair queue, so two of a camera's frames can be
inside the DAG at once. The shard therefore holds **one lock per camera**, not one lock: the
work under it is a Kalman predict plus a Hungarian solve over at most ``max_detections``
boxes — tens of microseconds — while fifty cameras keep fifty independent locks and do not
serialise against each other.

**MOT needs frames in order, and reassembly does not promise it.** See
:meth:`TrackerShard.update` for the invariant this relies on and what happens when it is
violated.

**Where the pixels are not.** The stage does not read the frame by default, because the image
is released as soon as the crop step is past and holding it until tracking would keep 6 MB
alive per in-flight frame — the exact retention :mod:`shipinfer.pipeline.graph.state` exists
to avoid. Only BoT-SORT can use pixels (camera-motion compensation), so
``tracking.needs_frame`` is the opt-in, and it is off unless a deployment asks.

**Where the per-camera table lives now.** :class:`TrackerShard` moved to
:mod:`shipinfer.topology.elements.track` in phase C4, because the chain's ``track`` element
needs the same object and two copies of an invariant with no symptom is how one of them stops
being correct. It is imported back here so this stage keeps working unchanged — the
coexistence ``docs/arch.md`` section 9 describes, and the reason ``pipeline`` may import
``topology`` (one-way; ``scripts/hooks/check_layers.py`` states the direction).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from shipinfer.core.errors import ConfigurationError
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.graph.stage import Cardinality, PipelineStage
from shipinfer.pipeline.graph.state import DETECTIONS, FRAME_INPUT
from shipinfer.topology.bridge import shipvision_available
from shipinfer.topology.elements.track import (  # noqa: F401  (re-export: see the docstring)
    TrackerShard,
    _CameraShard,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.core.settings.pipeline import TrackingSettings
    from shipinfer.pipeline.graph.state import FrameState

__all__ = [
    "MIXED_CLASS",
    "TRACK_IDS",
    "TRACK_STATES",
    "TrackStage",
    "TrackerShard",
    "tracking_available",
    "tracking_import_error",
]

#: The batch name carrying one track id per attributed detection.
TRACK_IDS = "track_id"
#: The batch name carrying that track's lifecycle state — ``confirmed``, in practice, since a
#: tracker publishes nothing else. Carried anyway because the value crosses to MTMC, and a
#: consumer that cannot see the state has to trust ours.
TRACK_STATES = "track_state"

#: ``ObjectBatch.class_name`` for the track batches. Every other batch in the DAG holds the
#: rows of exactly one detection class, because a crop set is cut per class; a track batch
#: does not, because one camera has one tracker and it is given the whole frame. Named rather
#: than left blank so a reader of a repr is told that, not left guessing.
MIXED_CLASS = "any"

#: State strings are ``"tentative"``/``"confirmed"``/``"lost"``/``"removed"``. Fixed width so
#: the batch is a plain rectangular array like every other one.
_STATE_DTYPE = "<U9"

try:  # the kernels + algorithms submodule is optional, and CI does not check it out
    from shipvision.mot import TRACKERS as _TRACKERS
    from shipvision.mot.association import associate as _associate
    from shipvision.types import Detection as _VisionDetection
    from shipvision.types import Detections as _VisionDetections
    from shipvision.types import FrameTag as _FrameTag
    from shipvision.types import iou_matrix as _iou_matrix

    _IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised on a checkout without the submodule
    _TRACKERS = None
    _associate = None
    _VisionDetection = None
    _VisionDetections = None
    _FrameTag = None
    _iou_matrix = None
    _IMPORT_ERROR = exc


def tracking_available() -> bool:
    """Whether ``shipvision`` can be imported on this host.

    Asked rather than assumed because ``3rdparty/shipvision`` is a submodule and CI
    deliberately does not check it out — that non-checkout is what keeps "every native
    component has a Python counterpart, so a machine with no build still runs" honest.
    Tracking is the one part of the DAG with *no* counterpart here: reimplementing five
    trackers in this repository to hedge against a missing library is exactly the
    duplication the ponytail principle refuses. So it degrades to "off", loudly.

    **Delegated to** :func:`~shipinfer.topology.bridge.shipvision_available` rather than
    answered from the ``_TRACKERS`` binding below, so that "is the submodule here" is one
    fact with one owner. The two could disagree, and did: the module-scope binding is decided
    once at import and stays decided, while :class:`TrackerShard` — which this stage now
    shares with the chain's ``track`` element — asks the bridge's loader on every call. Under
    a test that arranges the absence *after* this module was imported, the stale binding said
    "available" while the shard said "not", and the tests that should have skipped failed
    instead. The bridge's answer is the live one and the one the code path actually obeys.
    """
    return shipvision_available()


def tracking_import_error() -> ImportError | None:
    """Why the import failed, for an error message that names the fix."""
    return _IMPORT_ERROR


class TrackStage(PipelineStage):
    """Turn this frame's detections into stable per-camera identities.

    Placed after the embedders because it wants their output: an appearance vector is what
    carries an identity through the frames where geometry alone is ambiguous, and it is what
    the cross-camera tier downstream will compare. It does **not** wait for them, though —
    ``ship_embedding`` and ``person_embedding`` are declared :attr:`optional`, so a frame
    holding only people still tracks its people rather than skipping tracking entirely
    because the ship branch never ran.

    It has **no** ``requires``, and that is the one declaration here worth reading twice. A
    frame with zero detections still advances the tracker, because ageing is how a track
    dies: a stage that treated an empty frame as nothing to do would keep every departed
    object alive forever, and the pool would grow for as long as the process ran. It is
    skipped only when the *detector* did not answer, where "no objects" and "no answer" are
    genuinely different and :attr:`FrameState.detections_ready` is what tells them apart.

    Args:
        shard: the per-camera trackers. One shard per pipeline; sharing it across two
            pipelines in one process would be sharing the state it exists to separate.
        appearance: batch names to take embeddings from, in priority order. Declared
            ``optional`` so they never gate execution but still count for liveness.
        attribution_iou: minimum overlap for attributing a published track back to a
            detection row — see :meth:`_attribute`.
        needs_frame: read the frame's pixels and hand them to the tracker. Off by default:
            only BoT-SORT's camera-motion compensation can use them, and declaring the image
            makes this stage its last consumer, which keeps 6 MB alive per in-flight frame
            all the way through the DAG instead of freeing it after the crop step.
    """

    cardinality: ClassVar[Cardinality] = Cardinality.PER_OBJECT
    #: The second stage where cardinality changes: it reads the frame's detections and
    #: writes one row per *tracked* object, which is a subset of them.
    expands: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        *,
        shard: TrackerShard,
        appearance: Sequence[str] = (),
        attribution_iou: float = 0.3,
        needs_frame: bool = False,
    ) -> None:
        if not 0.0 < attribution_iou <= 1.0:
            raise ConfigurationError(
                f"stage {name!r}: attribution_iou must be in (0, 1], got {attribution_iou}"
            )
        super().__init__(
            name,
            consumes=(DETECTIONS, FRAME_INPUT) if needs_frame else (DETECTIONS,),
            optional=tuple(appearance),
            produces=(TRACK_IDS, TRACK_STATES),
        )
        self._shard = shard
        self._appearance = tuple(appearance)
        self._max_cost = 1.0 - attribution_iou
        self._needs_frame = needs_frame

    @property
    def shard(self) -> TrackerShard:
        return self._shard

    def _do_run(self, state: FrameState) -> int:
        detections = self._as_vision_detections(state)
        image = state.image if self._needs_frame and not state.image_released else None
        tracks = self._shard.update(detections, image=image)
        indices, ids, states = self._attribute(state, tracks)
        state.attach(
            ObjectBatch(
                name=TRACK_IDS,
                class_name=MIXED_CLASS,
                object_indices=indices,
                data=ids,
            )
        )
        state.attach(
            ObjectBatch(
                name=TRACK_STATES,
                class_name=MIXED_CLASS,
                object_indices=indices,
                data=states,
            )
        )
        return len(indices)

    def _as_vision_detections(self, state: FrameState) -> Any:
        """This frame's detections in the tracking library's vocabulary, embeddings attached.

        The conversion is per object and this is a hot path, but it is bounded by
        ``max_detections`` and it is the price of a shared vocabulary: the alternative is a
        second box format in the system, and ``shipvision.types`` names the exact bug that
        causes — a converter that wrote width where height belonged tracks square objects
        perfectly and falls apart on a ship.
        """
        vectors = self._embeddings(state)
        boxes = state.detections.boxes
        scores = state.detections.scores
        class_ids = state.detections.class_ids
        items = [
            _VisionDetection(
                box=boxes[index],
                # Clamped rather than passed through: `Detection` refuses a score outside
                # [0, 1], and an fp16 engine that reports 1.0000001 would otherwise fail
                # tracking on every frame it appeared in. The value is the detector's
                # confidence, and one ulp of it is not information worth a dropped frame.
                score=min(1.0, max(0.0, float(scores[index]))),
                class_id=int(class_ids[index]),
                embedding=vectors.get(index),
            )
            for index in range(len(state.detections))
        ]
        return _VisionDetections(
            tag=_FrameTag(
                camera_id=state.camera_id,
                frame_id=state.frame_id,
                timestamp=state.context.captured_unix_ns / 1e9,
            ),
            items=items,
            height=state.height,
            width=state.width,
        )

    def _embeddings(self, state: FrameState) -> dict[int, np.ndarray]:
        """Detection index -> appearance vector, from whichever embedders landed.

        A batch only ever holds the rows of its own class, so the ship embedder and the
        person embedder cannot both claim a row and the priority order never has to be
        exercised. It is still walked in declared order, because a deployment that adds a
        third embedder over an overlapping class should get the answer it declared first
        rather than the one dict iteration happened to reach last.
        """
        vectors: dict[int, np.ndarray] = {}
        for name in self._appearance:
            batch = state.batches.get(name)
            if batch is None or batch.is_empty:
                continue
            for index, row in batch.scatter():
                vectors.setdefault(index, row)
        return vectors

    def _attribute(
        self, state: FrameState, tracks: Sequence[Any]
    ) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        """Match published tracks back to the detection rows that produced them.

        The library returns tracks, not row indices, and it is right not to: a track's box is
        the *filtered* estimate, and which detection fed it is the tracker's business rather
        than its output. But an event record is per detection, so the mapping has to be
        recovered — and a published track is by construction the corrected state of one of
        this frame's detections, so the recovery is an assignment over IoU rather than a
        guess.

        One-to-one and globally optimal, via the same solver the trackers themselves use
        (:func:`shipvision.mot.association.associate`). Greedy per-track argmax was the
        obvious alternative and it is wrong in exactly the case that matters: two people
        walking close together each have high overlap with the other's box, and a greedy pass
        can give both track ids to one detection and none to the other — which reads
        downstream as one person who teleported and one who never existed.

        A track that matches nothing above the threshold is dropped rather than forced onto
        its nearest box. The threshold exists to refuse an answer, not to tune one.
        """
        count = len(state.detections)
        if not tracks or count == 0:
            return (), np.zeros((0, 1), dtype=np.int64), np.zeros((0, 1), dtype=_STATE_DTYPE)

        cost = 1.0 - _iou_matrix(np.stack([t.box for t in tracks]), state.detections.boxes)
        matches, _, _ = _associate(cost, self._max_cost)
        # Sorted by detection index so the batch's rows are in the frame's own order: two
        # runs over one frame then produce byte-identical batches, which is what makes a
        # parity or replay assertion possible at all.
        matches.sort(key=lambda pair: pair[1])
        indices = tuple(int(column) for _, column in matches)
        ids = np.array([[tracks[row].track_id] for row, _ in matches], dtype=np.int64)
        states = np.array([[tracks[row].state] for row, _ in matches], dtype=_STATE_DTYPE)
        return indices, ids, states

    def __repr__(self) -> str:
        return f"<TrackStage {self.name} {self._shard.algorithm}>"


def build_tracking_stage(
    settings: TrackingSettings,
    *,
    name: str = "track",
    appearance: Sequence[str] = (),
) -> TrackStage:
    """The tracking stage a deployment's settings describe.

    Separate from :func:`~shipinfer.pipeline.graph.graph.build_perception_graph` so the
    settings-to-stage mapping is one readable function, and so a test can build the stage
    without building five stub models around it.
    """
    return TrackStage(
        name,
        shard=TrackerShard(
            settings.algorithm, options=settings.options, backend=settings.backend
        ),
        appearance=appearance,
        attribution_iou=settings.attribution_iou,
        needs_frame=settings.needs_frame,
    )
