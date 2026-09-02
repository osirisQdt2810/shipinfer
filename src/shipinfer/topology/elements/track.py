"""The ``track`` element: one stateful tracker per camera, and the plane change.

This is arch.md section 5(6) on the chain — where a frame stops being pixels and becomes
identities. Two things live here, and the split is deliberate:

* :class:`TrackerShard` is the **per-camera table**: one tracker, one lock and one
  high-water mark per camera. It was proven in ``pipeline/graph/tracking.py`` and is
  *moved* here rather than rewritten, so the chain and the counting-simulation pipeline
  share one implementation of an invariant that has no symptom when it breaks (that module
  imports it back — the coexistence arch.md section 9 describes).
* :class:`ShipvisionTrack` is the element: it turns a :class:`ChainItem` into the tracker's
  vocabulary, calls the shard, and turns the answer back into metadata.

**Why the sharding is a correctness constraint and not a scaling one.** Kalman state, track
ids and ageing all belong to one camera's view. Two cameras on one tracker associate one
camera's objects with the other's, and the output is not degraded tracking — it is a real
identity reported somewhere nothing happened. ``shipvision``'s ``BaseTracker.begin``
refuses a camera change as a second line of defence, because the failure is invisible.

**Why the ordering guard is here and not upstream.** The fair lane preserves a camera's
frame order (one FIFO deque per key), but the *workers* do not: with more than one pipeline
worker, two of a camera's frames are walked concurrently and the later one can arrive
first. Feeding a tracker a frame it has already passed double-ages every track and
double-counts the hit that promotes one, so the frame that lost the race is refused — and
the refusal is **caught here**, counted, and the item emitted with ``track`` in
``missing_stages``. An element that raised would cost the frame its whole event, and a
frame with an honest gap is worth more than no frame at all (arch.md section 5(5)).

**Why the payload is dropped.** ``produces`` is ``meta@cpu`` and exactly one entry, so this
element stamps its own cap — the one place in the chain where that is right, because this
is the plane change the caps vocabulary exists to describe. Stamping ``meta@cpu`` while
carrying a device handle would relabel VRAM as host metadata, which is the laundering
``_substitute_donor`` refuses. So the payload goes with the label.

**Where shipvision is named.** Nowhere at module scope. Every symbol comes from
:mod:`shipinfer.topology.bridge` inside a function, so ``import shipinfer.topology`` stays
free of the submodule and a chain naming ``impl: shipvision`` is still *validatable* on a
host that never checked it out — it fails at ``open()``, with the command that fixes it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from shipinfer.core.errors import ConfigurationError, TrackingError, ValidationError
from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.bridge import load_mot, load_types
from shipinfer.topology.elements.detections import (
    MISSING_STAGES,
    TRACK_ROWS,
    Detections,
    parse_classes,
    per_row,
)
from shipinfer.topology.registry import registry_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.core.metrics import Counter, Gauge

__all__ = [
    "DEFAULT_ALGORITHM",
    "DEFAULT_ATTRIBUTION_IOU",
    "DEFAULT_REGRESSION_RESET",
    "MISSING_STAGES",
    "TRACK_ROWS",
    "ShipvisionTrack",
    "TrackerShard",
]

#: The tracker a slot gets when it does not say. ByteTrack, mirroring
#: ``pipeline.tracking.algorithm`` — a literal in a pure layer because ``topology`` may not
#: read the settings tree, and a test ties the two so they cannot drift.
DEFAULT_ALGORITHM = "bytetrack"

#: How far a camera's ``frame_id`` may go *backwards* before this element stops calling it a
#: reordering and starts calling it a restarted stream.
#:
#: Two populations, with no lifecycle event to tell them apart. A **reorder** is bounded by
#: the number of pipeline workers — at most that many of a camera's frames are in flight
#: (arch.md 5(3) sizes it at ~32) — so a handful of frames' regression is ordinary and must
#: be refused, not absorbed. A **restart** is an ingest actor that minted a fresh
#: ``FrameCounter`` without anybody calling ``remove_camera`` + ``add_camera``: the ids drop
#: from tens of thousands to zero, and refusing those means refusing every frame of that
#: camera for the process's life. 64 is comfortably above the first and far below the second.
#: ``0`` disables the recovery, which is the old behaviour.
#:
#: **The recovery has a window it does not cover, this many frames wide.** A restart is only
#: recognised once the new id is this far *below* the old high-water mark, so a camera that
#: ran to frame 39 and restarts at 0 has its first 40 frames refused — the regression is real
#: but too small to call — and the frame that passes 39 then continues the **old** tracker's
#: identities across the discontinuity, with no reset counted (measured). Both failures are
#: strictly better than refusing the camera for the process's life, and neither is as good as
#: the announced remove+add ADR-018 names: this is the floor under a deployment that never
#: sends it, not a substitute.
DEFAULT_REGRESSION_RESET = 64

#: Minimum IoU between a published track's box and a detection's before the two are called the
#: same object. Mirrors ``pipeline.tracking.attribution_iou``, whose default is the same 0.3
#: and whose docstring makes the argument: the only competitors are the frame's *other*
#: objects, so this exists to refuse an answer rather than to tune one. A track that matches
#: nothing above it gets no row, which serialises as a ``null`` track id rather than as a wrong
#: one.
DEFAULT_ATTRIBUTION_IOU = 0.3


class _CameraShard:
    """One camera's tracker, its lock, and how far through the stream it has been fed.

    ``__slots__`` and a plain class: one of these exists per camera for the process's life,
    so the shape matters less than it does per frame, but the lock and the high-water mark
    have to sit together — reading one without the other is the race this module prevents.
    """

    __slots__ = ("implicit_resets", "last_frame_id", "lock", "out_of_order", "tracker")

    def __init__(self, tracker: Any) -> None:
        self.lock = threading.Lock()
        self.tracker = tracker
        #: The highest ``frame_id`` fed to this tracker. ``-1`` because ``FrameTag`` allows
        #: frame 0 and a camera's first frame must not look like a replay.
        self.last_frame_id = -1
        self.out_of_order = 0
        self.implicit_resets = 0


class TrackerShard:
    """One tracker per camera, built lazily and never shared.

    The sharding is a correctness constraint, not a scaling one — see this module's docstring.
    Everything else here exists to make that constraint hold under the pipeline's workers.

    Args:
        algorithm: a name registered in ``shipvision.mot.TRACKERS`` (``sort``, ``bytetrack``,
            ``ocsort``, ``botsort``, ``deepsortv2``). Resolved through that registry, so adding
            a tracker there needs no edit here.
        options: constructor keyword arguments for that tracker.
        backend: pin ``python`` or ``native``. ``None`` takes the fastest one this host can
            build, which is the registry's documented behaviour and the numpy floor every caller
            in this tree wants.

    Raises:
        ConfigurationError: the library is absent, the algorithm is unknown, or ``options``
            holds a key the tracker's constructor does not accept. Raised **at construction** —
            one throwaway tracker is built for no other reason — because a typo in a
            deployment's tracking options must stop the deploy rather than surface identically
            on every frame from inside a worker thread.
    """

    def __init__(
        self,
        algorithm: str,
        *,
        options: Mapping[str, Any] | None = None,
        backend: str | None = None,
    ) -> None:
        # Through the bridge, inside the function: one message for a missing submodule, and
        # `import shipinfer.topology` still costs nothing. It is called *first* so that "no
        # shipvision" is answered by the bridge's sentence rather than by whatever the build
        # below would say about a registry that does not exist.
        load_mot()
        self._algorithm = algorithm
        self._options = dict(options or {})
        self._backend = backend
        # Guards insertion into and removal from `_cameras` only. Never held across
        # `tracker.update`, so a slow camera cannot stall the other forty-nine, and never held
        # across a per-camera lock, so a lifecycle call cannot queue behind one camera's frame.
        self._admit = threading.Lock()
        self._cameras: dict[str, _CameraShard] = {}
        self._build()

    def _build(self) -> Any:
        try:
            return load_mot().TRACKERS.build(
                self._algorithm, backend=self._backend, **self._options
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"tracking: cannot build tracker {self._algorithm!r}"
                f"{'' if self._backend is None else f' (backend {self._backend!r})'} "
                f"with options {sorted(self._options)}: {exc}"
            ) from exc

    # -- properties ----------------------------------------------------------------------

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def cameras(self) -> tuple[str, ...]:
        """Cameras that have a tracker, in the order they first appeared."""
        return tuple(self._cameras)

    @property
    def camera_count(self) -> int:
        """How many cameras hold a tracker.

        Separate from :attr:`cameras` because it is read on the per-frame path — a gauge is
        only written when the number *changes* — and materialising a tuple of fifty strings to
        take its length would be an allocation per frame to answer a question ``len`` answers
        in one bytecode.
        """
        return len(self._cameras)

    def tracker_for(self, camera_id: str) -> Any:
        """One camera's tracker, for a test that wants to look at the state directly."""
        return self._shard(camera_id).tracker

    def stats(self) -> dict[str, int]:
        """What an operator reads: how many cameras are tracked, and how many frames lost
        the ordering race. ``stages_failed{stage="track"}`` counts the same refusals; this
        is the per-shard view of the same fact."""
        return {
            "cameras": len(self._cameras),
            "tracks": sum(s.tracker.pool_size for s in self._cameras.values()),
            "out_of_order": sum(s.out_of_order for s in self._cameras.values()),
            "implicit_resets": sum(s.implicit_resets for s in self._cameras.values()),
        }

    # -- the contract --------------------------------------------------------------------

    def update(
        self,
        detections: Any,
        *,
        image: np.ndarray | None = None,
        regression_reset: int | None = None,
        on_implicit_reset: Callable[[str], None] | None = None,
    ) -> list[Any]:
        """Advance one camera by one frame and return its publishable tracks.

        The first argument is a ``shipvision.types.Detections``, tag included, so the camera and
        the frame cannot be passed separately and cannot disagree with the boxes.

        **The invariant.** A camera's tracker sees that camera's frames one at a time and in
        strictly increasing ``frame_id`` order. The per-camera lock gives the first half; the
        per-camera high-water mark gives the second, and it has to, because reassembly orders
        nothing: two of one camera's frames can be in flight and the later one can arrive first.

        **What a violation costs.** Feeding a tracker a frame it has already passed advances
        every filter a second time, ages every track a second time, and double-counts the hit
        that promotes a tentative track — so a replayed frame silently changes *which identities
        exist* downstream. The frame that lost the race is refused: the caller emits it with
        detections, embeddings and masks intact and no track ids, naming ``track`` in
        ``missing_stages``.

        **Not silently reordered, either.** A reorder buffer would wait for a frame that may
        never arrive — ingest drops on overflow and on an expired deadline — so it needs a
        timeout, and that timeout is latency paid by every frame of every camera to rescue the
        few that raced. If the refusal rate matters, the fix is upstream (fewer of one camera's
        frames in flight), and ``stats()["out_of_order"]`` is the number that says so.

        Args:
            detections: this frame's detections and its tag.
            image: the decoded frame, for the one tracker that compensates for camera motion.
            regression_reset: how far ``frame_id`` may go backwards before this is read as a
                restarted stream rather than a reordering — see
                :data:`DEFAULT_REGRESSION_RESET`. ``None`` (the default, and what the
                counting-simulation pipeline passes) refuses every regression. The decision is
                taken **here**, under the camera's lock, rather than by the caller reading
                :attr:`last_frame_id` and calling :meth:`reset`: those are three steps, and
                between any two another worker can advance the same camera.
            on_implicit_reset: called with the camera id when that recovery fires, from inside
                the camera's lock. A callback rather than a return value or a polled counter,
                because the event is rare and the alternatives put work on the frames where
                nothing happened. It must not block — it is holding one camera's tracker while
                it runs.

        Raises:
            TrackingError: the frame does not advance this camera's stream, and the regression
                is small enough to be a reordering.
        """
        tag = detections.tag
        shard = self._shard(tag.camera_id)
        with shard.lock:
            if tag.frame_id <= shard.last_frame_id:
                behind = shard.last_frame_id - tag.frame_id
                if regression_reset and behind >= regression_reset:
                    # A restarted stream nobody announced. Forget the tracks and take this
                    # frame as the new stream's first: the ids restart, which is what a
                    # reconnect means, and the alternative is refusing every frame of this
                    # camera until the process ends.
                    shard.tracker.reset()
                    shard.last_frame_id = -1
                    shard.implicit_resets += 1
                    if on_implicit_reset is not None:
                        on_implicit_reset(tag.camera_id)
                else:
                    shard.out_of_order += 1
                    raise TrackingError(
                        f"camera {tag.camera_id!r}: frame {tag.frame_id} reached the tracker "
                        f"after frame {shard.last_frame_id}. Tracking is stateful and ordered; "
                        f"replaying a frame double-ages every track and double-counts the hit "
                        f"that promotes one, so this frame is published without track ids "
                        f"rather than with wrong ones"
                    )
            shard.last_frame_id = tag.frame_id
            return shard.tracker.update(detections, image=image)

    def reset(self, camera_id: str) -> None:
        """Forget one camera's tracks and its stream position, building a tracker if needed.

        For a camera that reconnected: continuity is broken, so the ids must not continue,
        and the next frame's id may be lower than the last one seen.
        """
        self._reset(self._shard(camera_id))

    def reset_if_present(self, camera_id: str) -> bool:
        """:meth:`reset`, but never *builds* a tracker for a camera that has none.

        What a lifecycle hook wants. ``camera_added`` fires for every camera on the shard,
        including the forty-nine that were never on this element, and :meth:`reset` would mint a
        tracker for each — a Kalman filter, a track pool and a registry build for a camera that
        may never send a frame, on the thread holding the runner's lifecycle lock.

        The table is read under :attr:`_admit` and the lock is **released before** the camera's
        own lock is taken. Holding both would make every other camera's first frame queue behind
        this one camera's in-flight update.

        **It can still wait, on one camera, for one frame.** If a worker is inside
        ``tracker.update`` for *this* camera the reset queues behind that update (measured 0.40
        s against an artificially slow tracker; tens of microseconds ordinarily). :meth:`drop`
        has no such wait because unlinking a table entry needs no agreement with the frame in
        flight — a **reset** does, since resetting mid-update is the half-forgotten state the
        per-camera lock prevents. The wait is bounded by one update on one camera.

        Returns:
            Whether there was a tracker to reset.
        """
        with self._admit:
            shard = self._cameras.get(camera_id)
        if shard is None:
            return False
        self._reset(shard)
        return True

    def drop(self, camera_id: str) -> bool:
        """Forget a camera entirely: its tracker, its lock and its stream position.

        **Takes the table lock and nothing else.** A worker can be inside ``tracker.update``
        for this very camera at this very moment — the walk takes no lock against the
        lifecycle — and waiting for it would hold the runner's ``_lifecycle`` for as long as
        that frame takes, stalling every other camera's add, remove and the ``stop`` that would
        end the wait. So the entry is unlinked and the in-flight frame finishes against a shard
        nobody can reach any more; its result is discarded by the caller that is being removed.

        Idempotent, and a frame that arrives after the drop rebuilds the shard as a first
        frame. Both are ordinary: ``remove_camera`` answering ``False`` means the decoder was
        abandoned at its deadline rather than joined.

        Returns:
            Whether there was anything to drop.
        """
        with self._admit:
            return self._cameras.pop(camera_id, None) is not None

    @staticmethod
    def _reset(shard: _CameraShard) -> None:
        with shard.lock:
            shard.tracker.reset()
            shard.last_frame_id = -1

    def _shard(self, camera_id: str) -> _CameraShard:
        """This camera's shard, creating it on first sight.

        The unlocked ``get`` first is not a micro-optimisation: it is what keeps admitting a
        frame off the one lock every camera shares, in the steady state where the shard
        already exists. A ``dict`` lookup is a single bytecode under the GIL, and the
        re-check inside the lock is what makes the create path safe.
        """
        shard = self._cameras.get(camera_id)
        if shard is not None:
            return shard
        with self._admit:
            shard = self._cameras.get(camera_id)
            if shard is None:
                shard = _CameraShard(self._build())
                self._cameras[camera_id] = shard
            return shard

    def __repr__(self) -> str:
        return f"<TrackerShard {self._algorithm} cameras={len(self._cameras)}>"


class _TrackMetrics:
    """The element's metric handles, resolved once at ``open``.

    Same shape and the same reason as :class:`~shipinfer.runners.metrics.RunnerMetrics`: at a
    thousand frames a second a metric looked up by string per frame is a hash and a dict probe
    nobody needs to pay for. It exists as a class rather than five attributes so that
    ``context.metrics is None`` has one answer — a null object whose calls are no-ops — instead
    of an ``if`` on the per-frame path.

    ``None`` means the runner offered no registry, and then nothing is counted rather than a
    private registry being minted: a metric nobody scrapes reads as evidence and is worse than
    an absent one (:class:`~shipinfer.topology.base.ElementContext`).
    """

    __slots__ = ("cameras", "element", "out_of_order", "regressions", "untracked")

    def __init__(self, registry: Any, element: str) -> None:
        self.element = element
        counter = getattr(registry, "counter", None)
        if counter is None:
            self.out_of_order: Counter | None = None
            self.untracked: Counter | None = None
            self.regressions: Counter | None = None
            self.cameras: Gauge | None = None
            return
        self.out_of_order = registry.counter(
            "shipinfer_track_frames_out_of_order_total",
            "Frames refused by a camera's tracker because they did not advance its stream, "
            "per camera. The number that says whether the reordering the pipeline workers "
            "introduce is material; the fix is upstream (fewer of one camera's frames in "
            "flight), never a reorder buffer.",
        )
        self.untracked = registry.counter(
            "shipinfer_track_frames_untracked_total",
            "Frames the track element emitted with `track` in `missing_stages`, by reason. "
            "`no_detections` is a frame the detector never answered for; `out_of_order` is a "
            "frame that lost the ordering race. Both are emitted rather than failed: a frame "
            "with an honest gap is worth more than no frame at all.",
        )
        self.regressions = registry.counter(
            "shipinfer_track_implicit_resets_total",
            "Cameras whose frame_id went backwards far enough to be a restarted stream that "
            "nobody announced, per camera. Every track id under that camera changed at this "
            "moment; a deployment seeing these should be sending remove_camera + add_camera.",
        )
        self.cameras = registry.gauge(
            "shipinfer_track_cameras",
            "Cameras holding a tracker on this element. Labelled by element because two "
            "`track` slots in one chain keep two independent tables, and one gauge would "
            "report whichever wrote last.",
        )

    def out_of_order_frame(self, camera_id: str) -> None:
        if self.out_of_order is not None:
            self.out_of_order.inc(camera=camera_id)
        if self.untracked is not None:
            self.untracked.inc(reason="out_of_order")

    def untracked_frame(self, reason: str) -> None:
        if self.untracked is not None:
            self.untracked.inc(reason=reason)

    def implicit_reset(self, camera_id: str) -> None:
        if self.regressions is not None:
            self.regressions.inc(camera=camera_id)

    def camera_count(self, count: int) -> None:
        if self.cameras is not None:
            self.cameras.set(count, element=self.element)


@registry_for(ElementKind.TRACK).register("shipvision")
class ShipvisionTrack(Element):
    """Per-camera multi-object tracking over ``shipvision.mot``.

    Reads ``meta["detections"]`` — the decoded, source-pixel
    :class:`~shipinfer.topology.elements.detections.Detections` a ``pool`` detector files —
    and optionally ``meta["vectors"]``, and writes ``meta["tracks"]`` plus
    ``meta["track_rows"]``, the detection row each track came from, which is what lets an
    ``output`` element put a track id on the object it belongs to.

    **The caps.** ``accepts`` lists ``meta@cpu`` first because a tracker wants boxes and not
    pixels: fed by another metadata element it never touches a frame. ``bgr@cpu`` and
    ``nv12@gpu`` follow because today's chain is host memory end to end and phase D's is device
    memory end to end, and in both this element is where the chain leaves the frame behind.
    ``produces`` is one entry, so this element **does** stamp its own cap and clear the payload.

    **What it never does is fail a frame for a tracking refusal.** An out-of-order frame is
    counted and emitted with ``track`` in ``missing_stages``; so is a frame the detector never
    answered for. Anything raised from here is a genuine fault — a malformed payload, a broken
    configuration — and costs that one item.

    ``params:`` takes:

    * ``algorithm`` — a name in ``shipvision.mot.TRACKERS``. Default
      :data:`DEFAULT_ALGORITHM`.
    * ``options: {...}`` — constructor keyword arguments for that tracker (``max_age``,
      ``min_hits``, ``track_threshold``, ...). A key the tracker does not accept stops the
      deploy at ``open()`` rather than on the first frame.
    * ``classes: [ship, person]`` — the detection labels to track. Default: every label the
      detector produced. Names and not ids, because
      :class:`~shipinfer.topology.elements.detections.Detections` resolves the id table once
      and a second copy here is a second thing to get wrong.
    * ``regression_reset`` — see :data:`DEFAULT_REGRESSION_RESET`.

    There is deliberately **no ``backend:``**. ``TRACKERS.build`` with no backend resolves the
    fastest one this host can build with a numpy floor, which is what every caller here wants;
    naming ``native`` would make a chain that loads on the build machine refuse on a machine
    without the extension, for a tracker costing tens of microseconds against a millisecond
    frame budget.
    """

    kind: ClassVar[ElementKind] = ElementKind.TRACK
    #: ``meta@cpu`` first: a tracker fed by another metadata element never sees a frame. The
    #: other two are the same chain before and after phase D, and this element is the plane
    #: change in both.
    accepts: ClassVar[tuple[str, ...]] = ("meta@cpu", "bgr@cpu", "nv12@gpu")
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)
    #: It picks the rows it tracks with ``params: classes:``, exactly as a crop element picks
    #: the rows it embeds, so the loader refuses a ``when: class == …`` on this slot too and
    #: names the key that does the job.
    selects_rows: ClassVar[bool] = True
    # `requires_model_name`, `needs_model` and `needs_image_ops` all keep the ABC's `False`,
    # and each one is an answer rather than an omission: a MOT algorithm is not a repository
    # model (so no `model:` in the chain and no `InferenceServer` behind the runner), and it
    # reads boxes rather than pixels (so no letterbox and no `runtime.ops`).

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        self._algorithm = str(self.params.get("algorithm", DEFAULT_ALGORITHM))
        options = self.params.get("options") or {}
        if not isinstance(options, Mapping):
            raise ConfigurationError(
                f"track element {name!r}: `params: options:` must be a mapping of tracker "
                f"keyword arguments, got {type(options).__name__}"
            )
        self._options: dict[str, Any] = dict(options)
        self._classes = parse_classes(
            self.params.get("classes"), f"track element {self.name!r}"
        )
        self._regression_reset = self._parse_regression_reset(
            self.params.get("regression_reset", DEFAULT_REGRESSION_RESET)
        )
        self._max_attribution_cost = 1.0 - self._parse_attribution_iou(
            self.params.get("attribution_iou", DEFAULT_ATTRIBUTION_IOU)
        )
        self._shard: TrackerShard | None = None
        self._metrics = _TrackMetrics(None, name)
        # Bound once, next to the other resolve-once handles: `self._metrics.implicit_reset`
        # written at the call site mints a bound-method object on every frame, and this one is
        # on the per-frame path at a thousand frames a second (CONVENTIONS §2.5).
        self._on_implicit_reset: Callable[[str], None] = self._metrics.implicit_reset
        # What the gauge last reported, so the per-frame path is an int compare and the write
        # happens only when a camera appears or leaves.
        self._reported_cameras = -1
        # Resolved once at open: the three shipvision types this element builds per frame.
        # Bound as attributes rather than walked off the module every frame, because
        # `types.Detection` is two dict lookups and this runs fifteen thousand times a second.
        self._Detection: Any = None
        self._Detections: Any = None
        self._FrameTag: Any = None
        # The two halves of the row attribution below, resolved at open for the same reason
        # the three above are: `_attribute` runs once per frame with tracks on it.
        self._iou_matrix: Any = None
        self._associate: Any = None

    def _parse_attribution_iou(self, declared: Any) -> float:
        """The overlap below which a track is not attributed to a detection row.

        Raises:
            ConfigurationError: not a number, or outside ``(0, 1]``. ``0`` would attribute every
                track to whatever box it overlapped least badly, which is the guess this
                threshold exists to refuse.
        """
        try:
            value = float(declared)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"track element {self.name!r}: `params: attribution_iou:` must be a number "
                f"in (0, 1], got {declared!r}"
            ) from exc
        if not 0.0 < value <= 1.0:
            raise ConfigurationError(
                f"track element {self.name!r}: `params: attribution_iou:` must be in (0, 1], "
                f"got {value}"
            )
        return value

    def _parse_regression_reset(self, declared: Any) -> int:
        try:
            value = int(declared)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"track element {self.name!r}: `params: regression_reset:` must be a number "
                f"of frames, got {declared!r}"
            ) from exc
        if value < 0:
            raise ConfigurationError(
                f"track element {self.name!r}: `params: regression_reset:` must not be "
                f"negative, got {value}; 0 refuses every regression"
            )
        return value

    # -- lifecycle ---------------------------------------------------------------------

    def _do_open(self, context: ElementContext) -> None:
        """Build the per-camera table and resolve the vocabulary — both now, neither per frame.

        Raises:
            ConfigurationError: ``3rdparty/shipvision`` is not checked out or not installed (the
                bridge's refusal, carrying the command that fixes it), the algorithm is unknown,
                or ``options`` holds a key the tracker's constructor does not accept. All three
                stop the deploy rather than surfacing identically on every frame from inside a
                worker thread.
        """
        types = load_types()
        self._Detection = types.Detection
        self._Detections = types.Detections
        self._FrameTag = types.FrameTag
        self._iou_matrix = types.iou_matrix
        self._associate = load_mot().association.associate
        self._shard = TrackerShard(self._algorithm, options=self._options, backend=None)
        self._metrics = _TrackMetrics(context.metrics, self.name)
        self._on_implicit_reset = self._metrics.implicit_reset
        self._reported_cameras = -1
        self._note_cameras()

    def _do_close(self) -> None:
        # The trackers go with the element: their state is a camera's identity history and
        # keeping it across a close would mean a reopened chain continuing ids from before the
        # gap it cannot see.
        self._shard = None
        self._Detection = None
        self._Detections = None
        self._FrameTag = None
        self._iou_matrix = None
        self._associate = None

    def camera_added(self, camera_id: str) -> None:
        """Restart this camera's tracker if it already had one.

        A re-added camera is one whose ingest actor minted a fresh ``FrameCounter``, so its next
        frame is ``frame_id = 0`` — below the high-water mark the previous run left, and refused
        forever without this. ADR-018 names remove + add as the recovery for a lost camera, so
        this state has to be one the chain can be in.

        Never *builds* a tracker: this fires for every camera placed on the shard, and minting a
        Kalman filter for forty-nine cameras that are not on this element is work for nothing.

        Unlike :meth:`camera_removed`, this one **can wait** — for one in-flight frame on this
        one camera, while holding the runner's ``_lifecycle``. See
        :meth:`TrackerShard.reset_if_present` for why that residual is accepted: a reset has to
        agree with the update it interrupts, and a drop does not.
        """
        if self._shard is not None:
            self._shard.reset_if_present(camera_id)

    def camera_removed(self, camera_id: str) -> None:
        """Drop this camera's tracker, under the table lock and nothing else.

        See :meth:`TrackerShard.drop` for why it does not wait for an in-flight frame: this
        runs holding the runner's ``_lifecycle``, and every other lifecycle operation on the
        shard — including the ``stop`` that would end a wait — queues behind it.
        """
        if self._shard is not None and self._shard.drop(camera_id):
            self._note_cameras()

    # -- one frame ---------------------------------------------------------------------

    def _do_process(self, item: ChainItem) -> ChainItem:
        """Track one frame, and hand on metadata rather than pixels.

        Returns:
            The successor item: same tag, ``caps`` = ``meta@cpu``, ``payload`` cleared, and
            ``meta["tracks"]`` / ``meta["track_rows"]`` holding this frame's publishable tracks
            and the detection row each came from. ``meta["frame_hw"]`` rides along from the
            detector, because the boxes are in that extent's pixels and the payload that could
            have said so is gone.

            A frame the detector never answered for, and a frame that lost the ordering race,
            come back with the same shape and ``track`` in ``meta["missing_stages"]`` — never as
            ``None`` and never as an exception, because "no ships in this frame", "the detector
            is dead" and "this frame arrived late" are three events and only one is a fault
            (arch.md section 5(5)).

        Raises:
            ValidationError: ``meta["detections"]`` is not a
                :class:`~shipinfer.topology.elements.detections.Detections`, ``meta["vectors"]``
                cannot be attributed to detections, or the item carries no frame id. All three
                are a mis-wired chain rather than a late frame.
            ServerStateError: called before :meth:`Element.open`.
        """
        detections = item.meta.get("detections")
        if detections is None:
            # The detector did not answer. Distinct from "no objects": an empty `Detections`
            # still advances the tracker below, because ageing is how a track dies.
            self._metrics.untracked_frame("no_detections")
            return self._untracked(item)
        if not isinstance(detections, Detections):
            raise ValidationError(
                f"track element {self.name!r} was handed meta['detections'] of type "
                f"{type(detections).__name__} and needs a decoded Detections in source-frame "
                "pixels; a `pool` detector files one, and raw model outputs are not it"
            )

        assert self._shard is not None  # `process` refuses before `open`
        camera_id = item.context.camera_id
        keep = self._selected(detections)
        vision = self._as_vision_detections(item, detections, keep)
        try:
            tracks = self._shard.update(
                vision,
                regression_reset=self._regression_reset,
                on_implicit_reset=self._on_implicit_reset,
            )
        except TrackingError:
            # Counted and emitted, never raised on: the runner fails an item's future on any
            # exception, and this frame's boxes, masks and vectors are all still good.
            self._metrics.out_of_order_frame(camera_id)
            return self._untracked(item)
        self._note_cameras()
        # `tracks` is handed on as shipvision's own `Track` objects, deliberately: they
        # are exactly what the cross-camera tier consumes
        # (`mtmc.CameraTracks(tag, tracks, h, w)`), so a pure record here would be a shape
        # `mtmc` converts straight back, losing the embedding, the state and the tag.
        #
        # Buffering them past the camera's next frame is safe on **both** pool backends,
        # for two different reasons, and a reader who changes either has to keep the
        # property: the python pool mutates its `Track`s in place and hands back
        # `dataclasses.replace` copies; the native pool — what `backend=None` resolves to
        # wherever `shipvision._C` is built, so usually this one — hands back the live
        # list uncopied, and is safe because `_NativePool.decode` mints fresh `Track`s off
        # a freshly allocated array every frame. The property is pinned by
        # `test_the_published_tracks_are_not_rewritten_by_the_next_frame`, on whichever
        # backend this host built, rather than by either mechanism.
        return item.derive(
            caps=self.output_caps[0],
            payload=None,
            tracks=tracks,
            **{TRACK_ROWS: self._attribute(detections, keep, tracks)},
        )

    def _attribute(
        self,
        detections: Detections,
        keep: range | tuple[int, ...],
        tracks: Sequence[Any],
    ) -> tuple[int, ...]:
        """Which detection row each published track came from. Aligned with ``tracks``.

        The library returns tracks, not row indices, and it is right not to: a track's box is
        the *filtered* estimate, and which detection fed it is the tracker's business. But an
        emitted event is per detection — ``det_id``, ``bbox``, ``embedding`` and the identity
        fields are one row — so the mapping has to be recovered, and a published track is by
        construction the corrected state of one of this frame's detections, which makes the
        recovery an assignment over IoU rather than a guess.

        One-to-one and globally optimal, through the same solver the trackers use
        (``shipvision.mot.association.associate``) — this is
        ``pipeline/graph/tracking.py::_attribute``'s algorithm in the chain's vocabulary. Greedy
        per-track argmax is wrong in exactly the case that matters: two people walking close
        together each overlap the other's box, and a greedy pass can give both track ids to one
        detection and none to the other, which reads downstream as one person who teleported and
        one who never existed.

        A track that matches nothing above the threshold gets ``-1``. Dropped rather than forced
        onto its nearest box: the threshold exists to refuse an answer, not to tune one.

        Args:
            keep: the rows that were fed to the tracker, so a slot with ``classes:`` maps its
                tracks back to the frame's own row numbering rather than to the subset's.
        """
        rows = [-1] * len(tracks)
        if not tracks or not len(detections):
            return tuple(rows)
        candidates = detections.boxes_at(keep)
        if not candidates.shape[0]:
            return tuple(rows)
        # Both, not one: `_do_open` sets them on adjacent lines, so an assert on half of
        # that pair documents half a precondition.
        assert self._iou_matrix is not None and self._associate is not None  # `process` refuses
        cost = 1.0 - self._iou_matrix(np.stack([t.box for t in tracks]), candidates)
        matches, _, _ = self._associate(cost, self._max_attribution_cost)
        # `keep` indexes directly whether it is a `range` or a `tuple`; materialising it
        # allocated the whole selection once per frame on the attribution path.
        for track_index, column in matches:
            rows[int(track_index)] = int(keep[int(column)])
        return tuple(rows)

    def _untracked(self, item: ChainItem) -> ChainItem:
        """The successor for a frame this element could not track: same plane, no tracks."""
        return item.derive(
            caps=self.output_caps[0],
            payload=None,
            **{MISSING_STAGES: (*item.meta.get(MISSING_STAGES, ()), "track")},
        )

    def _as_vision_detections(
        self, item: ChainItem, detections: Detections, keep: range | tuple[int, ...]
    ) -> Any:
        """This frame's detections in the tracking library's vocabulary, embeddings attached.

        The conversion is per object and this is a hot path, but it is bounded by
        ``max_detections`` and it is the price of a shared vocabulary: the alternative is a
        second box format in the system, and ``shipvision.types`` names the exact bug that
        causes — a converter that wrote width where height belonged tracks square objects
        perfectly and falls apart on a ship.

        Raises:
            ValidationError: the item carries no frame id, or ``meta["vectors"]`` cannot be
                attributed to detection rows. The frame-id check is here rather than left to
                ``FrameTag`` because ``FrameTag`` refuses it with
                ``shipvision.errors.ConfigurationError`` — a foreign class, from the one
                hand-over where this element speaks another library's vocabulary, and the only
                place in the chain a caller could be handed somebody else's exception type.
        """
        frame_id = item.context.frame_id
        if frame_id < 0:
            raise ValidationError(
                f"track element {self.name!r} was handed an item for camera "
                f"{item.context.camera_id!r} with frame_id {frame_id}: tracking is ordered by "
                "frame id, and a negative one is the RequestContext default, which means "
                "nothing upstream ever tagged this frame"
            )
        vectors = self._embeddings(item, detections)
        boxes = detections.boxes
        scores = detections.scores
        class_ids = detections.class_ids
        height, width = item.meta.get("frame_hw", (0, 0))
        return self._Detections(
            tag=self._FrameTag(
                camera_id=item.context.camera_id,
                frame_id=frame_id,
                timestamp=item.context.captured_unix_ns / 1e9,
            ),
            items=[
                self._Detection(
                    box=boxes[index],
                    # Clamped rather than passed through: `Detection` refuses a score outside
                    # [0, 1], and an fp16 engine that reports 1.0000001 would otherwise fail
                    # tracking on every frame it appeared in. The value is the detector's
                    # confidence, and one ulp of it is not information worth a dropped frame.
                    score=min(1.0, max(0.0, float(scores[index]))),
                    class_id=int(class_ids[index]),
                    embedding=None if vectors is None else vectors[index],
                )
                for index in keep
            ],
            height=int(height),
            width=int(width),
        )

    def declared_classes(self) -> tuple[str, ...] | None:
        """See :meth:`~shipinfer.topology.base.Element.declared_classes`."""
        return self._classes

    def _selected(self, detections: Detections) -> range | tuple[int, ...]:
        """Which detection rows this element tracks — every one, or the declared classes.

        A ``range`` in the common case, so the "track everything" path allocates nothing per
        frame beyond what the tracker itself needs. The label match itself belongs to
        :meth:`~shipinfer.topology.elements.detections.Detections.indices_of_any`, so this
        element and every crop element select rows by one rule rather than by two copies of
        one.
        """
        if self._classes is None:
            return range(len(detections))
        return detections.indices_of_any(self._classes)

    def _embeddings(self, item: ChainItem, detections: Detections) -> Any:
        """``meta["vectors"]`` as one appearance vector per detection row, or ``None``.

        The attribution rule is :func:`~shipinfer.topology.elements.detections.per_row`, because
        the ``output`` element reads the same key under the same rule and two copies would be
        two places for "a mapping that covers no row is an off-by-N" to drift — a drift with no
        symptom, since both readers fail by quietly attaching nothing.

        What is local to *this* element is why the refusal matters here: appearance carries an
        identity through the frames where geometry alone is ambiguous, so a tracker that quietly
        ran without the vectors a deployment paid a re-ID network for is a measurable accuracy
        loss reported as a healthy chain.

        Returns:
            Something indexable by detection row, or ``None`` when the chain filed no vectors.

        Raises:
            ValidationError: the vectors are not attributable to detection rows.
        """
        return per_row(
            item.meta.get("vectors"),
            len(detections),
            what=f"track element {self.name!r}",
            key="vectors",
        )

    # -- metrics -----------------------------------------------------------------------

    def _note_cameras(self) -> None:
        """Publish the tracker count, and only when it changed.

        An int compare on the per-frame path and a gauge write when a camera appears or
        leaves. Two workers can race the compare; a gauge that is one frame late is a gauge,
        and taking a lock per frame to keep it exact would cost more than the number is worth.
        """
        if self._shard is None:
            return
        count = self._shard.camera_count
        if count != self._reported_cameras:
            self._reported_cameras = count
            self._metrics.camera_count(count)

    def __repr__(self) -> str:
        cameras = 0 if self._shard is None else self._shard.camera_count
        return f"<ShipvisionTrack {self.name} {self._algorithm} cameras={cameras}>"
