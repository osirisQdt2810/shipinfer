"""The ``mtmc`` element: one identity per object across every camera that can see it.

Cross-camera association is not a per-frame function. ``shipvision.mtmc`` consumes a
:class:`FrameTrackCluster` — *every* camera of a group at one synchronised instant — and
refuses anything less, because handing it one camera at a time turns cross-camera
association into within-camera deduplication. The chain, however, delivers **one frame at a
time**, on whichever pipeline worker happened to take it off the fair lane. Turning that
stream back into instants is :class:`~shipinfer.topology.barrier.InstantBarrier`'s job, in
its own pure module; this one is the element that gives the barrier's opaque payloads
meaning.

The split is the same one ``track.py`` makes. Everything worth testing about cross-camera
*synchronisation* — who waits, who closes, who is told to give up — is a property of the
barrier and is tested with integers and a callback in ``tests/topology/test_barrier.py``,
with no submodule anywhere near it. :class:`ShipvisionMtmc` turns a :class:`ChainItem` into a
``CameraTracks`` view, hands it to the barrier, and turns the answer back into metadata.

**Why the scatter is keyed and never positional.** ``FrameTrackCluster`` flattens every
camera's tracks into one observation list, and ``ClusterMTMCTracker.track`` answers in that
flattened order. Camera A's three tracks and camera B's one track come back as four results
whose positions mean nothing to either frame. The results are therefore indexed by
``(camera_id, track_id)`` — ``shipvision``'s own ``TrackKey`` — and each waiting item reads
its own tracks out of that map. Scattering by list position is the classic reassembly bug
(ADR-002's tag rule, one layer up), and it produces a plausible answer rather than an error.

**Why the association runs under the barrier's own lock.** Whichever thread closes a bucket
calls the tracker with the barrier lock held. That is one lock, which is what the GIL law
(``docs/arch.md`` §7, V142) allows: ``ClusterMTMCTracker`` already holds an ``RLock`` for the
whole of ``track()``, so a second lock of ours around the call would buy nothing, and the
barrier's lock is the one that has to be held anyway to publish the results to the waiters
atomically.

The cost is that a submit for the *next* instant queues behind the association for its
duration, and that duration is **milliseconds, not microseconds**. Measured on this host, CPU
only, with 128-d embeddings and a warm tracker:

===========  ================  ==============  ==============
Cameras      Tracks per frame  Observations    Median
===========  ================  ==============  ==============
2            2                 4               0.48 ms
8            2                 16              0.88 ms
8            15                120             4.70 ms
16           15                240             10.54 ms
50           15                750             54.32 ms
===========  ================  ==============  ==============

Fifteen tracks per frame is ``CLAUDE.md``'s own people-per-frame figure, so a realistic group
of eight cameras on a quay holds the barrier for ~4.7 ms — about 8% of a 60 ms window — and
the growth is **quadratic in observations**, not linear in cameras. That is what makes a
group a *group* rather than the whole fleet: fifty cameras in one ``mtmc`` slot is 54 ms of
serialised association per instant, which is a frame period, and the answer to it is more
groups rather than more locks. ``shipinfer_mtmc_association_us`` is the instrument.

**Where ``shipvision`` is named.** Nowhere at module scope — every symbol comes from
:mod:`shipinfer.topology.bridge` inside ``_do_open``, so ``import shipinfer.topology`` stays
free of the submodule and a chain naming ``impl: shipvision`` is still validatable on a host
that never checked it out.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from shipinfer.core.errors import ConfigurationError, ValidationError
from shipinfer.core.logging import get_logger
from shipinfer.topology.barrier import (
    DEFAULT_MAX_INSTANTS,
    DEFAULT_SYNC_WINDOW_MS,
    DROPPED_FAILED,
    DROPPED_SHUTDOWN,
    MISSED_LATE,
    MISSED_WOULD_STARVE,
    InstantBarrier,
    InstantEntry,
)
from shipinfer.topology.base import (
    CameraGroup,
    ChainItem,
    Element,
    ElementContext,
    ElementKind,
)
from shipinfer.topology.bridge import load_errors, load_mtmc, load_types
from shipinfer.topology.elements.track import MISSING_STAGES
from shipinfer.topology.registry import registry_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.core.metrics import Counter, Gauge, Histogram

__all__ = [
    "DEFAULT_ALGORITHM",
    "DEFAULT_CLUSTERER",
    "DEFAULT_MATRIX_BUILDER",
    "DEFAULT_MAX_INSTANTS",
    "DEFAULT_SYNC_WINDOW_MS",
    "MISSED_LATE",
    "MISSED_UNASSIGNABLE",
    "MISSED_WOULD_STARVE",
    "MISSING_TRACKS",
    "ShipvisionMtmc",
    "parse_group",
]

_LOG = get_logger("topology.mtmc")

# -- the element's own vocabulary ---------------------------------------------------------
#
# The barrier owns the rest of it: what happened to an *instant* (`complete`, `window`,
# `advanced`, `evicted`, `expired`, `shutdown`, `failed`) and what happened to a *frame* that
# missed one (`late`, `duplicate`, `would_starve`). Those live in `topology/barrier.py` and
# four of them are re-exported here because they are the `reason=` labels this element's
# metrics carry. The two below are the element's own, because only an element that knows what
# a track is can produce them.

#: ``meta["tracks"]`` was absent -- the ``track`` element never answered for this frame.
MISSING_TRACKS = "no_tracks"
#: The library refused to assign the instant: some track in it carries no appearance vector,
#: and cross-camera identity is decided on appearance. A *data* condition and not a fault --
#: an embedder that was spilled, a crop that produced nothing, or a chain with no embedder in
#: front of ``track`` at all -- so the frame is emitted with a gap and the first one is logged.
MISSED_UNASSIGNABLE = "unassignable"

#: The cross-camera algorithm a slot gets when it does not say.
DEFAULT_ALGORITHM = "cluster"
#: The matcher. ``gated`` is appearance vetoed by geometry and degrades to appearance-only
#: when no homographies are supplied, which is what makes it safe as the default.
DEFAULT_MATRIX_BUILDER = "gated"
#: The clusterer. One implementation exists; naming it here keeps the seam visible.
DEFAULT_CLUSTERER = "agglomerative"


def parse_group(params: Mapping[str, Any], *, where: str) -> tuple[str, tuple[str, ...]]:
    """The ``group:`` name and ``cameras:`` roster an ``mtmc`` slot declares.

    One parser, one caller: :class:`ShipvisionMtmc.__init__`. What every *other* reader needs
    is the parsed answer, and it gets it from :meth:`ShipvisionMtmc.camera_group` — the
    ``Element`` hook a runner asks — rather than by parsing ``params:`` a second time. Two
    readers of one YAML shape is exactly how a key drifts.

    Args:
        params: the slot's ``params:`` mapping, straight from the chain file.
        where: what to call the slot in a refusal -- ``"mtmc element 'mtmc'"``.

    Returns:
        ``(group, cameras)``. ``group`` falls back to ``""``, which the element resolves to
        its own slot name; ``cameras`` is empty when the file declared no roster.

    Raises:
        ConfigurationError: ``cameras:`` is not a list of names. A string there is the likely
            typo (``cameras: cam-a``) and would otherwise be read as five one-character
            camera ids.
    """
    group = str(params.get("group") or "")
    declared = params.get("cameras")
    if declared is None:
        return group, ()
    if isinstance(declared, str) or not isinstance(declared, Sequence):
        raise ConfigurationError(
            f"{where}: `params: cameras:` must be a list of camera ids, got "
            f"{type(declared).__name__}"
        )
    return group, tuple(str(entry) for entry in declared)


class _NeverRaised(BaseException):
    """The ``except`` clause an unopened element has: a class nothing ever raises.

    Cheaper and safer than an ``if`` on the per-frame path, and safer than ``BaseException``,
    which would swallow a ``KeyboardInterrupt`` in the window before ``_do_open`` resolved the
    real class.
    """


# -- the element ------------------------------------------------------------------------


class _MtmcMetrics:
    """The element's metric handles, resolved once at ``open``.

    Same null-object shape and the same reason as ``track.py``'s: a metric looked up by
    string on the per-frame path is a hash and a dict probe nobody needs to pay for, and
    ``context.metrics is None`` gets one answer instead of an ``if`` per call.
    """

    __slots__ = ("cameras", "element", "instants", "late", "latency", "missing", "starved")

    def __init__(self, registry: Any, element: str) -> None:
        self.element = element
        counter = getattr(registry, "counter", None)
        if counter is None:
            self.instants: Counter | None = None
            self.missing: Counter | None = None
            self.late: Counter | None = None
            self.starved: Counter | None = None
            self.cameras: Gauge | None = None
            self.latency: Histogram | None = None
            return
        self.instants = registry.counter(
            "shipinfer_mtmc_instants_total",
            "Synchronised instants this element resolved, by reason. `complete` is every "
            "live camera reporting in time and is the number that should dominate; `window` "
            "means a camera did not make it and the association ran without it; `evicted` "
            "and `expired` mean instants nobody could use. A rising `window` share is a "
            "clock or a lane-depth problem, not an MTMC one.",
        )
        self.missing = registry.counter(
            "shipinfer_mtmc_frames_missing_total",
            "Frames emitted with `mtmc` in `missing_stages`, by reason. Every one of these "
            "carries its boxes, vectors and per-camera track ids and lacks only the global "
            "id, which is worth far more than a dropped frame. The reasons mix two "
            "vocabularies on purpose: `late`, `duplicate` and `would_starve` are what "
            "happened to this frame, while `evicted`, `expired`, `failed` and `unassignable` "
            "are what happened to the *instant* it was in -- one such instant produces one "
            "of these per camera in it, so read this against "
            "`shipinfer_mtmc_instants_total` rather than as a count of instants.",
        )
        self.late = registry.counter(
            "shipinfer_mtmc_frames_late_total",
            "Frames that arrived after their instant had already been associated. Never "
            "retro-fitted: a second association for one late camera would issue global ids "
            "contradicting the ones already published for that moment.",
        )
        self.starved = registry.counter(
            "shipinfer_mtmc_would_starve_total",
            "Frames emitted immediately because waiting would have parked the last pipeline "
            "worker. Non-zero is normal under burst; sustained and high means the shard has "
            "too few workers for its camera group, and the fix is `pipeline.workers`.",
        )
        self.cameras = registry.gauge(
            "shipinfer_mtmc_cameras",
            "Cameras this element's barrier waits for. Labelled by element because two "
            "`mtmc` slots in one chain keep two independent groups.",
        )
        self.latency = registry.histogram(
            "shipinfer_mtmc_association_us",
            "Wall-clock microseconds inside one cross-camera association -- cluster "
            "construction, the tracker call and the scatter map. Held under the barrier lock, "
            "so this is also how long the next instant's first frame can queue.",
        )

    def instant(self, reason: str) -> None:
        if self.instants is not None:
            self.instants.inc(reason=reason)

    def frame_missing(self, reason: str) -> None:
        if self.missing is not None:
            self.missing.inc(reason=reason)
        if reason == MISSED_LATE and self.late is not None:
            self.late.inc()
        elif reason == MISSED_WOULD_STARVE and self.starved is not None:
            self.starved.inc()

    def association(self, microseconds: float) -> None:
        if self.latency is not None:
            self.latency.observe(microseconds)

    def camera_count(self, count: int) -> None:
        if self.cameras is not None:
            self.cameras.set(count, element=self.element)


@registry_for(ElementKind.MTMC).register("shipvision")
class ShipvisionMtmc(Element):
    """Cross-camera identity over ``shipvision.mtmc``, one group per element.

    Reads ``meta["tracks"]`` — the ``shipvision`` ``Track`` objects the ``track`` element
    files — and ``meta["frame_hw"]``, and writes ``meta["global_ids"]``.

    **``meta["global_ids"]`` is a list, aligned with this item's ``meta["tracks"]``**, one
    entry per track, ``int`` or ``None``. That is the shape ``meta['tracks']`` has and
    the shape an ``output`` element serialises beside the track ids it already carries. It is
    *built* from a mapping keyed on ``(camera_id, track_id)`` — the association answers for a
    whole group in one flattened list, and reading a camera's own ids out of it by position is
    the reassembly bug ADR-002 exists to prevent. ``None`` means the track was gated (too new
    or too small to identify), and it is ``None`` rather than ``-1`` or absent for the reason
    ``shipvision.types.GlobalTrack`` gives: ``-1`` sorts and serialises like an ordinary id.

    **The caps.** ``meta@cpu`` on both sides, and this element does **not** stamp a cap or
    touch the payload: it adds metadata to a frame that is already on the metadata plane. Only
    ``track`` changes plane, and it has already done so by the time an item reaches here.

    **A group of one camera is a legitimate no-op**, not a misconfiguration. Every instant
    completes on its first frame, the tracker clusters a single observation set, and each
    track gets its own global id. A deployment with one camera per group still wants stable
    ids across time, which is what ``GlobalIdAssigner`` gives it.

    ``params:`` takes:

    * ``algorithm`` — a name in ``shipvision.mtmc.MTMC``. Default :data:`DEFAULT_ALGORITHM`.
    * ``matrix_builder`` — the matcher, default :data:`DEFAULT_MATRIX_BUILDER`.
    * ``clusterer`` — default :data:`DEFAULT_CLUSTERER`.
    * ``group`` — the camera group's name. Defaults to the slot name. Used in refusals and
      published through :meth:`camera_group` to the runner, which places a group atomically.
    * ``cameras`` — the group's roster. **Does not drive the barrier** (see
      :meth:`~shipinfer.topology.barrier.InstantBarrier.camera_added`); it is what a runner
      reads through :meth:`camera_group` to keep the group on one shard, and what lets this
      element warn when a camera it was never told about turns up.
    * ``sync_window_ms`` — default :data:`DEFAULT_SYNC_WINDOW_MS`.
    * ``max_instants`` — default :data:`DEFAULT_MAX_INSTANTS`.
    * ``calibration`` — ``{camera_id: {matrix: [[...]], camera_width: ..., ...}}``. **Absent
      is a supported deployment, not a failure**: ``gated`` degrades to appearance-only
      without homographies, which is what a site that has not been surveyed yet gets. It is
      logged at ``open()`` so the degradation is on the record rather than inferred from an
      accuracy number.
    * ``options: {...}`` — the remaining ``ClusterMTMCTracker`` keyword arguments
      (``min_hits``, ``appearance_threshold``, ``distance_threshold``, ``max_age``,
      ``capacity``, …). A key the tracker does not accept stops the deploy at ``open()``.

    There is deliberately **no ``backend:``**, for the reason ``track.py`` gives: an unpinned
    ``MTMC.build`` takes the fastest matcher this host can build with a numpy floor, and
    naming ``native`` would make a chain that loads on the build machine refuse on one without
    the extension.

    **Placement is not checked here, and cannot be.** A camera group is an atomic unit of
    placement (``docs/arch.md`` §4) — split it across two shards and each half gets its own
    tracker, its own identity space, and two contradictory global ids for one object. This
    element cannot detect that: :attr:`ElementContext.shard_id` tells it which shard it is on
    and nothing tells it where any *camera* is, and ``open()`` runs before a single camera has
    been placed. The only component that knows the placement map is the runner, which owns
    ``{camera_id: shard_id}`` — so this element *declares* the membership through
    :meth:`camera_group` and :meth:`~shipinfer.runners.fleet.FleetRunner.add_camera` enforces
    it. The runner asks every node for a group and never asks what kind it is, which is why
    adding a second cross-camera element would need no edit to ``runners/``.
    """

    kind: ClassVar[ElementKind] = ElementKind.MTMC
    accepts: ClassVar[tuple[str, ...]] = ("meta@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)
    # `requires_model_name`, `needs_model` and `needs_image_ops` all keep the ABC's `False`:
    # cross-camera association is an algorithm over metadata, not a repository model, and it
    # never sees a pixel.

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        self._algorithm = str(self.params.get("algorithm", DEFAULT_ALGORITHM))
        self._matrix_builder = str(self.params.get("matrix_builder", DEFAULT_MATRIX_BUILDER))
        self._clusterer = str(self.params.get("clusterer", DEFAULT_CLUSTERER))
        group, roster = parse_group(self.params, where=f"mtmc element {name!r}")
        self._group = group or name
        self._roster = roster
        self._window_s = self._positive("sync_window_ms", DEFAULT_SYNC_WINDOW_MS) / 1e3
        self._max_instants = int(self._positive("max_instants", DEFAULT_MAX_INSTANTS))
        options = self.params.get("options")
        options = {} if options is None else options
        if not isinstance(options, Mapping):
            raise ConfigurationError(
                f"mtmc element {name!r}: `params: options:` must be a mapping of tracker "
                f"keyword arguments, got {type(options).__name__}"
            )
        self._options: dict[str, Any] = dict(options)
        calibration = self.params.get("calibration")
        calibration = {} if calibration is None else calibration
        if not isinstance(calibration, Mapping):
            raise ConfigurationError(
                f"mtmc element {name!r}: `params: calibration:` must be a mapping of camera "
                f"id to homography, got {type(calibration).__name__}"
            )
        self._calibration: dict[str, Any] = dict(calibration)
        self._tracker: Any = None
        self._barrier: InstantBarrier | None = None
        # The library's own refusal, resolved at open so the per-frame `except` names a class
        # rather than re-importing one. `BaseException` before that, which is unreachable --
        # `process` refuses before `open` -- and is a class no `except` clause will ever match
        # by accident.
        self._TrackingError: type[BaseException] = _NeverRaised
        self._warned_unassignable = False
        self._metrics = _MtmcMetrics(None, name)
        self._reported_cameras = -1
        #: Latched so the under-sized-group warning is one line per *crossing* rather than
        #: one per camera announcement.
        self._starved_group = False
        # Resolved once at open, so the per-frame path walks no module dictionaries.
        self._CameraTracks: Any = None
        self._FrameTrackCluster: Any = None
        self._FrameTag: Any = None

    def _positive(self, key: str, default: float) -> float:
        """A strictly positive number from ``params:``, or a typed refusal naming the key."""
        declared = self.params.get(key, default)
        try:
            value = float(declared)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"mtmc element {self.name!r}: `params: {key}:` must be a number, got "
                f"{declared!r}"
            ) from exc
        if value <= 0.0:
            raise ConfigurationError(
                f"mtmc element {self.name!r}: `params: {key}:` must be positive, got {value}"
            )
        return value

    @property
    def group(self) -> str:
        """The camera group this element associates over."""
        return self._group

    @property
    def roster(self) -> tuple[str, ...]:
        """The declared membership of the group; empty when the chain named none."""
        return self._roster

    def camera_group(self) -> CameraGroup | None:
        """This slot's declared roster, for the runner that places cameras.

        The :class:`~shipinfer.topology.base.Element` hook, answered from ``params:`` and
        therefore answerable before ``open()`` — which matters, because the runner asks when
        it is built and long before a camera exists.

        ``None`` when the chain declared no ``cameras:``. That is not "an empty group": a
        chain that did not say gets its cameras placed by load, and the group is whatever
        ended up together, which is the honest answer. Declaring the roster is what buys the
        atomic-placement invariant.
        """
        if not self._roster:
            return None
        return CameraGroup(self._group, self._roster)

    @property
    def barrier(self) -> InstantBarrier | None:
        """The barrier, for tests and for a health report. ``None`` before ``open()``."""
        return self._barrier

    # -- lifecycle ---------------------------------------------------------------------

    def _do_open(self, context: ElementContext) -> None:
        """Build the tracker and the barrier — both now, neither per frame.

        Raises:
            ConfigurationError: ``3rdparty/shipvision`` is not checked out or not installed
                (the bridge's refusal, carrying the command that fixes it); the algorithm,
                matcher or clusterer is unknown; ``options`` holds a key the tracker does not
                accept; or a ``calibration`` entry is not a homography. All of them stop the
                deploy rather than surfacing identically on every frame from a worker thread.
        """
        mtmc = load_mtmc()
        types = load_types()
        self._TrackingError = load_errors().TrackingError
        self._CameraTracks = mtmc.CameraTracks
        self._FrameTrackCluster = mtmc.FrameTrackCluster
        self._FrameTag = types.FrameTag
        self._tracker = mtmc.MTMC.build(
            self._algorithm,
            matrix_builder=self._matrix_builder,
            clusterer=self._clusterer,
            ground_plane=self._ground_plane(mtmc),
            **self._options,
        )
        self._metrics = _MtmcMetrics(context.metrics, self.name)
        self._barrier = InstantBarrier(
            sync_window_s=self._window_s,
            workers=context.workers,
            budget=context.waiter_budget,
            max_instants=self._max_instants,
            on_event=self._metrics.instant,
        )
        self._reported_cameras = -1
        self._starved_group = False
        self._note_cameras()
        if context.workers is None and self._barrier.budget.permits:
            # A supplied budget wins over the worker count, so this barrier *does* wait --
            # saying it would not would send an operator looking for the wrong symptom.
            _LOG.warning(
                "mtmc element %r was given no worker count but was given a waiter budget: "
                "group %r will wait for the rest of its instant on the budget's %d "
                "permit(s), not on a worker count. The runner sets ElementContext.workers",
                self.name,
                self._group,
                self._barrier.budget.permits,
            )
        elif context.workers is None:
            _LOG.warning(
                "mtmc element %r was given no worker count and no waiter budget; it will "
                "emit every frame immediately rather than wait for the rest of group %r. "
                "The runner sets ElementContext.workers",
                self.name,
                self._group,
            )
        if self._roster:
            self._warn_if_workers_cannot_cover(len(self._roster), "its declared roster")

    def _ground_plane(self, mtmc: Any) -> Any:
        """The group's homographies, or ``None`` for an appearance-only deployment.

        Args:
            mtmc: the ``shipvision.mtmc`` module, already loaded by the caller.

        Raises:
            ConfigurationError: an entry is not a mapping of ``Homography`` fields, or the
                matrix is not a usable homography. Named per camera, because "the calibration
                is wrong" without saying which camera sends an operator to re-survey a site.
        """
        if not self._calibration:
            _LOG.info(
                "mtmc element %r (group %r) has no `calibration:`; the %r matcher runs "
                "appearance-only. Cross-camera association will not use ground-plane "
                "geometry until homographies are supplied",
                self.name,
                self._group,
                self._matrix_builder,
            )
            return None
        homographies = {}
        for camera_id, spec in self._calibration.items():
            if not isinstance(spec, Mapping):
                raise ConfigurationError(
                    f"mtmc element {self.name!r}: calibration for camera {camera_id!r} must "
                    f"be a mapping with at least `matrix:`, got {type(spec).__name__}"
                )
            try:
                homographies[str(camera_id)] = mtmc.Homography(**dict(spec))
            except Exception as exc:
                raise ConfigurationError(
                    f"mtmc element {self.name!r}: calibration for camera {camera_id!r} is "
                    f"not a usable homography ({exc})"
                ) from exc
        return mtmc.GroundPlane(homographies)

    def _do_close(self) -> None:
        """Release every waiter, then drop the tracker.

        ``close_all`` first and unconditionally, and it is **belt and braces** rather than the
        guarantee: ``InprocessRunner._do_stop`` releases the cameras, closes the queue and
        *joins the workers* before it closes the chain, so in the ordinary shutdown no worker
        is parked in the barrier by the time this runs — and every wait is bounded by its own
        bucket's deadline anyway. It is here because a wait that is only bounded by a shutdown
        call becomes unbounded the first time somebody forgets to make it, and because a
        runner that closed a chain without joining its workers would otherwise pay a window
        per parked frame. See :meth:`~shipinfer.topology.barrier.InstantBarrier.close_all`.

        The identities go with the element for the reason the trackers do in ``track.py`` —
        keeping them across a close would mean a reopened chain continuing global ids over a
        gap it cannot see.
        """
        if self._barrier is not None:
            self._barrier.close_all(DROPPED_SHUTDOWN)
        self._barrier = None
        self._tracker = None
        self._TrackingError = _NeverRaised
        self._warned_unassignable = False
        self._starved_group = False
        self._CameraTracks = None
        self._FrameTrackCluster = None
        self._FrameTag = None

    def camera_added(self, camera_id: str) -> None:
        """This camera is live: instants now wait for it.

        Takes the barrier's lock and does no work under it — but it can **queue behind an
        association in progress**, which holds that lock for one ``tracker.track()`` call:
        ~0.5 ms for a pair of cameras, ~4.7 ms for a realistic group of eight (see the module
        docstring's table). That is inside the ABC's "return promptly" and well under the
        lifecycle deadlines, and it is not "returns immediately" — this runs on the thread
        holding the runner's lifecycle lock, behind which every other lifecycle call queues,
        so the bound is worth stating rather than assuming.

        Warns, but does not refuse, when the camera is outside a declared roster: a roster is
        written once and cameras are added by API at run time, so refusing here would make a
        stale list able to reject a live camera. The warning is the record that the two
        disagree.
        """
        if self._barrier is None:
            return
        if self._roster and camera_id not in self._roster:
            _LOG.warning(
                "mtmc element %r: camera %r is not in group %r's declared roster %s; "
                "associating it anyway. Update `params: cameras:` — the fleet reads it to "
                "keep the group on one shard",
                self.name,
                camera_id,
                self._group,
                list(self._roster),
            )
        self._barrier.camera_added(camera_id)
        self._note_cameras()

    def camera_removed(self, camera_id: str) -> None:
        """This camera is gone: stop waiting for it, in open instants too.

        See :meth:`~shipinfer.topology.barrier.InstantBarrier.drop_camera` — the second half
        is what keeps a removed camera from taxing every subsequent instant the full
        ``sync_window_ms``. It never runs an association itself, but like
        :meth:`camera_added` it can queue behind one for a few milliseconds; that bound, and
        not "immediately", is the promise.
        """
        if self._barrier is None:
            return
        self._barrier.drop_camera(camera_id)
        self._note_cameras()

    # -- one frame ---------------------------------------------------------------------

    def _do_process(self, item: ChainItem) -> ChainItem:
        """Join this frame's instant and hand back the group's global ids.

        Returns:
            The successor item: same tag, same caps, same payload, plus
            ``meta["global_ids"]`` — one entry per track in ``meta["tracks"]``.

            A frame the tracker never answered for, and a frame that missed its instant for
            any of the barrier's reasons, come back with the same shape and ``mtmc`` in
            ``meta["missing_stages"]`` — never as ``None`` and never as an exception, because
            "nothing was visible", "the tracker is dead" and "this frame was late" are three
            events (arch.md §5⑤).

        Raises:
            ValidationError: ``meta["tracks"]`` is not a sequence, ``meta["frame_hw"]`` is
                absent or not a positive ``(height, width)``, or ``captured_unix_ns`` is not
                positive. All three are a mis-wired chain rather than a late frame: the
                dimensions are what the height gate, the truncated-box test and the
                homography's domain are all computed from, and ``CameraTracks`` refuses a zero
                default precisely so all three cannot be silently wrong; the capture clock is
                what every instant is keyed on, so a source that never stamps it would put
                every frame of every camera into one instant that closes once and leaves the
                rest of the deployment late.
            ServerStateError: called before :meth:`Element.open`.
        """
        tracks = item.meta.get("tracks")
        if tracks is None:
            self._metrics.frame_missing(MISSING_TRACKS)
            return self._missing(item)
        if isinstance(tracks, str) or not isinstance(tracks, Sequence):
            raise ValidationError(
                f"mtmc element {self.name!r} was handed meta['tracks'] of type "
                f"{type(tracks).__name__} and needs the sequence of tracks a `track` element "
                "files"
            )

        assert self._barrier is not None  # `process` refuses before `open`
        camera_id = item.context.camera_id
        capture_s = self._capture_s(item)
        view = self._view(item, camera_id, capture_s, tracks)
        try:
            outcome = self._barrier.submit(
                camera_id,
                capture_s,
                view,
                associate=self._associate,
            )
        except self._TrackingError as exc:
            # The library refused the whole instant. Counted and emitted, never raised on: the
            # runner fails an item's future on any exception, and this frame's boxes, vectors
            # and per-camera track ids are all still good. Every other thread that was waiting
            # on this instant has already been released with `DROPPED_FAILED` by the barrier,
            # so they take the same gap through the branch below.
            self._warn_unassignable(exc)
            self._metrics.frame_missing(MISSED_UNASSIGNABLE)
            return self._missing(item)
        if outcome.results is None:
            self._metrics.frame_missing(outcome.reason)
            return self._missing(item)
        # Keyed, never positional: `results` covers the whole group in one flattened list and
        # this camera's rows are anywhere in it.
        global_ids = [outcome.results.get((camera_id, track.track_id)) for track in tracks]
        return item.derive(global_ids=global_ids)

    def _capture_s(self, item: ChainItem) -> float:
        """When this frame was taken, in seconds, or a typed refusal.

        :attr:`RequestContext.captured_unix_ns` defaults to ``0``, so a source that never
        stamps it is indistinguishable from one that stamps the epoch — and either way every
        frame of every camera lands in one instant, which closes once and makes every frame
        after it ``late`` for the life of the process. That is a static property of a
        mis-wired chain, identical on every frame, exactly like a zero ``frame_hw``, so it
        gets the same treatment: a refusal that names the fix rather than a per-frame gap that
        looks like clock skew. The production ingest path stamps it
        (``ingest/frame/tag.py``); a hand-built ``RequestContext`` is what does not.

        Raises:
            ValidationError: the capture clock is zero or negative.
        """
        captured = item.context.captured_unix_ns
        if captured <= 0:
            raise ValidationError(
                f"mtmc element {self.name!r}: this frame's captured_unix_ns is {captured} and "
                "cross-camera association buckets frames by capture time; every camera would "
                "land in one instant. Stamp the frame at ingest -- `FrameTag`/`ingest.frame` "
                "does it for every source the server reads"
            )
        return captured / 1e9

    def _view(
        self, item: ChainItem, camera_id: str, capture_s: float, tracks: Sequence[Any]
    ) -> Any:
        """This camera's contribution to the instant, in the library's vocabulary.

        Raises:
            ValidationError: ``meta["frame_hw"]`` is missing or degenerate. See
                :meth:`_do_process`.
        """
        frame_hw = item.meta.get("frame_hw")
        try:
            height, width = (int(frame_hw[0]), int(frame_hw[1]))  # type: ignore[index]
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            raise ValidationError(
                f"mtmc element {self.name!r}: meta['frame_hw'] must be the source frame's "
                f"(height, width) and was {frame_hw!r}. Cross-camera association needs the "
                "frame size for the height gate, the truncated-box test and the homography's "
                "domain; a `pool` detector files it and the `track` element carries it on"
            ) from exc
        if height <= 0 or width <= 0:
            raise ValidationError(
                f"mtmc element {self.name!r}: meta['frame_hw'] is {height}x{width} and must "
                "be positive; a zero would make the height gate, the truncated-box test and "
                "the homography's domain all silently wrong"
            )
        return self._CameraTracks(
            tag=self._FrameTag(
                camera_id=camera_id,
                frame_id=item.context.frame_id,
                timestamp=capture_s,
            ),
            tracks=tuple(tracks),
            height=height,
            width=width,
        )

    def _associate(self, entries: Sequence[InstantEntry]) -> dict[tuple[str, int], int | None]:
        """One cross-camera association, and the map every waiting frame reads itself out of.

        Called by whichever worker closed the instant, with the barrier's lock held. Holding
        it across ``tracker.track()`` is the one lock this codebase takes around a shipvision
        call (``docs/arch.md`` §7): the tracker has its own ``RLock`` for the whole of
        ``track()``, so a second one of ours would buy nothing, and the results have to be
        published to the waiters under this lock anyway.
        """
        started = time.perf_counter()
        cluster = self._FrameTrackCluster.from_views([entry.payload for entry in entries])
        results = self._tracker.track(cluster)
        self._metrics.association((time.perf_counter() - started) * 1e6)
        return {
            (result.track.camera_id, result.track.track_id): result.global_id
            for result in results
        }

    def _warn_unassignable(self, exc: BaseException) -> None:
        """Log the first refusal per open cycle, then stay quiet and count.

        Once, because the usual cause is a chain with no embedder in front of ``track`` and
        that produces the same line at a thousand frames a second — a log that drowns the
        deployment in the evidence for one static fact. The counter is what carries the rate;
        this is what carries the *reason*, which no counter can.
        """
        if self._warned_unassignable:
            return
        self._warned_unassignable = True
        _LOG.warning(
            "mtmc element %r could not associate an instant of group %r: %s. Every frame of "
            "that instant is emitted with `mtmc` in missing_stages, but under two reasons: "
            "the one frame whose thread ran the association counts as reason=%s and the rest "
            "of the group's, which the barrier had already released, as reason=%s -- so an "
            "8-camera group produces one of the first and seven of the second per instant. "
            "The fix is an embedder in front of `track`",
            self.name,
            self._group,
            exc,
            MISSED_UNASSIGNABLE,
            DROPPED_FAILED,
        )

    def _missing(self, item: ChainItem) -> ChainItem:
        """The successor for a frame with no global ids: everything else, and an honest gap."""
        return item.derive(
            **{MISSING_STAGES: (*item.meta.get(MISSING_STAGES, ()), "mtmc")},
        )

    # -- metrics -----------------------------------------------------------------------

    def _note_cameras(self) -> None:
        """Publish the live-camera count, and only when it changed.

        Also where a group that has outgrown the waiter budget is said out loud: the check
        is on the *live* set because cameras are added by API at run time, so a chain that
        declared a roster of four and was given eight cameras crosses the line hours after
        ``open()``. Latched on the crossing rather than counted per announcement, and
        cleared when the group comes back under the budget so a shard that loses and
        regains cameras says so each time. A chain that declared an over-large roster
        gets a line here *as well as* the one at ``open()``, and deliberately: the first
        says the configuration cannot work and the second says it has started happening.
        """
        if self._barrier is None:
            return
        count = len(self._barrier.live)
        if count != self._reported_cameras:
            self._reported_cameras = count
            self._metrics.camera_count(count)
            if count <= self._barrier.budget.permits + 1:
                self._starved_group = False
            elif not self._starved_group:
                self._starved_group = True
                self._warn_if_workers_cannot_cover(count, "live on this shard")

    def _warn_if_workers_cannot_cover(self, cameras: int, source: str) -> None:
        """Warn when the pipeline has too few workers to answer a whole instant.

        An instant closes when the **last** live camera of the group reports, and every
        frame that arrived before it is parked in the barrier until then -- so a group of
        ``n`` cameras needs ``n - 1`` waiter permits, which is ``pipeline.workers >= n``.
        Below that the never-starve guard is working rather than failing: the frames that
        could not park are emitted immediately, carrying their boxes, vectors and per-camera
        track ids and lacking only the global id. But at the shipped default of four workers
        an eight-camera group loses half of every instant's answers, and until now that was
        visible only to somebody already scraping ``shipinfer_mtmc_would_starve_total``.
        One line at the crossing costs nothing and names the setting that fixes it.

        Args:
            cameras: how many cameras an instant of this group waits for.
            source: where that number came from, so the line says whether it is the
                declared roster or what is actually running.
        """
        if self._barrier is None:
            return
        permits = self._barrier.budget.permits
        answerable = permits + 1
        if cameras <= answerable:
            return
        _LOG.warning(
            "mtmc element %r: group %r has %d cameras (%s) but only %d frame(s) of each "
            "instant can be answered -- %d worker(s) may park in a barrier at once and the "
            "one that closes the instant is the next. The other %d frame(s) per instant are "
            "emitted with a gap (shipinfer_mtmc_would_starve_total). Raise "
            "`pipeline.workers` to at least %d; the permits are process-wide, so a chain "
            "with a second mtmc slot needs the sum of its groups' sizes",
            self.name,
            self._group,
            cameras,
            source,
            answerable,
            permits,
            cameras - answerable,
            cameras,
        )

    def __repr__(self) -> str:
        cameras = 0 if self._barrier is None else len(self._barrier.live)
        return (
            f"<ShipvisionMtmc {self.name} group={self._group!r} {self._algorithm} "
            f"cameras={cameras}>"
        )
