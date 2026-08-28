"""The ``mtmc`` element: one identity per object across every camera that can see it.

Cross-camera association is not a per-frame function. ``shipvision.mtmc`` consumes a
:class:`FrameTrackCluster` — *every* camera of a group at one synchronised instant — and
refuses anything less, because handing it one camera at a time turns cross-camera
association into within-camera deduplication. The chain, however, delivers **one frame at a
time**, on whichever pipeline worker happened to take it off the fair lane. Something has to
turn a stream of single-camera frames back into instants, and that something is
:class:`InstantBarrier`.

Two classes, and the split is the same one ``track.py`` makes:

* :class:`InstantBarrier` is **pure** — no ``shipvision``, no numpy, no knowledge of what a
  "track" is. It buckets items by capture instant, decides when a bucket is complete, picks
  the thread that runs the association, and scatters the answer back. Every property worth
  testing about cross-camera synchronisation is a property of this class, and it is testable
  with integers and a callback (``tests/topology/test_mtmc_barrier.py``).
* :class:`ShipvisionMtmc` is the element: it turns a :class:`ChainItem` into a
  ``CameraTracks`` view, hands it to the barrier, and turns the answer back into metadata.

**Why the barrier must never block the last worker.** The walk is synchronous: a pipeline
worker that is waiting inside an element is a worker that is not draining its lane. If every
worker parks in the barrier waiting for cameras whose frames are still *queued*, no bucket
can ever complete on evidence — the only way out is the timeout, and the deployment has
converted itself into a fixed ``sync_window_ms`` of latency per frame with a stalled queue
behind it. So the barrier admits at most ``workers - 1`` waiters, and the frame that would
take the last one is emitted immediately with ``mtmc`` in ``missing_stages`` and counted
(``shipinfer_mtmc_would_starve_total``). A frame with an honest gap is worth more than a
stalled shard (arch.md §5⑤). ``ElementContext.workers`` is where the number comes from;
``None`` there means the runner did not say, and the barrier then refuses to wait at all
rather than guessing — the contract that field's docstring states.

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
atomically. The cost is that a submit for the *next* instant queues behind the association
for its duration; the association is a few hundred microseconds over a group's tracks and it
serialises against a single global tracker regardless.

**Where ``shipvision`` is named.** Nowhere at module scope — every symbol comes from
:mod:`shipinfer.topology.bridge` inside ``_do_open``, so ``import shipinfer.topology`` stays
free of the submodule and a chain naming ``impl: shipvision`` is still validatable on a host
that never checked it out.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from shipinfer.core.errors import ConfigurationError, ValidationError
from shipinfer.core.logging import get_logger
from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.bridge import load_errors, load_mtmc, load_types
from shipinfer.topology.elements.track import MISSING_STAGES
from shipinfer.topology.registry import registry_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.core.metrics import Counter, Gauge, Histogram

__all__ = [
    "CLOSED_COMPLETE",
    "CLOSED_WINDOW",
    "DEFAULT_ALGORITHM",
    "DEFAULT_CLUSTERER",
    "DEFAULT_MATRIX_BUILDER",
    "DEFAULT_MAX_INSTANTS",
    "DEFAULT_SYNC_WINDOW_MS",
    "DROPPED_EVICTED",
    "DROPPED_EXPIRED",
    "DROPPED_FAILED",
    "DROPPED_SHUTDOWN",
    "MISSED_DUPLICATE",
    "MISSED_LATE",
    "MISSED_UNASSIGNABLE",
    "MISSED_WOULD_STARVE",
    "MISSING_TRACKS",
    "InstantBarrier",
    "InstantEntry",
    "InstantOutcome",
    "ShipvisionMtmc",
    "parse_group",
]

_LOG = get_logger("topology.mtmc")

# -- the vocabulary of outcomes ---------------------------------------------------------
#
# Strings and not an enum, for the same reason `ElementKind` is a `str` enum: every one of
# these is written straight into a metric label, and a conversion at each site is a
# conversion to get wrong. They are split into two families because they answer two
# different questions -- what happened to an *instant* (there is one of these per closed
# bucket, and `InstantBarrier.on_event` reports them) and what happened to a *frame* (one
# per item, and the element reads it off the outcome).

#: The bucket closed because every live camera reported. The good case.
CLOSED_COMPLETE = "complete"
#: The bucket closed because ``sync_window_ms`` ran out with a camera still missing. The
#: association still ran, over whoever did report -- a partial instant is worth more than
#: no instant, and MTMC over a subset of a group is exactly what a camera outage looks like.
CLOSED_WINDOW = "window"
#: The bucket was pushed out by :data:`DEFAULT_MAX_INSTANTS` newer ones. A steady stream of
#: these means the group's clocks disagree by more than the window, or one camera is running
#: far ahead of the others.
DROPPED_EVICTED = "evicted"
#: The bucket passed its deadline with nobody waiting on it -- every frame in it had already
#: been emitted by the never-starve guard, so there was no answer for anyone to receive.
DROPPED_EXPIRED = "expired"
#: :meth:`InstantBarrier.close_all` resolved the bucket because the element is closing.
DROPPED_SHUTDOWN = "shutdown"
#: The association itself raised. The waiters are released with no result and the closing
#: thread gets the exception -- one frame carries the failure, the rest carry a gap.
DROPPED_FAILED = "failed"

#: The frame's instant had already been closed when it arrived. Counted, never retro-fitted:
#: re-running an association to add one late camera would issue a second, contradictory set
#: of global ids for tracks that have already been published under the first.
MISSED_LATE = "late"
#: Two frames of one camera landed in one instant. ``FrameTrackCluster`` refuses that (two
#: frames from one camera in one instant are two instants), so the second is emitted with a
#: gap rather than allowed to make the group's same-camera exclusion mask leak.
MISSED_DUPLICATE = "duplicate"
#: Waiting would have parked the last pipeline worker. See the module docstring.
MISSED_WOULD_STARVE = "would_starve"
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
#: How wide an instant is, in milliseconds. **A proposal, not a measurement** (the phase-C
#: plan's open question 3): nothing in ``docs/arch.md`` states one. 60 ms is about 1.2 frame
#: periods at 20 fps, so two cameras whose frames are a frame apart still land in one bucket
#: while two frames of one camera do not. It is also the worst-case latency this element can
#: add to a frame, which is why it is small.
DEFAULT_SYNC_WINDOW_MS = 60.0
#: How many instants may be open at once before the oldest is evicted. Eight is half a second
#: at the default window: enough to absorb one camera running a few frames behind, far too
#: few to hide a clock that is minutes out.
DEFAULT_MAX_INSTANTS = 8


def parse_group(params: Mapping[str, Any], *, where: str) -> tuple[str, tuple[str, ...]]:
    """The ``group:`` name and ``cameras:`` roster an ``mtmc`` slot declares.

    One parser with two callers, deliberately. :class:`ShipvisionMtmc` reads it to name its
    own group in logs and refusals; :class:`~shipinfer.runners.fleet.FleetRunner` reads it off
    the chain spec to learn which cameras belong together, because a camera group is an atomic
    unit of placement (``docs/arch.md`` §4) and the fleet is the only thing that places
    cameras. Two readers of one YAML shape is exactly how a key drifts, so there is one.

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


# -- the barrier ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstantEntry:
    """One camera's contribution to an instant: who, and whatever the caller put in.

    ``payload`` is typed ``object`` on purpose. The barrier is pure and must not learn what a
    ``CameraTracks`` is -- that is the element's vocabulary, and keeping it out of here is
    what lets the synchronisation properties be tested with strings.
    """

    camera_id: str
    payload: Any


@dataclass(frozen=True, slots=True)
class InstantOutcome:
    """What a submitted frame got back from the barrier.

    ``results`` is ``None`` for every reason the frame did **not** take part in an
    association, and a mapping otherwise -- never an empty mapping to mean failure (ADR-005).
    An instant in which nothing was visible on any camera legitimately returns an empty map,
    and that is a different fact from "this frame missed its instant".
    """

    reason: str
    results: Mapping[Any, Any] | None = None
    instant: int = 0

    @property
    def associated(self) -> bool:
        """Whether an association ran for this frame."""
        return self.results is not None


@dataclass(slots=True)
class _Bucket:
    """One instant, in flight.

    ``waiters`` is per-bucket as well as global because two questions need it: whether the
    barrier may admit another waiter at all (global, the never-starve guard) and whether
    anybody is still interested in this particular bucket's answer (per-bucket, which is what
    lets an abandoned instant be discarded instead of associated for nobody).
    """

    key: int
    deadline: float
    entries: list[InstantEntry] = field(default_factory=list)
    reported: set[str] = field(default_factory=set)
    waiters: int = 0
    #: Set by the thread that resolved this bucket. A waiter wakes, sees it, and returns.
    done: bool = False
    #: "Somebody noticed this bucket is complete but had no association function in hand."
    #: Only :meth:`InstantBarrier.drop_camera` sets it: the lifecycle thread must return
    #: promptly and must not run a tracker, so it wakes a waiter and lets *that* thread --
    #: which is a pipeline worker with the callback -- do the work.
    ready: bool = False
    reason: str = ""
    results: Mapping[Any, Any] | None = None


class InstantBarrier:
    """Turns a stream of per-camera frames back into synchronised instants. Pure.

    One bucket per instant, keyed on ``floor(capture_time / sync_window_s)`` — the **capture**
    clock and not the arrival clock, because arrival order is the pipeline's business (fair
    lanes, N workers, spill) and has nothing to do with whether two frames show the same
    moment. Two cameras 40 ms apart in arrival but 2 ms apart in capture are one instant; the
    reverse is two.

    A bucket closes when **every live camera has reported**, or when ``sync_window_ms`` has
    passed since it opened, whichever is first. Whichever thread closes it runs the
    association and publishes the answer to every thread waiting on that bucket, keyed by
    whatever the association function returned — the barrier never indexes results by
    position (see the module docstring).

    LOCKING. One :class:`threading.Condition` over one lock guards everything: the bucket
    map, the live set, the waiter count and the counters. It is held across the association
    callback, which is deliberate and is the *only* lock this codebase takes around
    ``tracker.track()`` (``docs/arch.md`` §7). Every wait is bounded by the bucket's own
    deadline, so no caller can be parked here for longer than one window even if the callback
    never runs.

    Args:
        sync_window_s: how wide an instant is. Also the maximum time any caller waits.
        workers: how many threads may be inside :meth:`submit` at once —
            :attr:`ElementContext.workers`. At most ``workers - 1`` of them ever wait.
            ``None`` means the runner did not say, and the barrier then never waits at all:
            guessing would park the only thread there is on a single-worker runner.
        max_instants: how many buckets may be open before the oldest is evicted.
        on_event: called once per *instant-level* event, under the lock, with one of
            :data:`CLOSED_COMPLETE`, :data:`CLOSED_WINDOW`, :data:`DROPPED_EVICTED`,
            :data:`DROPPED_EXPIRED`, :data:`DROPPED_SHUTDOWN` or :data:`DROPPED_FAILED`. For
            metrics, so it must be a counter increment and nothing else: it runs on the
            association's critical path. Frame-level outcomes are *not* reported here — the
            caller reads those off its own :class:`InstantOutcome`, because one closed instant
            resolves many frames and counting instants per frame would report the wrong number.

    Raises:
        ConfigurationError: a non-positive window, or fewer than one worker.
    """

    __slots__ = (
        "_announced",
        "_buckets",
        "_closed",
        "_cond",
        "_counts",
        "_hooked",
        "_max_instants",
        "_on_event",
        "_recent",
        "_recent_limit",
        "_seen",
        "_waiters",
        "_window_s",
        "_workers",
    )

    def __init__(
        self,
        *,
        sync_window_s: float,
        workers: int | None,
        max_instants: int = DEFAULT_MAX_INSTANTS,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        if sync_window_s <= 0.0:
            raise ConfigurationError(
                f"sync_window_s must be positive, got {sync_window_s}; a zero window makes "
                f"every frame its own instant and cross-camera association a no-op"
            )
        if max_instants < 1:
            raise ConfigurationError(f"max_instants must be at least 1, got {max_instants}")
        if workers is not None and workers < 1:
            raise ConfigurationError(
                f"workers must be at least 1, got {workers}; pass None for 'the runner did "
                f"not say', which makes this barrier never wait"
            )
        self._window_s = float(sync_window_s)
        # `None` collapses to 1, and 1 is what makes `_waiters + 1 >= _workers` true on the
        # first caller -- i.e. never wait. The two states are deliberately the same state:
        # "I do not know how many workers there are" and "there is one worker" have the same
        # correct behaviour, and spelling them differently would be a second code path.
        self._workers = 1 if workers is None else int(workers)
        self._max_instants = int(max_instants)
        self._on_event = on_event
        self._cond = threading.Condition(threading.Lock())
        self._buckets: dict[int, _Bucket] = {}
        #: Keys of instants already resolved, bounded. What makes a late frame *late* rather
        #: than the first member of a brand-new bucket with the same key.
        self._recent: OrderedDict[int, None] = OrderedDict()
        self._recent_limit = max(8, self._max_instants * 4)
        #: Cameras the runner announced through the lifecycle hooks.
        self._announced: set[str] = set()
        #: Cameras that have actually submitted a frame. Only consulted before the first
        #: announcement -- see :meth:`_live`.
        self._seen: set[str] = set()
        self._hooked = False
        self._waiters = 0
        self._closed = False
        self._counts: dict[str, int] = {}

    # -- reading -----------------------------------------------------------------------

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def workers(self) -> int:
        """How many threads may be inside :meth:`submit`; ``workers - 1`` may wait."""
        return self._workers

    @property
    def waiters(self) -> int:
        """How many callers are parked right now. Never reaches :attr:`workers`."""
        return self._waiters

    @property
    def live(self) -> frozenset[str]:
        """The cameras a bucket waits for. See :meth:`camera_added` for why it is not the roster."""
        with self._cond:
            return frozenset(self._live())

    @property
    def open_instants(self) -> int:
        with self._cond:
            return len(self._buckets)

    def instant_key(self, capture_s: float) -> int:
        """Which instant a capture timestamp belongs to.

        ``math.floor`` and not ``int()``: truncation rounds toward zero, which would put
        ``-0.5`` and ``+0.5`` in the same bucket. Capture times are unix seconds so it cannot
        arise today, and a bucket key that is wrong only for negative inputs is exactly the
        kind of thing that is discovered by a clock, at 3 a.m.
        """
        return math.floor(capture_s / self._window_s)

    def stats(self) -> dict[str, int]:
        """Every event this barrier has counted, by reason. What a growth test asserts on."""
        with self._cond:
            return dict(self._counts)

    # -- the camera set ----------------------------------------------------------------

    def camera_added(self, camera_id: str) -> None:
        """A camera is live on this shard: instants now wait for it.

        **The live set is the announced set, not the configured roster**, and that is the one
        decision in this class that a reader will want the reason for. A group's roster names
        every camera in it; a shard runs only the ones placed on it, and the rest are on other
        shards or not started. A barrier that waited for the roster would time out on every
        single instant, for the life of the process, and report it as a healthy chain running
        60 ms slower.

        Before the *first* announcement the barrier falls back to the cameras it has seen
        traffic from, so a runner that does not drive the lifecycle hooks at all degrades to a
        one-instant warm-up rather than to per-camera MTMC. The fallback latches off for good
        at the first announcement: a set that came back after ``camera_removed`` emptied it
        would resurrect the camera this hook exists to forget.

        Takes only this barrier's lock and returns immediately -- it runs on the thread
        holding the runner's lifecycle lock, behind which every other lifecycle call queues.
        """
        with self._cond:
            self._hooked = True
            self._announced.add(camera_id)

    def drop_camera(self, camera_id: str) -> None:
        """A camera is gone: stop waiting for it, including in instants already open.

        The second half is the point. Dropping the camera from the live set alone would leave
        every *currently open* bucket still counting it, so each one would sit out its full
        window before closing — and at 20 fps that is a permanent tax paid for a camera that
        will never report again. So the open buckets are re-checked here, and any that is now
        complete has its waiters woken with ``ready`` set.

        Woken rather than closed, because this thread must not run an association: it holds
        the runner's lifecycle lock, and every ``add_camera``, ``remove_camera``, ``drain``
        and ``stop`` on the shard is queued behind it. The waiter it wakes is a pipeline
        worker that already has the association callback in hand.

        A bucket with no waiters is left alone: every frame in it was already emitted by the
        never-starve guard, so nobody is owed an answer, and its own deadline will retire it.

        Idempotent, and safe for a camera that was never added -- a removal racing an
        in-flight frame is ordinary (``Element.camera_removed``).
        """
        with self._cond:
            self._announced.discard(camera_id)
            self._seen.discard(camera_id)
            live = self._live()
            woken = False
            for bucket in self._buckets.values():
                if bucket.waiters and live <= bucket.reported:
                    bucket.ready = True
                    woken = True
            if woken:
                self._cond.notify_all()

    # -- the frame path ----------------------------------------------------------------

    def submit(
        self,
        camera_id: str,
        capture_s: float,
        payload: Any,
        *,
        associate: Callable[[Sequence[InstantEntry]], Mapping[Any, Any]],
    ) -> InstantOutcome:
        """Put one camera's frame into its instant and come back with the group's answer.

        Args:
            camera_id: which camera. Two frames of one camera in one instant is a refusal,
                not a merge.
            capture_s: when the frame was **captured**, in seconds. Not when it arrived.
            payload: whatever the caller wants handed to ``associate`` for this camera.
            associate: called by whichever thread closes the bucket, with every entry in it,
                and must return a mapping from a caller-chosen key to a caller-chosen result.
                Called with this barrier's lock held (see the class docstring). If it raises,
                every waiter is released with :data:`DROPPED_FAILED` and the exception reaches
                the closing thread — one frame carries the failure and the rest carry a gap.

        Returns:
            An :class:`InstantOutcome`. ``results`` is the association's mapping when this
            frame took part, and ``None`` with a reason when it did not.
        """
        key = self.instant_key(capture_s)
        with self._cond:
            if self._closed:
                return self._missed(DROPPED_SHUTDOWN, key)
            now = time.monotonic()
            self._retire(now)
            self._seen.add(camera_id)
            if key in self._recent:
                return self._missed(MISSED_LATE, key)
            bucket = self._buckets.get(key)
            if bucket is None:
                self._evict()
                bucket = _Bucket(key=key, deadline=now + self._window_s)
                self._buckets[key] = bucket
            if camera_id in bucket.reported:
                return self._missed(MISSED_DUPLICATE, key)
            bucket.reported.add(camera_id)
            bucket.entries.append(InstantEntry(camera_id, payload))

            if self._live() <= bucket.reported:
                return self._close(bucket, CLOSED_COMPLETE, associate)
            # The never-starve guard. `+ 1` counts this caller: with `workers = 1` the very
            # first frame takes it, which is right -- a single-worker runner has nobody left
            # to deliver the camera this frame would be waiting for.
            if self._waiters + 1 >= self._workers:
                return self._missed(MISSED_WOULD_STARVE, key)

            self._waiters += 1
            bucket.waiters += 1
            try:
                self._cond.wait_for(
                    lambda: bucket.done or bucket.ready,
                    timeout=max(0.0, bucket.deadline - time.monotonic()),
                )
            finally:
                self._waiters -= 1
                bucket.waiters -= 1
            if bucket.done:
                return InstantOutcome(bucket.reason, bucket.results, key)
            if self._buckets.get(key) is bucket:
                # Either a `drop_camera` completed it while we slept (`ready`) or the window
                # ran out. Both mean: this thread is the closer.
                reason = CLOSED_COMPLETE if bucket.ready else CLOSED_WINDOW
                return self._close(bucket, reason, associate)
            # Removed from the map without being marked done. No path does that today; a
            # refusal is still better than a hang if one is ever added.
            return self._missed(DROPPED_EXPIRED, key)

    def close_all(self, reason: str = DROPPED_SHUTDOWN) -> int:
        """Resolve every open instant and refuse every later submit. Idempotent.

        Called from ``_do_close``. Without it a worker parked in :meth:`submit` would hold the
        shutdown for the rest of its window; with it, every waiter is released at once and
        ``stop()`` never spends its deadline on a barrier. The bound on the wait means this is
        belt and braces rather than the only guarantee, which is the right relationship
        between the two: a wait that is *only* bounded by a shutdown call becomes unbounded
        the first time somebody forgets to make it.

        Returns:
            How many instants were open.
        """
        with self._cond:
            self._closed = True
            open_buckets = list(self._buckets.values())
            for bucket in open_buckets:
                self._buckets.pop(bucket.key, None)
                self._remember(bucket.key)
                bucket.results = None
                bucket.reason = reason
                bucket.done = True
                self._event(reason)
            if open_buckets:
                self._cond.notify_all()
            return len(open_buckets)

    # -- internals (lock held) ---------------------------------------------------------

    def _live(self) -> set[str]:
        """The cameras an instant waits for. See :meth:`camera_added`."""
        return self._announced if self._hooked else self._seen

    def _missed(self, reason: str, key: int) -> InstantOutcome:
        """Count a frame-level miss and answer with it. No instant-level event."""
        self._counts[reason] = self._counts.get(reason, 0) + 1
        return InstantOutcome(reason, None, key)

    def _event(self, reason: str) -> None:
        """Count an instant-level event and tell the observer, if there is one."""
        self._counts[reason] = self._counts.get(reason, 0) + 1
        if self._on_event is not None:
            self._on_event(reason)

    def _remember(self, key: int) -> None:
        """Record a resolved instant so a frame arriving for it is late, not a new bucket."""
        self._recent[key] = None
        self._recent.move_to_end(key)
        while len(self._recent) > self._recent_limit:
            self._recent.popitem(last=False)

    def _close(
        self,
        bucket: _Bucket,
        reason: str,
        associate: Callable[[Sequence[InstantEntry]], Mapping[Any, Any]],
    ) -> InstantOutcome:
        """Run the association for one bucket and publish it to everyone waiting on it.

        The bucket leaves the map *before* the callback runs, so a frame that arrives for this
        instant while the association is in flight is late rather than a member of a bucket
        that is already being consumed.
        """
        self._buckets.pop(bucket.key, None)
        self._remember(bucket.key)
        try:
            results = associate(tuple(bucket.entries))
        except BaseException:
            bucket.results = None
            bucket.reason = DROPPED_FAILED
            bucket.done = True
            self._event(DROPPED_FAILED)
            self._cond.notify_all()
            raise
        bucket.results = results
        bucket.reason = reason
        bucket.done = True
        self._event(reason)
        self._cond.notify_all()
        return InstantOutcome(reason, results, bucket.key)

    def _retire(self, now: float) -> None:
        """Discard instants that ran out of time with nobody waiting on them.

        A bucket in this state holds only frames the never-starve guard already emitted, so
        there is no association anybody would receive — running one would cost a tracker call
        and a global-id assignment for an answer with no reader, and *not* running one would
        leave the bucket to be evicted later and read as clock skew. It is discarded and
        counted.
        """
        stale = [
            bucket
            for bucket in self._buckets.values()
            if bucket.waiters == 0 and bucket.deadline <= now
        ]
        for bucket in stale:
            self._buckets.pop(bucket.key, None)
            self._remember(bucket.key)
            bucket.results = None
            bucket.reason = DROPPED_EXPIRED
            bucket.done = True
            self._event(DROPPED_EXPIRED)
        if stale:
            self._cond.notify_all()

    def _evict(self) -> None:
        """Make room for a new instant by retiring the oldest, and say so.

        Oldest by *instant key*, i.e. by capture time — not by insertion order. The two differ
        exactly when frames arrive out of capture order, which is the case eviction exists for.
        """
        while len(self._buckets) >= self._max_instants:
            bucket = self._buckets.pop(min(self._buckets))
            self._remember(bucket.key)
            bucket.results = None
            bucket.reason = DROPPED_EVICTED
            bucket.done = True
            self._event(DROPPED_EVICTED)
            self._cond.notify_all()

    def __repr__(self) -> str:
        return (
            f"<InstantBarrier window={self._window_s * 1e3:.0f}ms workers={self._workers} "
            f"open={len(self._buckets)} waiting={self._waiters}>"
        )


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
            "id, which is worth far more than a dropped frame.",
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
    entry per track, ``int`` or ``None``. That is the shape ``elements/mock.py`` publishes and
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
      read off the chain spec by the fleet runner, which places a group atomically.
    * ``cameras`` — the group's roster. **Does not drive the barrier** (see
      :meth:`InstantBarrier.camera_added`); it is what the fleet reads to keep the group on
      one shard, and what lets this element warn when a camera it was never told about turns
      up.
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
    been placed. The only component that knows both the group membership (from this slot's
    ``params:``) and the placement map is the fleet runner, which owns
    ``{camera_id: shard_id}``, so the invariant is enforced in
    :meth:`~shipinfer.runners.fleet.FleetRunner.add_camera`.
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
            max_instants=self._max_instants,
            on_event=self._metrics.instant,
        )
        self._reported_cameras = -1
        self._note_cameras()
        if context.workers is None:
            _LOG.warning(
                "mtmc element %r was given no worker count; it will emit every frame "
                "immediately rather than wait for the rest of group %r. The runner sets "
                "ElementContext.workers",
                self.name,
                self._group,
            )

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

        ``close_all`` first and unconditionally: a worker parked in the barrier holds a future
        its producer is waiting on, and a shutdown that dropped the barrier without resolving
        them would leave each one to time out on its own window. The identities go with the
        element for the reason the trackers do in ``track.py`` — keeping them across a close
        would mean a reopened chain continuing global ids over a gap it cannot see.
        """
        if self._barrier is not None:
            self._barrier.close_all(DROPPED_SHUTDOWN)
        self._barrier = None
        self._tracker = None
        self._TrackingError = _NeverRaised
        self._warned_unassignable = False
        self._CameraTracks = None
        self._FrameTrackCluster = None
        self._FrameTag = None

    def camera_added(self, camera_id: str) -> None:
        """This camera is live: instants now wait for it.

        Takes only the barrier's lock and returns. Warns, but does not refuse, when the camera
        is outside a declared roster: a roster is written once and cameras are added by API at
        run time, so refusing here would make a stale list able to reject a live camera. The
        warning is the record that the two disagree.
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

        See :meth:`InstantBarrier.drop_camera` — the second half is what keeps a removed
        camera from taxing every subsequent instant the full ``sync_window_ms``.
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
            ValidationError: ``meta["tracks"]`` is not a sequence, or ``meta["frame_hw"]`` is
                absent or not a positive ``(height, width)``. Both are a mis-wired chain
                rather than a late frame: the dimensions are what the height gate, the
                truncated-box test and the homography's domain are all computed from, and
                ``CameraTracks`` refuses a zero default precisely so all three cannot be
                silently wrong.
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
        view = self._view(item, camera_id, tracks)
        try:
            outcome = self._barrier.submit(
                camera_id,
                item.context.captured_unix_ns / 1e9,
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

    def _view(self, item: ChainItem, camera_id: str, tracks: Sequence[Any]) -> Any:
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
                timestamp=item.context.captured_unix_ns / 1e9,
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
            "mtmc element %r could not associate an instant of group %r: %s. Every frame "
            "with an un-embedded track is emitted with `mtmc` in missing_stages and counted "
            "under reason=%s; the fix is an embedder in front of `track`",
            self.name,
            self._group,
            exc,
            MISSED_UNASSIGNABLE,
        )

    def _missing(self, item: ChainItem) -> ChainItem:
        """The successor for a frame with no global ids: everything else, and an honest gap."""
        return item.derive(
            **{MISSING_STAGES: (*item.meta.get(MISSING_STAGES, ()), "mtmc")},
        )

    # -- metrics -----------------------------------------------------------------------

    def _note_cameras(self) -> None:
        """Publish the live-camera count, and only when it changed."""
        if self._barrier is None:
            return
        count = len(self._barrier.live)
        if count != self._reported_cameras:
            self._reported_cameras = count
            self._metrics.camera_count(count)

    def __repr__(self) -> str:
        cameras = 0 if self._barrier is None else len(self._barrier.live)
        return (
            f"<ShipvisionMtmc {self.name} group={self._group!r} {self._algorithm} "
            f"cameras={cameras}>"
        )
