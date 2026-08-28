"""The ``mtmc`` element: one identity per object across every camera that can see it.

Cross-camera association is not a per-frame function. ``shipvision.mtmc`` consumes a
:class:`FrameTrackCluster` — *every* camera of a group at one synchronised instant — and
refuses anything less, because handing it one camera at a time turns cross-camera
association into within-camera deduplication. The chain, however, delivers **one frame at a
time**, on whichever pipeline worker happened to take it off the fair lane. Something has to
turn a stream of single-camera frames back into instants, and that something is
:class:`InstantBarrier`.

This module will hold two classes, and the split is the same one ``track.py`` makes.
:class:`InstantBarrier` — this commit — is **pure**: no ``shipvision``, no numpy, no knowledge
of what a "track" is. It buckets items by capture instant, decides when a bucket is complete,
picks the thread that runs the association, and scatters the answer back. Every property worth
testing about cross-camera synchronisation is a property of this class, and it is testable with
integers and a callback (``tests/topology/test_mtmc_barrier.py``). The element that hands it
real tracks lands next, and it is the only half that names ``shipvision``.

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

**Why the scatter is keyed and never positional.** ``shipvision``'s ``FrameTrackCluster``
flattens every camera's tracks into one observation list, and its tracker answers in that
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

**Where ``shipvision`` is named.** Nowhere in this class, and nowhere at module scope — the
element that follows takes every symbol from :mod:`shipinfer.topology.bridge` inside
``_do_open``, so ``import shipinfer.topology`` stays free of the submodule and a chain naming
``impl: shipvision`` is still validatable on a host that never checked it out.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shipinfer.core.errors import ConfigurationError

__all__ = [
    "CLOSED_COMPLETE",
    "CLOSED_WINDOW",
    "DEFAULT_MAX_INSTANTS",
    "DEFAULT_SYNC_WINDOW_MS",
    "DROPPED_EVICTED",
    "DROPPED_EXPIRED",
    "DROPPED_FAILED",
    "DROPPED_SHUTDOWN",
    "MISSED_DUPLICATE",
    "MISSED_LATE",
    "MISSED_WOULD_STARVE",
    "InstantBarrier",
    "InstantEntry",
    "InstantOutcome",
]

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
