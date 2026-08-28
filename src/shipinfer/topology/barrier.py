"""The instant barrier: a stream of per-camera frames, turned back into synchronised instants.

Cross-camera association is not a per-frame function. A cross-camera tracker consumes *every*
camera of a group at one synchronised instant and refuses anything less, because handing it
one camera at a time turns cross-camera association into within-camera deduplication. A chain,
however, delivers **one frame at a time**, on whichever pipeline worker happened to take it off
the fair lane. Something has to turn that stream back into instants, and that something is
:class:`InstantBarrier`.

This module is **pure** — no ``shipvision``, no numpy, no knowledge of what a "track" is. It
buckets opaque payloads by capture instant, decides when a bucket is complete, picks the thread
that runs the association, and scatters the answer back. Every property worth testing about
cross-camera synchronisation is a property of this file, and it is testable with strings and a
callback (``tests/topology/test_barrier.py``). The element that gives the payloads meaning is
:mod:`shipinfer.topology.elements.mtmc`.

**Why the instant is anchored and not a grid.** The obvious bucket key is
``floor(capture_s / window)`` on an absolute grid, and it is wrong at every setting. A window
*wider* than one frame period puts two consecutive frames of one camera into the same bucket
once every ``window / period`` frames — at 20 fps with a 60 ms window that is one frame in six,
each of which is refused an answer, with perfectly genlocked cameras and no jitter anywhere. A
window *narrower* than the frame period fixes that and breaks the other half: free-running
cameras spread their captures across a whole period, so no absolute cell holds the whole group.
The two constraints are incompatible, which is the proof that no absolute window is correct.

So an instant is **anchored by its first arrival**: the frame that opens a bucket sets the
instant's capture span, later frames join while the span stays inside ``sync_window_s``, and a
camera reporting a *second, later* frame is the signal that the instant it was in has all the
evidence it is going to get — that bucket closes and the next one opens with that frame. A
camera can no longer collide with itself whatever the window is, and a group whose arrival
spread is under the window still lands together.

**Why the barrier must never block the last worker.** The walk is synchronous: a pipeline
worker that is waiting inside an element is a worker that is not draining its lane. If every
worker parks in a barrier waiting for cameras whose frames are still *queued*, no bucket can
ever complete on evidence — the only way out is the timeout, and the deployment has converted
itself into a fixed ``sync_window_s`` of latency per frame with a stalled queue behind it. So a
:class:`WaiterBudget` caps the waiters, and the frame that would take the last permit is
emitted immediately with an honest gap. The budget is a **process-wide** object rather than a
per-barrier counter for the reason its own docstring gives: two barriers each admitting
``workers - 1`` waiters park every worker between them.

**Why the scatter is keyed and never positional.** A group's association answers for every
camera in one flattened list, and camera A's three results and camera B's one come back as four
whose positions mean nothing to either frame. The barrier therefore publishes whatever mapping
the association returned and never indexes it — each waiting caller reads its own entries out
by a key it chose. Scattering by list position is the classic reassembly bug (ADR-002's tag
rule, one layer up), and it produces a plausible answer rather than an error.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shipinfer.core.errors import ConfigurationError, ServerStateError

__all__ = [
    "CLOSED_ADVANCED",
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
    "WaiterBudget",
]

# -- the vocabulary of outcomes ---------------------------------------------------------
#
# Strings and not an enum, for the same reason `ElementKind` is a `str` enum: every one of
# these is written straight into a metric label, and a conversion at each site is a
# conversion to get wrong. They are split into two families because they answer two
# different questions -- what happened to an *instant* (there is one of these per closed
# bucket, and `InstantBarrier.on_event` reports them, counted in `instant_stats`) and what
# happened to a *frame* (one per item, counted in `frame_stats`). The two families are
# counted in two dictionaries and never summed into one, because `shutdown` is a member of
# both and an operator reading a single number would be reading frames plus instants.

#: The bucket closed because every live camera has reported. The good case.
CLOSED_COMPLETE = "complete"
#: The bucket closed because ``sync_window_s`` ran out with a camera still missing. The
#: association still ran, over whoever did report -- a partial instant is worth more than
#: no instant, and MTMC over a subset of a group is exactly what a camera outage looks like.
CLOSED_WINDOW = "window"
#: The bucket closed because a camera already in it reported its **next** frame, so no
#: further evidence for this instant can arrive from that camera. A steady stream of these
#: means one camera is running ahead of the rest of its group by less than a window; a rising
#: share is the number to look at before touching ``sync_window_ms``.
CLOSED_ADVANCED = "advanced"
#: The bucket was pushed out by :data:`DEFAULT_MAX_INSTANTS` newer ones. A steady stream of
#: these means the group's clocks disagree by more than the window, or one camera is running
#: far ahead of the others.
DROPPED_EVICTED = "evicted"
#: The bucket passed its deadline with nobody waiting on it -- every frame in it had already
#: been emitted by the never-starve guard, so there was no answer for anyone to receive.
DROPPED_EXPIRED = "expired"
#: :meth:`InstantBarrier.close_all` resolved the bucket because the element is closing. Also
#: the *frame*-level reason a submit after that gets, which is why the two families are
#: counted apart.
DROPPED_SHUTDOWN = "shutdown"
#: The association itself raised. The waiters are released with no result and the closing
#: thread gets the exception -- one frame carries the failure, the rest carry a gap.
DROPPED_FAILED = "failed"

#: The frame's instant had already been closed when it arrived. Counted, never retro-fitted:
#: re-running an association to add one late camera would issue a second, contradictory set
#: of results for objects that have already been published under the first.
MISSED_LATE = "late"
#: The same camera offered the same capture instant twice. A cross-camera cluster refuses
#: two frames from one camera in one instant (they are two instants), so the second is
#: emitted with a gap rather than allowed to make the group's same-camera exclusion mask
#: leak. A *later* capture from a camera already in the bucket is not this -- it is the next
#: instant, and it closes the open one (:data:`CLOSED_ADVANCED`).
MISSED_DUPLICATE = "duplicate"
#: Waiting would have parked the last pipeline worker. See the module docstring.
MISSED_WOULD_STARVE = "would_starve"

#: How wide an instant is, in milliseconds -- the maximum capture spread of one instant, and
#: also the longest any caller waits. **A proposal, not a measurement** (the phase-C plan's
#: open question 3): nothing in ``docs/arch.md`` states one. With the anchored instant it is
#: no longer constrained from below by the frame period: it has to be at least the group's
#: arrival spread, and 60 ms is a comfortable margin over the ~1 ms genlock skew of a wired
#: group while staying near one frame period at 20 fps, which is the worst-case latency this
#: bounds.
DEFAULT_SYNC_WINDOW_MS = 60.0
#: How many instants may be open at once before the oldest is evicted. Eight is half a second
#: at the default window: enough to absorb one camera running a few frames behind, far too
#: few to hide a clock that is minutes out.
DEFAULT_MAX_INSTANTS = 8


class WaiterBudget:
    """How many pipeline workers may be parked inside a barrier at once, **per process**.

    The never-starve guard the module docstring describes counts permits here rather than
    waiters in one barrier, and that is the whole reason this class exists. A chain may hold
    two ``mtmc`` slots — two independent camera groups is a supported configuration, and the
    loader takes an explicit ``kind:`` so two are expressible — and with a per-barrier count
    barrier A admits ``workers - 1`` waiters while barrier B, which sees zero, admits the
    last one. Every pipeline worker is then parked, neither barrier can close on evidence,
    and the shard degrades to exactly the "stall dressed as a wait" the guard exists to
    prevent. One budget for the process, constructed by the runner with ``workers - 1``
    permits, makes the invariant hold however many barriers there are.

    ``acquire`` never blocks: a barrier calls it while holding its own condition lock, and a
    budget that could block there would be a lock-ordering hazard between two barriers. It
    takes only this object's lock and calls nothing back, so a barrier's lock and this one
    are always taken in that order and never the reverse.

    Args:
        permits: how many waiters may be outstanding. ``0`` is legitimate and means "never
            wait" — a single-worker runner, or a runner that did not say how many workers it
            has, which is the same state and deliberately spelled the same way.

    Raises:
        ConfigurationError: a negative number of permits.
    """

    __slots__ = ("_held", "_lock", "_permits")

    def __init__(self, permits: int) -> None:
        if permits < 0:
            raise ConfigurationError(
                f"a waiter budget cannot have {permits} permits; 0 means 'never wait', which "
                f"is what a single-worker runner gets"
            )
        self._lock = threading.Lock()
        self._permits = int(permits)
        self._held = 0

    @property
    def permits(self) -> int:
        """How many waiters may be outstanding at once."""
        return self._permits

    @property
    def held(self) -> int:
        """How many are outstanding right now."""
        with self._lock:
            return self._held

    def acquire(self) -> bool:
        """Take a permit if there is one. Never blocks.

        Returns:
            Whether the caller may wait. ``False`` means every other permit is out and this
            caller must emit its frame with a gap instead of parking.
        """
        with self._lock:
            if self._held >= self._permits:
                return False
            self._held += 1
            return True

    def release(self) -> None:
        """Give a permit back.

        Raises:
            ServerStateError: more releases than acquires. Acquire and release are paired by
                one ``try``/``finally`` in :meth:`InstantBarrier.submit`, so this cannot
                happen from a correct caller and a silent decrement past zero would hand out
                permits that do not exist.
        """
        with self._lock:
            if self._held == 0:
                raise ServerStateError(
                    "a waiter budget was released more times than it was acquired; the "
                    "permit count is now meaningless and the never-starve guard with it"
                )
            self._held -= 1

    def __repr__(self) -> str:
        return f"<WaiterBudget {self.held}/{self._permits} permits held>"


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

    ``instant`` is the barrier's monotonic instant id, which is also its eviction order. It
    is ``0`` for a frame that never reached a bucket at all.
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

    ``first``/``last`` are the **capture** span of what has joined so far, and they are what
    makes the instant anchored rather than gridded: a frame may join while the span it would
    produce stays under one window, and the span of a closed instant is what a later frame is
    tested against to decide it is late.

    ``waiters`` is per-bucket as well as counted in the process-wide :class:`WaiterBudget`
    because two questions need it: whether the barrier may admit another waiter at all
    (the budget) and whether anybody is still interested in this particular bucket's answer
    (here, which is what lets an abandoned instant be discarded instead of associated for
    nobody).
    """

    instant: int
    deadline: float
    first: float
    last: float
    entries: list[InstantEntry] = field(default_factory=list)
    reported: set[str] = field(default_factory=set)
    waiters: int = 0
    #: Set by the thread that resolved this bucket. A waiter wakes, sees it, and returns.
    done: bool = False
    #: "Somebody sealed this bucket but had no association function in hand." Set by
    #: :meth:`InstantBarrier.drop_camera` -- the lifecycle thread must return promptly and
    #: must not run a tracker -- and by the frame path when a camera in the bucket reports its
    #: next frame. Either way a waiter, which *is* a pipeline worker with the callback, does
    #: the work. A sealed bucket accepts no further entries.
    ready: bool = False
    #: Which reason a waiter closes a ``ready`` bucket under.
    ready_reason: str = CLOSED_COMPLETE
    reason: str = ""
    results: Mapping[Any, Any] | None = None

    def spread_with(self, capture_s: float) -> float:
        """How wide this instant's capture span would be if ``capture_s`` joined it."""
        return max(self.last, capture_s) - min(self.first, capture_s)


class InstantBarrier:
    """Turns a stream of per-camera frames back into synchronised instants. Pure.

    An instant is a set of frames whose **capture** times lie within ``sync_window_s`` of each
    other — the capture clock and not the arrival clock, because arrival order is the
    pipeline's business (fair lanes, N workers, spill) and has nothing to do with whether two
    frames show the same moment. Two cameras 40 ms apart in arrival but 2 ms apart in capture
    are one instant; the reverse is two.

    The instant is **anchored, not gridded** (see the module docstring for why no absolute
    window is correct). The first frame to arrive opens a bucket and sets its span; a later
    frame joins the newest open bucket whose span it keeps inside one window; a camera already
    in that bucket offering a *later* capture seals it — its instant has all the evidence it
    will get — and opens the next with that frame; the same camera offering a capture inside
    the span it already contributed to is a duplicate and is refused.

    A bucket closes when **every live camera has reported**, when it is sealed, or when
    ``sync_window_s`` has passed since it opened, whichever is first. Whichever thread closes
    it runs the association and publishes the answer to every thread waiting on that bucket,
    keyed by whatever the association function returned — the barrier never indexes results by
    position.

    LOCKING. One :class:`threading.Condition` over one lock guards everything: the bucket map,
    the live set, the waiter count and the counters. It is held across the association
    callback, which is deliberate and is the *only* lock this codebase takes around a
    cross-camera tracker call (``docs/arch.md`` §7). Every wait is bounded by the bucket's own
    deadline, so no caller can be parked here for longer than one window even if the callback
    never runs.

    Args:
        sync_window_s: how wide an instant is — the largest capture spread one instant may
            hold, and also the maximum time any caller waits.
        workers: how many threads may be inside :meth:`submit` at once —
            :attr:`~shipinfer.topology.base.ElementContext.workers`. Used only to size a
            private :class:`WaiterBudget` when ``budget`` is not supplied. ``None`` means the
            runner did not say, and the barrier then never waits at all: guessing would park
            the only thread there is on a single-worker runner.
        budget: the process-wide waiter budget, from
            :attr:`~shipinfer.topology.base.ElementContext.waiter_budget`. ``None`` builds a
            private one with ``workers - 1`` permits, which is right for a barrier that is the
            only one in its process and is what the offline tier uses.
        max_instants: how many buckets may be open before the oldest is evicted.
        on_event: called once per *instant-level* event, under the lock, with one of
            :data:`CLOSED_COMPLETE`, :data:`CLOSED_WINDOW`, :data:`CLOSED_ADVANCED`,
            :data:`DROPPED_EVICTED`, :data:`DROPPED_EXPIRED`, :data:`DROPPED_SHUTDOWN` or
            :data:`DROPPED_FAILED`. For metrics, so it must be a counter increment and nothing
            else: it runs on the association's critical path. Frame-level outcomes are *not*
            reported here — the caller reads those off its own :class:`InstantOutcome`,
            because one closed instant resolves many frames and counting instants per frame
            would report the wrong number.

    Raises:
        ConfigurationError: a non-positive window, or fewer than one worker.
    """

    __slots__ = (
        "_announced",
        "_buckets",
        "_budget",
        "_closed",
        "_cond",
        "_frame_counts",
        "_hooked",
        "_instant_counts",
        "_live_set",
        "_max_instants",
        "_next_instant",
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
        budget: WaiterBudget | None = None,
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
        # `None` collapses to 1, and 1 gives a private budget of zero permits -- i.e. never
        # wait. The two states are deliberately the same state: "I do not know how many
        # workers there are" and "there is one worker" have the same correct behaviour, and
        # spelling them differently would be a second code path.
        self._workers = 1 if workers is None else int(workers)
        self._budget = WaiterBudget(self._workers - 1) if budget is None else budget
        self._max_instants = int(max_instants)
        self._on_event = on_event
        self._cond = threading.Condition(threading.Lock())
        #: Open instants, keyed by instant id. Insertion-ordered, so the first key is always
        #: the instant that has been open longest -- and, since every deadline is set one
        #: window after its bucket opened, the one that expires first.
        self._buckets: dict[int, _Bucket] = {}
        self._next_instant = 0
        #: The capture span of each resolved instant, bounded and newest-last. What makes a
        #: late frame *late* rather than the first member of a brand-new instant.
        self._recent: OrderedDict[int, tuple[float, float]] = OrderedDict()
        self._recent_limit = max(8, self._max_instants * 4)
        #: Cameras the runner announced through the lifecycle hooks.
        self._announced: set[str] = set()
        #: Cameras that have actually submitted a frame. Only consulted before the first
        #: announcement -- see :meth:`camera_added`.
        self._seen: set[str] = set()
        self._hooked = False
        #: The answer :meth:`_live` gives, recomputed only when the two sets above change, so
        #: the per-frame completeness test allocates nothing and still hands out an object
        #: nobody outside can mutate.
        self._live_set: frozenset[str] = frozenset()
        self._waiters = 0
        self._closed = False
        self._instant_counts: dict[str, int] = {}
        self._frame_counts: dict[str, int] = {}

    # -- reading -----------------------------------------------------------------------

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def workers(self) -> int:
        """How many threads may be inside :meth:`submit`; ``workers - 1`` may wait."""
        return self._workers

    @property
    def budget(self) -> WaiterBudget:
        """The waiter budget this barrier draws permits from, shared or private."""
        return self._budget

    @property
    def waiters(self) -> int:
        """How many callers are parked in *this* barrier right now."""
        return self._waiters

    @property
    def live(self) -> frozenset[str]:
        """The cameras a bucket waits for. See :meth:`camera_added` for why it is not the roster."""
        with self._cond:
            return self._live_set

    @property
    def open_instants(self) -> int:
        with self._cond:
            return len(self._buckets)

    @property
    def open_spans(self) -> tuple[tuple[int, float, float], ...]:
        """``(instant, first_capture, last_capture)`` per open instant, oldest first.

        For a health report and for the tests that assert *which* instant eviction took, which
        is a property worth pinning without reaching into the bucket map.
        """
        with self._cond:
            return tuple((b.instant, b.first, b.last) for b in self._buckets.values())

    def instant_stats(self) -> dict[str, int]:
        """One entry per resolved *instant*, by reason. See the vocabulary at the top."""
        with self._cond:
            return dict(self._instant_counts)

    def frame_stats(self) -> dict[str, int]:
        """One entry per *frame* that missed its instant, by reason.

        Kept apart from :meth:`instant_stats` rather than merged: :data:`DROPPED_SHUTDOWN` is
        a member of both families, and one dictionary would report frames plus instants under
        one key.
        """
        with self._cond:
            return dict(self._frame_counts)

    # -- the camera set ----------------------------------------------------------------

    def camera_added(self, camera_id: str) -> None:
        """A camera is live on this shard: instants now wait for it.

        **The live set is the announced set, not the configured roster**, and that is the one
        decision in this class that a reader will want the reason for. A group's roster names
        every camera in it; a shard runs only the ones placed on it, and the rest are on other
        shards or not started. A barrier that waited for the roster would time out on every
        single instant, for the life of the process, and report it as a healthy chain running
        one window slower.

        Before the *first* announcement the barrier falls back to the cameras it has seen
        traffic from, so a runner that does not drive the lifecycle hooks at all degrades to a
        one-instant warm-up rather than to per-camera MTMC. The fallback latches off for good
        at the first announcement: a set that came back after ``camera_removed`` emptied it
        would resurrect the camera this hook exists to forget.

        Takes only this barrier's lock and does no work under it, so the runner's lifecycle
        lock -- behind which every other lifecycle call queues -- is held for a set insertion.
        It can still queue behind an association in progress, which is bounded by one
        :meth:`submit` and is why the ABC asks for "promptly" rather than "immediately".
        """
        with self._cond:
            self._hooked = True
            self._announced.add(camera_id)
            self._refresh_live()

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
        worker that already has the association callback in hand. Like :meth:`camera_added`
        this does no work under the barrier's lock, but it can queue behind an association in
        progress -- bounded by one instant, not by the window.

        A bucket with no waiters is left alone: every frame in it was already emitted by the
        never-starve guard, so nobody is owed an answer, and its own deadline will retire it.

        Idempotent, and safe for a camera that was never added -- a removal racing an
        in-flight frame is ordinary (``Element.camera_removed``).
        """
        with self._cond:
            self._announced.discard(camera_id)
            self._seen.discard(camera_id)
            self._refresh_live()
            live = self._live_set
            woken = False
            for bucket in self._buckets.values():
                if bucket.waiters and not bucket.ready and live <= bucket.reported:
                    bucket.ready = True
                    bucket.ready_reason = CLOSED_COMPLETE
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
                not a merge; a *later* frame from a camera already in the open instant seals
                that instant and opens the next one with this frame.
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
        with self._cond:
            if self._closed:
                return self._missed(DROPPED_SHUTDOWN, 0)
            now = time.monotonic()
            self._retire(now)
            if camera_id not in self._seen:
                self._seen.add(camera_id)
                self._refresh_live()

            bucket = self._match(capture_s)
            if bucket is None:
                late = self._late_instant(capture_s)
                if late is not None:
                    return self._missed(MISSED_LATE, late)
                bucket = self._open(capture_s, now)
            elif camera_id in bucket.reported:
                if capture_s <= bucket.last:
                    return self._missed(MISSED_DUPLICATE, bucket.instant)
                # This camera has moved on, so that instant has every frame it will ever get
                # from it. Seal it -- a waiter, which holds a callback, closes it -- and put
                # this frame in the instant that starts here.
                self._seal(bucket, CLOSED_ADVANCED)
                bucket = self._open(capture_s, now)

            bucket.reported.add(camera_id)
            bucket.entries.append(InstantEntry(camera_id, payload))
            bucket.first = min(bucket.first, capture_s)
            bucket.last = max(bucket.last, capture_s)

            if self._live_set <= bucket.reported:
                return self._close(bucket, CLOSED_COMPLETE, associate)
            # The never-starve guard, counted process-wide: two barriers each admitting
            # `workers - 1` waiters would park every worker between them.
            if not self._budget.acquire():
                return self._missed(MISSED_WOULD_STARVE, bucket.instant)

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
                self._budget.release()
            if bucket.done:
                return InstantOutcome(bucket.reason, bucket.results, bucket.instant)
            if self._buckets.get(bucket.instant) is bucket:
                # Either somebody sealed it while we slept or the window ran out. Both mean:
                # this thread is the closer.
                reason = bucket.ready_reason if bucket.ready else CLOSED_WINDOW
                return self._close(bucket, reason, associate)
            # Removed from the map without being marked done. No path does that today; a
            # refusal is still better than a hang if one is ever added.
            return self._missed(DROPPED_EXPIRED, bucket.instant)

    def close_all(self, reason: str = DROPPED_SHUTDOWN) -> int:
        """Resolve every open instant and refuse every later submit. Idempotent.

        Called from an element's ``_do_close``. Without it a worker parked in :meth:`submit`
        would hold the shutdown for the rest of its window; with it, every waiter is released
        at once and ``stop()`` never spends its deadline on a barrier. The bound on the wait
        means this is belt and braces rather than the only guarantee, which is the right
        relationship between the two: a wait that is *only* bounded by a shutdown call becomes
        unbounded the first time somebody forgets to make it.

        Returns:
            How many instants were open.
        """
        with self._cond:
            self._closed = True
            open_buckets = list(self._buckets.values())
            for bucket in open_buckets:
                self._buckets.pop(bucket.instant, None)
                self._remember(bucket)
                bucket.results = None
                bucket.reason = reason
                bucket.done = True
                self._event(reason)
            if open_buckets:
                self._cond.notify_all()
            return len(open_buckets)

    # -- internals (lock held) ---------------------------------------------------------

    def _refresh_live(self) -> None:
        """Recompute the cached live set. Called only when one of the two sets changed."""
        self._live_set = frozenset(self._announced if self._hooked else self._seen)

    def _match(self, capture_s: float) -> _Bucket | None:
        """The open instant this capture belongs to: the newest whose span it stays inside.

        Newest first, because when two open buckets could both take a capture the later one is
        the instant the group is currently filling; joining the older would put this frame in
        an instant its own camera may already have contributed to.

        O(len(_buckets)), which is at most ``max_instants`` — eight by default, and in a
        healthy group one or two. A scan of that on a path already holding the lock is cheaper
        than the dict of spans that would be needed to avoid it.
        """
        for bucket in reversed(self._buckets.values()):
            if bucket.ready or bucket.done:
                continue
            if bucket.spread_with(capture_s) < self._window_s:
                return bucket
        return None

    def _late_instant(self, capture_s: float) -> int | None:
        """The resolved instant this capture belongs inside, if there is one.

        The test is the closed instant's **actual** capture span, not the window it was
        allowed: a group whose cameras are genlocked closes instants whose span is a single
        point, and testing against the full window would call every one of that camera's
        subsequent frames late — which is precisely the failure the absolute grid had.
        """
        for instant, (first, last) in reversed(self._recent.items()):
            if first <= capture_s <= last:
                return instant
        return None

    def _open(self, capture_s: float, now: float) -> _Bucket:
        """Start a new instant anchored on this capture, evicting the oldest if the map is full."""
        self._evict()
        self._next_instant += 1
        bucket = _Bucket(
            instant=self._next_instant,
            deadline=now + self._window_s,
            first=capture_s,
            last=capture_s,
        )
        self._buckets[bucket.instant] = bucket
        return bucket

    def _seal(self, bucket: _Bucket, reason: str) -> None:
        """Take no more entries for this instant and wake whoever can close it.

        Sealed rather than closed on the spot because the thread that seals is not a member of
        this instant: if it ran the association and the callback raised, the exception would
        fail *its* frame — a frame from a different instant — while the instant that actually
        failed was somebody else's. A waiter is a pipeline worker holding the callback and
        owns the answer it is waiting for, so it is the right thread to run it.

        A sealed bucket with no waiters simply expires: every frame in it was emitted by the
        never-starve guard and nobody is owed an answer.
        """
        bucket.ready = True
        bucket.ready_reason = reason
        if bucket.waiters:
            self._cond.notify_all()

    def _missed(self, reason: str, instant: int) -> InstantOutcome:
        """Count a frame-level miss and answer with it. No instant-level event."""
        self._frame_counts[reason] = self._frame_counts.get(reason, 0) + 1
        return InstantOutcome(reason, None, instant)

    def _event(self, reason: str) -> None:
        """Count an instant-level event and tell the observer, if there is one."""
        self._instant_counts[reason] = self._instant_counts.get(reason, 0) + 1
        if self._on_event is not None:
            self._on_event(reason)

    def _remember(self, bucket: _Bucket) -> None:
        """Record a resolved instant's capture span so a frame inside it is late, not new."""
        self._recent[bucket.instant] = (bucket.first, bucket.last)
        self._recent.move_to_end(bucket.instant)
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
        self._buckets.pop(bucket.instant, None)
        self._remember(bucket)
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
        return InstantOutcome(reason, results, bucket.instant)

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
            self._buckets.pop(bucket.instant, None)
            self._remember(bucket)
            bucket.results = None
            bucket.reason = DROPPED_EXPIRED
            bucket.done = True
            self._event(DROPPED_EXPIRED)
        if stale:
            self._cond.notify_all()

    def _evict(self) -> None:
        """Make room for a new instant by retiring the one open longest, and say so.

        Oldest by **instant id**, which is the order buckets were opened and therefore also
        the order their deadlines expire — every deadline is one window after its bucket
        opened. Evicting by capture time instead would let a single camera with a stale clock
        push out the instant the rest of the group is actively filling, which is the opposite
        of what eviction is for.
        """
        while len(self._buckets) >= self._max_instants:
            bucket = self._buckets.pop(next(iter(self._buckets)))
            self._remember(bucket)
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
