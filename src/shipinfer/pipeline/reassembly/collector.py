"""Joining a frame's stage results back together: complete, or timed out, never vanished.

The shape is deliberately the reference system's. ``BodyDataCollector`` kept a
``camera -> frame -> results`` buffer, marked a frame complete when it had as many results as
the frame had objects, and emitted it after ``aLiveTimeInMS`` (1500 ms) whether or not it was
complete. That much was right, and it is kept — including the timeout's value.

Three things are new, and each one is a failure mode the original had:

* **it is bounded, and the bound is per-fleet rather than per-nothing.** A camera whose
  results never arrive cannot grow the buffer without limit, and every internal structure
  is bounded, not just the obvious one — a per-camera index that keeps an entry for every
  camera ever seen is a slow leak on a 24/7 process.
* **eviction is fair.** The camera holding the most incomplete frames loses a slot, never
  the longest-waiting frame. Evicting the oldest is the inherited bug: it let a crowded
  camera starve a quiet one, and it is what ADR-005 exists to prevent. The alternative
  policy is still shipped, in :mod:`shipinfer.pipeline.reassembly.policy`, purely so the
  difference can be demonstrated.
* **a timeout emits.** A frame that lost its embedder is published *partial*, naming the
  stages that never answered, rather than deleted. A dropped frame that is counted is an
  operational fact; one that vanishes is a week of debugging.

**Sealing** is the mechanism that keeps the timeout as a safety net rather than the normal
path. When a worker finishes a frame's graph it seals the key, which emits immediately with
whatever landed — so a frame whose detector raised is published in microseconds instead of
1500 ms later. The timeout then only fires for a worker that died or wedged inside a stage,
which is the case the reference system's timeout was really for.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.settings.pipeline import ReassemblySettings
from shipinfer.pipeline.graph.state import FrameState
from shipinfer.pipeline.metrics import PipelineMetrics
from shipinfer.pipeline.reassembly.policy import EVICTION_POLICIES, EvictionPolicy, PendingKey

__all__ = ["FrameCollector", "FrameResult", "PendingFrame"]

_LOG = get_logger("pipeline.reassembly")

#: Why a frame was emitted. Carried on the event, because "no people in this frame" and "the
#: person embedder never answered" must not look the same to a consumer.
COMPLETE = "complete"
INCOMPLETE = "incomplete"
TIMEOUT = "timeout"
SHUTDOWN = "shutdown"
#: Dropped to make room. Reported like any other exit so that "every frame that is opened is
#: reported exactly once" holds without exception — the runner then decides that an evicted
#: frame is counted rather than published, because publishing a mostly-empty event while the
#: system is behind pushes the overload onto a consumer that has to reason about it.
EVICTED = "evicted"

#: One eviction in this many is logged at WARNING; the rest go to DEBUG. A saturated fleet
#: evicting at 1000 frames a second would otherwise spend real CPU formatting log records
#: that all say the same thing, and the metric is the number an operator acts on anyway —
#: the same split :class:`shipinfer.ingest.CameraActor` makes for a refused frame.
_WARN_EVERY = 256


@dataclass(frozen=True, slots=True)
class FrameResult:
    """One frame, finished — completely or otherwise."""

    state: FrameState
    #: Stages that delivered, in delivery order.
    delivered: tuple[str, ...]
    #: Stages that were expected and never delivered. Empty for a complete frame.
    missing: tuple[str, ...]
    reason: str = COMPLETE
    waited_us: int = 0
    #: The objects to publish, captured **under the collector's lock** at the moment the
    #: frame was finished.
    #:
    #: `FrameState` is documented as owned by exactly one worker for the frame's whole life
    #: (ADR-002), and the sweeper broke that: it built a result under the lock but the
    #: emission read `state` afterwards, outside it, while the owning worker was still
    #: inside `graph.execute` and could `set_detections` or `attach`. With a 1500 ms
    #: reassembly timeout against a 5000 ms stage timeout, any stage waiting on a backed-up
    #: model queue guarantees the overlap. The reader saw an empty detection list, the
    #: worker then filled it, and the scatter indexed past the end.
    #:
    #: ``None`` means "no snapshot was taken" — a `FrameResult` built directly in a test —
    #: and the caller falls back to reading the state, which is safe there because nothing
    #: else is touching it.
    objects: tuple[Any, ...] | None = None

    @property
    def key(self) -> PendingKey:
        return self.state.key

    @property
    def camera_id(self) -> str:
        return self.state.camera_id

    @property
    def frame_id(self) -> int:
        return self.state.frame_id

    @property
    def is_partial(self) -> bool:
        return bool(self.missing)

    def __repr__(self) -> str:
        return (
            f"<FrameResult {self.key} {self.reason} delivered={list(self.delivered)} "
            f"missing={list(self.missing)}>"
        )


class PendingFrame:
    """A frame waiting for its stages.

    ``__slots__`` because up to ``capacity`` of these exist at once and they are created at
    the frame rate — 1000 a second at fleet scale.
    """

    __slots__ = ("_delivered", "_expected", "deadline_ns", "opened_ns", "state")

    def __init__(self, state: FrameState, *, opened_ns: int, deadline_ns: int) -> None:
        self.state = state
        self.opened_ns = opened_ns
        self.deadline_ns = deadline_ns
        self._expected: set[str] = set()
        # A list, not a set: the order stages answered in is diagnostic, and the set
        # membership test it would give us is over at most a handful of names.
        self._delivered: list[str] = []

    @property
    def key(self) -> PendingKey:
        return self.state.key

    @property
    def camera_id(self) -> str:
        return self.state.camera_id

    def expect(self, stages: Iterable[str]) -> None:
        """Add to the set this frame is waiting for. Idempotent, by construction."""
        self._expected.update(stages)

    def deliver(self, stage: str) -> bool:
        """Record a stage's answer. Returns whether nothing is outstanding *right now*.

        "Right now" is the caveat that matters: the expected set grows as the graph decides
        which branches to run, so this is not the completion test. :meth:`FrameCollector.seal`
        is.
        """
        self._expected.add(stage)
        if stage not in self._delivered:
            self._delivered.append(stage)
        return self.is_complete

    @property
    def is_complete(self) -> bool:
        return self._expected.issubset(self._delivered)

    @property
    def expected(self) -> tuple[str, ...]:
        return tuple(sorted(self._expected))

    @property
    def delivered(self) -> tuple[str, ...]:
        return tuple(self._delivered)

    @property
    def missing(self) -> tuple[str, ...]:
        """Expected stages that never answered, sorted.

        Sorted rather than in the graph's order because a partial event's ``missing_stages``
        is read by a human and by a test, and both want the same list every time — the
        graph's order is not available here and set iteration order is not an answer.
        """
        outstanding = self._expected.difference(self._delivered)
        return tuple(sorted(outstanding))

    def is_expired(self, now_ns: int) -> bool:
        return now_ns >= self.deadline_ns

    def result(
        self,
        reason: str,
        now_ns: int,
        snapshot: Callable[[FrameState], tuple[Any, ...]] | None = None,
    ) -> FrameResult:
        """Finish this frame, capturing what emission needs while the lock is still held.

        ``snapshot`` is supplied by the collector's owner because only it knows how to turn
        a state into records — the collector has no field map. It runs here, under the lock,
        so the worker cannot be mid-`attach` when it reads.
        """
        return FrameResult(
            state=self.state,
            delivered=self.delivered,
            missing=self.missing,
            reason=reason,
            waited_us=max(0, (now_ns - self.opened_ns) // 1000),
            objects=snapshot(self.state) if snapshot is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"<PendingFrame {self.key} delivered={list(self.delivered)} "
            f"missing={list(self.missing)}>"
        )


class FrameCollector:
    """Bounded, camera-fair reassembly of per-frame stage results.

    Args:
        emit: called exactly once per frame, with the frame's result. It **must not raise**:
            the caller owns the sink and is the only component that can name it in an error,
            so it also owns catching the sink's failures. A raise here would take out the
            sweeper thread and with it every frame's timeout.
        settings: the capacity, the timeout and the eviction policy.
        metrics: shared handles. Optional so a unit test can run the collector alone, but a
            server always passes them — the eviction counter is the number that tells an
            operator which camera is flooding.
        clock: monotonic nanoseconds. Injectable so a timeout test is deterministic instead
            of sleeping through 1500 ms of real time.
    """

    def __init__(
        self,
        emit: Callable[[FrameResult], None],
        *,
        settings: ReassemblySettings | None = None,
        metrics: PipelineMetrics | None = None,
        policy: EvictionPolicy | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        snapshot: Callable[[FrameState], tuple[Any, ...]] | None = None,
    ) -> None:
        self._settings = settings or ReassemblySettings()
        self._emit = emit
        self._metrics = metrics
        self._clock = clock
        self._snapshot = snapshot
        self._policy = policy or EVICTION_POLICIES.create(self._settings.eviction_policy)
        self._timeout_ns = self._settings.timeout_ms * 1_000_000
        self._lock = threading.Lock()
        self._pending: dict[PendingKey, PendingFrame] = {}
        # camera -> keys in arrival order. A plain dict is an ordered set here, so "this
        # camera's oldest" is `next(iter(...))` rather than a scan.
        self._by_camera: dict[str, dict[PendingKey, None]] = {}
        self.opened = 0
        #: Frames handed to ``emit``, whatever their reason. Every opened frame is reported
        #: exactly once, which is the invariant an end-to-end "none lost, none duplicated"
        #: assertion rests on.
        self.reported = 0
        self.complete = 0
        self.partial = 0
        self.evicted = 0
        self.late = 0
        self.rejected = 0
        self.duplicates = 0

    # -- the PendingIndex protocol, for the eviction policy ----------------------------

    def camera_counts(self) -> Mapping[str, int]:
        return {camera: len(keys) for camera, keys in self._by_camera.items()}

    def oldest(self, camera_id: str) -> PendingKey | None:
        keys = self._by_camera.get(camera_id)
        if not keys:
            return None
        return next(iter(keys))

    def oldest_overall(self) -> PendingKey | None:
        return next(iter(self._pending), None)

    # -- producer side ------------------------------------------------------------------

    def open(self, state: FrameState, *, expected: Sequence[str] = ()) -> bool:
        """Start waiting for one frame's stages.

        Returns False when the frame could not be admitted at all, which only happens if the
        eviction policy declines to free a slot. Making room is the normal path and it is
        *charged to a camera*: the metric and the log line name the camera that lost the
        entry, so an operator can see which camera is flooding rather than only that
        something was dropped.
        """
        now = self._clock()
        deadline = now + self._timeout_ns
        if state.deadline_ns:
            # The frame's own budget, if it is tighter. Waiting past the point where the
            # result is useful spends memory to publish something nobody will act on.
            deadline = min(deadline, state.deadline_ns)
        frame = PendingFrame(state, opened_ns=now, deadline_ns=deadline)
        frame.expect(expected)

        evicted: FrameResult | None = None
        with self._lock:
            if frame.key in self._pending:
                # Two live frames with one tag. ADR-002 says this never happens — the frame
                # counter belongs to the camera's actor for its whole life — and if it does,
                # replacing the in-flight one would silently discard its partial results and
                # break the "every opened frame is reported exactly once" invariant. So the
                # newcomer is refused and the collision is named.
                self.duplicates += 1
                _LOG.error(
                    "duplicate reassembly key %s: a frame with this tag is already in "
                    "flight, so the new one is refused. A camera's frame_id must never "
                    "repeat (ADR-002); check first_frame_id after a re-add.",
                    frame.key,
                    extra=log_context(camera_id=state.camera_id, frame_id=state.frame_id),
                )
                return False
            while len(self._pending) >= self._settings.capacity:
                victim = self._evict_locked()
                if victim is None:
                    self.rejected += 1
                    return False
                evicted = victim
            self._pending[frame.key] = frame
            self._by_camera.setdefault(state.camera_id, {})[frame.key] = None
            self.opened += 1

        if evicted is not None:
            self._report_eviction(evicted)
        return True

    def _evict_locked(self) -> FrameResult | None:
        """Make one slot. Caller holds the lock."""
        key = self._policy.choose(self)
        if key is None:
            return None
        frame = self._remove_locked(key)
        if frame is None:  # pragma: no cover - a policy naming a key it was not given
            return None
        self.evicted += 1
        return frame.result(EVICTED, self._clock(), self._snapshot)

    def _report_eviction(self, evicted: FrameResult) -> None:
        """Count, log and report an eviction — naming the camera that caused it.

        The camera is the whole point. "We dropped 86 frames" is not actionable; "camera
        ``quay_west`` lost 86 frames while holding 14 incomplete ones" tells an operator
        which stream to look at. That number is what the previous system could not produce at
        all, because its buffer had no idea whose entry it was deleting.
        """
        if self._metrics is not None:
            self._metrics.frames_evicted.inc(camera=evicted.camera_id)
        log = (
            _LOG.warning if self.evicted == 1 or self.evicted % _WARN_EVERY == 0 else _LOG.debug
        )
        log(
            "reassembly full (%d/%d): dropped %s frame %d after %d us, missing %s "
            "[%d evicted so far]",
            len(self._pending),
            self._settings.capacity,
            evicted.camera_id,
            evicted.frame_id,
            evicted.waited_us,
            list(evicted.missing),
            self.evicted,
            extra=log_context(camera_id=evicted.camera_id, frame_id=evicted.frame_id),
        )
        self._publish(evicted)

    def expect(self, key: PendingKey, stages: Sequence[str]) -> None:
        """Widen the set of stages this frame is waiting for. Idempotent."""
        if not stages:
            return
        with self._lock:
            frame = self._pending.get(key)
            if frame is not None:
                frame.expect(stages)

    def deliver(self, key: PendingKey, stage: str) -> None:
        """Record one stage's answer.

        Recording, not emitting. A frame is *not* published the moment its currently-expected
        set is satisfied, because that set grows as branches are decided: after the detector
        answers, ``{detect}`` is momentarily complete while five stages are still to come.
        Only the worker knows when a frame is finished, and it says so by calling
        :meth:`seal` — with the sweeper as the safety net for a worker that never does.

        A delivery for a frame that is no longer pending is counted as a **late arrival**:
        its frame was already timed out or evicted, and the result now has nowhere to go.
        That number is worth watching — a non-zero rate means the reassembly timeout is
        tighter than the pipeline's real latency.
        """
        with self._lock:
            frame = self._pending.get(key)
            if frame is None:
                self.late += 1
                if self._metrics is not None:
                    self._metrics.late_arrivals.inc(camera=key[0], stage=stage)
                return
            frame.deliver(stage)

    def seal(self, key: PendingKey) -> None:
        """No further stages are coming for this frame — emit it now.

        The normal end of a frame. It is separate from :meth:`deliver` because a frame can be
        finished *without* being complete: a stage that raised, or a branch whose input was
        empty and whose downstream was therefore never planned. Waiting for the timeout in
        those cases would add 1500 ms of latency to every failure.
        """
        with self._lock:
            frame = self._pending.pop(key, None)
            if frame is None:
                return
            self._forget_camera_locked(key)
            reason = COMPLETE if frame.is_complete else INCOMPLETE
            result = frame.result(reason, self._clock(), self._snapshot)
        self._publish(result)

    # -- the timeout safety net ---------------------------------------------------------

    def sweep(self, now_ns: int | None = None) -> int:
        """Emit every frame past its deadline. Returns how many.

        Called on a timer rather than armed per frame: one wake-up every
        ``sweep_interval_ms`` for the whole fleet, against 1000 timers a second. The cost is
        that the timeout is accurate to within one interval, which for a 1500 ms budget is
        noise.
        """
        now = self._clock() if now_ns is None else now_ns
        with self._lock:
            expired = [key for key, frame in self._pending.items() if frame.is_expired(now)]
            results = [
                frame.result(TIMEOUT, now, self._snapshot)
                for frame in (self._remove_locked(key) for key in expired)
                if frame is not None
            ]
            if self._metrics is not None:
                for camera, count in self.camera_counts().items():
                    self._metrics.pending_frames.set(count, camera=camera)
        for result in results:
            self._publish(result)
        return len(results)

    def drain(self) -> int:
        """Emit everything still pending, as ``shutdown``. Returns how many.

        A shutdown that silently discards four hundred half-finished frames is not an orderly
        shutdown — the same argument :meth:`shipinfer.scheduling.queues.RequestQueue.close`
        makes for failing its queued items explicitly.
        """
        now = self._clock()
        with self._lock:
            results = [
                frame.result(SHUTDOWN, now, self._snapshot) for frame in self._pending.values()
            ]
            self._pending.clear()
            self._by_camera.clear()
        for result in results:
            self._publish(result)
        return len(results)

    # -- internals ----------------------------------------------------------------------

    def _remove_locked(self, key: PendingKey) -> PendingFrame | None:
        frame = self._pending.pop(key, None)
        if frame is not None:
            self._forget_camera_locked(key)
        return frame

    def _forget_camera_locked(self, key: PendingKey) -> None:
        """Drop the key from its camera's index, and the camera when it empties.

        Deleting the empty camera entry is the difference between a bounded structure and a
        slow leak: a 24/7 process that has seen a thousand transient camera ids would
        otherwise keep a thousand empty dicts forever.
        """
        keys = self._by_camera.get(key[0])
        if keys is None:
            return
        keys.pop(key, None)
        if not keys:
            del self._by_camera[key[0]]

    def _publish(self, result: FrameResult) -> None:
        self.reported += 1
        if result.is_partial:
            self.partial += 1
        else:
            self.complete += 1
        if self._metrics is not None and result.reason != EVICTED:
            self._metrics.frames_emitted.inc(camera=result.camera_id, reason=result.reason)
            if result.is_partial:
                self._metrics.frames_partial.inc(camera=result.camera_id, reason=result.reason)
        self._emit(result)

    # -- introspection ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pending)

    @property
    def capacity(self) -> int:
        return self._settings.capacity

    def pending_per_camera(self) -> dict[str, int]:
        """Incomplete frames per camera — the fairness assertion, and the dashboard."""
        with self._lock:
            return dict(self.camera_counts())

    def sizes(self) -> dict[str, int]:
        """The length of every internal structure, so boundedness can be asserted rather
        than assumed."""
        with self._lock:
            return {
                "pending": len(self._pending),
                "cameras": len(self._by_camera),
                "camera_entries": sum(len(keys) for keys in self._by_camera.values()),
            }

    def stats(self) -> dict[str, int | float]:
        return {
            "pending": len(self._pending),
            "capacity": self._settings.capacity,
            "opened": self.opened,
            "reported": self.reported,
            "complete": self.complete,
            "partial": self.partial,
            "evicted": self.evicted,
            "late": self.late,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "timeout_ms": self._settings.timeout_ms,
            "eviction_policy": self._policy.name,
        }

    def __repr__(self) -> str:
        return (
            f"<FrameCollector {len(self._pending)}/{self._settings.capacity} "
            f"reported={self.reported} partial={self.partial} evicted={self.evicted}>"
        )
