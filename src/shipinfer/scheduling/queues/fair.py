"""The default queue: priority lanes, round-robin fair within a lane."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Sequence

from shipinfer.core.errors import QueueFullError, RequestCancelledError
from shipinfer.core.logging import LOG
from shipinfer.core.request import Priority
from shipinfer.core.settings import OverflowPolicy
from shipinfer.scheduling.queues.base import BatchWindow, QueueStats, RequestQueue
from shipinfer.scheduling.queues.lanes import Lane
from shipinfer.scheduling.queues.registry import QUEUES
from shipinfer.scheduling.work import WorkItem

__all__ = ["FairPriorityQueue"]

_PRIORITY_LEVELS = len(Priority)


@QUEUES.register("fair", "fair_priority")
class FairPriorityQueue(RequestQueue):
    """Priority-ordered across lanes, camera-fair within a lane.

    This class is the direct answer to the failure documented in the reference system's
    ``docs/flow.md``: every camera fed one shared 1000-slot buffer that evicted the
    *oldest* entry when full, so a crowded camera silently starved a quiet one.

    Two choices fix it, and both live here:

    1. **Fair queueing.** Requests are bucketed by ``camera_id`` and drained round-robin,
       so a camera producing 30 crops per frame cannot occupy 30 consecutive batch slots.
    2. **Honest overflow.** A full queue raises :class:`QueueFullError` by default.
       Backpressure that reaches the producer is a signal; a silent eviction three stages
       downstream is a bug that takes a week to find.

    Everything here is pure Python and hardware-free, which is what makes the invariants
    testable without a GPU. The compiled ``shipinfer._C`` queue implements the same
    contract with a lock-free ring; ``tests/scheduling/test_queue_parity.py`` asserts the
    two agree.
    """

    name = "fair"

    def __init__(
        self,
        name: str,
        capacity: int,
        *,
        overflow: OverflowPolicy = OverflowPolicy.REJECT,
        block_timeout_ms: int = 50,
        drop_expired: bool = True,
    ) -> None:
        super().__init__(
            name,
            capacity,
            overflow=overflow,
            block_timeout_ms=block_timeout_ms,
            drop_expired=drop_expired,
        )
        self._lanes: list[Lane] = [Lane() for _ in range(_PRIORITY_LEVELS)]
        self._size = 0
        self._closed = False
        self._cond = threading.Condition(threading.Lock())
        self._accepted = 0
        self._rejected = 0
        self._evicted = 0
        self._expired = 0
        #: The per-camera half of ADR-005. A `Counter` because every increment is
        #: `+= 1` on a key that may not exist yet, and all three are written under
        #: `self._cond` — the same lock that guards the lanes they describe.
        self._rejected_by_camera: Counter[str] = Counter()
        self._evicted_by_camera: Counter[str] = Counter()
        self._expired_by_camera: Counter[str] = Counter()

    # -- introspection -----------------------------------------------------------------

    @property
    def depth(self) -> int:
        return self._size

    @property
    def is_closed(self) -> bool:
        return self._closed

    def stats(self) -> QueueStats:
        """A snapshot, taken under the lock and handed out as copies.

        The lock is why ``depth_by_camera`` is computed here rather than maintained: a walk
        of every lane's keys is O(cameras x priorities) — 200 dict entries at the design
        point — and it happens once per stats call, not once per frame. Counting on
        :meth:`put` and :meth:`get_batch` instead would put that bookkeeping on the path
        that runs 15 000 times a second to save a walk an operator triggers every few
        seconds.

        Every map is a copy. :attr:`depth` is unavoidably one (it is built here), but the
        three counters would otherwise be live objects the caller could mutate — and a
        health document that gets trimmed in place would silently rewrite the queue's own
        attribution.
        """
        with self._cond:
            depth_by_camera: Counter[str] = Counter()
            for lane in self._lanes:
                depth_by_camera.update(lane.depths())
            return QueueStats(
                depth=self._size,
                capacity=self.capacity,
                accepted=self._accepted,
                rejected=self._rejected,
                evicted=self._evicted,
                expired=self._expired,
                depth_by_camera=dict(depth_by_camera),
                rejected_by_camera=dict(self._rejected_by_camera),
                evicted_by_camera=dict(self._evicted_by_camera),
                expired_by_camera=dict(self._expired_by_camera),
            )

    # -- producer ----------------------------------------------------------------------

    def put(self, item: WorkItem) -> None:
        with self._cond:
            if self._closed:
                raise RequestCancelledError(f"queue {self.name!r} is closed")

            if self._size >= self.capacity and not self._make_room_locked():
                # `_make_room_locked` has two false exits and they are different events.
                # Under BLOCK a producer asleep in it is woken by `close()` as well as by
                # the timeout, and the two were indistinguishable here: a shutdown charged
                # `rejected_by_camera` and raised `QueueFullError("full (0/1)")`, so an
                # orderly stop read as a camera flooding in the one view an operator uses
                # to find floods. `QueueStats` promises close() charges nobody; this is
                # where that promise is kept for the producer that was still asleep.
                if self._closed:
                    raise RequestCancelledError(f"queue {self.name!r} is closed")
                self._rejected += 1
                self._rejected_by_camera[item.fairness_key] += 1
                raise QueueFullError(self.name, self._size, self.capacity)

            item.request.timings.queued_ns = item.enqueued_ns
            self._lanes[int(item.request.priority)].push(item)
            self._size += 1
            self._accepted += 1
            self._cond.notify()

    def _make_room_locked(self) -> bool:
        """Try to free one slot. Returns True if the caller may now enqueue."""
        if self.overflow is OverflowPolicy.REJECT:
            return False

        if self.overflow is OverflowPolicy.BLOCK:
            deadline = time.monotonic() + self.block_timeout_s
            while self._size >= self.capacity and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return not self._closed

        # DROP_OLDEST: sacrifice from the *lowest*-priority non-empty lane, so a
        # BACKGROUND request can never displace a TRACKING_CRITICAL one.
        for lane in reversed(self._lanes):
            victim = lane.evict_from_longest()
            if victim is not None:
                self._size -= 1
                self._evicted += 1
                # The victim's camera is the greediest by construction — `evict_from_longest`
                # picks the key hogging the lane — so this counter names the flood, not the
                # camera whose frame merely happened to be oldest. That inversion is the
                # whole of ADR-005, and until now it was invisible in the numbers.
                self._evicted_by_camera[victim.fairness_key] += 1
                victim.fail(QueueFullError(self.name, self._size + 1, self.capacity))
                return True
        return False

    # -- consumer ----------------------------------------------------------------------

    def get_batch(self, window: BatchWindow, *, poll_s: float = 0.05) -> list[WorkItem]:
        """Two-phase wait — the classic dynamic-batching shape.

        1. Wait (indefinitely, but wake-able) for the *first* request. An idle model must
           not burn a core spinning.
        2. Once one has arrived, wait at most ``max_delay_us`` more for the batch to fill,
           returning early the moment it reaches ``max_batch_size`` or a preferred size.
        """
        with self._cond:
            while self._size == 0:
                if self._closed:
                    return []
                self._cond.wait(poll_s)

            if window.max_delay_us > 0 and self._size < window.max_batch_size:
                self._wait_to_fill_locked(window)

            return self._drain_locked(window.max_batch_size)

    def _wait_to_fill_locked(self, window: BatchWindow) -> None:
        deadline = time.monotonic() + window.max_delay_us / 1_000_000.0
        preferred = set(window.preferred_sizes)
        # `self._size` counts items and `max_batch_size` counts rows, so this is a lower
        # bound on fullness: with multi-row requests the batch reaches its row budget before
        # the item count does, and waiting past that only adds latency. Deliberately not
        # made exact — summing every queued request's rows on each wake would walk the whole
        # queue thousands of times a second to refine a wait heuristic.
        while self._size < window.max_batch_size and not self._closed:
            if self._size in preferred:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._cond.wait(remaining)

    def _drain_locked(self, max_rows: int) -> list[WorkItem]:
        """Pop up to ``max_rows`` **rows** highest-priority-first, round-robin in a lane.

        Rows, not items. A per-object request carries one row per crop, so counting items
        against a row budget overfills the batch: sixteen person-embedder requests carrying
        a frame's worth of crops each assembled 24 rows against ``max_batch_size: 16``, the
        assembler refused it, and every request in it failed. Observed on the first real
        50-camera run — the offline tests never saw it because their requests are one row.

        An item whose own row count already exceeds the budget is still returned, alone.
        Refusing to dequeue it would park it at the head of its lane forever and stall the
        model; letting it through gives the assembler a chance to name the real problem,
        which is a request too large for the engine rather than a scheduling decision.
        """
        now = time.monotonic_ns()
        batch: list[WorkItem] = []
        rows = 0
        try:
            for lane in self._lanes:
                while lane.size:
                    head = lane.peek()
                    if head is None:
                        break
                    head_rows = head.request.batch_size or 1
                    if batch and rows + head_rows > max_rows:
                        return batch
                    item = lane.pop()
                    self._size -= 1
                    if self.drop_expired and item.request.is_expired(now):
                        self._expired += 1
                        self._expired_by_camera[item.fairness_key] += 1
                        item.fail(
                            RequestCancelledError("request deadline passed before execution")
                        )
                        continue
                    batch.append(item)
                    rows += head_rows
                    if rows >= max_rows:
                        return batch
            return batch
        finally:
            # `finally`, not a line after the loops. Converting the row-budget exits from
            # `break` to `return` — which they have to be, to leave a partially filled lane
            # alone — jumped straight over this, and the exit at the row budget is the
            # *common* one under load, so a blocked producer slept the full
            # `block_timeout_ms` instead of waking the instant a slot freed. Measured at 500
            # against 50 ms. Worse than the latency: when the deadline beat the drain, `put`
            # raised `QueueFullError` and the camera actor charged a drop to a camera that
            # had done nothing wrong, which is the ADR-005 misattribution this project
            # exists to prevent. `fifo.py` kept its `break` and never had the bug.
            if self.overflow is OverflowPolicy.BLOCK:
                self._cond.notify_all()  # producers may be waiting for space

    # -- lifecycle ---------------------------------------------------------------------

    def close(self, error: BaseException | None = None) -> Sequence[WorkItem]:
        with self._cond:
            self._closed = True
            drained: list[WorkItem] = []
            for lane in self._lanes:
                drained.extend(lane.drain())
            self._size = 0
            self._cond.notify_all()

        reason = error or RequestCancelledError(f"queue {self.name!r} closed during shutdown")
        for item in drained:
            item.fail(reason)
        if drained:
            LOG.warning("queue %s closed with %d in-flight request(s)", self.name, len(drained))
        return drained
