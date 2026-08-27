"""Plain FIFO, priority-blind and fairness-blind."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Sequence

from shipinfer.core.errors import QueueFullError, RequestCancelledError
from shipinfer.core.settings import OverflowPolicy
from shipinfer.scheduling.queues.base import BatchWindow, QueueStats, RequestQueue
from shipinfer.scheduling.queues.registry import QUEUES
from shipinfer.scheduling.work import WorkItem

__all__ = ["FifoQueue"]


@QUEUES.register("fifo")
class FifoQueue(RequestQueue):
    """First in, first out.

    Kept as the honest baseline. Benchmarks that claim fair queueing helps need something
    to be better *than*, and "the obvious implementation" is the fairest comparison. It is
    also the right queue for a model with a single producer, where fairness machinery is
    pure overhead.
    """

    name = "fifo"

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
        self._items: deque[WorkItem] = deque()
        self._closed = False
        self._cond = threading.Condition(threading.Lock())
        self._accepted = 0
        self._rejected = 0
        self._evicted = 0
        self._expired = 0
        #: Per-camera attribution, kept even though this queue is fairness-blind: the
        #: breakdown is how an operator *sees* that it is. A FIFO under DROP_OLDEST charges
        #: whichever camera happened to be at the head, and that is the reading ADR-005
        #: wants visible rather than argued about.
        self._rejected_by_camera: Counter[str] = Counter()
        self._evicted_by_camera: Counter[str] = Counter()
        self._expired_by_camera: Counter[str] = Counter()

    @property
    def depth(self) -> int:
        return len(self._items)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def stats(self) -> QueueStats:
        """A snapshot under the lock, with every map copied.

        ``depth_by_camera`` is a walk of the deque, O(depth), taken here and not maintained
        on :meth:`put`/:meth:`get_batch` — same trade as the fair queue: a snapshot an
        operator reads every few seconds must not cost anything on the path that runs
        15 000 times a second.
        """
        with self._cond:
            depth_by_camera: Counter[str] = Counter(item.fairness_key for item in self._items)
            return QueueStats(
                depth=len(self._items),
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

    def put(self, item: WorkItem) -> None:
        with self._cond:
            if self._closed:
                raise RequestCancelledError(f"queue {self.name!r} is closed")
            if len(self._items) >= self.capacity and not self._make_room_locked():
                # A BLOCK producer woken by `close()` is cancelled, not refused — see
                # `FairPriorityQueue.put` for what charging it instead reported.
                if self._closed:
                    raise RequestCancelledError(f"queue {self.name!r} is closed")
                self._rejected += 1
                self._rejected_by_camera[item.fairness_key] += 1
                raise QueueFullError(self.name, len(self._items), self.capacity)
            item.request.timings.queued_ns = item.enqueued_ns
            self._items.append(item)
            self._accepted += 1
            self._cond.notify()

    def _make_room_locked(self) -> bool:
        if self.overflow is OverflowPolicy.REJECT:
            return False
        if self.overflow is OverflowPolicy.BLOCK:
            deadline = time.monotonic() + self.block_timeout_s
            while len(self._items) >= self.capacity and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return not self._closed
        victim = self._items.popleft()
        self._evicted += 1
        self._evicted_by_camera[victim.fairness_key] += 1
        victim.fail(QueueFullError(self.name, len(self._items) + 1, self.capacity))
        return True

    def get_batch(self, window: BatchWindow, *, poll_s: float = 0.05) -> list[WorkItem]:
        with self._cond:
            while not self._items:
                if self._closed:
                    return []
                self._cond.wait(poll_s)

            if window.max_delay_us > 0 and len(self._items) < window.max_batch_size:
                deadline = time.monotonic() + window.max_delay_us / 1_000_000.0
                preferred = set(window.preferred_sizes)
                while len(self._items) < window.max_batch_size and not self._closed:
                    if len(self._items) in preferred:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)

            now = time.monotonic_ns()
            batch: list[WorkItem] = []
            rows = 0
            # Rows, not items — a per-object request carries one row per crop. See
            # `FairQueue._drain_locked` for what counting items instead cost.
            while self._items:
                head_rows = self._items[0].request.batch_size or 1
                if batch and rows + head_rows > window.max_batch_size:
                    break
                item = self._items.popleft()
                if self.drop_expired and item.request.is_expired(now):
                    self._expired += 1
                    self._expired_by_camera[item.fairness_key] += 1
                    item.fail(RequestCancelledError("request deadline passed before execution"))
                    continue
                batch.append(item)
                rows += head_rows
                if rows >= window.max_batch_size:
                    break
            if self.overflow is OverflowPolicy.BLOCK:
                self._cond.notify_all()
            return batch

    def close(self, error: BaseException | None = None) -> Sequence[WorkItem]:
        with self._cond:
            self._closed = True
            drained = list(self._items)
            self._items.clear()
            self._cond.notify_all()
        reason = error or RequestCancelledError(f"queue {self.name!r} closed during shutdown")
        for item in drained:
            item.fail(reason)
        return drained
