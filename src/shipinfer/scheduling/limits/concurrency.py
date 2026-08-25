"""Bound how many of a model's instances may be executing at the same time.

The bound a burst needs. Eight instances each holding a full queue will all enter compute
the moment their batching windows close, and they share a memory bus, a PCIe root complex
and — on a shared box — the devices. Capping concurrent executions turns that spike into a
queue, which is a thing the rest of this system already measures and sheds.

The instances that do not get a slot **wait**; they do not shed. Shedding is the queue's
job, and it already happens at the right place — the edge, where the caller learns about it
and a counter records it. A limiter that dropped work would be a second, invisible eviction
policy, which is the failure this whole project was rebuilt to remove.
"""

from __future__ import annotations

import threading
import time
from typing import ClassVar

from shipinfer.core.errors import ConfigurationError
from shipinfer.scheduling.limits.base import RateLimiter
from shipinfer.scheduling.limits.registry import RATE_LIMITERS

__all__ = ["ConcurrencyRateLimiter"]


@RATE_LIMITERS.register(
    "concurrency",
    "max_concurrent",
    description="Cap simultaneous executions of one model across its instances",
)
class ConcurrencyRateLimiter(RateLimiter):
    """A counting semaphore over a model's executions.

    Args:
        max_concurrent_executions: how many instances of this model may be inside their
            execute path at once. Must be at least 1 — 0 would mean a model that can never
            run, which is a config mistake worth failing at start-up rather than a deadlock
            worth debugging at 3am.

    ``BoundedSemaphore`` rather than ``Semaphore``: an unpaired :meth:`release` — a bug that
    is easy to write around an exception path — would silently raise the ceiling and the
    limiter would stop limiting without anything saying so. Bounded turns that into a
    ``ValueError`` at the moment the pairing broke.
    """

    name: ClassVar[str] = "concurrency"

    def __init__(self, max_concurrent_executions: int = 0) -> None:
        super().__init__(max_concurrent_executions)
        if self.limit < 1:
            raise ConfigurationError(
                "the concurrency rate limiter needs max_concurrent_executions >= 1; "
                "use kind 'off' to disable rate limiting"
            )
        self._slots = threading.BoundedSemaphore(self.limit)
        self._held = 0
        self._counter_lock = threading.Lock()

    def acquire(self, timeout_s: float | None = None) -> bool:
        # The uncontended case must not pay for a timed wait, and it must not be counted as
        # a wait either: `waited` is what tells an operator the limiter is actually binding,
        # so counting every acquire would make it useless for exactly that question.
        if self._slots.acquire(blocking=False):
            self._on_acquired(0)
            return True

        started_ns = time.monotonic_ns()
        acquired = (
            self._slots.acquire()
            if timeout_s is None
            else self._slots.acquire(timeout=timeout_s)
        )
        waited_ns = time.monotonic_ns() - started_ns
        if not acquired:
            with self._counter_lock:
                self.timed_out += 1
                self.waited += 1
                self.wait_ns += waited_ns
            return False
        self._on_acquired(waited_ns)
        return True

    def _on_acquired(self, waited_ns: int) -> None:
        with self._counter_lock:
            self.granted += 1
            self._held += 1
            self.peak_in_flight = max(self.peak_in_flight, self._held)
            if waited_ns:
                self.waited += 1
                self.wait_ns += waited_ns

    def release(self) -> None:
        """Give a slot back. Safe to call at most once per :meth:`acquire`.

        Both steps under one lock, and the ordering inside it still matters.

        The semaphore goes first: `BoundedSemaphore.release` raises on an unpaired call, and
        decrementing before it meant the counter had already moved when the raise happened —
        so `in_flight` went permanently negative and every later reading of it was wrong,
        including the one an operator uses to decide whether a pool is saturated.

        And the decrement is under the *same* lock as the release, not a second acquisition of
        it. Between the two, a waiter parked in `_slots.acquire()` can wake and run
        `_on_acquired` against a `_held` that has not yet come down — so `in_flight` and
        `peak_in_flight` both read `limit + 1`. The semaphore never over-admits, so this is not
        a safety bug; it is a reporting one, and `peak_in_flight` is the one number that says
        whether the bound is doing anything. A limiter whose peak reads above its own limit
        reads to an operator as a limiter that is not holding. Review reproduced a peak of 4
        against a bound of 2; two hammers here did not — the window is small and CPython
        serialises acquire/release tightly — which is why the fix is argued from the code
        rather than from a failing run.
        """
        with self._counter_lock:
            self._slots.release()
            self._held -= 1

    @property
    def in_flight(self) -> int:
        return self._held
