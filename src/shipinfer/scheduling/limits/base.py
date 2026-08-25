"""The rate-limiter contract: how many of a model's instances may compute at once.

This is a different bound from the one the queues apply, and the difference is the reason
the module exists. `scheduler.max_queue_size` bounds how much work may be *waiting*; nothing
bounds how much may be *running*. A model with eight instances and a full queue on each puts
all eight into compute the moment the batches close, which is precisely the burst an
operator wants to shape — because the eight share a memory bus, a PCIe root complex and, on
a multi-tenant box, the devices themselves.

Triton bounds it with a rate limiter over named resources
(``rate_limiter { resources [{ name, count }] }``); an instance must acquire its resources
before it executes. This is the same idea with the general resource model left out: the only
resource this pipeline has ever needed to bound is "an execution", so the bound is expressed
directly as a count of concurrent executions. Adding a named-resource limiter later is a new
file and a decorator, which is the point of the registry.

**Off by default**, and off means free: :class:`UnlimitedRateLimiter` is two non-blocking
method calls per batch, not per request.
"""

from __future__ import annotations

import abc
import contextlib
from collections.abc import Iterator
from typing import Any, ClassVar

from shipinfer.core.errors import ConfigurationError

__all__ = ["RateLimiter"]


class RateLimiter(abc.ABC):
    """Gates entry to a model's execution path.

    One limiter is shared by every instance of one model, because the bound is on the
    model's aggregate concurrency; the instances are what it is bounding.

    The constructor signature is deliberately uniform across the whole family — a single
    optional bound — so the model config can build any registered implementation without a
    branch on its name. A limiter that needs richer configuration should say so by raising
    :class:`~shipinfer.core.errors.ConfigurationError` from its own ``__init__``.

    Args:
        max_concurrent_executions: the bound, or 0 meaning "no bound".
    """

    name: ClassVar[str] = "abstract"

    def __init__(self, max_concurrent_executions: int = 0) -> None:
        if max_concurrent_executions < 0:
            raise ConfigurationError(
                f"max_concurrent_executions must be >= 0, got {max_concurrent_executions}"
            )
        self.limit = max_concurrent_executions
        self.granted = 0
        self.waited = 0
        self.timed_out = 0
        self.wait_ns = 0
        #: The most slots ever held at once. The one number that says whether the bound is
        #: doing anything: a limiter configured at 8 whose peak is 3 is not shaping the
        #: burst, it is decoration, and nothing else in `stats()` distinguishes the two.
        self.peak_in_flight = 0

    # -- the contract --------------------------------------------------------------------

    @abc.abstractmethod
    def acquire(self, timeout_s: float | None = None) -> bool:
        """Take one execution slot.

        Args:
            timeout_s: how long to wait, ``None`` meaning forever. A caller that must stay
                responsive to shutdown passes a short timeout and loops, which is why this
                returns a bool rather than raising: "not yet" is an ordinary outcome, not a
                failure.

        Returns:
            True when a slot was taken and :meth:`release` must be called for it.
        """

    @abc.abstractmethod
    def release(self) -> None:
        """Give one slot back. Must be called exactly once per successful acquire."""

    @property
    @abc.abstractmethod
    def in_flight(self) -> int:
        """Slots currently held. 0 when unbounded, which has none to hold."""

    @contextlib.contextmanager
    def execution(self, timeout_s: float | None = None) -> Iterator[bool]:
        """Hold a slot for the duration of the block.

        Yields whether a slot was actually taken, so the caller decides what an exhausted
        limiter means — the instance worker treats it as "try again while we are still
        running", and a test may treat it as an assertion. Yielding rather than raising
        keeps the release paired with the acquire in one place, which is the part that is
        easy to get wrong under an exception.
        """
        acquired = self.acquire(timeout_s)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()

    # -- introspection -------------------------------------------------------------------

    @property
    def mean_wait_us(self) -> float:
        """Average time a *blocked* acquire waited. 0.0 when none has blocked."""
        return 0.0 if not self.waited else self.wait_ns / self.waited / 1_000.0

    def stats(self) -> dict[str, Any]:
        return {
            "limiter": self.name,
            "limit": self.limit,
            "in_flight": self.in_flight,
            "peak_in_flight": self.peak_in_flight,
            "granted": self.granted,
            "waited": self.waited,
            "timed_out": self.timed_out,
            "mean_wait_us": round(self.mean_wait_us, 1),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} limit={self.limit} in_flight={self.in_flight}>"
