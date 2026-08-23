"""Exponential backoff with jitter and a cap.

The reference implementation retried a dead camera every twenty seconds, forever, from a
shared monitor thread (``VideoReaderManager._watch_running_thread``). Two things go wrong
with that. A camera that blipped for 200 ms stays dark for twenty seconds; and fifty
cameras behind one switch retry in lockstep, so the switch coming back up is met with fifty
simultaneous RTSP DESCRIBEs, which is how a recovery turns into a second outage.

Exponential growth fixes the first, jitter fixes the second, and the cap keeps a camera
that has been down all night from waiting hours to notice it is back.
"""

from __future__ import annotations

import random

__all__ = ["ExponentialBackoff"]


class ExponentialBackoff:
    """Successive retry delays: ``initial``, ``initial*factor``, ... capped at ``cap``.

    Pure and hardware-free, so the delay *sequence* is asserted in the offline tier rather
    than inferred from a log. That matters: "it retried" is easy to observe and says
    nothing, while "it retried at 0.5 s, 1 s, 2 s, 4 s, 5 s, 5 s" is the actual policy.

    Args:
        initial_s: the first delay, in seconds.
        cap_s: the largest delay ever returned, jitter included.
        factor: growth per attempt; must be > 1 or the backoff does not back off.
        jitter: fraction of each delay to remove at random, in ``[0, 1)``. 0.2 means the
            returned delay is uniform in ``[0.8 * d, d]``. Subtractive rather than
            additive so the cap is a real bound.
        rng: injectable for tests; defaults to the module-level :mod:`random`, which is
            fine here because nothing about a retry delay needs to be unpredictable to an
            adversary.
    """

    __slots__ = ("_attempts", "_rng", "cap_s", "factor", "initial_s", "jitter")

    def __init__(
        self,
        initial_s: float = 0.5,
        cap_s: float = 30.0,
        *,
        factor: float = 2.0,
        jitter: float = 0.2,
        rng: random.Random | None = None,
    ) -> None:
        if initial_s <= 0:
            raise ValueError("initial_s must be > 0")
        if cap_s < initial_s:
            raise ValueError("cap_s must be >= initial_s")
        if factor <= 1.0:
            raise ValueError("factor must be > 1")
        if not 0.0 <= jitter < 1.0:
            raise ValueError("jitter must be in [0, 1)")
        self.initial_s = initial_s
        self.cap_s = cap_s
        self.factor = factor
        self.jitter = jitter
        self._rng = rng or random.Random()
        self._attempts = 0

    @property
    def attempts(self) -> int:
        """Delays handed out since the last :meth:`reset` — the consecutive-failure count."""
        return self._attempts

    def peek(self) -> float:
        """The next delay's un-jittered value, without consuming an attempt."""
        return min(self.cap_s, self.initial_s * self.factor**self._attempts)

    def next_delay(self) -> float:
        """The next delay, in seconds, and advance the sequence."""
        base = self.peek()
        self._attempts += 1
        if self.jitter <= 0.0:
            return base
        return base * (1.0 - self.jitter * self._rng.random())

    def reset(self) -> None:
        """Back to the first delay. Called the moment a connection succeeds."""
        self._attempts = 0

    def __repr__(self) -> str:
        return (
            f"<ExponentialBackoff initial={self.initial_s}s cap={self.cap_s}s "
            f"factor={self.factor} jitter={self.jitter} attempts={self._attempts}>"
        )
