"""Frame pacing with drift catch-up.

``time.sleep(1/fps)`` between frames is the obvious implementation and it is wrong: the
sleep is only part of the loop, so the real period is ``1/fps + decode + publish`` and a
"20 fps" replay source delivers 17. The error is systematic, it compounds over a long run,
and it silently makes every throughput measurement taken with it optimistic about the
server and pessimistic about the load.

Accumulating an absolute deadline fixes it, which is what
``references/counting-simulation/sim_pipeline_v2.py`` does. The other half — resetting the
deadline to *now* when the loop has fallen behind — is what stops a hiccup from being
repaid as a burst of back-to-back frames that no downstream queue asked for.
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["DeadlinePacer"]


class DeadlinePacer:
    """Holds a loop at ``fps`` by advancing an absolute deadline.

    Args:
        fps: target rate. ``<= 0`` disables pacing entirely and :meth:`wait` becomes a
            no-op, which is what a live RTSP source wants — the camera sets the rate.
        monotonic: clock source, injectable so the offline tier can assert the arithmetic
            without spending real seconds.
        sleep: sleep function, injectable for the same reason.
    """

    __slots__ = ("_deadline", "_monotonic", "_sleep", "behind", "interval_s")

    def __init__(
        self,
        fps: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval_s = 1.0 / fps if fps > 0 else 0.0
        self._monotonic = monotonic
        self._sleep = sleep
        self._deadline = 0.0
        #: How many times the loop was already late when :meth:`wait` was called. A
        #: non-zero value on a replay run means the *consumer* could not keep up, which is
        #: exactly what a stress test wants to know.
        self.behind = 0

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0.0

    def reset(self) -> None:
        """Start the schedule from now. Called when a source opens or reopens."""
        self._deadline = self._monotonic()

    # doc: long the interrupt, because a pacer that cannot be interrupted holds a stop
    def wait(self, interrupt: Callable[[float], bool] | None = None) -> bool:
        """Block until the next frame is due. ``True`` if ``interrupt`` cut it short.

        Returns immediately, without advancing time, when the loop is already late: the
        deadline is reset to *now* rather than left in the past, so lateness is absorbed
        instead of being repaid as a burst.

        ``interrupt`` is given the seconds owed and returns whether it was woken rather
        than timing out -- ``Event.wait`` has exactly that shape. This is the same seam as
        ``csrc/…/sources/replay.cpp``'s ``pacer_.wait``, which takes the same callback: a
        pacer that only ``sleep``s makes a stop wait out the whole frame period, and at
        1 fps that is a second per camera on every shutdown. **The deadline does not advance
        on an interrupted wait**, so the frame schedule survives one.
        """
        if not self.enabled:
            return False
        if self._deadline == 0.0:
            self.reset()
        now = self._monotonic()
        due = self._deadline + self.interval_s
        if due > now:
            if interrupt is not None and interrupt(due - now):
                return True
            if interrupt is None:
                self._sleep(due - now)
            self._deadline = due
        else:
            self.behind += 1
            self._deadline = now
        return False

    def __repr__(self) -> str:
        rate = 1.0 / self.interval_s if self.enabled else 0.0
        return f"<DeadlinePacer {rate:g} fps behind={self.behind}>"
