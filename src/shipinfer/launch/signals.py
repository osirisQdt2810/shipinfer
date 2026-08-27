"""Ctrl-C and SIGTERM, routed to the fleet instead of past it.

Its own module rather than a method on :class:`~shipinfer.launch.supervisor.Fleet`, for the
reason the function's docstring gives: signal handlers are process-global state, and a class
that installed them when you constructed it could not be embedded in a process that has its
own idea of what SIGTERM means. The caller opts in.

It is also the seam that will not change when the control plane does. A shard supervised over
gRPC still has to be told to go when the parent is told to go (arch.md §2), and "the handler
records, the supervising thread does the blocking work" is the invariant that makes that safe
regardless of what the stopping consists of.
"""

from __future__ import annotations

import signal
from typing import Protocol

from shipinfer.core.logging import get_logger

__all__ = ["Stoppable", "forward_signals"]

# Renamed with `server/` itself; the reason is in `launch/supervisor.py`, once.
_LOG = get_logger("launch.signals")


class Stoppable(Protocol):
    """Anything that can be *asked* to stop without being stopped on the caller's thread.

    Structural rather than a base class, and one method wide, because that is the whole of
    what a signal handler may do. :class:`~shipinfer.launch.supervisor.Fleet` satisfies it and
    so does every :class:`~shipinfer.runners.base.Runner`, which is what lets ``shipinfer run``
    install these handlers over the runner it holds without ``launch`` importing ``runners``.
    """

    def request_stop(self) -> None: ...


def forward_signals(target: Stoppable) -> None:
    """Make Ctrl-C and SIGTERM stop ``target`` instead of orphaning it.

    Installed by the caller rather than by the target itself: signal handlers are
    process-global and a library that installs them behind your back is a library you cannot
    embed.
    """

    def _handle(signum: int, _frame: object) -> None:
        # Record only. The terminating happens on the supervising thread, which is the one
        # that can block: a handler that drains would deadlock on the second signal.
        _LOG.info("received signal %d; stopping", signum)
        target.request_stop()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
