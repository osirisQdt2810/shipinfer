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

from shipinfer.core.logging import get_logger
from shipinfer.launch.supervisor import Fleet

__all__ = ["forward_signals"]

# The logger name stays "server.launcher" on purpose: an operator's log filter is
# behaviour, and this move promises none changed. It is retargeted to "launch…" when
# server/ is deleted (A2 PR-6).
_LOG = get_logger("server.launcher")


def forward_signals(fleet: Fleet) -> None:
    """Make Ctrl-C and SIGTERM stop the fleet instead of orphaning it.

    Installed by the caller rather than by ``Fleet`` itself: signal handlers are process-global
    and a library that installs them behind your back is a library you cannot embed.
    """

    def _handle(signum: int, _frame: object) -> None:
        # Record only. The terminating happens on the supervising thread, which is the one
        # that can block: a handler that drains would deadlock on the second signal.
        _LOG.info("received signal %d; stopping fleet", signum)
        fleet.request_stop()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
