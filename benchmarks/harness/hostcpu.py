"""Host CPU-seconds for a measurement window, from the kernel rather than a sampler.

`C1-WHAT-IS-THE-5x-AGAINST?` recorded that the honest comparison had no denominator --
"GPU-seconds is the honest measure ... but `sim_pipeline_v2` reports no counterpart, so there
is nothing to divide by". The kernel keeps a counterpart for every process, and #190 put the
baseline's on its result. This is the same reading for OUR arm, so both appear in one log.

Exact, not sampled: `getrusage` returns accumulated `utime + stime`, and a delta bounds a
window with nothing to miss. Two readings, because the two arms are shaped differently -- the
baseline is a child this harness supervises, while our `single` topology IS this process and
our sharded topologies are its children.
"""

from __future__ import annotations

import resource

__all__ = ["children_since", "now", "since"]


def _totals() -> tuple[float, float]:
    """``(self, children)`` CPU-seconds so far, children counted when they are reaped."""
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (me.ru_utime + me.ru_stime, kids.ru_utime + kids.ru_stime)


def now() -> tuple[float, float]:
    """The reading a window opens with. Pass it back to :func:`since` or :func:`children_since`."""
    return _totals()


def since(before: tuple[float, float]) -> float:
    """CPU-seconds this process AND its reaped children have used since ``before``.

    What our own arm costs, in either topology: `single` runs the plane in this process and
    the sharded ones run it in children while the parent serves RTSP. Both are ours, so both
    terms count.
    """
    me, kids = _totals()
    return (me - before[0]) + (kids - before[1])


def children_since(before: tuple[float, float]) -> float:
    """CPU-seconds this process's REAPED children have used since ``before``.

    What the baseline costs, and it excludes the harness's own work by construction. `os.wait4`
    would give the same number for one child and cannot be used there: `baseline._terminate`
    reaps the process itself on the ordinary path, because the binary has no `--seconds` and is
    always stopped by signal.
    """
    return _totals()[1] - before[1]
