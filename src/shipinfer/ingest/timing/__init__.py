"""When: the two clock policies the ingest plane needs, as objects rather than sleeps.

Both are pure, hardware-free and injectable, which is the entire point. A retry schedule and
a frame rate are *policies*, and a policy tangled into a thread loop can only be tested by
running the loop and watching the wall clock — so in practice it is never tested at all, and
"it retried" gets mistaken for "it retried correctly".

* :class:`~shipinfer.ingest.timing.backoff.ExponentialBackoff` — how long to wait before
  trying a dead camera again.
* :class:`~shipinfer.ingest.timing.pacing.DeadlinePacer` — how long to wait before
  delivering the next frame, with the drift catch-up that ``sleep(1/fps)`` does not have.
"""

from shipinfer.ingest.timing.backoff import ExponentialBackoff
from shipinfer.ingest.timing.pacing import DeadlinePacer

__all__ = ["DeadlinePacer", "ExponentialBackoff"]
