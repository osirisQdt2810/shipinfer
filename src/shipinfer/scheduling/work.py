"""The unit that travels through the scheduler.

A :class:`WorkItem` is a request plus the future that will carry its result back. It lives
here, above both the queues and the batcher, because both need it and neither owns it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from shipinfer.core.request import InferenceRequest, ResponseFuture

__all__ = ["WorkItem", "summarise_fairness"]


@dataclass(slots=True)
class WorkItem:
    """One queued request and its pending result."""

    request: InferenceRequest
    future: ResponseFuture
    enqueued_ns: int = field(default_factory=time.monotonic_ns)

    @property
    def fairness_key(self) -> str:
        """The identity fairness is measured against.

        Camera id, falling back to a shared lane for non-video callers so they queue
        together rather than each inventing a private lane and starving the cameras.
        """
        return self.request.context.camera_id or "-"

    def fail(self, error: BaseException) -> None:
        """Complete the future with an error unless the caller already gave up."""
        if self.future.set_running_or_notify_cancel():
            self.future.set_exception(error)

    def __repr__(self) -> str:
        ctx = self.request.context
        return (
            f"<WorkItem {self.request.model_name} req={self.request.request_id} "
            f"cam={ctx.camera_id} frame={ctx.frame_id} prio={self.request.priority.name}>"
        )


def summarise_fairness(items: list[WorkItem] | tuple[WorkItem, ...]) -> dict[str, int]:
    """Count a batch by camera.

    Used by tests and the ``shipinfer bench`` report to *show* that fair queueing spread a
    batch across cameras rather than merely claiming it does.
    """
    counts: dict[str, int] = {}
    for item in items:
        key = item.fairness_key
        counts[key] = counts.get(key, 0) + 1
    return counts
