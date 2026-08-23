"""Errors raised while a request is in flight."""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError

__all__ = [
    "InferenceError",
    "QueueFullError",
    "RequestCancelledError",
    "RequestTimeoutError",
    "ServerStateError",
    "ValidationError",
]


class ValidationError(ShipInferError):
    """A request's tensors do not match the model's declared inputs."""


class InferenceError(ShipInferError):
    """The backend failed while executing a batch."""


class QueueFullError(ShipInferError):
    """Backpressure: the target queue is at capacity and the policy is to reject.

    Carrying the depth and capacity turns "we dropped a frame" into a number an operator
    can act on — exactly what the previous system's shared eviction buffer could never say.
    """

    def __init__(self, queue_name: str, depth: int, capacity: int) -> None:
        super().__init__(f"queue {queue_name!r} is full ({depth}/{capacity})")
        self.queue_name = queue_name
        self.depth = depth
        self.capacity = capacity


class RequestCancelledError(ShipInferError):
    """The request was dropped before execution (shutdown, or its deadline passed)."""


class RequestTimeoutError(ShipInferError):
    """The request did not complete within its deadline."""


class ServerStateError(ShipInferError):
    """An operation was attempted in the wrong lifecycle state (infer before start, ...)."""
