"""The future a caller waits on."""

from __future__ import annotations

from concurrent.futures import Future

from shipinfer.core.request.request import InferenceRequest

__all__ = ["ResponseFuture"]


class ResponseFuture(Future):  # type: ignore[type-arg]
    """A :class:`concurrent.futures.Future` that remembers what it is waiting for.

    Subclassing rather than wrapping keeps ``as_completed`` and ``wait`` usable, which
    matters for the pipeline stages that fan one frame out into many crops and then need
    to rejoin them.
    """

    __slots__ = ("context", "model_name", "request_id")

    def __init__(self, request: InferenceRequest) -> None:
        super().__init__()
        self.request_id = request.request_id
        self.model_name = request.model_name
        self.context = request.context

    def __repr__(self) -> str:
        return (
            f"<ResponseFuture model={self.model_name} req={self.request_id} "
            f"cam={self.context.camera_id} frame={self.context.frame_id}>"
        )
