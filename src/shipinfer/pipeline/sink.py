"""The ingest to scheduler adapter: a tagged frame becomes a queued inference request.

This is the production :class:`shipinfer.ingest.sink.FrameSink` that ADR-011 says lives
here rather than in ``ingest``. The reasoning is worth restating because it is the reason
this file is in this package: deciding that a frame becomes a request *for a particular
model, in a particular queue, at a particular priority* is dispatch policy, and the code
that performs that mapping is the same code that has to undo it when it reassembles the
frame's results (:mod:`shipinfer.pipeline.reassembly`). One decision split across two
packages is one that drifts.

Three properties are load-bearing, and each one is a test:

**No translation layer.** ``RequestQueue.put`` raises exactly the two errors the
``FrameSink`` contract names — :class:`~shipinfer.core.errors.QueueFullError` (drop this
frame, keep going) and :class:`~shipinfer.core.errors.RequestCancelledError` (the consumer
is gone, stop) — and both already live in ``core.errors``. So they propagate untouched to
the camera actor, which is the only component that knows *which* camera to charge the drop
to. Wrapping them here would take that away, and taking it away is the whole substance of
the bug this project exists to fix (ADR-005).

**Per-camera policy is resolved by camera id, once.** ``put(frame)`` carries no policy: a
:class:`Frame` is data and a priority is configuration. Looking the priority up per *camera*
rather than carrying it on a thousand frames a second is both cheaper and harder to get
wrong — there is one place it can be wrong, and it is the config file.

**The deadline is measured from capture, not from enqueue.** The gap between the two *is*
the queue latency, and the point of a frame deadline is to discard a frame that is already
too late to act on. Measuring from the moment the adapter happened to run would make a
deeply queued frame look fresh, which is precisely backwards.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from shipinfer.core.logging import LOG, log_context
from shipinfer.core.request import InferenceRequest, Priority, RequestContext, ResponseFuture
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.core.types import Tensor
from shipinfer.scheduling.queues import RequestQueue
from shipinfer.scheduling.work import WorkItem

__all__ = ["QueueFrameSink", "TaggedFrame"]


@runtime_checkable
class TaggedFrame(Protocol):
    """The four things this adapter needs from a decoded frame.

    Structural rather than an import of :class:`shipinfer.ingest.frame.Frame`, for the same
    reason ADR-001 refers to device memory through the ``MemoryHandle`` protocol: the
    layering DAG in ``scripts/hooks/check_layers.py`` gives ``pipeline`` no edge to
    ``ingest``, and a protocol satisfies the dependency without asking for one. ``Frame``
    satisfies it exactly as written.

    Four members is also the honest measure of how narrow this seam is — the same argument
    the ``Placeable`` protocol makes for placement policies.
    """

    @property
    def camera_id(self) -> str:
        """Which camera produced this frame."""

    @property
    def context(self) -> RequestContext:
        """The ``(camera_id, frame_id, timestamps)`` tag, as every request carries it."""

    @property
    def captured_ns(self) -> int:
        """Monotonic nanoseconds at the moment of decode."""

    def as_batch(self) -> np.ndarray:
        """The image as a batch-major ``(1, H, W, C)`` array."""


@dataclass(frozen=True, slots=True)
class _CameraPolicy:
    """The dispatch decisions that belong to a camera rather than to a frame."""

    model: str
    priority: Priority


class QueueFrameSink:
    """Publishes decoded frames into a :class:`RequestQueue` as inference requests.

    One sink serves the whole fleet, because the fairness that matters is *between* cameras
    and can only be arbitrated where they meet — and it is arbitrated by the queue, which
    already buckets by ``camera_id`` and drains round-robin. There is deliberately no
    second, subtly different fairness mechanism here (ADR-005).

    Args:
        queue: the pipeline's ingest queue. Usually a
            :class:`~shipinfer.scheduling.queues.FairPriorityQueue`.
        settings: fleet ingest settings — the target model, the input tensor name and the
            frame deadline. These live in ``ingest`` settings because an operator
            configuring the video path expects to find them together, and they are read
            *here* because they are dispatch policy (see
            :mod:`shipinfer.ingest.sink`).
        cameras: the camera list, for the per-camera priority and model overrides. Defaults
            to ``settings.cameras``; pass it explicitly when the fleet came from a camera
            database rather than from the settings tree.
    """

    __slots__ = (
        "_deadline_ns",
        "_default",
        "_input",
        "_lock",
        "_policies",
        "_queue",
        "accepted",
    )

    def __init__(
        self,
        queue: RequestQueue,
        *,
        settings: IngestSettings | None = None,
        cameras: Iterable[CameraConfig] | None = None,
    ) -> None:
        resolved = settings or IngestSettings()
        self._queue = queue
        self._input = resolved.input_name
        self._deadline_ns = resolved.frame_deadline_ms * 1_000_000
        self._default = _CameraPolicy(model=resolved.target_model, priority=Priority.NORMAL)
        source = resolved.cameras if cameras is None else cameras
        self._policies: dict[str, _CameraPolicy] = {
            camera.camera_id: _CameraPolicy(
                model=camera.model or resolved.target_model, priority=camera.priority
            )
            for camera in source
        }
        # Only taken when a camera is seen for the first time, which is once per camera for
        # the life of the process — never on the steady-state path.
        self._lock = threading.Lock()
        self.accepted = 0

    # -- the FrameSink contract --------------------------------------------------------

    def put(self, frame: TaggedFrame) -> None:
        """Map one frame onto one queued request.

        Raises:
            QueueFullError: the pipeline is saturated. Propagated untouched so the camera
                actor can drop this frame and count it against the camera that sent it.
            RequestCancelledError: the queue is closed. The actor stops.
        """
        policy = self._policies.get(frame.camera_id) or self._policy_for(frame.camera_id)
        request = InferenceRequest(
            model_name=policy.model,
            inputs={self._input: Tensor.from_numpy(frame.as_batch())},
            context=frame.context,
            priority=policy.priority,
            deadline_ns=(frame.captured_ns + self._deadline_ns if self._deadline_ns else 0),
        )
        self._queue.put(WorkItem(request, ResponseFuture(request)))
        self.accepted += 1

    def _policy_for(self, camera_id: str) -> _CameraPolicy:
        """Resolve, log and memoise the policy for a camera that is not in the config.

        A camera added over the API at runtime is a normal event (a fifty-camera site gains
        cameras during commissioning), so it gets the fleet default rather than an error.
        It is logged **once** because "my new camera is not being prioritised" is otherwise
        an invisible configuration gap, and memoised because paying for that discovery on
        every frame would make it a performance bug as well.
        """
        with self._lock:
            existing = self._policies.get(camera_id)
            if existing is not None:
                return existing
            LOG.info(
                "camera %s is not in the ingest config; using model=%s priority=%s",
                camera_id,
                self._default.model,
                self._default.priority.name,
                extra=log_context(camera_id=camera_id, model=self._default.model),
            )
            self._policies[camera_id] = self._default
            return self._default

    # -- introspection -----------------------------------------------------------------

    @property
    def queue(self) -> RequestQueue:
        return self._queue

    @property
    def input_name(self) -> str:
        return self._input

    @property
    def deadline_ns(self) -> int:
        """The per-frame deadline in nanoseconds; 0 when deadlines are disabled."""
        return self._deadline_ns

    def policies(self) -> Mapping[str, tuple[str, Priority]]:
        """``camera_id -> (model, priority)``, for a health endpoint or a test."""
        return {c: (p.model, p.priority) for c, p in self._policies.items()}

    def __len__(self) -> int:
        return self.accepted

    def __repr__(self) -> str:
        return (
            f"<QueueFrameSink queue={self._queue.name} accepted={self.accepted} "
            f"cameras={len(self._policies)}>"
        )
