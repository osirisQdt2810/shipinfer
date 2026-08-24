"""Where a camera actor puts a frame — a protocol this package owns.

The ingest plane's job ends at "here is a tagged frame". Turning that frame into an
inference request for a particular model, in a particular queue, is **dispatch policy**, and
dispatch policy belongs next to the DAG that consumes it: the same code that maps a frame
onto a request has to undo that mapping when it reassembles the results, and splitting one
decision across two packages is how the two halves drift apart.

So ``ingest`` depends on this protocol instead of on :mod:`shipinfer.scheduling`. Three
things follow, and all three are the point rather than a side effect:

* the layering DAG in ADR-001 is unchanged — ``ingest`` still imports only ``core`` and
  ``runtime``, and nothing here knows a scheduler exists;
* ``pipeline`` owns both halves of the frame-to-request mapping;
* the part that has to scale — 50 cameras at 20 fps, 1000 frames a second — can be
  measured against :class:`CountingSink` with no scheduler in the process at all.

**Raising is the contract, not a failure mode.** An RTSP camera cannot be backpressured: it
will send the next frame whether or not anyone is ready. So a sink that cannot accept a
frame says so by raising, and the *caller* decides what to do — and the caller is the camera
actor, which is the only thing in the system that knows which camera is being greedy and can
count the drop against it (ADR-005).

Two errors, both from :mod:`shipinfer.core.errors`, so a sink implementation needs nothing
from the scheduler either:

* :class:`~shipinfer.core.errors.QueueFullError` — no room. The actor drops this one frame,
  counts it against the camera, and carries on.
* :class:`~shipinfer.core.errors.RequestCancelledError` — the consumer is gone. The actor
  finishes cleanly instead of logging one line per frame until the process dies.

Those are exactly the two exceptions
:meth:`shipinfer.scheduling.queues.RequestQueue.put` already raises, and they already live
in ``core``, so **the production adapter needs no translation layer** — it is a
frame-to-``WorkItem`` mapping and nothing else:

.. code-block:: python

    # in shipinfer/pipeline/ — NOT here; that package owns dispatch policy.
    class WorkQueueSink:
        \"\"\"Publishes frames into a RequestQueue as inference requests.\"\"\"

        def __init__(self, queue, settings, cameras):
            self._queue = queue
            self._model = settings.target_model
            self._input = settings.input_name
            self._deadline_ns = settings.frame_deadline_ms * 1_000_000
            # Per-camera policy the protocol deliberately does not carry on the frame:
            # a Frame is data, a priority is policy. Resolved by camera_id, once.
            self._priority = {c.camera_id: c.priority for c in cameras}

        def put(self, frame):
            request = InferenceRequest(
                model_name=self._model,
                inputs={self._input: Tensor.from_numpy(frame.as_batch())},
                context=frame.context,
                priority=self._priority.get(frame.camera_id, Priority.NORMAL),
                deadline_ns=frame.captured_ns + self._deadline_ns if self._deadline_ns else 0,
            )
            self._queue.put(WorkItem(request, ResponseFuture(request)))

The sinks below are **not** that production path, and are the honest two cases either side
of it: one that never refuses (measure the producer) and one that refuses at a known depth
(test the consumer's reaction). Their docstrings say so.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Protocol, runtime_checkable

from shipinfer.core.errors import QueueFullError, RequestCancelledError
from shipinfer.ingest.frame import Frame

__all__ = ["BoundedSink", "CountingSink", "FrameSink"]


@runtime_checkable
class FrameSink(Protocol):
    """The one method a camera actor needs from whatever consumes its frames."""

    def put(self, frame: Frame) -> None:
        """Accept a frame, or raise if it cannot be accepted.

        Raises:
            QueueFullError: there is no room. The caller drops this frame and continues;
                carrying the depth and capacity is what turns "we lost a frame" into a
                number an operator can act on.
            RequestCancelledError: the consumer has shut down and will accept nothing more.
                The caller stops.
        """
        ...


class CountingSink:
    """Counts frames per camera and keeps none of them. Never refuses.

    The measurement harness, not a production sink. Ingest is the part of the system that
    must hold 1000 frames a second, and the only way to measure *that* rather than the
    scheduler behind it is to run against a consumer whose cost is one lock and one integer.

    Optionally keeps the most recent frame per camera, which is enough for a preview or a
    smoke test and is bounded regardless of how long the run lasts.
    """

    def __init__(self, *, keep_last: bool = False) -> None:
        self.keep_last = keep_last
        self._lock = threading.Lock()
        self._per_camera: dict[str, int] = {}
        self._latest: dict[str, Frame] = {}
        self.total = 0

    def put(self, frame: Frame) -> None:
        with self._lock:
            self.total += 1
            self._per_camera[frame.camera_id] = self._per_camera.get(frame.camera_id, 0) + 1
            if self.keep_last:
                self._latest[frame.camera_id] = frame

    def counts(self) -> dict[str, int]:
        """Frames accepted, per camera."""
        with self._lock:
            return dict(self._per_camera)

    def latest(self, camera_id: str) -> Frame | None:
        """The most recent frame from one camera, if ``keep_last`` was set."""
        with self._lock:
            return self._latest.get(camera_id)

    def __len__(self) -> int:
        return self.total

    def __repr__(self) -> str:
        return f"<CountingSink total={self.total} cameras={len(self._per_camera)}>"


class BoundedSink:
    """A bounded in-memory buffer that refuses when full, and keeps what it took.

    For a single process with no scheduler in it — a demo, a smoke test, a fixture — and for
    exercising a producer's reaction to a sink that says no. It is deliberately **not**
    camera-fair: it is one FIFO, so the greediest camera fills it. Fairness is the
    scheduler's job and lives in :mod:`shipinfer.scheduling.queues`; a second, subtly
    different implementation of it here is exactly what ADR-005 warns against.

    Args:
        capacity: frames held before :meth:`put` starts raising.
        name: used in the error message, so a log line says which sink was full.
    """

    def __init__(self, capacity: int = 64, *, name: str = "bounded_sink") -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.name = name
        self._lock = threading.Lock()
        self._frames: deque[Frame] = deque()
        self._closed = False
        self.accepted = 0
        self.refused = 0

    def put(self, frame: Frame) -> None:
        with self._lock:
            if self._closed:
                raise RequestCancelledError(f"sink {self.name!r} is closed")
            if len(self._frames) >= self.capacity:
                self.refused += 1
                raise QueueFullError(self.name, len(self._frames), self.capacity)
            self._frames.append(frame)
            self.accepted += 1

    def drain(self) -> list[Frame]:
        """Take everything currently buffered."""
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
            return frames

    def close(self) -> None:
        """Refuse everything from now on. Idempotent."""
        with self._lock:
            self._closed = True

    @property
    def depth(self) -> int:
        return len(self._frames)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def counts(self) -> dict[str, int]:
        """Buffered frames, per camera — what a fairness assertion reads."""
        with self._lock:
            counts: dict[str, int] = {}
            for frame in self._frames:
                counts[frame.camera_id] = counts.get(frame.camera_id, 0) + 1
            return counts

    def __len__(self) -> int:
        return len(self._frames)

    def __repr__(self) -> str:
        return (
            f"<BoundedSink {self.name} {len(self._frames)}/{self.capacity} "
            f"accepted={self.accepted} refused={self.refused}>"
        )
