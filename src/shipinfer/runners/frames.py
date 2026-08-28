"""The ingest-to-chain adapter: a decoded frame becomes an admitted chain item.

The runner's half of the seam :mod:`shipinfer.ingest.sink` describes. ``ingest`` publishes
into a ``FrameSink`` protocol it owns and knows nothing about queues, chains or scheduling;
this is the implementation the runner supplies, and it is the exact analogue of
:class:`shipinfer.pipeline.sink.QueueFrameSink` one layer up — a frame in, a
:class:`~shipinfer.topology.base.ChainItem` through :meth:`Runner.submit`, and nothing else.

It is a **separate implementation and not an import of that one**, deliberately.
``QueueFrameSink`` maps a frame onto an :class:`~shipinfer.core.request.InferenceRequest`
for one model, because the pipeline's first stage *is* a model call; a runner maps it onto a
chain item for a topology whose first element may be anything, and the runner's own
``submit`` is what builds the carrier request behind it. Two mappings, two consumers; sharing
one of them would mean one of the two packages learning the other's dispatch policy.

**Both refusals travel untouched, and that is the whole contract** (ADR-005). ``submit``
raises exactly the two errors the ``FrameSink`` protocol names —
:class:`~shipinfer.core.errors.QueueFullError` when this camera's lane is full and
:class:`~shipinfer.core.errors.RequestCancelledError` when the runner has stopped — and both
already live in ``core.errors``, so there is no translation layer here at all. They reach the
camera actor, which is the only component in the system that knows *which* camera produced
the frame and can therefore drop that camera's own newest frame and charge it to the right
camera. Wrapping either one is exactly the bug this project exists to fix.

**One dropped frame is counted twice, on purpose, and the pair is the point.** A
``QueueFullError`` here increments ``shipinfer_runner_items_dropped_total{camera}`` on the way
past (``inprocess.py::_do_submit``) *and*
``shipinfer_ingest_frames_dropped_total{camera,reason=sink_full}`` in the actor
(``ingest/camera/actor.py::_publish``). They are not redundant: the first is the admission
door's own ledger, read against ``items_accepted`` for the shard, and the second is the
camera's, read against ``frames_read`` to answer "what fraction of this camera did we lose".
An operator reading either alone gets a true answer to a different question.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from shipinfer.core.request import RequestContext, ResponseFuture
from shipinfer.core.types import Tensor
from shipinfer.topology import Caps, ChainItem

__all__ = ["ChainFrameSink", "TaggedFrame"]


@runtime_checkable
class TaggedFrame(Protocol):
    """The four things this adapter needs from a decoded frame.

    Structural rather than an import of :class:`shipinfer.ingest.frame.Frame`, and for a
    harder reason than the identical protocol in ``pipeline/sink.py`` has: importing
    ``shipinfer.ingest`` at module scope would pull a decode runtime — and through
    ``sources/gstreamer.py``, ``shipinfer.runtime``, and torch behind that on a host where a
    device source is importable — behind ``import shipinfer.runners``, which
    ``tests/test_architecture.py`` refuses. A protocol satisfies the dependency without asking
    for the import. ``Frame`` satisfies it exactly as written.

    Four members is also the honest measure of how narrow the seam is.
    """

    @property
    def camera_id(self) -> str:
        """Which camera produced this frame."""

    @property
    def context(self) -> RequestContext:
        """The ``(camera_id, frame_id, timestamps)`` tag, as every item carries it."""

    @property
    def captured_ns(self) -> int:
        """Monotonic nanoseconds at the moment of decode."""

    def as_batch(self) -> np.ndarray:
        """The image as a batch-major ``(1, H, W, C)`` array."""


class ChainFrameSink:
    """Publishes decoded frames into a runner's chain.

    Args:
        submit: what admits one item — a runner's submission entry point. A callable rather
            than the runner itself, so this class cannot reach for anything else on it and a
            test can drive it with a lambda.
        caps: the chain's **head cap**: what the payload put on the item is, in the
            vocabulary the loader negotiated for the edges out of the decode element. Passed
            in rather than derived here, because it is a property of the topology and the
            runner is the one holding it (``inprocess.py::_head``).
    """

    __slots__ = ("_caps", "_fps_of", "_submit")

    def __init__(
        self,
        submit: Callable[[ChainItem], ResponseFuture],
        caps: Caps,
        fps_of: Callable[[str], float] | None = None,
    ) -> None:
        self._submit = submit
        self._caps = caps
        #: The camera's negotiated rate, per frame — the one per-camera fact an event needs
        #: that no element can discover (V148's first real run shipped img_fps=0 everywhere).
        self._fps_of = fps_of

    @property
    def caps(self) -> Caps:
        """The head cap every item leaving this sink carries."""
        return self._caps

    def put(self, frame: TaggedFrame) -> None:
        """Map one frame onto one admitted chain item.

        The payload is a :class:`~shipinfer.core.types.Tensor` and not the raw array, which
        is not decoration: a ``pool`` element refuses any other payload by type
        (``topology/elements/pool.py``), so a frame that arrived as a bare ``ndarray`` would
        walk the whole chain and fail at the first model with a
        :class:`~shipinfer.core.errors.ValidationError`. ``Tensor.from_numpy`` over
        ``Frame.as_batch`` is a view of a contiguous array, so this costs no copy — the same
        wrap ``pipeline/sink.py`` performs, for the same reason.

        **The future is discarded, and that is a decision rather than an oversight.** A
        camera actor has nothing to do with it: it cannot wait on one without becoming the
        chain's pacer (an RTSP camera sends the next frame regardless), and keeping twenty a
        second per camera would be an unbounded set of objects with no reaper. The two
        outcomes an actor *can* act on arrive as exceptions from this call, which is precisely
        why the ``FrameSink`` contract is written in exceptions; every other outcome of the
        walk is already counted per camera by the runner's metrics and delivered to the item's
        future by the walk itself, so nothing vanishes untyped (ADR-005).

        Raises:
            QueueFullError: this camera's lane is full. Untouched, so the actor drops this
                frame and charges it to this camera.
            RequestCancelledError: the runner has stopped and will accept nothing more. The
                actor finishes cleanly instead of logging one line per frame.
        """
        fps = self._fps_of(frame.camera_id) if self._fps_of is not None else 0.0
        self._submit(
            ChainItem(
                context=frame.context,
                caps=self._caps,
                payload=Tensor.from_numpy(frame.as_batch()),
                meta={"fps": fps} if fps else {},
            )
        )

    def __repr__(self) -> str:
        return f"<ChainFrameSink caps={self._caps}>"
