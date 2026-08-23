"""Where a finished frame goes — one method, and the two rules around it.

**A sink must not raise into the pipeline.** It is called from the worker thread that just
finished a frame and from the sweeper thread that just timed one out; an exception there
would take out a worker or, worse, the sweeper, and the sweeper is what guarantees every
frame is eventually emitted. So :meth:`ResultSink.emit` is a template method: the subclass
hook may fail, and the wrapper counts the failure and returns. The count is the signal —
``pipeline_sink_failures_total`` is the metric an operator alerts on.

**A sink must not block for long.** Emission happens on the thread that would otherwise be
running the next frame's graph, so a network sink buffers and flushes asynchronously rather
than waiting for an acknowledgement per frame. ``confluent_kafka`` already does exactly
that, which is why the Kafka sink is thin.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

from shipinfer.core.logging import get_logger
from shipinfer.pipeline.schema import PerceptionEvent

__all__ = ["ResultSink"]

_LOG = get_logger("pipeline.sinks")


class ResultSink(abc.ABC):
    """Publishes perception events. One implementation per file, registered by name."""

    name: ClassVar[str] = "abstract"

    def __init__(self) -> None:
        self.emitted = 0
        self.failed = 0
        self._closed = False

    # -- the contract ------------------------------------------------------------------

    def emit(self, event: PerceptionEvent) -> None:
        """Publish one event. Never raises.

        Returning quietly on failure is not swallowing: the failure is counted and logged
        with the frame's tag, and a sink that is failing is visible in one counter. What the
        pipeline must not do is lose a *worker thread* because a broker went away.
        """
        if self._closed:
            self.failed += 1
            _LOG.debug("sink %s is closed; dropping %s", self.name, event.key)
            return
        try:
            self._do_emit(event)
        except Exception:
            self.failed += 1
            _LOG.exception(
                "sink %s failed to publish camera %s frame %d",
                self.name,
                event.camera_id,
                event.frame_id,
            )
            return
        self.emitted += 1

    @abc.abstractmethod
    def _do_emit(self, event: PerceptionEvent) -> None:
        """Publish one event, or raise."""

    def flush(self) -> None:
        """Push anything buffered. Called on a timer and at shutdown."""

    def close(self) -> None:
        """Flush and release. Idempotent, and safe to call from a shutdown path."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        except Exception:
            _LOG.exception("sink %s failed to flush on close", self.name)
        self._do_close()

    def _do_close(self) -> None:
        """Release resources. Must tolerate a partially constructed sink."""

    # -- introspection -----------------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        return self._closed

    def stats(self) -> dict[str, Any]:
        return {"sink": self.name, "emitted": self.emitted, "failed": self.failed}

    def __enter__(self) -> ResultSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.emitted

    def __repr__(self) -> str:
        return f"<{type(self).__name__} emitted={self.emitted} failed={self.failed}>"
