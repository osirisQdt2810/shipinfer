"""A sink that counts and publishes nothing — the default, and the measurement harness."""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

from shipinfer.pipeline.schema import PerceptionEvent
from shipinfer.pipeline.sinks.base import ResultSink
from shipinfer.pipeline.sinks.registry import RESULT_SINKS

__all__ = ["NullResultSink"]


@RESULT_SINKS.register("null", "none", "count")
class NullResultSink(ResultSink):
    """Counts events, keeps at most ``keep_last`` of them, publishes nowhere.

    The **default** because a pipeline should not need a broker to start: a deployment that
    has not chosen where results go is better off producing none than failing to boot, and
    the counter still says the pipeline is working. It is also how a bench measures the
    pipeline rather than the sink — the throughput number for a DAG is worthless if it
    includes a JSON encoder.

    ``keep_last`` is bounded by construction, so a long run cannot turn a smoke test into an
    out-of-memory kill.
    """

    name: ClassVar[str] = "null"

    def __init__(self, *, keep_last: int = 0) -> None:
        super().__init__()
        if keep_last < 0:
            raise ValueError("keep_last must be >= 0")
        self.keep_last = keep_last
        self._events: deque[PerceptionEvent] = deque(maxlen=keep_last or 1)

    def _do_emit(self, event: PerceptionEvent) -> None:
        if self.keep_last:
            self._events.append(event)

    def events(self) -> tuple[PerceptionEvent, ...]:
        """The most recent events, oldest first. Empty unless ``keep_last`` was set."""
        return tuple(self._events) if self.keep_last else ()

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "kept": len(self._events) if self.keep_last else 0}
