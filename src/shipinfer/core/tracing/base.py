"""Request tracing: the seven named timestamps, and where they go.

:class:`~shipinfer.core.request.Timings` already stamps six points along a request's
journey, and the metrics registry histograms the derived spans. What was missing was a way
to read *one* request's stamps: a histogram says "p99 queue wait is 4 ms", it cannot say
"frame 8213 from cam07 waited 40 ms because it was batched late". That is the question an
operator asks when one camera is behaving badly, and answering it needs the individual
record, not the aggregate.

The names are **Triton's**, verbatim, for the same reason
:mod:`shipinfer.runtime.profiling` uses Triton's phase names: an operator who already reads
Triton traces should not have to learn a second vocabulary, and a trace from this server
should diff against one from Triton without a translation table.

Sampling is part of the contract, not an afterthought. At the design point of 1000 frames a
second this path runs 1000 times a second per model; tracing every request would make the
instrument the bottleneck. Triton solves it with ``--trace-config rate=N`` (trace one
request in N) and so does :attr:`TraceSink.rate`.
"""

from __future__ import annotations

import abc
import itertools
from dataclasses import dataclass
from typing import Any, ClassVar

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import LOG
from shipinfer.core.request import InferenceResponse

__all__ = ["TRACE_EVENTS", "RequestTrace", "TraceSink"]


#: Trace event names, as Triton's trace API emits them, in the order they occur.
#:
#: Six of the seven come from a distinct stamp in :class:`Timings`. ``COMPUTE_END`` and
#: ``REQUEST_END`` share ``completed_ns``, because this server does not stamp the response
#: hand-off separately from the end of scatter — the two are consecutive statements on the
#: same thread. Emitting both from one stamp keeps the vocabulary complete for a client that
#: expects seven events; it does not claim a measurement that was not taken, which is why it
#: is said here rather than left for a reader to infer from two equal numbers.
TRACE_EVENTS = (
    "REQUEST_START",
    "QUEUE_START",
    "COMPUTE_START",
    "COMPUTE_INPUT_END",
    "COMPUTE_OUTPUT_START",
    "COMPUTE_END",
    "REQUEST_END",
)


@dataclass(frozen=True, slots=True)
class RequestTrace:
    """One request's named timestamps, plus the identity needed to find it again.

    ``frozen``/``slots`` because this is built on the completion path: it must not grow a
    ``__dict__`` and it must not be mutated after the sink has been handed it — a sink that
    buffers would otherwise write whatever the record became later.
    """

    request_id: int
    model_name: str
    model_version: int
    camera_id: str
    frame_id: int
    trace_id: str
    device: str
    #: ``(event name, monotonic nanoseconds)`` in :data:`TRACE_EVENTS` order.
    timestamps: tuple[tuple[str, int], ...]

    @classmethod
    def from_response(cls, response: InferenceResponse) -> RequestTrace:
        """Map the six stamps onto Triton's seven event names.

        The mapping is the whole point of this class, so it is written out rather than
        derived:

        ==================== ===========================================================
        ``REQUEST_START``    the request was accepted (``received_ns``)
        ``QUEUE_START``      it entered an instance's queue (``queued_ns``)
        ``COMPUTE_START``    its batch was closed and assembly began (``batched_ns``)
        ``COMPUTE_INPUT_END`` assembly and staging finished (``compute_start_ns``)
        ``COMPUTE_OUTPUT_START`` the backend returned (``compute_end_ns``)
        ``COMPUTE_END``      scatter finished (``completed_ns``)
        ``REQUEST_END``      same stamp; see :data:`TRACE_EVENTS`
        ==================== ===========================================================
        """
        t = response.timings
        return cls(
            request_id=response.request_id,
            model_name=response.model_name,
            model_version=response.model_version,
            camera_id=response.context.camera_id,
            frame_id=response.context.frame_id,
            trace_id=response.context.trace_id,
            device=str(response.executed_on),
            timestamps=(
                ("REQUEST_START", t.received_ns),
                ("QUEUE_START", t.queued_ns),
                ("COMPUTE_START", t.batched_ns),
                ("COMPUTE_INPUT_END", t.compute_start_ns),
                ("COMPUTE_OUTPUT_START", t.compute_end_ns),
                ("COMPUTE_END", t.completed_ns),
                ("REQUEST_END", t.completed_ns),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """The wire shape: Triton's ``timestamps`` array of ``{name, ns}`` objects."""
        return {
            "id": self.request_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "trace_id": self.trace_id,
            "device": self.device,
            "timestamps": [{"name": name, "ns": ns} for name, ns in self.timestamps],
        }

    def span_us(self, start: str, end: str) -> float:
        """Microseconds between two named events.

        Raises:
            ConfigurationError: for an event name that is not in :data:`TRACE_EVENTS`.
                Returning 0.0 for a typo would look exactly like a span that took no time.
        """
        stamps = dict(self.timestamps)
        for name in (start, end):
            if name not in stamps:
                raise ConfigurationError(
                    f"unknown trace event {name!r}; known: {list(TRACE_EVENTS)}"
                )
        return max(0, stamps[end] - stamps[start]) / 1_000.0


class TraceSink(abc.ABC):
    """Where request traces go. One implementation per file, registered by name.

    Two rules, both learned elsewhere in this codebase:

    **A sink must not raise into the serving path.** :meth:`record` is called from the
    worker thread that just finished a batch. An exception there would fail a request that
    already succeeded, so the hook may fail, the wrapper counts it and returns ``False``.

    **The caller must be able to tell.** ``pipeline.sinks.ResultSink`` shipped "never
    raises" while meaning "the caller learns nothing", and the failure counter it documented
    became unreachable. So the bool is real: ``False`` means the trace was not written,
    whether because it was sampled out, the sink is closed, or the write failed.
    """

    name: ClassVar[str] = "abstract"

    def __init__(self, *, rate: int = 1) -> None:
        if rate < 1:
            raise ConfigurationError(f"trace rate must be >= 1, got {rate}")
        self.rate = rate
        self.recorded = 0
        self.sampled_out = 0
        self.failed = 0
        # itertools.count is atomic under the GIL, which is cheaper and less error-prone
        # here than a lock around an integer that several worker threads increment.
        self._counter = itertools.count()
        self._closed = False

    # -- the contract ------------------------------------------------------------------

    def should_record(self) -> bool:
        """Whether to build a trace for the request now completing.

        Checked *before* :class:`RequestTrace` is constructed, so a sampled-out request
        costs one counter increment and no allocation. That ordering is why this is a
        separate method rather than a branch inside :meth:`record`: the dispatch path may
        not allocate per request, and a record built only to be discarded is exactly that.
        """
        if self._closed:
            return False
        if next(self._counter) % self.rate:
            self.sampled_out += 1
            return False
        return True

    def record(self, trace: RequestTrace) -> bool:
        """Write one trace. Never raises. ``True`` if it was written."""
        if self._closed:
            self.failed += 1
            return False
        try:
            self._do_record(trace)
        except Exception:
            self.failed += 1
            LOG.exception(
                "trace sink %s failed on request %d (%s frame %d)",
                self.name,
                trace.request_id,
                trace.camera_id,
                trace.frame_id,
            )
            return False
        self.recorded += 1
        return True

    @abc.abstractmethod
    def _do_record(self, trace: RequestTrace) -> None:
        """Write one trace, or raise."""

    def flush(self) -> None:
        """Push anything buffered. Called at shutdown."""

    def close(self) -> None:
        """Flush and release. Idempotent, and safe on a shutdown path."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        except Exception:
            LOG.exception("trace sink %s failed to flush on close", self.name)
        self._do_close()

    def _do_close(self) -> None:
        """Release resources. Must tolerate a partially constructed sink."""

    # -- introspection -----------------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        return self._closed

    def stats(self) -> dict[str, Any]:
        return {
            "sink": self.name,
            "rate": self.rate,
            "recorded": self.recorded,
            "sampled_out": self.sampled_out,
            "failed": self.failed,
        }

    def __enter__(self) -> TraceSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} rate={self.rate} recorded={self.recorded}>"
