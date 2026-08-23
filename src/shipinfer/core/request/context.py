"""The provenance tag that must survive every reorder in the system."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RequestContext"]


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Who produced this work, and when.

    The one invariant worth stating loudly: this tag rides along untouched from ingest to
    the very last response. Every stage that batches, reorders or spills work to another
    GPU is allowed to do so *precisely because* reassembly downstream keys on this tag
    rather than on arrival order (ADR-002).
    """

    camera_id: str = ""
    frame_id: int = -1
    #: Monotonic nanoseconds captured at ingest — for end-to-end latency, never display.
    captured_ns: int = 0
    #: Wall-clock nanoseconds, for anything a human or Kafka consumer will read.
    captured_unix_ns: int = 0
    #: Free-form correlation id for tracing across process boundaries (Kafka, HTTP).
    trace_id: str = ""

    @property
    def key(self) -> tuple[str, int]:
        """The reassembly key. :mod:`shipinfer.pipeline.assembler` groups on this."""
        return (self.camera_id, self.frame_id)
