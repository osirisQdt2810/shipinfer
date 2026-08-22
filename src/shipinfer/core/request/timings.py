"""Per-request nanosecond stamps."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Timings"]


@dataclass(slots=True)
class Timings:
    """Stamps for one request's journey.

    Deliberately mutable and flat: this object is written on the hot path and must not
    allocate a dict per stage. ``slots`` keeps it to six machine words.
    """

    received_ns: int = 0
    queued_ns: int = 0
    batched_ns: int = 0
    compute_start_ns: int = 0
    compute_end_ns: int = 0
    completed_ns: int = 0

    @property
    def queue_us(self) -> float:
        return max(0, self.batched_ns - self.queued_ns) / 1_000.0

    @property
    def compute_us(self) -> float:
        return max(0, self.compute_end_ns - self.compute_start_ns) / 1_000.0

    @property
    def total_us(self) -> float:
        return max(0, self.completed_ns - self.received_ns) / 1_000.0
