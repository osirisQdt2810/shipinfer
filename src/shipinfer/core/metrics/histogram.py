"""Bucketed distributions."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from shipinfer.core.metrics.base import LATENCY_BUCKETS_US, Labels, Metric, labels_key

__all__ = ["Histogram"]


@dataclass(slots=True)
class _Cell:
    counts: list[int]
    total: float = 0.0
    count: int = 0


class Histogram(Metric):
    """Latency and batch-size distributions.

    ``observe`` is O(log b) via bisect over a sorted bucket list — cheap enough to call on
    every request, which matters because a mean latency hides exactly the tail this system
    is tuned for.
    """

    __slots__ = ("_cells", "buckets")
    kind = "histogram"

    def __init__(
        self,
        name: str,
        help: str,
        buckets: Sequence[float] = LATENCY_BUCKETS_US,
    ) -> None:
        super().__init__(name, help)
        self.buckets = tuple(buckets)
        self._cells: dict[Labels, _Cell] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = labels_key(labels)
        cell = self._cells.get(key)
        if cell is None:
            with self._lock:
                cell = self._cells.setdefault(key, _Cell(counts=[0] * (len(self.buckets) + 1)))
        cell.counts[bisect_left(self.buckets, value)] += 1
        cell.total += value
        cell.count += 1

    def snapshot(self, **labels: str) -> tuple[int, float]:
        """``(count, sum)`` — enough for a mean without a full export."""
        cell = self._cells.get(labels_key(labels))
        return (cell.count, cell.total) if cell else (0, 0.0)

    def quantile(self, q: float, **labels: str) -> float:
        """Bucket-resolution quantile.

        Approximate by construction: it returns the upper edge of the bucket the quantile
        falls in. Good enough to alert on p99 without shipping a t-digest, and honest
        about it rather than interpolating a precision it does not have.
        """
        cell = self._cells.get(labels_key(labels))
        if cell is None or cell.count == 0:
            return 0.0
        target = q * cell.count
        seen = 0
        for i, count in enumerate(cell.counts):
            seen += count
            if seen >= target:
                return self.buckets[i] if i < len(self.buckets) else self.buckets[-1]
        return self.buckets[-1]

    def samples(self) -> Iterator[tuple[str, Labels, float]]:
        for key, cell in list(self._cells.items()):
            cumulative = 0
            for i, edge in enumerate(self.buckets):
                cumulative += cell.counts[i]
                yield f"{self.name}_bucket", (*key, ("le", str(edge))), cumulative
            yield f"{self.name}_bucket", (*key, ("le", "+Inf")), cell.count
            yield f"{self.name}_sum", key, cell.total
            yield f"{self.name}_count", key, cell.count
