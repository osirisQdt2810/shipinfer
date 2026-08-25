"""Read a :class:`~shipinfer.core.metrics.histogram.Histogram`, and difference it in time.

WHY THIS EXISTS
---------------
The algo tier profiles the **steady window** — the warm-up boundary to the end of the run. A
run's first seconds are engines deserialising and first calls paying JIT: exactly the tail a
service-time profile has to exclude, and the same window the system tier's growth fit already
uses (``ShipInferResult.steady_*``). A histogram is cumulative, so the steady window is the
histogram at the end *minus* the histogram at the boundary — and that subtraction needs the
per-bucket counts, not only ``(count, sum)``.

``Histogram`` exports exactly that through ``samples()``, Prometheus-style: cumulative bucket
counts under an ``le`` label, then ``_sum`` and ``_count``. This module reads that export into a
plain record. It does not reach into the histogram's cells and it does not time anything.

WHAT A QUANTILE MEANS HERE
--------------------------
:meth:`HistogramCell.quantile` mirrors ``Histogram.quantile``: the **upper edge of the bucket**
the quantile falls in. With 2-2.5x bucket steps that is distribution colour, not a cost. The
cost model uses :attr:`HistogramCell.mean`, which is exact — ``total / count``, both carried at
full precision. The first version of the algo tier built its per-frame costs on p50 and rendered
a 2.3x difference between two stages in one bucket as a tie; review caught it, this is the fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shipinfer.core.metrics.base import labels_key

__all__ = ["HistogramCell", "read_cell"]


@dataclass(frozen=True, slots=True)
class HistogramCell:
    """One label set's histogram as plain numbers, at one instant or over one window."""

    #: Bucket upper edges, ascending. The implicit last bucket is ``+Inf``.
    edges: tuple[float, ...]
    #: Observations **per bucket** (not cumulative): ``len(edges) + 1`` entries.
    counts: tuple[int, ...]
    #: Exact number of observations and their exact sum.
    count: int
    total: float

    def __post_init__(self) -> None:
        if len(self.counts) != len(self.edges) + 1:
            raise ValueError(
                f"a cell over {len(self.edges)} edges needs {len(self.edges) + 1} bucket counts, "
                f"got {len(self.counts)}"
            )

    @property
    def mean(self) -> float:
        """The exact per-observation mean — the number a cost model is built on."""
        return 0.0 if self.count <= 0 else self.total / self.count

    def quantile(self, q: float) -> float:
        """Bucket-resolution quantile: the upper edge of the bucket ``q`` falls in."""
        if self.count <= 0 or not self.edges:
            return 0.0
        target = q * self.count
        seen = 0
        for i, n in enumerate(self.counts):
            seen += n
            if seen >= target:
                return self.edges[i] if i < len(self.edges) else self.edges[-1]
        return self.edges[-1]

    def minus(self, earlier: HistogramCell | None) -> HistogramCell:
        """This cell over the window since ``earlier`` was read. ``None`` means "since the start"."""
        if earlier is None:
            return self
        if earlier.edges != self.edges:
            raise ValueError("cannot difference two histograms with different buckets")
        counts = tuple(
            now - then for now, then in zip(self.counts, earlier.counts, strict=True)
        )
        if self.count < earlier.count or any(n < 0 for n in counts):
            raise ValueError(
                "the earlier snapshot holds more observations than the later one; a histogram "
                "only grows, so these two were not read from the same cell"
            )
        return HistogramCell(
            self.edges, counts, self.count - earlier.count, self.total - earlier.total
        )


def read_cell(histogram: Any, **labels: str) -> HistogramCell:
    """The cell for exactly ``labels``, read through ``samples()``.

    A label set the histogram has never observed reads as an empty cell over the histogram's
    buckets, so a stage that never ran is a zero, not an exception.
    """
    key = tuple(labels_key(labels))
    edges = tuple(float(edge) for edge in histogram.buckets)
    cumulative: dict[str, int] = {}
    count = 0
    total = 0.0
    for name, row_labels, value in histogram.samples():
        row = tuple(row_labels)
        if name.endswith("_bucket"):
            if row[:-1] == key and row[-1][0] == "le":
                cumulative[row[-1][1]] = int(value)
        elif row == key:
            if name.endswith("_sum"):
                total = float(value)
            elif name.endswith("_count"):
                count = int(value)
    if not cumulative:
        return HistogramCell(edges, (0,) * (len(edges) + 1), 0, 0.0)
    running = [cumulative.get(str(edge), 0) for edge in histogram.buckets]
    running.append(cumulative.get("+Inf", count))
    counts = tuple(now - then for then, now in zip([0, *running[:-1]], running, strict=True))
    return HistogramCell(edges, counts, count, total)
