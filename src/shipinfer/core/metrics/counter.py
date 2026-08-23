"""Monotonically increasing totals."""

from __future__ import annotations

from collections.abc import Iterator

from shipinfer.core.metrics.base import Labels, Metric, labels_key

__all__ = ["Counter"]


class Counter(Metric):
    """A total that only goes up: requests, drops, batches, spills."""

    __slots__ = ("_values",)
    kind = "counter"

    def __init__(self, name: str, help: str) -> None:
        super().__init__(name, help)
        self._values: dict[Labels, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Add to a cell.

        The read-modify-write is not atomic across threads. That is a deliberate trade: a
        lost increment on a *counter* under extreme contention is cheaper than taking a
        lock on the dispatch path, and the lock is still taken the first time a label set
        is seen, which is when the dict actually mutates its structure.
        """
        key = labels_key(labels)
        current = self._values.get(key)
        if current is None:
            with self._lock:
                self._values[key] = self._values.get(key, 0.0) + amount
        else:
            self._values[key] = current + amount

    def value(self, **labels: str) -> float:
        return self._values.get(labels_key(labels), 0.0)

    def total(self) -> float:
        return sum(self._values.values())

    def samples(self) -> Iterator[tuple[str, Labels, float]]:
        for key, value in list(self._values.items()):
            yield self.name, key, value
