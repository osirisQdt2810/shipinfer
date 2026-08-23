"""Values that go up and down."""

from __future__ import annotations

from collections.abc import Iterator

from shipinfer.core.metrics.base import Labels, Metric, labels_key

__all__ = ["Gauge"]


class Gauge(Metric):
    """Queue depth, ready instances, free device memory."""

    __slots__ = ("_values",)
    kind = "gauge"

    def __init__(self, name: str, help: str) -> None:
        super().__init__(name, help)
        self._values: dict[Labels, float] = {}

    def set(self, value: float, **labels: str) -> None:
        self._values[labels_key(labels)] = value

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = labels_key(labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: str) -> float:
        return self._values.get(labels_key(labels), 0.0)

    def samples(self) -> Iterator[tuple[str, Labels, float]]:
        for key, value in list(self._values.items()):
            yield self.name, key, value
