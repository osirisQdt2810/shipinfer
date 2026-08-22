"""Owns the metric objects for one server."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import TypeVar

from shipinfer.core.metrics.base import LATENCY_BUCKETS_US, Metric
from shipinfer.core.metrics.counter import Counter
from shipinfer.core.metrics.gauge import Gauge
from shipinfer.core.metrics.histogram import Histogram

__all__ = ["MetricsRegistry"]

M = TypeVar("M", bound=Metric)


class MetricsRegistry:
    """A namespace of metrics.

    Not a module-level global: a test creates its own and asserts on it, and two servers
    in one process do not silently share counters.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str) -> Counter:
        return self._get_or_create(name, lambda: Counter(name, help), Counter)

    def gauge(self, name: str, help: str) -> Gauge:
        return self._get_or_create(name, lambda: Gauge(name, help), Gauge)

    def histogram(
        self,
        name: str,
        help: str,
        buckets: Sequence[float] = LATENCY_BUCKETS_US,
    ) -> Histogram:
        return self._get_or_create(name, lambda: Histogram(name, help, buckets), Histogram)

    def _get_or_create(self, name: str, factory: Callable[[], M], expected: type[M]) -> M:
        existing = self._metrics.get(name)
        if existing is None:
            with self._lock:
                existing = self._metrics.get(name)
                if existing is None:
                    existing = factory()
                    self._metrics[name] = existing
        if not isinstance(existing, expected):
            raise TypeError(f"metric {name!r} already registered as {type(existing).__name__}")
        return existing

    def collect(self) -> list[Metric]:
        return sorted(self._metrics.values(), key=lambda m: m.name)

    def __len__(self) -> int:
        return len(self._metrics)

    def __contains__(self, name: object) -> bool:
        return name in self._metrics
