"""The metric contract and label plumbing.

Why not ``prometheus_client``? Because these counters are incremented on the dispatch path
at a target of ~15 000 requests/s, and the hot path here is one dict lookup plus an integer
``+=`` under the GIL. It also keeps the pure core free of a dependency whose absence would
make :mod:`shipinfer.scheduling` unimportable.
"""

from __future__ import annotations

import abc
import threading
from collections.abc import Iterator, Mapping

__all__ = ["LATENCY_BUCKETS_US", "Labels", "Metric", "labels_key", "render_labels"]

Labels = tuple[tuple[str, str], ...]

#: 50 us .. 4 s. Covers a pinned-memory copy at one end and a stalled batch at the other.
#: Deliberately not the prometheus default, which is shaped for HTTP handlers in seconds.
LATENCY_BUCKETS_US: tuple[float, ...] = (
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    4_000_000,
)


def labels_key(mapping: Mapping[str, str] | None) -> Labels:
    """Canonicalise a label mapping into a hashable, order-independent key."""
    return tuple(sorted(mapping.items())) if mapping else ()


def render_labels(labels: Labels) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"


class Metric(abc.ABC):
    """One named, labelled measurement family."""

    __slots__ = ("_lock", "help", "name")

    #: Prometheus type name, used by the exporters.
    kind: str = "untyped"

    def __init__(self, name: str, help: str) -> None:
        self.name = name
        self.help = help
        #: Guards *creation* of a label cell only. Updates to an existing cell run lock-free.
        self._lock = threading.Lock()

    @abc.abstractmethod
    def samples(self) -> Iterator[tuple[str, Labels, float]]:
        """``(sample_name, labels, value)`` triples for the exporters."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
