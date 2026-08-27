"""The request-queue contract.

A queue owns three decisions: what order requests come out in, what happens when it is
full, and how long a consumer waits for a batch to fill. Making it an interface means
those three can be varied independently of the instance that drains it — and it means the
fair-queueing behaviour that fixes this system's inherited starvation bug is a *choice*
that can be A/B'd against plain FIFO rather than a hardcoded assumption.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from shipinfer.core.settings import OverflowPolicy
from shipinfer.scheduling.work import WorkItem

__all__ = ["BatchWindow", "QueueStats", "RequestQueue"]


@dataclass(frozen=True, slots=True)
class BatchWindow:
    """The batching contract a consumer asks the queue to honour.

    The trade this encodes is the whole point of dynamic batching: waiting
    ``max_delay_us`` costs every request that latency, but lets the GPU run one large
    kernel launch instead of many small ones. At 50 cameras x 20 fps a 5 ms window fills a
    batch of 32 without being visible end-to-end.
    """

    max_batch_size: int
    max_delay_us: int = 0
    #: Sizes worth stopping early at. A TensorRT engine profiled for {8, 16, 32} runs an
    #: unprofiled batch of 31 on a fallback path that can be much slower than padding to 32.
    preferred_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if any(s < 1 or s > self.max_batch_size for s in self.preferred_sizes):
            raise ValueError("preferred_sizes must be within [1, max_batch_size]")


@dataclass(frozen=True, slots=True)
class QueueStats:
    """A snapshot an operator can act on.

    The four ``*_by_camera`` maps are the per-camera half of ADR-005. The totals say a
    queue lost work; only the breakdown says *whose* work, which is the difference between
    "the server dropped 4 000 frames" and "camera 17 flooded and paid for it". They are
    keyed by :attr:`~shipinfer.scheduling.work.WorkItem.fairness_key`, so a caller with no
    camera lands in the honest ``"-"`` bucket rather than inventing a lane of its own.

    Every map is defaulted to empty: a queue that cannot attribute an outcome — a third
    implementation, a compiled adapter — constructs without them and reports nothing rather
    than reporting a zero it did not measure.

    ``close()`` deliberately feeds none of these. Shutdown loss is not a per-camera fault
    and the runner's ``items_queue_closed`` already owns that outcome; charging it here
    would make a clean stop look like a flood.
    """

    depth: int
    capacity: int
    accepted: int
    rejected: int
    evicted: int
    expired: int
    depth_by_camera: Mapping[str, int] = field(default_factory=dict)
    rejected_by_camera: Mapping[str, int] = field(default_factory=dict)
    evicted_by_camera: Mapping[str, int] = field(default_factory=dict)
    expired_by_camera: Mapping[str, int] = field(default_factory=dict)

    @property
    def utilisation(self) -> float:
        return self.depth / self.capacity if self.capacity else 0.0

    def as_dict(self) -> dict[str, Any]:
        """The wire shape, with the maps copied.

        Copied, not handed out: this dict is what ``/v2/statistics`` serialises and what a
        health handler nests into its own document, and a caller that trims or re-keys it
        must not be editing a live queue's counters through the back door.
        """
        return {
            "depth": self.depth,
            "capacity": self.capacity,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "evicted": self.evicted,
            "expired": self.expired,
            "depth_by_camera": dict(self.depth_by_camera),
            "rejected_by_camera": dict(self.rejected_by_camera),
            "evicted_by_camera": dict(self.evicted_by_camera),
            "expired_by_camera": dict(self.expired_by_camera),
        }


class RequestQueue(abc.ABC):
    """A bounded queue with a dynamic-batching consumer.

    One instance owns one of these. Producers call :meth:`put`; the instance's worker
    thread calls :meth:`get_batch`.
    """

    name: ClassVar[str] = "abstract"

    def __init__(
        self,
        name: str,
        capacity: int,
        *,
        overflow: OverflowPolicy = OverflowPolicy.REJECT,
        block_timeout_ms: int = 50,
        drop_expired: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("queue capacity must be >= 1")
        self.name = name
        self.capacity = capacity
        self.overflow = overflow
        self.block_timeout_s = block_timeout_ms / 1000.0
        self.drop_expired = drop_expired

    # -- producer ----------------------------------------------------------------------

    @abc.abstractmethod
    def put(self, item: WorkItem) -> None:
        """Enqueue one request, applying the configured overflow policy.

        Raises:
            QueueFullError: under ``REJECT``, or when ``BLOCK`` times out.
            RequestCancelledError: if the queue is closing.
        """

    # -- consumer ----------------------------------------------------------------------

    @abc.abstractmethod
    def get_batch(self, window: BatchWindow, *, poll_s: float = 0.05) -> list[WorkItem]:
        """Block until a batch is ready, then return it.

        Returns an empty list only when the queue has been closed, which is how a worker
        thread learns to exit without a separate sentinel.
        """

    # -- lifecycle ---------------------------------------------------------------------

    @abc.abstractmethod
    def close(self, error: BaseException | None = None) -> Sequence[WorkItem]:
        """Close and fail everything still queued.

        Returns the drained items so the caller can report exactly how much work was lost.
        A shutdown that silently discards 400 requests is not an orderly shutdown.
        """

    # -- introspection -----------------------------------------------------------------

    @property
    @abc.abstractmethod
    def depth(self) -> int:
        """Requests currently waiting.

        Read without a lock by the placement policies: a slightly stale depth changes
        which of two near-equal GPUs wins, nothing more, and a lock here would be taken
        thousands of times a second.
        """

    @property
    @abc.abstractmethod
    def is_closed(self) -> bool: ...

    @abc.abstractmethod
    def stats(self) -> QueueStats: ...

    def __len__(self) -> int:
        return self.depth

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} {self.depth}/{self.capacity}>"
