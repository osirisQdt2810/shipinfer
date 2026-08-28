"""Reference implementation: a size-bucketed caching allocator."""

from __future__ import annotations

import threading
from collections import defaultdict, deque

from shipinfer.core.logging import LOG
from shipinfer.runtime.memory.base import Allocator, Buffer, align_up
from shipinfer.runtime.memory.registry import ALLOCATORS

__all__ = ["CustomCachingAllocator"]


@ALLOCATORS.register("custom_caching")
class CustomCachingAllocator(Allocator):
    """Reuse freed blocks instead of returning them to the driver.

    **This is the single most valuable page in the ``custom`` tree**, because it is a
    minimal, readable version of what torch's caching allocator does for you. The whole
    idea in thirty lines: round sizes into buckets so requests collide, keep a bounded free
    list per bucket, pop on allocate, push on free.

    What it deliberately does *not* do — and what makes torch's version worth using instead
    — is split and coalesce blocks, track which stream last used a block before handing it
    to another, cooperate with graph capture, or retry an allocation after releasing cached
    memory. Each of those is where the real difficulty lives.

    Bucketing alone is nonetheless enough for this workload's shape (a handful of distinct
    batch x model-I/O sizes), which is why it is a usable fallback and not just a demo.
    """

    name = "custom_caching"

    def __init__(
        self,
        inner: Allocator,
        *,
        granularity: int = 256,
        max_blocks_per_bucket: int = 8,
    ) -> None:
        super().__init__(inner.device, inner.kind)
        self._inner = inner
        self._granularity = max(1, granularity)
        self._max_per_bucket = max(0, max_blocks_per_bucket)
        self._free: defaultdict[int, deque[Buffer]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._returned = 0
        self._cached_bytes = 0

    @property
    def inner(self) -> Allocator:
        return self._inner

    def allocate(self, nbytes: int) -> Buffer:
        size = align_up(nbytes, self._granularity)
        with self._lock:
            bucket = self._free.get(size)
            if bucket:
                buffer = bucket.pop()
                self._hits += 1
                self._cached_bytes -= size
                return _revive(buffer, self)
            self._misses += 1
        return _reown(self._inner.allocate(size), self)

    def deallocate(self, buffer: Buffer) -> None:
        size = buffer.nbytes
        with self._lock:
            bucket = self._free[size]
            if len(bucket) < self._max_per_bucket:
                bucket.append(buffer)
                self._returned += 1
                self._cached_bytes += size
                return
        _reown(buffer, self._inner)
        self._inner.deallocate(buffer)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict[str, int]:
        with self._lock:
            stats = {
                "hits": self._hits,
                "misses": self._misses,
                "returned": self._returned,
                "cached_blocks": sum(len(b) for b in self._free.values()),
                "cached_bytes": self._cached_bytes,
                "buckets": len(self._free),
            }
        stats.update({f"inner_{k}": v for k, v in self._inner.stats().items()})
        return stats

    def close(self) -> None:
        with self._lock:
            buffers = [b for bucket in self._free.values() for b in bucket]
            self._free.clear()
            self._cached_bytes = 0
        for buffer in buffers:
            _reown(buffer, self._inner)
            self._inner.deallocate(buffer)
        LOG.debug("released %d cached block(s) for %s", len(buffers), self._device)
        self._inner.close()

    def __repr__(self) -> str:
        return (
            f"<CustomCachingAllocator {self._kind.value}@{self._device} "
            f"hit_rate={self.hit_rate:.0%} wrapping {self._inner!r}>"
        )


def _revive(buffer: Buffer, owner: Allocator) -> Buffer:
    """Hand a cached block back out; it was marked freed on return."""
    buffer._freed = False
    buffer._owner = owner
    return buffer


def _reown(buffer: Buffer, owner: Allocator) -> Buffer:
    buffer._owner = owner
    return buffer
