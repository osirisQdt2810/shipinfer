"""A bounded LRU response cache."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping

from shipinfer.core.types import Tensor
from shipinfer.engine.cache.base import ResponseCache, freeze_outputs
from shipinfer.engine.cache.registry import RESPONSE_CACHES

__all__ = ["LruResponseCache"]


@RESPONSE_CACHES.register("lru")
class LruResponseCache(ResponseCache):
    """Least-recently-used, bounded by both entry count and total bytes.

    Two bounds because either alone fails: an entry limit lets a few large detection
    outputs eat gigabytes, and a byte limit alone lets millions of tiny embeddings blow up
    the dict overhead. Both are cheap to maintain on insert.

    The lock is held only for the ``OrderedDict`` operations, never across a copy, so a
    lookup costs a hash and a move-to-end. What :meth:`get` returns is read-only, sealed
    once on the way in by :func:`~shipinfer.engine.cache.base.freeze_outputs`.
    """

    name = "lru"

    def __init__(self, max_entries: int = 4096, max_bytes: int = 256 * 1024 * 1024) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, dict[str, Tensor]] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Tensor] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            # A fresh mapping so a caller adding or dropping a key cannot reshape the
            # stored entry; the tensors inside it are non-writeable and safely shared.
            return dict(entry)

    def put(self, key: str, outputs: Mapping[str, Tensor]) -> None:
        payload = freeze_outputs(outputs)
        nbytes = sum(t.nbytes for t in payload.values())
        if nbytes > self._max_bytes:
            return  # one entry may not evict the entire cache
        with self._lock:
            if key in self._entries:
                self._bytes -= self._sizes[key]
            self._entries[key] = payload
            self._entries.move_to_end(key)
            self._sizes[key] = nbytes
            self._bytes += nbytes
            while self._entries and (
                len(self._entries) > self._max_entries or self._bytes > self._max_bytes
            ):
                victim, _ = self._entries.popitem(last=False)
                self._bytes -= self._sizes.pop(victim, 0)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._sizes.clear()
            self._bytes = 0

    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._entries),
            "bytes": self._bytes,
            "evictions": self._evictions,
        }
