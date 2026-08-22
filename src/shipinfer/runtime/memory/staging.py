"""Pinned host staging buffers.

Torch already has a **caching pinned-memory allocator**, so ``torch.empty(pin_memory=True)``
is not the millisecond-scale ``cudaHostAlloc`` it would be through raw bindings — repeated
same-size requests are served from its cache. This pool therefore does not re-implement
caching; it adds the one thing torch does not provide and this system needs: **buffers with
stable addresses**, which is a hard precondition for CUDA graph capture (ADR-008).

Keyed on ``(shape, dtype)``. A model's staging shapes are fixed by its config, so the pool
converges to a handful of entries and then never allocates again.
"""

from __future__ import annotations

import threading
from typing import Any

from shipinfer.core.logging import get_logger
from shipinfer.runtime.platform import require_torch

__all__ = ["PinnedStagingPool"]

_LOG = get_logger("runtime.memory.staging")


class PinnedStagingPool:
    """Reusable pinned host tensors, keyed on shape and dtype.

    Pinned memory matters for one specific reason: ``cudaMemcpyAsync`` from *pageable*
    memory silently degrades to a synchronous copy through a driver bounce buffer. That
    serialises the stream and quietly cancels the whole multi-stream design, with no error
    and no obvious symptom beyond "the GPU is only 60% busy".
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._buffers: dict[tuple[tuple[int, ...], Any], Any] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, shape: tuple[int, ...], dtype: Any) -> Any:
        """A pinned tensor of exactly this shape and dtype, reused across calls.

        The tensor is **shared, not owned** by the caller: it is valid until the next
        request for the same key. Callers stage into it and issue the copy immediately;
        holding one across a batch boundary is a use-after-reuse bug.
        """
        key = (tuple(shape), dtype)
        buffer = self._buffers.get(key)
        if buffer is not None:
            self._hits += 1
            return buffer

        torch = require_torch()
        with self._lock:
            buffer = self._buffers.get(key)
            if buffer is None:
                buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
                if len(self._buffers) >= self._max_entries:
                    # Evict an arbitrary entry rather than growing without bound. Shape
                    # churn past this point means a model with truly dynamic staging, which
                    # cannot be graph-captured anyway.
                    self._buffers.pop(next(iter(self._buffers)))
                self._buffers[key] = buffer
                self._misses += 1
                _LOG.debug("allocated pinned staging buffer %s %s", shape, dtype)
        return buffer

    def stage(self, tensor: Any) -> Any:
        """Copy a host tensor into pinned memory and return the pinned view."""
        buffer = self.get(tuple(tensor.shape), tensor.dtype)
        buffer.copy_(tensor)
        return buffer

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._buffers),
            "hits": self._hits,
            "misses": self._misses,
            "bytes": sum(b.numel() * b.element_size() for b in self._buffers.values()),
        }

    def __repr__(self) -> str:
        return f"<PinnedStagingPool entries={len(self._buffers)}>"
