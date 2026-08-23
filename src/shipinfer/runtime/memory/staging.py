"""Pinned host staging buffers.

Torch already has a **caching pinned-memory allocator**, so ``torch.empty(pin_memory=True)``
is not the millisecond-scale ``cudaHostAlloc`` it would be through raw bindings — repeated
same-size requests are served from its cache. This pool therefore does not re-implement
caching; it adds the one thing torch does not provide and this system needs: **buffers with
stable addresses**, which is a hard precondition for CUDA graph capture (ADR-008).

Keyed on ``(name, shape, dtype)`` and owned by exactly one thread. Both halves of that are
load-bearing — see :class:`PinnedStagingPool`.
"""

from __future__ import annotations

import threading
from typing import Any

from shipinfer.core.logging import get_logger
from shipinfer.runtime.platform import require_torch

__all__ = ["PinnedStagingPool"]

_LOG = get_logger("runtime.memory.staging")


class PinnedStagingPool:
    """Reusable pinned host tensors, owned by one thread, keyed on name, shape and dtype.

    Pinned memory matters for one specific reason: ``cudaMemcpyAsync`` from *pageable*
    memory silently degrades to a synchronous copy through a driver bounce buffer. That
    serialises the stream and quietly cancels the whole multi-stream design, with no error
    and no obvious symptom beyond "the GPU is only 60% busy".

    Reuse is what makes it affordable, and reuse is also where it goes wrong. A buffer
    handed out twice while the first copy is still in flight is a data race with no
    diagnostic: the DMA reads whatever the second writer put there. Two rules prevent it,
    and both are structural rather than advisory:

    **One pool per owner.** Get one from :meth:`MemoryPool.staging_for`, never a shared
    instance. A single process runs one worker thread per model instance, so a shared pool
    means two threads staging ``(8, 3, 640, 640) float32`` hand each other the same buffer —
    and GPU 0 ends up inferring on GPU 1's frames, returned under the wrong camera's tag.

    **The caller's name is part of the key.** Shape and dtype alone are not enough even
    inside one thread: a model with two inputs of the same shape would stage the second
    over the first while the first's async H2D was still reading it.

    Within one owner the buffer for a given name is still reused across calls, which is the
    entire point — the address stays stable, which CUDA graph capture requires (ADR-008).
    A caller must therefore issue its copy before staging that same name again.
    """

    def __init__(self, owner: str = "shared", max_entries: int = 64) -> None:
        self.owner = owner
        self._buffers: dict[tuple[str, tuple[int, ...], Any], Any] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
        """The pinned tensor for ``name`` at this shape and dtype.

        Args:
            name: what the buffer is for — an input's name, a stage's name. Anything that
                distinguishes two buffers a single caller holds at the same time.

        The lookup is taken under the lock rather than read first and re-checked. The
        lock-free fast path is tempting and wrong here: it can hand out a buffer that
        eviction is in the middle of dropping, which is a use-after-free with a delay fuse.
        The lock is uncontended by construction — one pool, one thread.
        """
        key = (name, tuple(shape), dtype)
        with self._lock:
            buffer = self._buffers.get(key)
            if buffer is not None:
                self._hits += 1
                return buffer

            torch = require_torch()
            buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
            if len(self._buffers) >= self._max_entries:
                # Bounded rather than growing forever. Shape churn past this point means a
                # model with truly dynamic staging, which cannot be graph-captured anyway.
                self._buffers.pop(next(iter(self._buffers)))
            self._buffers[key] = buffer
            self._misses += 1
            _LOG.debug(
                "allocated pinned staging buffer %s/%s %s %s", self.owner, name, shape, dtype
            )
            return buffer

    def stage(self, name: str, tensor: Any) -> Any:
        """Copy a host tensor into this owner's pinned buffer for ``name``."""
        buffer = self.get(name, tuple(tensor.shape), tensor.dtype)
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
        return f"<PinnedStagingPool {self.owner} entries={len(self._buffers)}>"
