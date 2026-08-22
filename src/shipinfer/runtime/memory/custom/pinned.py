"""Reference implementation: pinned host memory via raw ``cudaHostAlloc``."""

from __future__ import annotations

from shipinfer.core.types import Device, MemoryKind
from shipinfer.runtime.memory.base import Allocator, Buffer
from shipinfer.runtime.memory.registry import ALLOCATORS
from shipinfer.runtime.providers import CudaProvider, get_cuda_provider

__all__ = ["CustomPinnedAllocator"]


@ALLOCATORS.register("custom_pinned")
class CustomPinnedAllocator(Allocator):
    """``cudaHostAlloc`` with no caching — what torch's pinned allocator replaces.

    Worth reading precisely because of what it *lacks*. Every ``allocate`` here is a real
    ``cudaHostAlloc``: a synchronising, millisecond-scale call that stalls the device. At
    1000 batches/s that alone would cost more than the inference, which is why the default
    path uses torch's caching host allocator and why this class is only useful wrapped in
    :class:`~shipinfer.runtime.memory.custom.caching.CustomCachingAllocator`.
    """

    name = "custom_pinned"

    def __init__(self, provider: CudaProvider | None = None) -> None:
        super().__init__(Device.cpu(), MemoryKind.PINNED)
        self._provider = provider or get_cuda_provider()
        self._live = 0
        self._allocated = 0

    def allocate(self, nbytes: int) -> Buffer:
        ptr = self._provider.host_alloc_pinned(nbytes)
        self._live += nbytes
        self._allocated += 1
        return Buffer(
            ptr=ptr,
            nbytes=nbytes,
            kind=MemoryKind.PINNED,
            device=self._device,
            owner=self,
            array=self._provider.host_array(ptr, nbytes),
        )

    def deallocate(self, buffer: Buffer) -> None:
        self._provider.host_free_pinned(buffer.ptr)
        self._live -= buffer.nbytes

    def stats(self) -> dict[str, int]:
        return {"live_bytes": self._live, "allocations": self._allocated}
