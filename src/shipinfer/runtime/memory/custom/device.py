"""Reference implementation: device memory via raw ``cudaMalloc``."""

from __future__ import annotations

from shipinfer.core.types import Device, MemoryKind
from shipinfer.runtime.memory.base import Allocator, Buffer
from shipinfer.runtime.memory.registry import ALLOCATORS
from shipinfer.runtime.providers import CudaProvider, get_cuda_provider

__all__ = ["CustomDeviceAllocator"]


@ALLOCATORS.register("custom_device")
class CustomDeviceAllocator(Allocator):
    """``cudaMalloc``/``cudaFree`` on one device — what torch's caching allocator replaces.

    Note the ``set_device`` before every allocation: ``cudaMalloc`` allocates on whatever
    device is *current*, so a thread that drifted produces memory on the wrong GPU and an
    error thousands of lines from the cause. Torch handles this by making the allocator
    per-device; here it has to be done by hand, every time, which is the sort of detail
    that makes hand-rolling this layer a bad trade.
    """

    name = "custom_device"

    def __init__(self, device: Device, provider: CudaProvider | None = None) -> None:
        if not device.is_cuda:
            raise ValueError(f"CustomDeviceAllocator needs a cuda device, got {device}")
        super().__init__(device, MemoryKind.DEVICE)
        self._provider = provider or get_cuda_provider()
        self._live = 0
        self._allocated = 0

    def allocate(self, nbytes: int) -> Buffer:
        self._provider.set_device(self._device.index)
        ptr = self._provider.device_malloc(nbytes)
        self._live += nbytes
        self._allocated += 1
        return Buffer(
            ptr=ptr, nbytes=nbytes, kind=MemoryKind.DEVICE, device=self._device, owner=self
        )

    def deallocate(self, buffer: Buffer) -> None:
        self._provider.device_free(buffer.ptr)
        self._live -= buffer.nbytes

    def stats(self) -> dict[str, int]:
        return {"live_bytes": self._live, "allocations": self._allocated}
