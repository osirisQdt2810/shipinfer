"""The default allocators: thin adapters over torch's caching allocators."""

from __future__ import annotations

from typing import Any

import numpy as np

from shipinfer.core.errors import DeviceOutOfMemoryError
from shipinfer.core.types import Device, MemoryKind
from shipinfer.runtime.memory.base import Allocator, Buffer
from shipinfer.runtime.memory.registry import ALLOCATORS
from shipinfer.runtime.platform import require_torch

__all__ = ["TorchDeviceAllocator", "TorchPinnedAllocator"]


@ALLOCATORS.register("torch_device", "device")
class TorchDeviceAllocator(Allocator):
    """GPU memory from torch's caching allocator.

    Deliberately a *thin adapter*, not a cache of its own. Torch already keeps freed blocks
    in per-size, per-stream free lists, coalesces and splits them, retries an allocation
    after releasing cached blocks, and cooperates with CUDA graph capture. Layering another
    cache on top would add bookkeeping and take the graph-awareness away.
    """

    name = "torch_device"

    def __init__(self, device: Device) -> None:
        if not device.is_cuda:
            raise ValueError(f"TorchDeviceAllocator needs a cuda device, got {device}")
        super().__init__(device, MemoryKind.DEVICE)
        self._torch = require_torch()
        self._live: dict[int, Any] = {}
        self._allocated = 0

    def allocate(self, nbytes: int) -> Buffer:
        try:
            block = self._torch.empty(
                nbytes, dtype=self._torch.uint8, device=f"cuda:{self._device.index}"
            )
        except RuntimeError as exc:
            raise DeviceOutOfMemoryError(
                f"device alloc of {nbytes} B on {self._device} failed: {exc}"
            ) from exc
        ptr = int(block.data_ptr())
        self._live[ptr] = block
        self._allocated += 1
        return Buffer(
            ptr=ptr,
            nbytes=nbytes,
            kind=MemoryKind.DEVICE,
            device=self._device,
            owner=self,
            keepalive=block,
        )

    def deallocate(self, buffer: Buffer) -> None:
        # Dropping the reference returns the block to torch's cache, not to the driver —
        # which is exactly what should happen.
        self._live.pop(buffer.ptr, None)

    def stats(self) -> dict[str, int]:
        stats = self._torch.cuda.memory_stats(self._device.index)
        return {
            "live_blocks": len(self._live),
            "allocations": self._allocated,
            "torch_allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
            "torch_reserved_bytes": int(stats.get("reserved_bytes.all.current", 0)),
            "torch_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        }

    def close(self) -> None:
        self._live.clear()


@ALLOCATORS.register("torch_pinned", "pinned")
class TorchPinnedAllocator(Allocator):
    """Page-locked host memory from torch's caching host allocator.

    Pinned memory is the pipeline's staging currency, for one specific reason:
    ``cudaMemcpyAsync`` from *pageable* memory silently degrades to a synchronous copy
    through a driver bounce buffer, serialising the stream with no error and no obvious
    symptom beyond "the GPU is only 60% busy".

    Torch caches these allocations, so this is not the millisecond-scale ``cudaHostAlloc``
    the same call would be through raw bindings.
    """

    name = "torch_pinned"

    def __init__(self, device: Device | None = None) -> None:
        super().__init__(device or Device.cpu(), MemoryKind.PINNED)
        self._torch = require_torch()
        self._live: dict[int, Any] = {}
        self._allocated = 0

    def allocate(self, nbytes: int) -> Buffer:
        block = self._torch.empty(nbytes, dtype=self._torch.uint8, pin_memory=True)
        ptr = int(block.data_ptr())
        self._live[ptr] = block
        self._allocated += 1
        return Buffer(
            ptr=ptr,
            nbytes=nbytes,
            kind=MemoryKind.PINNED,
            device=self._device,
            owner=self,
            array=block.numpy(),
            keepalive=block,
        )

    def deallocate(self, buffer: Buffer) -> None:
        self._live.pop(buffer.ptr, None)

    def stats(self) -> dict[str, int]:
        return {"live_blocks": len(self._live), "allocations": self._allocated}

    def close(self) -> None:
        self._live.clear()


@ALLOCATORS.register("host", "numpy")
class HostAllocator(Allocator):
    """Ordinary pageable host memory. The CPU-backend and offline-test path."""

    name = "host"

    def __init__(self, device: Device | None = None) -> None:
        super().__init__(device or Device.cpu(), MemoryKind.HOST)
        self._live = 0
        self._allocated = 0

    def allocate(self, nbytes: int) -> Buffer:
        array = np.empty(nbytes, dtype=np.uint8)
        self._live += nbytes
        self._allocated += 1
        return Buffer(
            ptr=int(array.ctypes.data),
            nbytes=nbytes,
            kind=MemoryKind.HOST,
            device=self._device,
            owner=self,
            array=array,
            keepalive=array,
        )

    def deallocate(self, buffer: Buffer) -> None:
        self._live -= buffer.nbytes

    def stats(self) -> dict[str, int]:
        return {"live_bytes": self._live, "allocations": self._allocated}
