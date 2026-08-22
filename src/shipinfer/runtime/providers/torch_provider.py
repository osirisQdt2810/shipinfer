"""The low-level provider contract, satisfied by torch."""

from __future__ import annotations

import threading
from typing import Any

from shipinfer.core.errors import DeviceError, DeviceOutOfMemoryError
from shipinfer.runtime.providers.base import CudaProvider, RawDeviceProperties, StreamHandle
from shipinfer.runtime.providers.registry import PROVIDERS

__all__ = ["TorchCudaProvider"]


@PROVIDERS.register("torch", "pytorch")
class TorchCudaProvider(CudaProvider):
    """Driver primitives backed by ``torch.cuda``.

    Present so the ``custom`` allocator and graph cache can run on a host that has torch
    but not ``cuda-python`` — and, usefully, so a parity test can drive the *same*
    hand-written allocator through two different substrates and assert it behaves
    identically.

    It also covers ROCm at no extra cost, because on ROCm ``torch.cuda`` is the HIP API.
    """

    name = "torch"
    priority = 50

    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module
        self._device_blocks: dict[int, Any] = {}
        self._pinned_blocks: dict[int, Any] = {}
        self._streams: dict[int, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def probe(cls) -> CudaProvider | None:
        try:
            import torch
        except ImportError:
            return None
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                return None
        except Exception:
            return None
        return cls(torch)

    def device_count(self) -> int:
        return int(self._torch.cuda.device_count())

    def properties(self, index: int) -> RawDeviceProperties:
        props = self._torch.cuda.get_device_properties(index)
        return RawDeviceProperties(
            index=index,
            name=props.name,
            total_memory=int(props.total_memory),
            compute_capability=(int(props.major), int(props.minor)),
            multi_processor_count=int(getattr(props, "multi_processor_count", 0)),
        )

    def set_device(self, index: int) -> None:
        self._torch.cuda.set_device(index)

    def synchronize(self, index: int | None = None) -> None:
        self._torch.cuda.synchronize(index)

    def memory_info(self, index: int) -> tuple[int, int]:
        free, total = self._torch.cuda.mem_get_info(index)
        return int(free), int(total)

    def device_malloc(self, nbytes: int) -> int:
        try:
            block = self._torch.empty(nbytes, dtype=self._torch.uint8, device="cuda")
        except RuntimeError as exc:
            raise DeviceOutOfMemoryError(
                f"torch device alloc of {nbytes} B failed: {exc}"
            ) from exc
        ptr = int(block.data_ptr())
        with self._lock:
            self._device_blocks[ptr] = block  # hold the tensor: GC must not free it
        return ptr

    def device_free(self, ptr: int) -> None:
        with self._lock:
            self._device_blocks.pop(ptr, None)

    def host_alloc_pinned(self, nbytes: int) -> int:
        block = self._torch.empty(nbytes, dtype=self._torch.uint8, pin_memory=True)
        ptr = int(block.data_ptr())
        with self._lock:
            self._pinned_blocks[ptr] = block
        return ptr

    def host_free_pinned(self, ptr: int) -> None:
        with self._lock:
            self._pinned_blocks.pop(ptr, None)

    def _block(self, ptr: int, nbytes: int, *, on_device: bool) -> Any:
        registry = self._device_blocks if on_device else self._pinned_blocks
        block = registry.get(ptr)
        if block is None:
            raise DeviceError(f"pointer 0x{ptr:x} was not allocated by the torch provider")
        return block[:nbytes]

    def memcpy_h2d(self, dst: int, src: int, nbytes: int, stream: int = 0) -> None:
        self._block(dst, nbytes, on_device=True).copy_(
            self._block(src, nbytes, on_device=False), non_blocking=bool(stream)
        )

    def memcpy_d2h(self, dst: int, src: int, nbytes: int, stream: int = 0) -> None:
        self._block(dst, nbytes, on_device=False).copy_(
            self._block(src, nbytes, on_device=True), non_blocking=bool(stream)
        )

    def create_stream(self) -> StreamHandle:
        stream = self._torch.cuda.Stream()
        handle = int(stream.cuda_stream)
        with self._lock:
            self._streams[handle] = stream
        return handle

    def destroy_stream(self, stream: StreamHandle) -> None:
        with self._lock:
            self._streams.pop(stream, None)

    def stream_synchronize(self, stream: StreamHandle) -> None:
        obj = self._streams.get(stream)
        if obj is not None:
            obj.synchronize()
