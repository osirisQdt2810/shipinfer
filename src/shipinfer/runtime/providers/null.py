"""No accelerator. Not an error — the ordinary state on a dev laptop."""

from __future__ import annotations

from shipinfer.core.errors import DeviceError
from shipinfer.runtime.providers.base import CudaProvider, RawDeviceProperties, StreamHandle
from shipinfer.runtime.providers.registry import PROVIDERS

__all__ = ["NullCudaProvider"]


@PROVIDERS.register("null", "none", "cpu")
class NullCudaProvider(CudaProvider):
    """Zero devices; every device call fails loudly.

    Distinguishing "zero devices" (fine, run CPU backends) from "a device call failed" (a
    real error) is the whole reason this is an object rather than a ``None`` every call
    site has to remember to check.
    """

    name = "null"
    priority = 900

    @classmethod
    def probe(cls) -> CudaProvider:
        return cls()

    def device_count(self) -> int:
        return 0

    @staticmethod
    def _no_cuda() -> DeviceError:
        return DeviceError("no CUDA runtime available on this host")

    def properties(self, index: int) -> RawDeviceProperties:
        raise self._no_cuda()

    def set_device(self, index: int) -> None:
        raise self._no_cuda()

    def synchronize(self, index: int | None = None) -> None:
        return None

    def memory_info(self, index: int) -> tuple[int, int]:
        raise self._no_cuda()

    def device_malloc(self, nbytes: int) -> int:
        raise self._no_cuda()

    def device_free(self, ptr: int) -> None:
        raise self._no_cuda()

    def host_alloc_pinned(self, nbytes: int) -> int:
        raise self._no_cuda()

    def host_free_pinned(self, ptr: int) -> None:
        raise self._no_cuda()

    def memcpy_h2d(self, dst: int, src: int, nbytes: int, stream: int = 0) -> None:
        raise self._no_cuda()

    def memcpy_d2h(self, dst: int, src: int, nbytes: int, stream: int = 0) -> None:
        raise self._no_cuda()

    def create_stream(self) -> StreamHandle:
        raise self._no_cuda()

    def destroy_stream(self, stream: StreamHandle) -> None:
        return None

    def stream_synchronize(self, stream: StreamHandle) -> None:
        return None

    def describe(self) -> str:
        return "null (no accelerator)"
