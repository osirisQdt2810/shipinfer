"""Low-level driver access — the contract behind the ``custom`` implementations.

**Read this package to understand what torch is doing for you.** Nothing on the default
path uses it: :mod:`shipinfer.runtime.platform` talks to ``torch.cuda``, which is faster,
better tested and covers ROCm for free (ADR-003).

What this exists for is twofold. It is the substrate for the ``custom`` allocator and
``custom`` graph cache — deliberately readable re-implementations of machinery torch hides
— and it is the escape hatch for the rare case that needs a driver call torch does not
expose. Both are opt-in through a registry; neither is the recommended path.
"""

from __future__ import annotations

import abc
import ctypes
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from shipinfer.core.errors import DeviceError

__all__ = [
    "CudaProvider",
    "DevicePtr",
    "HostPtr",
    "RawDeviceProperties",
    "StreamHandle",
    "decode",
]

StreamHandle = int
DevicePtr = int
HostPtr = int


@dataclass(frozen=True, slots=True)
class RawDeviceProperties:
    """Device facts as the driver reports them."""

    index: int
    name: str
    total_memory: int
    compute_capability: tuple[int, int]
    multi_processor_count: int = 0

    def __str__(self) -> str:
        major, minor = self.compute_capability
        mib = self.total_memory // (1024 * 1024)
        return f"cuda:{self.index} {self.name} ({mib} MiB, sm_{major}{minor})"


class CudaProvider(abc.ABC):
    """Enumerate devices, bind a thread, move bytes, manage streams and graphs."""

    name: ClassVar[str] = "abstract"
    priority: ClassVar[int] = 100
    supports_graphs: ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def probe(cls) -> CudaProvider | None:
        """A working instance, or ``None``. Must not raise: probing runs on every shape of
        host, and a driver mismatch is a reason to try the next provider, not to crash."""

    @property
    def is_available(self) -> bool:
        return self.device_count() > 0

    @abc.abstractmethod
    def device_count(self) -> int: ...

    @abc.abstractmethod
    def properties(self, index: int) -> RawDeviceProperties: ...

    @abc.abstractmethod
    def set_device(self, index: int) -> None: ...

    @abc.abstractmethod
    def synchronize(self, index: int | None = None) -> None: ...

    @abc.abstractmethod
    def memory_info(self, index: int) -> tuple[int, int]: ...

    @abc.abstractmethod
    def device_malloc(self, nbytes: int) -> DevicePtr: ...

    @abc.abstractmethod
    def device_free(self, ptr: DevicePtr) -> None: ...

    @abc.abstractmethod
    def host_alloc_pinned(self, nbytes: int) -> HostPtr: ...

    @abc.abstractmethod
    def host_free_pinned(self, ptr: HostPtr) -> None: ...

    @abc.abstractmethod
    def memcpy_h2d(self, dst: int, src: int, nbytes: int, stream: StreamHandle = 0) -> None: ...

    @abc.abstractmethod
    def memcpy_d2h(self, dst: int, src: int, nbytes: int, stream: StreamHandle = 0) -> None: ...

    @abc.abstractmethod
    def create_stream(self) -> StreamHandle: ...

    @abc.abstractmethod
    def destroy_stream(self, stream: StreamHandle) -> None: ...

    @abc.abstractmethod
    def stream_synchronize(self, stream: StreamHandle) -> None: ...

    # -- graphs (optional capability) ----------------------------------------------------

    def begin_capture(self, stream: StreamHandle) -> None:
        raise DeviceError(f"provider {self.name!r} cannot capture graphs")

    def end_capture(self, stream: StreamHandle) -> int:
        raise DeviceError(f"provider {self.name!r} cannot capture graphs")

    def launch_graph(self, graph_exec: int, stream: StreamHandle) -> None:
        raise DeviceError(f"provider {self.name!r} cannot launch graphs")

    def destroy_graph(self, graph_exec: int) -> None:
        raise DeviceError(f"provider {self.name!r} cannot destroy graphs")

    # -- helpers ---------------------------------------------------------------------------

    @staticmethod
    def host_array(ptr: HostPtr, nbytes: int) -> np.ndarray:
        """A numpy view over pinned host memory — no copy.

        The trick that makes raw pinned buffers usable from Python: the pool owns the
        allocation, numpy borrows it, so staging is an array assignment.
        """
        if not ptr:
            raise DeviceError("cannot build a host view over a null pointer")
        return np.frombuffer((ctypes.c_byte * nbytes).from_address(ptr), dtype=np.uint8)

    def describe(self) -> str:
        return f"{self.name} ({self.device_count()} device(s))"

    def __repr__(self) -> str:
        return f"<CudaProvider {self.name} devices={self.device_count()}>"


def decode(value: Any) -> str:
    """cuda-python returns bytes for names; torch returns ``str``."""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
