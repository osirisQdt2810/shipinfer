"""Raw driver access through NVIDIA's ``cuda-python`` bindings."""

from __future__ import annotations

import threading
from typing import Any

from shipinfer.core.errors import DeviceError, DeviceOutOfMemoryError
from shipinfer.core.logging import LOG
from shipinfer.runtime.providers.base import (
    CudaProvider,
    RawDeviceProperties,
    StreamHandle,
    decode,
)
from shipinfer.runtime.providers.registry import PROVIDERS

__all__ = ["CudaPythonProvider"]


@PROVIDERS.register("cuda_python", "cuda")
class CudaPythonProvider(CudaProvider):
    """The thinnest possible layer over the CUDA runtime API.

    Reading this class is the fastest way to see what ``torch.cuda`` does on your behalf:
    every method here is one driver call plus error unwrapping, and the ``custom``
    allocator and graph cache are built out of exactly these primitives.
    """

    name = "cuda_python"
    priority = 10
    supports_graphs = True

    def __init__(self, runtime: Any) -> None:
        self._rt = runtime
        self._pinned: dict[int, int] = {}
        self._lock = threading.Lock()

    @classmethod
    def probe(cls) -> CudaProvider | None:
        runtime: Any
        try:  # cuda-python >= 12.3
            from cuda.bindings import runtime
        except ImportError:
            try:  # cuda-python < 12.3
                from cuda import cudart as runtime
            except ImportError:
                return None
        try:
            err, count = runtime.cudaGetDeviceCount()
        except Exception as exc:
            LOG.debug("cuda-python present but unusable: %s", exc)
            return None
        if int(err) != 0 or count == 0:
            return None
        return cls(runtime)

    def _check(self, result: Any, what: str) -> Any:
        """Unwrap cuda-python's ``(error, *values)`` convention.

        Every binding call returns that tuple, so centralising the unwrap is the difference
        between one readable module and two hundred lines of ``if err != 0``.
        """
        if isinstance(result, tuple):
            err, *values = result
        else:
            err, values = result, []
        if int(err) != 0:
            _, err_name = self._rt.cudaGetErrorName(err)
            _, err_text = self._rt.cudaGetErrorString(err)
            name, text = decode(err_name), decode(err_text)
            message = f"{what} failed: {name} - {text}"
            if "outofmemory" in name.lower().replace("_", ""):
                raise DeviceOutOfMemoryError(message)
            raise DeviceError(message)
        if not values:
            return None
        return values[0] if len(values) == 1 else tuple(values)

    def device_count(self) -> int:
        err, count = self._rt.cudaGetDeviceCount()
        return int(count) if int(err) == 0 else 0

    def properties(self, index: int) -> RawDeviceProperties:
        props = self._check(
            self._rt.cudaGetDeviceProperties(index), f"cudaGetDeviceProperties({index})"
        )
        return RawDeviceProperties(
            index=index,
            name=decode(props.name),
            total_memory=int(props.totalGlobalMem),
            compute_capability=(int(props.major), int(props.minor)),
            multi_processor_count=int(getattr(props, "multiProcessorCount", 0)),
        )

    def set_device(self, index: int) -> None:
        self._check(self._rt.cudaSetDevice(index), f"cudaSetDevice({index})")

    def synchronize(self, index: int | None = None) -> None:
        if index is not None:
            self.set_device(index)
        self._check(self._rt.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def memory_info(self, index: int) -> tuple[int, int]:
        self.set_device(index)
        free, total = self._check(self._rt.cudaMemGetInfo(), "cudaMemGetInfo")
        return int(free), int(total)

    def device_malloc(self, nbytes: int) -> int:
        return int(self._check(self._rt.cudaMalloc(nbytes), f"cudaMalloc({nbytes})"))

    def device_free(self, ptr: int) -> None:
        self._check(self._rt.cudaFree(ptr), "cudaFree")

    def host_alloc_pinned(self, nbytes: int) -> int:
        ptr = int(self._check(self._rt.cudaHostAlloc(nbytes, 0), f"cudaHostAlloc({nbytes})"))
        with self._lock:
            self._pinned[ptr] = nbytes
        return ptr

    def host_free_pinned(self, ptr: int) -> None:
        self._check(self._rt.cudaFreeHost(ptr), "cudaFreeHost")
        with self._lock:
            self._pinned.pop(ptr, None)

    def pinned_bytes(self) -> int:
        return sum(self._pinned.values())

    def memcpy_h2d(self, dst: int, src: int, nbytes: int, stream: int = 0) -> None:
        kind = self._rt.cudaMemcpyKind.cudaMemcpyHostToDevice
        if stream:
            self._check(
                self._rt.cudaMemcpyAsync(dst, src, nbytes, kind, stream), "memcpyAsync H2D"
            )
        else:
            self._check(self._rt.cudaMemcpy(dst, src, nbytes, kind), "memcpy H2D")

    def memcpy_d2h(self, dst: int, src: int, nbytes: int, stream: int = 0) -> None:
        kind = self._rt.cudaMemcpyKind.cudaMemcpyDeviceToHost
        if stream:
            self._check(
                self._rt.cudaMemcpyAsync(dst, src, nbytes, kind, stream), "memcpyAsync D2H"
            )
        else:
            self._check(self._rt.cudaMemcpy(dst, src, nbytes, kind), "memcpy D2H")

    def create_stream(self) -> StreamHandle:
        return int(self._check(self._rt.cudaStreamCreate(), "cudaStreamCreate"))

    def destroy_stream(self, stream: StreamHandle) -> None:
        self._check(self._rt.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def stream_synchronize(self, stream: StreamHandle) -> None:
        self._check(self._rt.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    # -- graphs ----------------------------------------------------------------------------

    def begin_capture(self, stream: StreamHandle) -> None:
        # ThreadLocal, not Global: a global capture would swallow work issued by other
        # worker threads on their own streams, producing a graph that replays somebody
        # else's inference. torch.cuda.graph() makes the same choice for the same reason.
        mode = self._rt.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        self._check(self._rt.cudaStreamBeginCapture(stream, mode), "cudaStreamBeginCapture")

    def end_capture(self, stream: StreamHandle) -> int:
        graph = self._check(self._rt.cudaStreamEndCapture(stream), "cudaStreamEndCapture")
        graph_exec = self._check(
            self._rt.cudaGraphInstantiate(graph, 0), "cudaGraphInstantiate"
        )
        # The template graph is dead weight once instantiated; keeping it leaks a few
        # hundred KB per captured batch size, times every model, times every instance.
        self._check(self._rt.cudaGraphDestroy(graph), "cudaGraphDestroy")
        return int(graph_exec)

    def launch_graph(self, graph_exec: int, stream: StreamHandle) -> None:
        self._check(self._rt.cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")

    def destroy_graph(self, graph_exec: int) -> None:
        self._check(self._rt.cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy")
