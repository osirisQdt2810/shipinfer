"""Reference implementation: CUDA graph capture on raw driver calls."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from shipinfer.core.errors import DeviceError
from shipinfer.core.logging import LOG
from shipinfer.core.types import Device
from shipinfer.runtime.graphs.base import CapturedGraph, GraphCache
from shipinfer.runtime.graphs.registry import GRAPH_CACHES
from shipinfer.runtime.providers import CudaProvider, get_cuda_provider
from shipinfer.runtime.stream import Stream

__all__ = ["CustomCapturedGraph", "CustomGraphCache"]


class CustomCapturedGraph(CapturedGraph):
    """A ``cudaGraphExec_t`` held as an integer handle."""

    def __init__(
        self,
        handle: int,
        batch_size: int,
        device: Device,
        provider: CudaProvider,
        stream: Stream,
        static_inputs: Mapping[str, Any],
        static_outputs: Mapping[str, Any],
    ) -> None:
        self._handle = handle
        self._device = device
        self._provider = provider
        self._stream = stream
        self.batch_size = batch_size
        self.static_inputs = dict(static_inputs)
        self.static_outputs = dict(static_outputs)
        self._replays = 0
        self._closed = False

    @property
    def replays(self) -> int:
        return self._replays

    def replay(self, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise DeviceError("cannot replay a destroyed CUDA graph")
        if inputs:
            for name, value in inputs.items():
                target = self.static_inputs.get(name)
                if target is None:
                    raise DeviceError(f"captured graph has no input {name!r}")
                target[: value.shape[0]].copy_(value, non_blocking=True)
        self._provider.launch_graph(self._handle, self._stream.handle)
        self._replays += 1
        return self.static_outputs

    def close(self) -> None:
        if not self._closed:
            self._provider.destroy_graph(self._handle)
            self._closed = True
        self.static_inputs.clear()
        self.static_outputs.clear()

    def __repr__(self) -> str:
        return f"<CustomCapturedGraph batch={self.batch_size} replays={self._replays}>"


@GRAPH_CACHES.register("custom", "raw")
class CustomGraphCache(GraphCache):
    """Capture with ``cudaStreamBeginCapture`` / ``cudaGraphInstantiate`` directly.

    **Read this to see what** :class:`~shipinfer.runtime.graphs.torch_graph.TorchGraphCache`
    **spares you.** The sequence is short — begin capture, issue work, end capture,
    instantiate — and every hard part is missing:

    * no side-stream warm-up, so whatever one-time allocation or autotuning the first call
      performs gets baked into the graph and replayed forever;
    * no shared memory pool, so N graphs for N batch sizes each reserve their own arena;
    * no allocator cooperation, so a block allocated during capture can be handed to
      another stream afterwards and silently corrupt a later replay.

    Correct only when the captured region allocates nothing and has been warmed by the
    caller — which is exactly the discipline torch enforces for you. Use it to understand
    the mechanism, or on a host where torch's graph API is unavailable; not in production
    (ADR-003).
    """

    name = "custom"

    def __init__(
        self,
        device: Device,
        *,
        enabled: bool = True,
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        max_failures: int = 3,
        provider: CudaProvider | None = None,
    ) -> None:
        super().__init__(
            device, enabled=enabled, batch_sizes=batch_sizes, max_failures=max_failures
        )
        self._provider = provider or get_cuda_provider()
        self._graphs: dict[int, CustomCapturedGraph] = {}
        self._lock = threading.Lock()
        self._enabled = enabled and device.is_cuda and self._provider.supports_graphs
        if enabled and not self._enabled:
            LOG.debug(
                "custom graph cache disabled on %s: provider %s cannot capture",
                device,
                self._provider.name,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled and self._failures < self._max_failures

    def get(self, batch_size: int) -> CapturedGraph | None:
        return self._graphs.get(batch_size)

    def capture(
        self,
        batch_size: int,
        stream: Stream,
        static_inputs: Mapping[str, Any],
        run: Callable[[], Mapping[str, Any]],
    ) -> CapturedGraph | None:
        if not self.should_capture(batch_size):
            return self._graphs.get(batch_size)

        with self._lock:
            if batch_size in self._graphs:
                return self._graphs[batch_size]
            try:
                # The caller must have warmed this shape already: there is no side-stream
                # warm-up here, and capturing a first-call allocation would poison every
                # replay. This is the sharpest edge the torch implementation removes.
                self._provider.begin_capture(stream.handle)
                outputs = run()
                handle = self._provider.end_capture(stream.handle)
            except Exception as exc:
                self._failures += 1
                log = LOG.warning if self._failures >= self._max_failures else LOG.debug
                log(
                    "custom graph capture failed on %s at batch %d (%d/%d): %s",
                    self._device,
                    batch_size,
                    self._failures,
                    self._max_failures,
                    exc,
                )
                return None
            captured = CustomCapturedGraph(
                handle,
                batch_size,
                self._device,
                self._provider,
                stream,
                static_inputs,
                outputs,
            )
            self._graphs[batch_size] = captured
            LOG.info("captured raw CUDA graph on %s for batch %d", self._device, batch_size)
            return captured

    def close(self) -> None:
        with self._lock:
            for graph in self._graphs.values():
                graph.close()
            self._graphs.clear()

    def stats(self) -> dict[str, int]:
        return {
            "captured": len(self._graphs),
            "replays": sum(g.replays for g in self._graphs.values()),
            "failures": self._failures,
        }

    def __repr__(self) -> str:
        return f"<CustomGraphCache {self._device} enabled={self.enabled}>"
