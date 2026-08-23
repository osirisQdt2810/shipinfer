"""CUDA graph capture and replay, via ``torch.cuda.CUDAGraph``.

This is the biggest CPU-side win available to this workload, and the arithmetic is simple:
a 512-d ReID embedder at batch 8 issues on the order of a hundred kernel launches, each
costing a few microseconds of CPU and driver time, and the kernels themselves take about as
long. Replay collapses the whole thing into one ``cudaGraphLaunch``. vLLM applies exactly
this to its decode step, for exactly this reason.

We use **torch's** graph API rather than raw ``cudaStreamBeginCapture`` because torch does
three things correctly that are easy to get wrong by hand (ADR-003):

* it warms up on a **side stream** first, so one-time lazy allocations and cuBLAS
  autotuning are not baked into the graph;
* it captures into a **shared memory pool**, so N graphs for N batch sizes do not each
  reserve a private arena;
* it makes the caching allocator graph-aware, so a block allocated during capture is not
  handed to another stream afterwards.

The preconditions are still strict and still ours to honour: static shapes (hence one graph
per batch size), stable I/O addresses (hence the persistent static buffers here), and no
host synchronisation inside the captured region.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from shipinfer.core.errors import DeviceError
from shipinfer.core.logging import get_logger
from shipinfer.core.types import Device
from shipinfer.runtime.graphs.base import CapturedGraph, GraphCache
from shipinfer.runtime.graphs.registry import GRAPH_CACHES
from shipinfer.runtime.platform import is_available, require_torch
from shipinfer.runtime.stream import Stream

__all__ = ["TorchCapturedGraph", "TorchGraphCache"]

_LOG = get_logger("runtime.graph")


class TorchCapturedGraph(CapturedGraph):
    """One captured graph plus the static buffers it reads and writes.

    The buffers are the contract. Replay does not take arguments: it re-executes the exact
    kernels recorded, against the exact addresses recorded. So a caller copies inputs *into*
    :attr:`static_inputs`, calls :meth:`replay`, and reads :attr:`static_outputs`.
    """

    def __init__(
        self,
        graph: Any,
        batch_size: int,
        static_inputs: Mapping[str, Any],
        static_outputs: Mapping[str, Any],
    ) -> None:
        self._graph = graph
        self.batch_size = batch_size
        self.static_inputs = dict(static_inputs)
        self.static_outputs = dict(static_outputs)
        self._replays = 0

    @property
    def replays(self) -> int:
        return self._replays

    def replay(self, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Copy inputs into the static buffers, replay, return the static outputs.

        ``copy_`` rather than rebinding: rebinding would point at new memory the graph has
        never heard of, and the replay would happily read the old buffer instead.
        """
        if inputs:
            for name, value in inputs.items():
                target = self.static_inputs.get(name)
                if target is None:
                    raise DeviceError(f"captured graph has no input {name!r}")
                target[: value.shape[0]].copy_(value, non_blocking=True)
        self._graph.replay()
        self._replays += 1
        return self.static_outputs

    def close(self) -> None:
        self._graph = None
        self.static_inputs.clear()
        self.static_outputs.clear()

    def __repr__(self) -> str:
        return f"<TorchCapturedGraph batch={self.batch_size} replays={self._replays}>"


@GRAPH_CACHES.register("torch", "default")
class TorchGraphCache(GraphCache):
    """Per-instance cache of captured graphs, keyed on batch size.

    Owns the *failure* policy as well as the cache: "we tried to capture and it did not
    work" has to be remembered, or an uncapturable model pays a failed capture attempt on
    every single batch — overhead added to the path it was meant to accelerate.
    """

    name = "torch"

    def __init__(
        self,
        device: Device,
        *,
        enabled: bool = True,
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        max_failures: int = 3,
        warmup_iterations: int = 3,
    ) -> None:
        super().__init__(
            device, enabled=enabled, batch_sizes=batch_sizes, max_failures=max_failures
        )
        self._warmup_iterations = warmup_iterations
        self._graphs: dict[int, CapturedGraph] = {}
        self._failures = 0
        self._lock = threading.Lock()
        self._enabled = enabled and device.is_cuda and is_available()
        #: One pool shared by every graph on this instance, so N batch sizes reserve one
        #: arena rather than N. This is the part that is genuinely hard to hand-roll.
        self._pool: Any = None

        if enabled and not self._enabled:
            _LOG.debug("CUDA graphs disabled on %s: no accelerator", device)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._failures < self._max_failures

    @property
    def captured_sizes(self) -> tuple[int, ...]:
        return tuple(sorted(self._graphs))

    def get(self, batch_size: int) -> CapturedGraph | None:
        return self._graphs.get(batch_size)

    def capture(
        self,
        batch_size: int,
        stream: Stream,
        static_inputs: Mapping[str, Any],
        run: Callable[[], Mapping[str, Any]],
    ) -> CapturedGraph | None:
        """Warm up, then record ``run`` into a graph for ``batch_size``.

        Args:
            static_inputs: the persistent input buffers ``run`` reads. They must be the
                same objects at replay time.
            run: issues the work and returns its output tensors. Must not synchronise.

        Returns:
            The captured graph, or ``None`` when capture is disabled or failed. ``None`` is
            not an error — it means "take the ordinary launch path", which is slower and
            equally correct.
        """
        if not self.should_capture(batch_size):
            return self._graphs.get(batch_size)

        torch = require_torch()
        with self._lock:
            if batch_size in self._graphs:
                return self._graphs[batch_size]
            try:
                self._warmup(torch, stream, run)
                if self._pool is None:
                    self._pool = torch.cuda.graph_pool_handle()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._pool, stream=stream.torch_stream):
                    outputs = run()
                captured = TorchCapturedGraph(graph, batch_size, static_inputs, outputs)
            except Exception as exc:
                self._failures += 1
                log = _LOG.warning if self._failures >= self._max_failures else _LOG.debug
                log(
                    "CUDA graph capture failed on %s at batch %d (%d/%d): %s",
                    self._device,
                    batch_size,
                    self._failures,
                    self._max_failures,
                    exc,
                )
                return None
            self._graphs[batch_size] = captured
            _LOG.info("captured CUDA graph on %s for batch %d", self._device, batch_size)
            return captured

    def _warmup(self, torch: Any, stream: Stream, run: Callable[[], Mapping[str, Any]]) -> None:
        """Run the work on a side stream before capturing.

        Without this the graph records whatever one-time work the first call does — lazy
        module loading, workspace allocation, an autotuning pass — and replays it forever.
        A side stream is used so the warm-up's allocations do not land in the graph's pool.
        """
        side = torch.cuda.Stream(device=self._device.index)
        side.wait_stream(torch.cuda.current_stream(self._device.index))
        with torch.cuda.stream(side):
            for _ in range(self._warmup_iterations):
                run()
        torch.cuda.current_stream(self._device.index).wait_stream(side)
        torch.cuda.synchronize(self._device.index)

    def close(self) -> None:
        with self._lock:
            for graph in self._graphs.values():
                graph.close()
            self._graphs.clear()
            self._pool = None

    def stats(self) -> dict[str, int]:
        return {
            "captured": len(self._graphs),
            "replays": sum(g.replays for g in self._graphs.values()),
            "failures": self._failures,
        }

    def __repr__(self) -> str:
        return f"<TorchGraphCache {self._device} enabled={self.enabled} sizes={self.captured_sizes}>"
