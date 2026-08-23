"""CUDA graph capture — the default over torch, plus a raw reference implementation.

Graph replay is the biggest CPU-side win available to this workload: a small, launch-bound
model spends as much time issuing kernels as running them, and replay collapses the whole
inference into one launch. vLLM applies the same technique to its decode step.

:class:`TorchGraphCache` is the default and does the three hard things correctly — side
stream warm-up, a shared memory pool, and an allocator that knows a capture is in progress.
:class:`CustomGraphCache` does the same four driver calls with none of that, and exists so
those three things are legible rather than magic (ADR-003).
"""

from shipinfer.runtime.graphs.base import CapturedGraph, GraphCache
from shipinfer.runtime.graphs.custom_graph import CustomCapturedGraph, CustomGraphCache
from shipinfer.runtime.graphs.registry import GRAPH_CACHES
from shipinfer.runtime.graphs.torch_graph import TorchCapturedGraph, TorchGraphCache

__all__ = [
    "GRAPH_CACHES",
    "CapturedGraph",
    "CustomCapturedGraph",
    "CustomGraphCache",
    "GraphCache",
    "TorchCapturedGraph",
    "TorchGraphCache",
]
