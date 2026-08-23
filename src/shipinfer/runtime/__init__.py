"""Runtime: devices, streams, memory, CUDA graphs, image ops, the native loader.

The only package allowed to know an accelerator exists. Everything above it — scheduling,
the repository, the server's control flow — is written against :mod:`shipinfer.core` and
runs unchanged on a host with no driver (ADR-001).

The substrate is **torch**, deliberately. It already provides a caching device allocator, a
caching pinned-host allocator, streams, events, CUDA graph capture with memory-pool sharing,
and one API that covers CUDA and ROCm alike. This project writes the layer *above* that
(scheduling, batching, placement) and the fused kernels torch has no equivalent for
(``native/``) — nothing in between (ADR-003).

Sub-packages, each with a registry:

* :mod:`~shipinfer.runtime.memory` — allocators, the staging pool, the facade (:data:`ALLOCATORS`)
* :mod:`~shipinfer.runtime.graphs` — graph capture (:data:`GRAPH_CACHES`)
* :mod:`~shipinfer.runtime.ops` — batched image kernels (:data:`IMAGE_OPS`)
* :mod:`~shipinfer.runtime.providers` — raw driver access, for the ``custom`` variants only
"""

from shipinfer.runtime.device import DeviceManager, bind_thread, current_device
from shipinfer.runtime.graphs import (
    GRAPH_CACHES,
    CapturedGraph,
    CustomGraphCache,
    GraphCache,
    TorchGraphCache,
)
from shipinfer.runtime.memory import (
    ALLOCATORS,
    Allocator,
    Buffer,
    MemoryPool,
    MemoryReport,
    PinnedStagingPool,
)
from shipinfer.runtime.native import (
    is_native_available,
    native_module,
    native_version,
    require_native,
    resolve_provider,
)
from shipinfer.runtime.ops import (
    IMAGE_OPS,
    ImageOps,
    LetterboxResult,
    NormalizeParams,
    NumpyImageOps,
    TorchImageOps,
    get_image_ops,
)
from shipinfer.runtime.platform import (
    AcceleratorKind,
    DeviceProperties,
    accelerator_kind,
    device_count,
    device_properties,
    is_available,
    require_torch,
    torch_module,
)
from shipinfer.runtime.stream import Stream, StreamPool
from shipinfer.runtime.tensor import batch_to_torch, from_torch, to_torch

__all__ = [
    "ALLOCATORS",
    "GRAPH_CACHES",
    "IMAGE_OPS",
    "AcceleratorKind",
    "Allocator",
    "Buffer",
    "CapturedGraph",
    "CustomGraphCache",
    "DeviceManager",
    "DeviceProperties",
    "GraphCache",
    "ImageOps",
    "LetterboxResult",
    "MemoryPool",
    "MemoryReport",
    "NormalizeParams",
    "NumpyImageOps",
    "PinnedStagingPool",
    "Stream",
    "StreamPool",
    "TorchGraphCache",
    "TorchImageOps",
    "accelerator_kind",
    "batch_to_torch",
    "bind_thread",
    "current_device",
    "device_count",
    "device_properties",
    "from_torch",
    "get_image_ops",
    "is_available",
    "is_native_available",
    "native_module",
    "native_version",
    "require_native",
    "require_torch",
    "resolve_provider",
    "to_torch",
    "torch_module",
]
