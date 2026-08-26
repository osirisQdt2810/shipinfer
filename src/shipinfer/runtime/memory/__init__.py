"""Memory: buffers, allocators, the staging pool and the per-process facade.

The default allocators are thin adapters over **torch's** caching device and pinned-host
allocators — stream-aware, graph-aware, and better than anything written here would be.
The ``custom_*`` allocators re-implement the same contract on raw driver calls; they are
selectable through the registry, exercised by the parity tests, and exist so that "torch
handles it" is something you can read rather than take on faith (ADR-003).

:class:`PinnedStagingPool` is the one piece with no torch equivalent: pinned buffers with
**stable addresses**, which CUDA graph capture requires.
"""

from shipinfer.runtime.memory.base import Allocator, Buffer, align_up
from shipinfer.runtime.memory.custom import (
    CustomCachingAllocator,
    CustomDeviceAllocator,
    CustomPinnedAllocator,
)
from shipinfer.runtime.memory.pool import MemoryPool
from shipinfer.runtime.memory.registry import ALLOCATORS
from shipinfer.runtime.memory.report import MemoryReport, device_report
from shipinfer.runtime.memory.shared_ring import (
    RingHeader,
    RingLayout,
    SharedRing,
    SlotState,
    reap_pending_closes,
)
from shipinfer.runtime.memory.staging import PinnedStagingPool
from shipinfer.runtime.memory.torch_alloc import (
    HostAllocator,
    TorchDeviceAllocator,
    TorchPinnedAllocator,
)

__all__ = [
    "ALLOCATORS",
    "Allocator",
    "Buffer",
    "CustomCachingAllocator",
    "CustomDeviceAllocator",
    "CustomPinnedAllocator",
    "HostAllocator",
    "MemoryPool",
    "MemoryReport",
    "PinnedStagingPool",
    "RingHeader",
    "RingLayout",
    "SharedRing",
    "SlotState",
    "TorchDeviceAllocator",
    "TorchPinnedAllocator",
    "align_up",
    "device_report",
    "reap_pending_closes",
]
