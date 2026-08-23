"""Host and device allocator behaviour."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MemorySettings"]


class MemorySettings(BaseModel):
    """Pool sizes and reuse policy.

    Pinned memory is the pipeline's currency: it is the only host memory a CUDA async copy
    can read, so a pool of it is reserved up front and recycled. ``cudaHostAlloc`` is a
    synchronising, millisecond-scale call — doing one per request would cost more than the
    inference it was preparing for.
    """

    model_config = ConfigDict(extra="forbid")

    #: Which allocator family to use. ``torch`` (the default) delegates to torch's caching
    #: device and pinned-host allocators. ``custom`` selects this project's hand-written
    #: bucketed allocator over raw driver calls — a readable reference implementation, kept
    #: for comparison and parity testing, not for production (ADR-003).
    allocator: Literal["torch", "custom"] = "torch"

    pinned_pool_mb: int = Field(default=512, ge=0)
    #: 0 == allocate device blocks on demand and cache them, rather than reserving a slab.
    device_pool_mb: int = Field(default=0, ge=0)
    #: Round allocations up to this many bytes so the caching allocator actually gets hits
    #: instead of holding a hundred near-identical blocks.
    allocation_granularity: int = Field(default=256, ge=1)
    #: Blocks kept per size bucket before memory goes back to the driver.
    max_cached_blocks_per_bucket: int = Field(default=8, ge=0)
    #: Fail loudly if a pool would exceed this share of a device's memory. Guards against
    #: a config that leaves no room for the TensorRT engines themselves.
    max_device_fraction: float = Field(default=0.9, gt=0.0, le=1.0)
