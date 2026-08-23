"""Allocator statistics, read straight from torch."""

from __future__ import annotations

from dataclasses import dataclass

from shipinfer.core.types import Device
from shipinfer.runtime.platform import memory_info, require_torch

__all__ = ["MemoryReport", "device_report"]


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """One device's memory picture.

    ``alloc_retries`` is the number to watch. Torch retries an allocation after freeing
    cached blocks, so a rising count means the pool is fragmented or a batch is too large —
    and it rises well before an OOM actually happens, which makes it the useful alert.
    """

    device: Device
    free: int
    total: int
    allocated: int
    reserved: int
    alloc_retries: int
    ooms: int

    @property
    def used_fraction(self) -> float:
        return 1.0 - (self.free / self.total) if self.total else 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "device": str(self.device),
            "free_mb": self.free // (1 << 20),
            "total_mb": self.total // (1 << 20),
            "torch_allocated_mb": self.allocated // (1 << 20),
            "torch_reserved_mb": self.reserved // (1 << 20),
            "alloc_retries": self.alloc_retries,
            "ooms": self.ooms,
            "used": round(self.used_fraction, 3),
        }


def device_report(device: Device) -> MemoryReport:
    if not device.is_cuda:
        return MemoryReport(device, 0, 0, 0, 0, 0, 0)
    torch = require_torch()
    free, total = memory_info(device.index)
    stats = torch.cuda.memory_stats(device.index)
    return MemoryReport(
        device=device,
        free=free,
        total=total,
        allocated=int(stats.get("allocated_bytes.all.current", 0)),
        reserved=int(stats.get("reserved_bytes.all.current", 0)),
        alloc_retries=int(stats.get("num_alloc_retries", 0)),
        ooms=int(stats.get("num_ooms", 0)),
    )
