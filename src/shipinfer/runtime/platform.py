"""Accelerator detection — one place that answers "what can this host do?".

Everything below this line is built on **torch**, deliberately. Torch already ships a
caching device allocator, a caching *pinned-host* allocator, stream and event objects,
CUDA graph capture with memory-pool sharing, and a single API that covers both CUDA and
ROCm. Re-implementing any of that against raw driver bindings would produce something
slower, buggier and platform-specific — which is exactly why vLLM, SGLang and TensorRT-LLM
all build on torch rather than beside it (ADR-003).

What this project *does* write itself is the layer above (scheduling, batching, placement)
and the fused kernels torch has no equivalent for (``native/``). Nothing in between.
"""

from __future__ import annotations

import enum
import functools
from dataclasses import dataclass
from typing import Any

from shipinfer.core.errors import DeviceError
from shipinfer.core.logging import LOG

__all__ = [
    "AcceleratorKind",
    "DeviceProperties",
    "accelerator_kind",
    "device_count",
    "device_properties",
    "is_available",
    "require_torch",
    "torch_module",
]


class AcceleratorKind(str, enum.Enum):
    """Which accelerator stack torch was built against."""

    CUDA = "cuda"
    ROCM = "rocm"  # torch.cuda talks to HIP; the tensor API is identical
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class DeviceProperties:
    """What the server needs to know about one device."""

    index: int
    name: str
    total_memory: int
    compute_capability: tuple[int, int]
    multi_processor_count: int = 0

    @property
    def total_memory_mb(self) -> int:
        return self.total_memory // (1024 * 1024)

    def __str__(self) -> str:
        major, minor = self.compute_capability
        return f"cuda:{self.index} {self.name} ({self.total_memory_mb} MiB, sm_{major}{minor})"


@functools.lru_cache(maxsize=1)
def torch_module() -> Any | None:
    """Import torch once, or ``None`` if it is not installed.

    Cached because a failed import is not free and this is asked from constructors.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        LOG.debug("torch unavailable: %s", exc)
        return None
    return torch


def require_torch() -> Any:
    """Torch, or an actionable error.

    Raises:
        DeviceError: naming the install command. Torch is a declared dependency, so
            reaching this means a broken environment rather than an unsupported one.
    """
    torch = torch_module()
    if torch is None:
        raise DeviceError(
            "PyTorch is required by shipinfer.runtime but is not importable. "
            "Install it from https://pytorch.org for your CUDA/ROCm version."
        )
    return torch


@functools.lru_cache(maxsize=1)
def accelerator_kind() -> AcceleratorKind:
    """CUDA, ROCm or CPU.

    ROCm is detected through ``torch.version.hip`` rather than a separate code path,
    because on ROCm ``torch.cuda`` *is* the HIP API — the same calls, the same semantics.
    That is why this project supports AMD without a parallel implementation.
    """
    torch = torch_module()
    if torch is None or not torch.cuda.is_available():
        return AcceleratorKind.CPU
    return AcceleratorKind.ROCM if getattr(torch.version, "hip", None) else AcceleratorKind.CUDA


def is_available() -> bool:
    return accelerator_kind() is not AcceleratorKind.CPU


def device_count() -> int:
    torch = torch_module()
    if torch is None or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def device_properties(index: int) -> DeviceProperties:
    torch = require_torch()
    props = torch.cuda.get_device_properties(index)
    return DeviceProperties(
        index=index,
        name=props.name,
        total_memory=int(props.total_memory),
        compute_capability=(int(props.major), int(props.minor)),
        multi_processor_count=int(getattr(props, "multi_processor_count", 0)),
    )


def memory_info(index: int) -> tuple[int, int]:
    """``(free, total)`` bytes as the driver reports them."""
    torch = require_torch()
    free, total = torch.cuda.mem_get_info(index)
    return int(free), int(total)


def describe() -> str:
    kind = accelerator_kind()
    if kind is AcceleratorKind.CPU:
        torch = torch_module()
        version = getattr(torch, "__version__", "not installed") if torch else "not installed"
        return f"cpu (torch {version}, no accelerator)"
    torch = require_torch()
    devices = ", ".join(str(device_properties(i)) for i in range(device_count()))
    return f"{kind.value} (torch {torch.__version__}): {devices}"
