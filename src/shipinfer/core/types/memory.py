"""Where the bytes behind a tensor physically live."""

from __future__ import annotations

import enum
from typing import Protocol, runtime_checkable

from shipinfer.core.types.device import Device

__all__ = ["MemoryHandle", "MemoryKind"]


class MemoryKind(enum.Enum):
    """Host, pinned host, or device memory.

    The distinction is not cosmetic: ``PINNED`` is the only host kind a CUDA async copy
    may read from, and spilling a batch to another GPU is only cheap when the payload is
    already ``PINNED`` (ADR-004).
    """

    HOST = "host"  # ordinary pageable numpy memory
    PINNED = "pinned"  # page-locked host memory, DMA-able
    DEVICE = "device"  # GPU global memory


@runtime_checkable
class MemoryHandle(Protocol):
    """A block owned by :mod:`shipinfer.runtime.memory`.

    ``core`` never allocates one, it only carries them. Expressing the dependency as a
    Protocol rather than an import is what keeps the pure core free of CUDA (ADR-001).
    """

    @property
    def ptr(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def kind(self) -> MemoryKind: ...

    @property
    def device(self) -> Device: ...
