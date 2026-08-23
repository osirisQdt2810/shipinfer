"""Buffers and the allocator contract.

Two implementations satisfy it and they exist for different reasons:

* :mod:`~shipinfer.runtime.memory.torch_alloc` — the default. It delegates to torch's
  caching allocator, which is stream-aware, graph-aware, battle-tested and faster than
  anything written here would be.
* :mod:`~shipinfer.runtime.memory.custom` — a readable re-implementation of the same
  behaviour on raw driver calls. Selectable through the registry, used by the parity tests,
  and the place to look when you want to know *what torch is actually doing*.

Keeping both behind one interface costs a few dozen lines and buys a runnable explanation
of the fast path.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

import numpy as np

from shipinfer.core.errors import DeviceError
from shipinfer.core.types import Device, MemoryKind

__all__ = ["Allocator", "Buffer", "align_up"]


def align_up(nbytes: int, granularity: int) -> int:
    """Round a size up to a multiple of ``granularity``.

    Bucketing is what makes a caching allocator actually cache: without it, batches of 7
    and 8 images request two different sizes forever and the free lists never hit.
    """
    if granularity <= 1:
        return nbytes
    return -(-nbytes // granularity) * granularity


class Buffer:
    """One allocated block, returned to its allocator on :meth:`free`.

    Structurally satisfies :class:`shipinfer.core.types.MemoryHandle`, which is how a
    device tensor can reference GPU memory without ``core`` importing anything CUDA-shaped.
    """

    __slots__ = (
        "_array",
        "_device",
        "_freed",
        "_keepalive",
        "_kind",
        "_nbytes",
        "_owner",
        "_ptr",
    )

    def __init__(
        self,
        ptr: int,
        nbytes: int,
        kind: MemoryKind,
        device: Device,
        owner: Allocator | None = None,
        array: np.ndarray | None = None,
        keepalive: Any = None,
    ) -> None:
        self._ptr = ptr
        self._nbytes = nbytes
        self._kind = kind
        self._device = device
        self._owner = owner
        self._freed = False
        #: Host-visible view, when there is one.
        self._array = array
        #: The object that actually owns the memory (e.g. a torch tensor). Held so garbage
        #: collection cannot free memory a live pointer still refers to.
        self._keepalive = keepalive

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def kind(self) -> MemoryKind:
        return self._kind

    @property
    def device(self) -> Device:
        return self._device

    @property
    def is_freed(self) -> bool:
        return self._freed

    @property
    def owner_object(self) -> Any:
        """The backing object, if any — a ``torch.Tensor`` for the torch allocators."""
        return self._keepalive

    def as_array(
        self, dtype: Any = np.uint8, shape: tuple[int, ...] | None = None
    ) -> np.ndarray:
        """A numpy view over this buffer — no copy.

        Raises:
            DeviceError: for device memory, which has no host address. Ask for an explicit
                D2H instead of hiding a synchronising copy behind an attribute access.
        """
        if self._kind is MemoryKind.DEVICE:
            raise DeviceError("device memory has no host view; copy it back explicitly")
        if self._array is None:
            raise DeviceError("this buffer was created without a host view")
        view = self._array.view(dtype)
        return view.reshape(shape) if shape is not None else view

    def free(self) -> None:
        """Return the block to its allocator. Idempotent."""
        if self._freed:
            return
        self._freed = True
        if self._owner is not None:
            self._owner.deallocate(self)

    def __len__(self) -> int:
        return self._nbytes

    def __repr__(self) -> str:
        state = "freed" if self._freed else f"0x{self._ptr:x}"
        return f"<Buffer {self._nbytes}B {self._kind.value}@{self._device} {state}>"


class Allocator(abc.ABC):
    """Allocates and releases blocks of one memory kind on one device."""

    name: ClassVar[str] = "abstract"

    def __init__(self, device: Device, kind: MemoryKind) -> None:
        self._device = device
        self._kind = kind

    @property
    def device(self) -> Device:
        return self._device

    @property
    def kind(self) -> MemoryKind:
        return self._kind

    @abc.abstractmethod
    def allocate(self, nbytes: int) -> Buffer:
        """A block of at least ``nbytes``.

        Raises:
            DeviceOutOfMemoryError: when the device cannot satisfy it.
        """

    @abc.abstractmethod
    def deallocate(self, buffer: Buffer) -> None:
        """Release a block. Called by :meth:`Buffer.free`, not usually by hand."""

    def stats(self) -> dict[str, int]:
        """Allocation counters. An allocator that cannot report is one you cannot tune."""
        return {}

    def close(self) -> None:
        """Release everything still held. Called at server shutdown."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self._kind.value}@{self._device}>"
