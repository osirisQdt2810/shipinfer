"""The per-process memory facade."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from shipinfer.core.logging import get_logger
from shipinfer.core.settings import MemorySettings
from shipinfer.core.types import Device, MemoryKind
from shipinfer.runtime.memory.base import Allocator, Buffer
from shipinfer.runtime.memory.registry import ALLOCATORS
from shipinfer.runtime.memory.report import MemoryReport, device_report
from shipinfer.runtime.memory.staging import PinnedStagingPool
from shipinfer.runtime.platform import is_available

__all__ = ["MemoryPool"]

_LOG = get_logger("runtime.memory.pool")


class MemoryPool:
    """The allocators one process uses, wired according to :class:`MemorySettings`.

    The wiring is the point. Callers never pick an allocator by hand, because the *correct*
    choice is context-dependent and easy to get wrong: torch's caching allocator on a GPU,
    plain numpy on CPU, and — if someone explicitly asks for the reference implementation —
    the hand-written one wrapped in its bucketed cache, never bare.

    Device allocators are created lazily, so a process pinned to GPU 3 never touches GPU 0.
    """

    def __init__(self, settings: MemorySettings | None = None) -> None:
        self._settings = settings or MemorySettings()
        self._lock = threading.Lock()
        self._device_allocators: dict[Device, Allocator] = {}
        self._host: Allocator = ALLOCATORS.create("host")
        self._pinned: Allocator | None = None
        self._staging_pools: dict[str, PinnedStagingPool] = {}

    # -- allocators ---------------------------------------------------------------------

    @property
    def host(self) -> Allocator:
        """Pageable host memory. Never the source of an async copy."""
        return self._host

    def staging_for(self, owner: str) -> PinnedStagingPool:
        """This owner's private pinned staging pool, created on first use.

        Per owner, not shared, and deliberately not exposed as a single ``.staging``
        property. One worker thread runs per model instance; a shared pool hands two of
        them the same buffer for the same shape, and the second one's copy overwrites the
        first's while its DMA is still reading — GPU 0 then infers on GPU 1's frames and
        returns them under the wrong camera's tag, with no error anywhere.

        Callers pass something stable and unique: an instance name, not a shape.
        """
        pool = self._staging_pools.get(owner)
        if pool is None:
            with self._lock:
                pool = self._staging_pools.get(owner)
                if pool is None:
                    pool = PinnedStagingPool(owner=owner)
                    self._staging_pools[owner] = pool
        return pool

    @property
    def pinned(self) -> Allocator:
        """Page-locked host memory.

        Falls back to pageable host memory when there is no accelerator, so CPU-only code
        paths work without a branch at every call site.
        """
        if self._pinned is None:
            with self._lock:
                if self._pinned is None:
                    self._pinned = self._build_pinned()
        return self._pinned

    def _build_pinned(self) -> Allocator:
        if not is_available():
            _LOG.debug("no accelerator: pinned pool falls back to pageable host memory")
            return self._host
        if self._settings.allocator == "custom":
            return ALLOCATORS.create(
                "custom_caching",
                ALLOCATORS.create("custom_pinned"),
                granularity=self._settings.allocation_granularity,
                max_blocks_per_bucket=self._settings.max_cached_blocks_per_bucket,
            )
        return ALLOCATORS.create("torch_pinned")

    def device(self, device: Device) -> Allocator:
        """The device allocator for one GPU, created on first use."""
        if not device.is_cuda:
            return self._host
        allocator = self._device_allocators.get(device)
        if allocator is None:
            with self._lock:
                allocator = self._device_allocators.get(device)
                if allocator is None:
                    allocator = self._build_device(device)
                    self._device_allocators[device] = allocator
                    _LOG.debug("created %s for %s", allocator.name, device)
        return allocator

    def _build_device(self, device: Device) -> Allocator:
        if self._settings.allocator == "custom":
            return ALLOCATORS.create(
                "custom_caching",
                ALLOCATORS.create("custom_device", device),
                granularity=self._settings.allocation_granularity,
                max_blocks_per_bucket=self._settings.max_cached_blocks_per_bucket,
            )
        return ALLOCATORS.create("torch_device", device)

    def for_kind(self, kind: MemoryKind, device: Device) -> Allocator:
        if kind is MemoryKind.DEVICE:
            return self.device(device)
        if kind is MemoryKind.PINNED:
            return self.pinned
        return self._host

    # -- scoped use ---------------------------------------------------------------------

    @contextmanager
    def borrow(self, nbytes: int, kind: MemoryKind, device: Device) -> Iterator[Buffer]:
        """Allocate for the duration of a block, then release it.

        With a caching allocator underneath, "release" means "push onto a free list", so
        this is cheap enough to use per batch — which is exactly why the cache exists.
        """
        allocator = self.for_kind(kind, device)
        buffer = allocator.allocate(nbytes)
        try:
            yield buffer
        finally:
            buffer.free()

    # -- lifecycle ----------------------------------------------------------------------

    def report(self, device: Device) -> MemoryReport:
        return device_report(device)

    def stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {
            "host": self._host.stats(),
            **{f"staging[{o}]": p.stats() for o, p in self._staging_pools.items()},
        }
        if self._pinned is not None:
            out["pinned"] = self._pinned.stats()
        for device, allocator in self._device_allocators.items():
            out[str(device)] = allocator.stats()
        return out

    def close(self) -> None:
        with self._lock:
            allocators = list(self._device_allocators.values())
            self._device_allocators.clear()
            pinned, self._pinned = self._pinned, None
        for allocator in allocators:
            allocator.close()
        if pinned is not None and pinned is not self._host:
            pinned.close()
        for pool in self._staging_pools.values():
            pool.clear()
        self._staging_pools.clear()
        self._host.close()

    def __repr__(self) -> str:
        return f"<MemoryPool devices={sorted(map(str, self._device_allocators))}>"
