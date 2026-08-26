"""Device discovery, validation and thread binding — over ``torch.cuda``."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from shipinfer.core.errors import ConfigurationError, DeviceError
from shipinfer.core.logging import get_logger
from shipinfer.core.settings import DeviceSettings
from shipinfer.core.types import Device
from shipinfer.runtime.platform import (
    AcceleratorKind,
    DeviceProperties,
    accelerator_kind,
    describe,
    device_count,
    device_properties,
    memory_info,
    require_torch,
)

__all__ = ["DeviceManager", "bind_thread", "current_device"]

_LOG = get_logger("runtime.device")

#: Which device each worker thread is bound to. Used to *assert* the invariant "one thread,
#: one context, one GPU" rather than trusting it (ADR-002).
_THREAD_DEVICE = threading.local()


class DeviceManager:
    """The GPUs this process may use, validated once at start-up.

    Fail-fast is the entire design. A missing device found at start-up is a config error
    with a clear message; the same device found at the first inference is a CUDA error
    inside a worker thread, three layers from the cause, on a running service.
    """

    def __init__(self, settings: DeviceSettings | None = None) -> None:
        self._settings = settings or DeviceSettings()
        self._visible: tuple[int, ...] = self._resolve_visible()
        if self._settings.validate_on_start:
            self._validate()

    # -- discovery ----------------------------------------------------------------------

    def _resolve_visible(self) -> tuple[int, ...]:
        present = list(range(device_count()))
        requested = self._settings.visible_gpus
        if not requested:
            return tuple(present)
        missing = sorted(set(requested) - set(present))
        if missing:
            raise ConfigurationError(
                f"visible_gpus names device(s) {missing} but torch reports {present or 'none'}"
            )
        return tuple(requested)

    def _validate(self) -> None:
        if not self._visible:
            if not self._settings.allow_cpu_only:
                raise ConfigurationError(
                    "no accelerators visible and devices.allow_cpu_only is false"
                )
            _LOG.warning("no accelerators visible; CPU backends only")
            return
        for index in self._visible:
            free, total = memory_info(index)
            _LOG.info(
                "%s  free=%d MiB / %d MiB",
                device_properties(index),
                free // (1 << 20),
                total // (1 << 20),
            )

    # -- queries ------------------------------------------------------------------------

    @property
    def kind(self) -> AcceleratorKind:
        return accelerator_kind()

    @property
    def visible_gpus(self) -> tuple[int, ...]:
        return self._visible

    @property
    def shared_by(self) -> dict[int, int]:
        """Processes sharing each visible device, keyed by device index; absent means one.

        Raises:
            ConfigurationError: ``shared_by`` was given for a different number of devices than
                are visible — a misaligned list would silently halve the wrong device.
        """
        sharing = self._settings.shared_by
        if not sharing:
            return {}
        if len(sharing) != len(self._visible):
            raise ConfigurationError(
                f"devices.shared_by has {len(sharing)} entr(y/ies) but {len(self._visible)} "
                f"device(s) are visible; the two lists must align"
            )
        return dict(zip(self._visible, sharing, strict=True))

    @property
    def has_accelerator(self) -> bool:
        return bool(self._visible)

    def devices(self) -> list[Device]:
        return [Device.cuda(i) for i in self._visible]

    def properties(self, device: Device) -> DeviceProperties:
        if not device.is_cuda:
            raise DeviceError("cpu has no accelerator properties")
        return device_properties(device.index)

    def memory_info(self, device: Device) -> tuple[int, int]:
        return memory_info(device.index)

    def require(self, device: Device) -> Device:
        """Assert a device is usable by this process, returning it for chaining."""
        if device.is_cuda and device.index not in self._visible:
            raise ConfigurationError(
                f"{device} is not visible to this process (visible: {list(self._visible)})"
            )
        return device

    # -- binding ------------------------------------------------------------------------

    def bind_current_thread(self, device: Device) -> None:
        """Bind the calling thread to ``device`` for the rest of its life.

        Called once per worker thread at start-up, never per request. Torch keeps a
        thread-local current device, so binding once means every allocation, stream and
        kernel that thread issues lands on the right GPU with no further ceremony.
        """
        self.require(device)
        if device.is_cuda:
            require_torch().cuda.set_device(device.index)
        _THREAD_DEVICE.device = device
        _LOG.debug("thread %s bound to %s", threading.current_thread().name, device)

    @contextmanager
    def activate(self, device: Device) -> Iterator[Device]:
        """Temporarily make ``device`` current. Set-up code only, not the hot path."""
        previous = current_device()
        self.bind_current_thread(device)
        try:
            yield device
        finally:
            if previous is not None:
                self.bind_current_thread(previous)

    def synchronize(self, device: Device | None = None) -> None:
        if not self.has_accelerator:
            return
        torch = require_torch()
        torch.cuda.synchronize(device.index if device and device.is_cuda else None)

    def empty_cache(self) -> None:
        """Return torch's cached blocks to the driver.

        Almost never the right call: torch's caching allocator exists precisely so that
        freed blocks are *not* handed back, and dropping the cache makes the next
        allocations synchronise. Useful only when another process must be given room.
        """
        if self.has_accelerator:
            require_torch().cuda.empty_cache()

    def memory_stats(self, device: Device) -> dict[str, int]:
        """Torch allocator statistics for one device — reserved, allocated, retries.

        ``num_alloc_retries`` climbing is the signal that a pool is fragmented or a batch
        size is too large; it is the number to watch before OOM actually happens.
        """
        if not device.is_cuda:
            return {}
        stats = require_torch().cuda.memory_stats(device.index)
        return {
            "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
            "reserved_bytes": int(stats.get("reserved_bytes.all.current", 0)),
            "active_blocks": int(stats.get("active.all.current", 0)),
            "alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "ooms": int(stats.get("num_ooms", 0)),
        }

    def describe(self) -> str:
        return describe()

    def __repr__(self) -> str:
        return f"<DeviceManager kind={self.kind.value} visible={list(self._visible)}>"


def current_device() -> Device | None:
    """The device this thread is bound to, or ``None`` if it never was."""
    return getattr(_THREAD_DEVICE, "device", None)


def bind_thread(device: Device) -> None:
    """Bind without a manager. For worker threads spawned outside the server."""
    if device.is_cuda:
        require_torch().cuda.set_device(device.index)
    _THREAD_DEVICE.device = device
