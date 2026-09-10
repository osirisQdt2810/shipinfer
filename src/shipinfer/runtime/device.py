"""Device discovery, validation and thread binding — over ``torch.cuda``."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
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

__all__ = [
    "BLOCKING_SYNC_ENV",
    "DeviceManager",
    "bind_thread",
    "blocking_sync_requested",
    "current_device",
    "prefer_blocking_sync",
]

_LOG = get_logger("runtime.device")

#: `cudaDeviceScheduleBlockingSync` from `cuda_runtime_api.h`. A literal because this reaches
#: libcudart through `ctypes`, which has no header to read it from; the C++ plane uses the
#: symbol (`core/platform.h`) and a test ties the two so they cannot drift.
_CUDA_DEVICE_SCHEDULE_BLOCKING_SYNC = 0x04

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
        # doc: long why the knob is applied HERE and nowhere later
        # HERE, and this is the last point at which it can work: `_resolve_visible` calls
        # `device_count()`, which takes no context, and `_validate` below calls
        # `memory_info(index)` for every visible device -- `cudaMemGetInfo` under a device
        # guard, which INITIALISES that device's primary context. The driver refuses
        # `cudaSetDeviceFlags` once a context exists, so a call from anywhere further out
        # (`InferenceServer.start`, an instance's `start`) is a no-op that warns 216 per
        # device and leaves the threads spinning -- measured, and the reason this moved.
        if blocking_sync_requested():
            applied = prefer_blocking_sync(self._visible)
            _LOG.info("blocking synchronise on device(s) %s", list(applied) or "none")
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
    def share_rank(self) -> dict[int, int]:
        """This process's rank among those sharing each visible device; absent means 0."""
        ranks = self._settings.share_rank
        if not ranks:
            return {}
        if len(ranks) != len(self._visible):
            raise ConfigurationError(
                f"devices.share_rank has {len(ranks)} entr(y/ies) but {len(self._visible)} "
                f"device(s) are visible; the two lists must align"
            )
        return dict(zip(self._visible, ranks, strict=True))

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


#: The knob both planes read. `csrc/shipinfer/core/env.h` has the same three rules, and the
#: reason for the third is `docker run -e VAR`: an unset host variable is forwarded as EMPTY,
#: so empty has to mean "not asked for" or every container run flips a scheduling flag.
BLOCKING_SYNC_ENV = "SHIPINFER_CUDA_BLOCKING_SYNC"


def blocking_sync_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the operator asked for a blocking synchronise: set, non-empty, and not ``0``."""
    value = (os.environ if environ is None else environ).get(BLOCKING_SYNC_ENV)
    return value is not None and value != "" and value != "0"


# doc: long the flag, the ctypes route, and what the C++ plane measured
def prefer_blocking_sync(devices: Iterable[int]) -> tuple[int, ...]:
    """Ask each device's synchronise to BLOCK rather than spin. Before its first CUDA call.

    CUDA's default is `cudaDeviceScheduleAuto`, which spins when the active contexts do not
    outnumber the logical processors. The C++ plane measured what that costs at the design
    load (`THE-INSTANCE-THREADS-SPIN-ON-cudaStreamSynchronize`): the model-instance threads'
    host CPU HALVED, total host CPU fell 24%, the pipeline workers got 32% more because the
    spin had been starving them, and events rose ~15%. This plane's instance threads wait in
    the same place, so they owe the same knob.

    `ctypes` into libcudart because torch exposes no wrapper -- with the prototypes DECLARED,
    since an undeclared call truncates a pointer-sized handle and segfaults, which no
    `try/except` can catch. Best effort otherwise: a diagnostic knob must not be able to stop
    a server, so a device that refuses (it already has a context: `cudaErrorSetOnActiveProcess`
    = 216) is skipped and reported rather than raised.

    Returns:
        The devices the flag was actually applied to, in order.
    """
    wanted = list(devices)
    if not wanted:
        return ()  # a CPU-only host must not dlopen the driver's runtime to learn that
    # doc: long the three names, and which images each one is there for
    # `find_library` first, then the SONAME, then the dev symlink -- the same shape
    # `core/thread_name.py` uses for libc, for the same reason: the name on the developer's
    # box is not the name where the measurement is taken. MEASURED in the three images this
    # repository runs: `shipinfer-gst:jammy` and `:jammy-nvdec` (where every benchmark runs)
    # resolve `cudart` to `libcudart.so.12` and load all three names, while
    # `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` (the offline TEST image) has none of
    # them on the loader path -- torch keeps its own under `torch/lib/`. That image runs no
    # GPU benchmark, so the warning below is the right outcome there rather than a failure.
    candidates = [ctypes.util.find_library("cudart"), "libcudart.so.12", "libcudart.so"]
    libcudart = None
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            libcudart = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    if libcudart is None:
        _LOG.warning(
            "%s asked for, but libcudart is not loadable here (tried %s)",
            BLOCKING_SYNC_ENV,
            [name for name in candidates if name],
        )
        return ()
    try:
        libcudart.cudaSetDevice.restype = ctypes.c_int
        libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
        libcudart.cudaSetDeviceFlags.restype = ctypes.c_int
        libcudart.cudaSetDeviceFlags.argtypes = [ctypes.c_uint]
    except AttributeError:  # pragma: no cover - a libcudart without the symbols
        _LOG.warning("%s asked for, but libcudart has no cudaSetDeviceFlags", BLOCKING_SYNC_ENV)
        return ()

    applied: list[int] = []
    for index in wanted:
        if libcudart.cudaSetDevice(index) != 0:
            continue
        status = libcudart.cudaSetDeviceFlags(_CUDA_DEVICE_SCHEDULE_BLOCKING_SYNC)
        if status == 0:
            applied.append(index)
        else:
            _LOG.warning(
                "device %d already has a context, so its synchronise keeps spinning "
                "(cudaSetDeviceFlags returned %d); ask for %s before the first CUDA call",
                index,
                status,
                BLOCKING_SYNC_ENV,
            )
    return tuple(applied)
