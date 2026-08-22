"""Device discovery, streams and the memory facade."""

from __future__ import annotations

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings import DeviceSettings, MemorySettings
from shipinfer.core.types import Device, MemoryKind
from shipinfer.runtime.device import DeviceManager, current_device
from shipinfer.runtime.memory import MemoryPool
from shipinfer.runtime.platform import AcceleratorKind, accelerator_kind, device_count


def test_accelerator_kind_is_reported() -> None:
    assert accelerator_kind() in set(AcceleratorKind)


def test_manager_accepts_a_cpu_only_host() -> None:
    manager = DeviceManager(DeviceSettings(visible_gpus=[], allow_cpu_only=True))
    assert manager.describe()


def test_requesting_an_absent_device_fails_at_construction() -> None:
    """Fail-fast: a missing device found at start-up is a config error with a clear
    message; found at the first inference it is a CUDA error three layers from the cause."""
    with pytest.raises(ConfigurationError, match="visible_gpus"):
        DeviceManager(DeviceSettings(visible_gpus=[device_count() + 99]))


def test_host_allocator_round_trips() -> None:
    pool = MemoryPool(MemorySettings())
    buffer = pool.host.allocate(1024)
    try:
        assert buffer.nbytes == 1024
        assert buffer.kind is MemoryKind.HOST
        view = buffer.as_array()
        view[:] = 7
        assert int(view[0]) == 7
    finally:
        buffer.free()
        pool.close()


def test_borrow_releases_on_exit() -> None:
    pool = MemoryPool(MemorySettings())
    with pool.borrow(256, MemoryKind.HOST, Device.cpu()) as buffer:
        assert not buffer.is_freed
    assert buffer.is_freed
    pool.close()


@pytest.mark.gpu
class TestGpu:
    def test_manager_sees_devices(self) -> None:
        manager = DeviceManager(DeviceSettings())
        assert manager.has_accelerator
        assert len(manager.devices()) == device_count()
        free, total = manager.memory_info(manager.devices()[0])
        assert 0 < free <= total

    def test_binding_a_thread_is_recorded(self) -> None:
        manager = DeviceManager(DeviceSettings())
        device = manager.devices()[0]
        manager.bind_current_thread(device)
        assert current_device() == device

    def test_device_allocator_and_stats(self) -> None:
        manager = DeviceManager(DeviceSettings())
        device = manager.devices()[0]
        pool = MemoryPool(MemorySettings())
        buffer = pool.device(device).allocate(1 << 20)
        try:
            assert buffer.ptr != 0
            assert buffer.kind is MemoryKind.DEVICE
            assert pool.report(device).total > 0
        finally:
            buffer.free()
            pool.close()

    def test_pinned_allocator_gives_a_host_view(self) -> None:
        pool = MemoryPool(MemorySettings())
        buffer = pool.pinned.allocate(4096)
        try:
            assert buffer.kind is MemoryKind.PINNED
            buffer.as_array()[:] = 3
        finally:
            buffer.free()
            pool.close()

    def test_staging_pool_reuses_a_buffer(self) -> None:
        """Stable addresses are the CUDA-graph precondition, so reuse is load-bearing."""
        import torch

        pool = MemoryPool(MemorySettings())
        first = pool.staging.get((4, 8), torch.float32)
        second = pool.staging.get((4, 8), torch.float32)
        assert first is second
        assert first.is_pinned()
        assert pool.staging.stats()["hits"] >= 1
        pool.close()

    def test_streams_are_distinct_and_synchronise(self) -> None:
        from shipinfer.runtime.stream import StreamPool

        device = DeviceManager(DeviceSettings()).devices()[0]
        streams = StreamPool(device, 3)
        try:
            handles = {s.handle for s in streams.streams}
            assert len(handles) == 3
            with streams.borrow() as stream:
                assert stream.handle != 0
        finally:
            streams.close()


@pytest.mark.multigpu
def test_allocations_land_on_the_requested_device() -> None:
    """The invariant the whole worker model rests on: a buffer for cuda:1 is on cuda:1."""
    import torch

    pool = MemoryPool(MemorySettings())
    try:
        for index in (0, 1):
            buffer = pool.device(Device.cuda(index)).allocate(1 << 20)
            tensor = buffer.owner_object
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.index == index
            buffer.free()
    finally:
        pool.close()
