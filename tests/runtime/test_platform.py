"""Device discovery, streams and the memory facade."""

from __future__ import annotations

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings import DeviceSettings, MemorySettings
from shipinfer.core.types import Device, MemoryKind
from shipinfer.runtime.device import DeviceManager, current_device
from shipinfer.runtime.memory import MemoryPool
from shipinfer.runtime.platform import AcceleratorKind, accelerator_kind, device_count


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
        staging = pool.staging_for("instance-0")
        first = staging.get("images", (4, 8), torch.float32)
        second = staging.get("images", (4, 8), torch.float32)
        assert first is second
        assert first.is_pinned()
        assert staging.stats()["hits"] >= 1
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


class TestDeviceDiscovery:
    """What the runtime reports about this host, and how it refuses a device it cannot see."""

    def test_accelerator_kind_is_reported(self) -> None:
        assert accelerator_kind() in set(AcceleratorKind)

    def test_manager_accepts_a_cpu_only_host(self) -> None:
        manager = DeviceManager(DeviceSettings(visible_gpus=[], allow_cpu_only=True))
        assert manager.describe()

    def test_requesting_an_absent_device_fails_at_construction(self) -> None:
        """Fail-fast: a missing device found at start-up is a config error with a clear
        message; found at the first inference it is a CUDA error three layers from the cause."""
        with pytest.raises(ConfigurationError, match="visible_gpus"):
            DeviceManager(DeviceSettings(visible_gpus=[device_count() + 99]))


class TestHostMemory:
    """Host buffers round-trip through the pool and are released when borrowed."""

    def test_host_allocator_round_trips(self) -> None:
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

    def test_borrow_releases_on_exit(self) -> None:
        pool = MemoryPool(MemorySettings())
        with pool.borrow(256, MemoryKind.HOST, Device.cpu()) as buffer:
            assert not buffer.is_freed
        assert buffer.is_freed
        pool.close()


class TestDeviceAffinity:
    """A buffer requested for one GPU is allocated on that GPU and no other."""

    @pytest.mark.multigpu
    def test_allocations_land_on_the_requested_device(self) -> None:
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


class TestSharingIsKeyedByDevice:
    """`DeviceSettings.shared_by` is aligned with the visible devices; the manager turns it into
    the mapping `InstanceGroup.expand` divides by. Offline: the driver is a stub."""

    @pytest.fixture()
    def two_devices(self, monkeypatch):
        import shipinfer.runtime.device as device_module

        monkeypatch.setattr(device_module, "device_count", lambda: 2)

    def test_aligned_with_the_visible_devices(self, two_devices) -> None:
        manager = DeviceManager(DeviceSettings(shared_by=[2, 1], validate_on_start=False))
        assert manager.shared_by == {0: 2, 1: 1}

    def test_aligned_with_an_explicit_visible_list(self, two_devices) -> None:
        manager = DeviceManager(
            DeviceSettings(visible_gpus=[1], shared_by=[3], validate_on_start=False)
        )
        assert manager.shared_by == {1: 3}

    def test_empty_means_nobody_shares(self, two_devices) -> None:
        assert DeviceManager(DeviceSettings(validate_on_start=False)).shared_by == {}

    def test_a_misaligned_list_is_refused(self, two_devices) -> None:
        manager = DeviceManager(DeviceSettings(shared_by=[2], validate_on_start=False))
        with pytest.raises(ConfigurationError, match="must align"):
            _ = manager.shared_by

    def test_the_rank_is_keyed_the_same_way(self, two_devices) -> None:
        manager = DeviceManager(
            DeviceSettings(shared_by=[2, 2], share_rank=[1, 0], validate_on_start=False)
        )
        assert manager.share_rank == {0: 1, 1: 0}
        assert DeviceManager(DeviceSettings(validate_on_start=False)).share_rank == {}

    def test_a_misaligned_rank_list_is_refused(self, two_devices) -> None:
        manager = DeviceManager(DeviceSettings(share_rank=[0], validate_on_start=False))
        with pytest.raises(ConfigurationError, match="must align"):
            _ = manager.share_rank
