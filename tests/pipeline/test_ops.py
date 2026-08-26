"""Per-thread image ops: one instance per worker, one device per instance.

These three properties are the fix for three GPU faults that all report as something else
(see :mod:`shipinfer.pipeline.graph.ops`), and all three are assertable with a counting fake
and no driver at all — which is the point of testing them here rather than in the GPU tier.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from shipinfer.pipeline.graph.ops import ThreadLocalImageOps, staging_owner
from shipinfer.runtime.ops import NormalizeParams
from shipinfer.runtime.ops.numpy_ops import NumpyImageOps

pytestmark = pytest.mark.timeout(30)


class CountingOps(NumpyImageOps):
    """Numpy ops that remember which device they were built for, and by which thread."""

    def __init__(self, device: int) -> None:
        self.device = device
        self.built_by = threading.current_thread().name
        self.crops = 0

    def crop_batch(self, image, boxes, dst_size, params):
        self.crops += 1
        return super().crop_batch(image, boxes, dst_size, params)


def run_in_threads(ops: ThreadLocalImageOps, count: int, work) -> None:
    threads = [
        threading.Thread(target=work, args=(ops,), name=f"worker-{index}")
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)


class TestEachThreadGetsItsOwnInstance:
    """An ImageOps owns a staging ring; sharing one across workers corrupts it mid-DMA."""

    def test_two_threads_never_see_the_same_instance(self):
        built: list[CountingOps] = []
        lock = threading.Lock()

        def factory(device: int) -> CountingOps:
            instance = CountingOps(device)
            with lock:
                built.append(instance)
            return instance

        ops = ThreadLocalImageOps(factory, devices=(0,))
        seen: list[int] = []

        def work(shared: ThreadLocalImageOps) -> None:
            shared.crop_batch(
                np.zeros((8, 8, 3), np.uint8),
                np.array([[0, 0, 4, 4]], np.float32),
                (4, 4),
                NormalizeParams(),
            )
            with lock:
                seen.append(id(shared._ops))

        run_in_threads(ops, 4, work)

        assert len(built) == 4
        assert len(set(seen)) == 4
        assert all(instance.crops == 1 for instance in built)

    def test_an_instance_is_built_by_the_thread_that_uses_it(self):
        """A CUDA context belongs to the thread that created it, so eager construction from
        ``start()`` would bind the wrong one."""
        built: list[CountingOps] = []
        ops = ThreadLocalImageOps(lambda device: built.append(CountingOps(device)) or built[-1])

        def work(shared: ThreadLocalImageOps) -> None:
            shared.on_device  # noqa: B018 - first touch is what builds the instance

        run_in_threads(ops, 2, work)

        assert {instance.built_by for instance in built} == {"worker-0", "worker-1"}

    def test_one_thread_builds_exactly_once(self):
        calls = []
        ops = ThreadLocalImageOps(lambda device: calls.append(device) or CountingOps(device))
        for _ in range(5):
            ops.crop_batch(
                np.zeros((8, 8, 3), np.uint8),
                np.zeros((0, 4), np.float32),
                (4, 4),
                NormalizeParams(),
            )
        assert calls == [0]


class TestPreprocessingIsSpreadAcrossDevices:
    """Letterboxing every frame on cuda:0 re-creates this project's founding bug one layer up."""

    def test_threads_are_assigned_devices_round_robin(self):
        ops = ThreadLocalImageOps(CountingOps, devices=(0, 1, 2, 3))

        def work(shared: ThreadLocalImageOps) -> None:
            shared.on_device  # noqa: B018 - first touch assigns the device

        run_in_threads(ops, 8, work)

        assert ops.assignments() == {0: 2, 1: 2, 2: 2, 3: 2}

    def test_a_single_device_host_still_works(self):
        ops = ThreadLocalImageOps(CountingOps)
        assert ops.on_device is False
        assert ops.assignments() == {0: 1}

    def test_an_empty_device_list_falls_back_to_zero(self):
        """A GPU-less host reports no visible devices, and must not divide by zero."""
        ops = ThreadLocalImageOps(CountingOps, devices=())
        assert ops.on_device is False
        assert ops.assignments() == {0: 1}


class TestStagingOwners:
    """The pinned pool is keyed by this string, so two live threads must never produce one.

    A shared key is a shared buffer, and a shared buffer is one worker's crops overwritten by
    another's mid-DMA: plausible pixels under the wrong camera's tag, with no error to notice.
    """

    def test_every_worker_gets_its_own_key(self):
        """The barrier is the test, not scenery: keys only have to differ between threads
        that are *alive together*, and without it these eight are short enough that the
        interpreter hands the same identity to the next one."""
        owners: list[str] = []
        lock = threading.Lock()
        together = threading.Barrier(8)

        def work(_shared) -> None:
            together.wait(5.0)
            with lock:
                owners.append(staging_owner(2))

        run_in_threads(ThreadLocalImageOps(CountingOps), 8, work)

        assert len(owners) == 8
        assert len(set(owners)) == 8
        assert all("cuda:2" in owner for owner in owners)

    def test_two_live_threads_with_the_same_name_still_differ(self):
        """Two ``PipelineRunner`` instances over one server both name their workers
        ``pipeline-worker-0``; the name alone would collide, which is why the identity is in
        the key as well."""
        owners: list[str] = []
        lock = threading.Lock()
        together = threading.Barrier(2)

        def record() -> None:
            together.wait(5.0)
            with lock:
                owners.append(staging_owner(0))

        threads = [threading.Thread(target=record, name="pipeline-worker-0") for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)

        assert len(set(owners)) == 2

    def test_one_thread_keeps_one_key(self):
        """The pool is looked up per instance, not cached — the same thread asking twice has
        to get the same buffer back, or reuse is not reuse."""
        assert staging_owner(1) == staging_owner(1)
        assert staging_owner(1) != staging_owner(3), "the device belongs in the key"


class TestItIsStillAnImageOps:
    """Every stage takes an ImageOps and must not know threads exist."""

    def test_it_delegates_the_whole_contract(self):
        ops = ThreadLocalImageOps(CountingOps)
        image = np.zeros((8, 12, 3), np.uint8)

        letterboxed = ops.letterbox_batch([image], (16, 16), NormalizeParams())
        crops = ops.crop_batch(
            image, np.array([[0, 0, 6, 6]], np.float32), (4, 4), NormalizeParams()
        )
        kept = ops.nms(
            np.array([[0, 0, 10, 10]], np.float32), np.array([0.9], np.float32), 0.5, 0.1, 10
        )

        assert letterboxed.tensor.shape == (1, 3, 16, 16)
        assert crops.shape == (1, 3, 4, 4)
        assert kept.tolist() == [0]
        assert "thread-local" in ops.describe()
