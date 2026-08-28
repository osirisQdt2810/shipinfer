"""Per-thread image ops: one instance per worker, one device per instance.

These three properties are the fix for three GPU faults that all report as something else
(see :mod:`shipinfer.runtime.ops.thread_local`), and all three are assertable with a counting
fake and no driver at all — which is the point of testing them here rather than in the GPU
tier.

Moved with the module out of ``tests/pipeline/``: the class is an ``ImageOps`` decorator that
has never imported anything from ``pipeline``, and it now has three callers — the pipeline
runner, ``shipinfer run`` and ``shipinfer-shard``. This file is the parity suite for all
three, and the wiring of each is asserted where it lives.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from shipinfer.core.errors import DeviceError
from shipinfer.runtime.ops import NormalizeParams
from shipinfer.runtime.ops.numpy_ops import NumpyImageOps
from shipinfer.runtime.ops.thread_local import ThreadLocalImageOps, staging_owner

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


class TestTheLedgerCountsDelegatesThatExist:
    """``assignments()`` is what a ``-m multigpu`` run is read for, so it must not overcount.

    The ledger used to be written *before* the factory ran, so a build that raised still
    credited its device with a delegate that does not exist -- and "the spread is even" then
    reads as true of a process where half the constructions failed. There is nothing else to
    catch that: a failed build raises into the worker, which dies, and the number left behind
    is the only account of what happened.
    """

    def test_a_failed_build_leaves_the_device_uncredited(self):
        def explode(device_index: int):
            raise DeviceError(f"no context on cuda:{device_index}")

        ops = ThreadLocalImageOps(explode, devices=(0, 1))

        for _ in range(3):
            with pytest.raises(DeviceError):
                ops.on_device  # noqa: B018 - the first touch is the build

        assert ops.assignments() == {}, "a delegate that was never built was counted"
        assert "threads=0" in repr(ops)

    def test_the_cursor_still_moves_so_a_retry_lands_on_the_next_device(self):
        """The failure is not free: the *cursor* advances, and deliberately.

        A thread whose construction raised retries onto the next device rather than back onto
        the one that just refused it — on a box with one sick GPU that is the difference
        between a worker that comes up and a worker that cannot. What must not move is the
        ledger, which is the assertion above and the second one here.
        """
        attempted: list[int] = []

        def explode_once(device_index: int):
            attempted.append(device_index)
            if len(attempted) == 1:
                raise DeviceError("the first device said no")
            return CountingOps(device_index)

        ops = ThreadLocalImageOps(explode_once, devices=(0, 1))

        with pytest.raises(DeviceError):
            ops.on_device  # noqa: B018 - the failing build
        assert ops.on_device is False  # the retry, on the same thread

        assert attempted == [0, 1]
        assert ops.assignments() == {1: 1}, "only the delegate that exists is counted"


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


class TestTheOldImportPathStillResolvesToTheSameObjects:
    """``pipeline`` imports this by its old name in two places, and both re-export it.

    A copy rather than a re-export would give the tree two ``ThreadLocalImageOps`` classes,
    and ``isinstance`` across the two paths would start answering ``False`` — the exact
    failure the detections shim was written to avoid one slice earlier.
    """

    def test_the_shim_hands_back_the_same_class_and_function(self):
        import shipinfer.pipeline.graph as graph
        from shipinfer.pipeline.graph import ops as shim

        assert shim.ThreadLocalImageOps is ThreadLocalImageOps
        assert shim.staging_owner is staging_owner
        assert graph.ThreadLocalImageOps is ThreadLocalImageOps


class TestTheCompositionRootsGetItInOneCall:
    """``get_thread_local_image_ops`` — the wiring ``run`` and ``shard`` would otherwise copy.

    It exists because the copy is not obvious: the delegate has to be built *on* the thread
    that will use it, bound to that thread's device first, and handed its own pinned pool.
    Three composition roots need exactly that, and two of them had none of it.
    """

    def test_it_spreads_threads_over_the_devices_it_is_given(self):
        from shipinfer.runtime.ops import get_thread_local_image_ops

        ops = get_thread_local_image_ops(devices=(0, 1, 2, 3))

        def work(shared) -> None:
            shared.on_device  # noqa: B018 - first touch assigns the device

        run_in_threads(ops, 8, work)

        assert ops.assignments() == {0: 2, 1: 2, 2: 2, 3: 2}

    def test_a_host_with_no_devices_resolves_numpy_on_index_zero(self):
        """``devices=()`` is what a CPU-only host reports, and it must not divide by zero."""
        from shipinfer.runtime.ops import get_thread_local_image_ops

        ops = get_thread_local_image_ops(devices=())

        assert ops.assignments() == {}
        assert ops.describe() == "thread-local over devices [0]"

    def test_no_device_manager_means_no_binding_and_no_staging(self, monkeypatch):
        """The offline shape: nothing is bound, nothing is claimed, numpy comes back.

        A caller with no manager must not reach a ``bind_current_thread`` or a
        ``staging_for``, because on a driverless host there is nothing on the other end of
        either — and a benchmark double that answers ``has_accelerator`` falsely would then
        take a CUDA path in the offline tier.
        """
        from shipinfer.runtime import ops as ops_module

        asked: list[tuple[int, object]] = []

        def fake_get_image_ops(provider, *, device_index=0, staging=None):
            asked.append((device_index, staging))
            return NumpyImageOps()

        monkeypatch.setattr(ops_module, "get_image_ops", fake_get_image_ops)
        built = ops_module.get_thread_local_image_ops(devices=(3,), memory=object())
        built.on_device  # noqa: B018 - first touch builds the delegate

        assert asked == [(3, None)]

    def test_a_caller_that_must_release_its_pools_claims_them_through_its_own_hook(
        self, monkeypatch
    ):
        """``claim=`` is the one line :class:`~shipinfer.pipeline.PipelineRunner` needs.

        It stops and starts inside one process, and the owner keys embed the worker's thread
        ident, so every cycle mints fresh ones: without recording them, ``stop()`` has nothing
        to release and each cycle strands its pinned pages for the server's life. That is a
        hook rather than a second copy of this wiring — the second copy had already drifted
        from this one in two places before it was removed.
        """
        from shipinfer.runtime import ops as ops_module

        class FakeManager:
            has_accelerator = True

            def bind_current_thread(self, device) -> None:
                pass

        class FakeMemory:
            def staging_for(self, owner: str) -> str:  # pragma: no cover - `claim` wins
                raise AssertionError("the hook was bypassed")

        recorded: list[tuple[object, str]] = []

        def claim(memory, owner: str) -> str:
            recorded.append((memory, owner))
            return owner

        monkeypatch.setattr(
            ops_module,
            "get_image_ops",
            lambda provider, *, device_index=0, staging=None: NumpyImageOps(),
        )
        memory = FakeMemory()
        built = ops_module.get_thread_local_image_ops(
            devices=(1,), device_manager=FakeManager(), memory=memory, claim=claim
        )
        built.on_device  # noqa: B018 - first touch builds the delegate

        assert [owner for _, owner in recorded] == [staging_owner(1)]
        assert recorded[0][0] is memory, "the hook is handed the pool it claims out of"

    def test_an_accelerated_manager_binds_the_thread_and_claims_its_own_pool(self, monkeypatch):
        """One instance per thread is only half the fix; the pinned pool has to split too.

        Two workers sharing one ``PinnedStagingPool`` is one buffer between two DMAs, which
        CONVENTIONS 2.8 says produces plausible output and no error at all. The owner key is
        per *thread*, so two threads on one device must still claim two pools.
        """
        from shipinfer.runtime import ops as ops_module

        bound: list[object] = []
        owners: list[str] = []
        lock = threading.Lock()

        class FakeManager:
            has_accelerator = True

            def bind_current_thread(self, device) -> None:
                with lock:
                    bound.append(device)

        class FakeMemory:
            def staging_for(self, owner: str) -> str:
                with lock:
                    owners.append(owner)
                return owner

        monkeypatch.setattr(
            ops_module,
            "get_image_ops",
            lambda provider, *, device_index=0, staging=None: NumpyImageOps(),
        )
        built = ops_module.get_thread_local_image_ops(
            devices=(2,), device_manager=FakeManager(), memory=FakeMemory()
        )

        def work(shared) -> None:
            shared.on_device  # noqa: B018 - first touch builds the delegate

        run_in_threads(built, 2, work)

        assert [device.index for device in bound] == [2, 2]
        assert len(set(owners)) == 2, "two workers on one device shared one pinned pool"
        assert all("cuda:2" in owner for owner in owners)
