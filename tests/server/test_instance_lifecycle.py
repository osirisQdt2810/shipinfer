"""Shutting a model instance down must never free a backend that is still in use.

``stop()`` used to join with a timeout, log a warning when the worker was still running,
and then finalise the backend anyway. For TensorRT that destroys the execution context and
every binding tensor while a batch is inside ``execute_async_v3`` — a use-after-free that
surfaces as corrupted output or a driver fault somewhere unrelated, long after the cause.

Leaking the backend is strictly better, so a failed join now abandons it and says so.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import wait

import numpy as np

from shipinfer.core.metrics import ServerMetrics
from shipinfer.core.request import InferenceRequest, RequestContext, ResponseFuture
from shipinfer.core.settings import SchedulerSettings
from shipinfer.core.types import DataType, Device, Tensor, TensorSpec
from shipinfer.runtime.device import DeviceManager
from shipinfer.scheduling.batching import StackingBatcher
from shipinfer.scheduling.queues import BatchWindow, FairPriorityQueue
from shipinfer.scheduling.work import WorkItem
from shipinfer.server.instance import ModelInstance

INPUTS = (TensorSpec("x", DataType.FP32, (4,)),)
OUTPUTS = (TensorSpec("y", DataType.FP32, (4,)),)


class SpyBackend:
    """A backend that can be made to block, and that records its own teardown."""

    def __init__(self, *, block: threading.Event | None = None) -> None:
        self.finalized = 0
        self.initialized = 0
        self._block = block
        self.device = Device.cpu()
        self.context = _Ctx()

    def initialize(self) -> None:
        self.initialized += 1

    def warmup(self, iterations: int) -> None:
        return None

    def finalize(self) -> None:
        self.finalized += 1

    def execute(self, inputs, batch_size):
        if self._block is not None:
            # Blocks until the test releases it, standing in for a batch still inside a
            # TensorRT enqueue when shutdown begins.
            self._block.wait(timeout=30)
        return {"y": Tensor.from_numpy(np.zeros((batch_size, 4), dtype=np.float32))}

    def stats(self) -> dict:
        return {}


class _Artifact:
    name = "m"
    version = 1


class _Execution:
    warmup_iterations = 0


class _Ctx:
    artifact = _Artifact()
    execution = _Execution()
    instance_name = "m:1@cpu"


def _instance(backend: SpyBackend) -> ModelInstance:
    return ModelInstance(
        name="m_0_cpu",
        backend=backend,
        queue=FairPriorityQueue("m_0_cpu", capacity=8),
        batcher=StackingBatcher(INPUTS, OUTPUTS, max_batch_size=4),
        window=BatchWindow(max_batch_size=4),
        devices=DeviceManager(),
        metrics=ServerMetrics(),
        scheduler=SchedulerSettings(),
    )


def _item() -> WorkItem:
    request = InferenceRequest(
        model_name="m",
        inputs={"x": Tensor.from_numpy(np.zeros((1, 4), dtype=np.float32))},
        context=RequestContext(camera_id="cam0", frame_id=0),
    )
    return WorkItem(request, ResponseFuture(request))


class TestBackendTeardownOnStop:
    """Whether stop() may finalize the backend depends on whether the worker actually left."""

    def test_a_clean_stop_finalizes_the_backend(self) -> None:
        backend = SpyBackend()
        instance = _instance(backend)
        instance.start()
        assert instance.wait_ready(10)

        instance.stop(grace_s=5.0)

        assert backend.finalized == 1
        assert not instance.is_abandoned

    def test_a_stuck_worker_is_abandoned_not_finalized(self) -> None:
        """The regression test: finalising here would free memory the GPU is still reading."""
        block = threading.Event()
        backend = SpyBackend(block=block)
        instance = _instance(backend)
        instance.start()
        assert instance.wait_ready(10)

        instance.enqueue(_item())
        time.sleep(0.2)  # let the worker pick it up and block inside execute

        instance.stop(grace_s=0.3)  # shorter than the batch will take

        assert instance.is_abandoned
        assert backend.finalized == 0, "the backend was torn down under a running batch"
        assert instance.stats()["abandoned"] is True

        block.set()  # let the worker finish so the test does not leak it

    def test_stop_is_idempotent(self) -> None:
        backend = SpyBackend()
        instance = _instance(backend)
        instance.start()
        assert instance.wait_ready(10)

        instance.stop(grace_s=5.0)
        instance.stop(grace_s=5.0)
        instance.stop(grace_s=5.0)

        assert backend.finalized == 1, "finalize ran more than once"


class TestQueuedWorkOnStop:
    """Work the instance still owns when it stops resolves, rather than hanging forever."""

    def test_queued_work_is_failed_on_stop(self) -> None:
        """Everything the instance still owns resolves; nothing is left pending forever."""
        block = threading.Event()
        backend = SpyBackend(block=block)
        instance = _instance(backend)
        instance.start()
        assert instance.wait_ready(10)

        queued = [_item() for _ in range(3)]
        instance.enqueue(queued[0])
        time.sleep(0.2)  # this one is in flight
        for item in queued[1:]:
            instance.enqueue(item)

        instance.stop(grace_s=0.3)

        # The two still in the queue are failed by close(); the in-flight one belongs to the
        # worker and is resolved by it, which is why the test releases the block and waits.
        done, not_done = wait([i.future for i in queued[1:]], timeout=5)
        assert not not_done
        assert all(f.exception() is not None for f in done)

        block.set()
        wait([queued[0].future], timeout=10)


class TestReadinessMetric:
    """The readiness gauge tracks reality, including for an instance that never started."""

    def test_readiness_gauge_is_not_decremented_below_zero(self) -> None:
        """An instance that never became ready must not decrement a gauge it never
        incremented — a negative "instances ready" is worse than no metric at all."""
        metrics = ServerMetrics()
        backend = SpyBackend()
        instance = ModelInstance(
            name="m_0_cpu",
            backend=backend,
            queue=FairPriorityQueue("m_0_cpu", capacity=8),
            batcher=StackingBatcher(INPUTS, OUTPUTS, max_batch_size=4),
            window=BatchWindow(max_batch_size=4),
            devices=DeviceManager(),
            metrics=metrics,
            scheduler=SchedulerSettings(),
        )

        instance.stop()  # never started

        assert metrics.instances_ready.value(model="m") == 0
