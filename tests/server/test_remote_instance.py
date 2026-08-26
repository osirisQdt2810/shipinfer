"""A peer's instance through a ring, offline: two "processes" as objects in one, the rings real.

The owner side is a `RingIngress` over a fake model whose `infer` computes in a thread; the
submitter side is a `RemoteInstance` behind a real `Dispatcher` next to a fake local instance.
The rings are real shared memory, so a thread here sees exactly what a peer process would.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future

import numpy as np
import pytest

from shipinfer.core.errors import (
    PeerLostError,
    QueueFullError,
    RingClosedError,
    RingFullError,
    ServerStateError,
)
from shipinfer.core.request import InferenceRequest, InferenceResponse, RequestContext
from shipinfer.core.request.future import ResponseFuture
from shipinfer.core.types import Device, Tensor
from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing
from shipinfer.scheduling.dispatcher import Dispatcher
from shipinfer.scheduling.policies.locality_spillover import LocalityAwareSpilloverPolicy
from shipinfer.scheduling.work import WorkItem
from shipinfer.server.remote_instance import RemoteInstance, ResultReader, RingIngress


def _name(tag: str) -> str:
    return f"shipinfer-test-{tag}-{uuid.uuid4().hex[:10]}"


class FakeModel:
    """Doubles every input in a worker thread; fails when told to."""

    def __init__(self, *, fail: bool = False, delay_s: float = 0.0) -> None:
        self.fail = fail
        self.delay_s = delay_s
        self.seen: list[str] = []
        self.depth = 0

    def infer(self, request: InferenceRequest) -> Future:
        self.seen.append(request.context.camera_id)
        future: Future = Future()
        if self.fail:
            raise ValueError("engine refused the batch")

        def work() -> None:
            time.sleep(self.delay_s)
            outputs = {
                name: Tensor.from_numpy(t.numpy() * 2) for name, t in request.inputs.items()
            }
            response = InferenceResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version=1,
                outputs=outputs,
                context=request.context,
                timings=request.timings,
                executed_on=Device.cuda(3),
            )
            future.set_result(response)

        threading.Thread(target=work, daemon=True).start()
        return future


class FakeLocal:
    """A local instance that is always deep: the policy spills away from it."""

    def __init__(self, depth: int = 100, full: bool = True) -> None:
        self._depth = depth
        self.full = full
        self.enqueued: list[WorkItem] = []

    device = Device.cuda(0)
    ewma_latency_us = 1000.0
    is_ready = True

    @property
    def depth(self) -> int:
        return self._depth

    def enqueue(self, item: WorkItem) -> None:
        if self.full:
            raise QueueFullError("local", self._depth, self._depth)
        self.enqueued.append(item)


@pytest.fixture()
def pair():
    """Submitter A, owner B, one model: the request ring (B owns, A writes) and the result
    ring (A owns, B writes), plus the two threads."""
    layout = RingLayout(slots=4, slot_bytes=64 * 1024)
    request_ring_owner = SharedRing.create(_name("req"), layout, owner="B")
    request_ring_writer = SharedRing.open(request_ring_owner.name, layout)
    result_ring_owner = SharedRing.create(_name("res"), layout, owner="A")
    result_ring_writer = SharedRing.open(result_ring_owner.name, layout)
    model = FakeModel()
    ingress = RingIngress(
        submitter="A",
        inbound=request_ring_owner,
        results=result_ring_writer,
        infer=model.infer,
        load=lambda: (model.depth, 500.0),
        stamp_every_s=0.05,
    )
    reader = ResultReader(lost_after_s=0.5)
    reader.add_result_ring("B", result_ring_owner)
    proxy = RemoteInstance(
        owner="B", model_name="m", submit=request_ring_writer, reader=reader, lost_after_s=0.5
    )
    ingress.start()
    reader.start()
    time.sleep(0.05)  # the ingress stamps once at start; the proxy is ready after that
    try:
        yield model, proxy, reader, ingress
    finally:
        reader.stop()
        ingress.stop()
        ingress.join(timeout=2)
        reader.join(timeout=2)
        for ring in (request_ring_writer, result_ring_writer, result_ring_owner):
            ring.close()
        with pytest.raises(RingClosedError):  # the ingress closed and unlinked its inbound ring
            SharedRing.open(request_ring_owner.name, layout)


def _request(camera: str, frame: int, rows: int = 3) -> InferenceRequest:
    return InferenceRequest(
        model_name="m",
        inputs={"x": Tensor.from_numpy(np.full((rows, 8), frame, dtype=np.float32))},
        context=RequestContext(camera_id=camera, frame_id=frame),
    )


class TestTheProxyIsAPlaceable:
    def test_it_reads_the_owners_load_off_the_header(self, pair) -> None:
        model, proxy, _, _ = pair
        assert proxy.is_ready
        assert proxy.depth == 0 and proxy.ewma_latency_us == 500.0
        model.depth = 7
        time.sleep(0.15)  # a stamp interval
        assert proxy.depth == 7

    def test_it_is_never_the_resident_device(self, pair) -> None:
        _, proxy, _, _ = pair
        assert proxy.device == Device.cpu() and not proxy.device.is_cuda


class TestARequestGoesRemoteAndComesBack:
    def test_end_to_end_through_the_dispatcher(self, pair) -> None:
        model, proxy, reader, _ = pair
        local = FakeLocal()
        dispatcher = Dispatcher(
            model_name="m",
            instances=[local, proxy],
            policy=LocalityAwareSpilloverPolicy(spill_threshold=4),
        )
        request = _request("quay-7", 42)
        request.resident_device = Device.cuda(0)
        future = ResponseFuture(request)
        result = dispatcher.dispatch(
            WorkItem(request, future), lambda inst, item: inst.enqueue(item)
        )

        # The local queue is past the spill threshold, so the policy's shortest-queue fallback
        # picks the proxy outright — no spill, the remote was simply the better candidate.
        assert result.instance is proxy
        response = future.result(timeout=2)
        assert response.executed_on == Device.cuda(3), "where it really ran comes back"
        assert (response.context.camera_id, response.context.frame_id) == ("quay-7", 42)
        np.testing.assert_array_equal(
            response.outputs["x"].numpy(), np.full((3, 8), 84, dtype=np.float32)
        )
        assert model.seen == ["quay-7"], "the owner queued it under the submitter's camera id"
        assert reader.pending == 0

    def test_many_requests_in_flight_at_once(self, pair) -> None:
        model, proxy, reader, _ = pair
        model.delay_s = 0.01
        futures = []
        for frame in range(20):  # five times the slot count: the ring fills and refuses
            request = _request("cam", frame)
            future = ResponseFuture(request)
            deadline = time.monotonic() + 5.0
            while True:  # what the dispatcher's spill loop does with the next candidate — here
                try:  # there is only one, so it waits for a release
                    proxy.enqueue(WorkItem(request, future))
                    break
                except RingFullError:
                    assert time.monotonic() < deadline, "the owner never released a slot"
                    time.sleep(0.002)
            futures.append((frame, future))
        for frame, future in futures:
            np.testing.assert_array_equal(
                future.result(timeout=5).outputs["x"].numpy()[0, 0], frame * 2
            )
        assert reader.pending == 0

    def test_the_owners_failure_fails_the_future_with_the_owners_error(self, pair) -> None:
        model, proxy, _, ingress = pair
        model.fail = True
        request = _request("cam", 1)
        future = ResponseFuture(request)
        proxy.enqueue(WorkItem(request, future))
        with pytest.raises(
            ServerStateError, match="remote 'B': ValueError: engine refused the batch"
        ):
            future.result(timeout=2)
        assert ingress.failed == 1


class TestBackpressureAndLoss:
    def test_a_full_ring_is_a_queue_full_error_the_dispatcher_can_spill_on(self, pair) -> None:
        model, proxy, _, _ingress = pair
        model.delay_s = 10.0  # the owner holds every slot it takes until the work settles
        for frame in range(4):
            proxy.enqueue(
                WorkItem(_request("cam", frame), ResponseFuture(_request("cam", frame)))
            )
        with pytest.raises(RingFullError) as caught:
            proxy.enqueue(WorkItem(_request("cam", 9), ResponseFuture(_request("cam", 9))))
        assert isinstance(caught.value, QueueFullError)
        assert (caught.value.depth, caught.value.capacity) == (4, 4)

        # ...and through the dispatcher the spill goes back to a local instance that has room.
        local = FakeLocal(depth=0, full=False)
        dispatcher = Dispatcher(
            model_name="m",
            instances=[proxy, local],
            policy=LocalityAwareSpilloverPolicy(spill_threshold=0),
        )
        request = _request("cam", 10)
        result = dispatcher.dispatch(
            WorkItem(request, ResponseFuture(request)), lambda inst, item: inst.enqueue(item)
        )
        assert result.instance is local

    def test_a_lost_owner_fails_its_requests_with_their_tags(self, pair) -> None:
        model, proxy, reader, ingress = pair
        model.delay_s = 10.0  # the owner "hangs" on the work...
        request = _request("quay-3", 77)
        future = ResponseFuture(request)
        proxy.enqueue(WorkItem(request, future))
        ingress.stop()  # ...and stops stamping its heartbeat
        ingress.join(timeout=2)
        with pytest.raises(PeerLostError) as caught:
            future.result(timeout=3)
        assert caught.value.owner == "B" and caught.value.tags == (("quay-3", 77),)
        assert not proxy.is_ready
        assert reader.pending == 0


class TestRepliesThatCannotLand:
    """A reply is held through a burst and dropped only for a corpse — never lost silently."""

    def _ingress(self, results_slots: int = 1, patience_s: float = 0.05):
        layout = RingLayout(slots=4, slot_bytes=64 * 1024)
        inbound = SharedRing.create(_name("inb"), layout, owner="B")
        results_owner = SharedRing.create(
            _name("res"), RingLayout(slots=results_slots, slot_bytes=64 * 1024), owner="A"
        )
        results_writer = SharedRing.open(results_owner.name, results_owner.layout)
        ingress = RingIngress(
            submitter="A",
            inbound=inbound,
            results=results_writer,
            infer=FakeModel().infer,
            load=lambda: (0, 0.0),
            result_timeout_s=0.01,
            result_patience_s=patience_s,
        )
        return ingress, inbound, results_owner, results_writer

    def test_a_closed_result_ring_means_the_submitter_is_gone(self) -> None:
        ingress, inbound, results_owner, results_writer = self._ingress()
        results_owner.close()  # the submitter left; its reader failed its futures already
        ingress._reply_failure(7, "m", ValueError("late"))
        assert ingress.dropped == 1, "counted, not raised"
        inbound.close()
        results_writer.close()

    def test_a_full_result_ring_is_waited_out_then_given_up_with_a_count(self) -> None:
        ingress, inbound, results_owner, results_writer = self._ingress(
            results_slots=1, patience_s=0.05
        )
        blocker = results_writer.claim(timeout_s=0.1)  # the one slot, held for the whole test
        started = time.monotonic()
        ingress._reply_failure(9, "m", ValueError("burst"))
        waited = time.monotonic() - started
        assert ingress.dropped == 1
        assert waited >= 0.05, "the patience was actually spent waiting, not skipped"
        results_writer.abandon(blocker)
        results_owner.close()
        inbound.close()
        results_writer.close()
