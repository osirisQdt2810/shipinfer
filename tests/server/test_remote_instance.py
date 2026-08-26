"""A peer's instance through a ring, offline: two "processes" as objects in one, the rings real.

The owner side is a `RingIngress` over a fake model that admits once and computes on ONE
worker thread; the
submitter side is a `RemoteInstance` behind a real `Dispatcher` next to a fake local instance.
The rings are real shared memory, so a thread here sees exactly what a peer process would.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid

import numpy as np
import pytest

from shipinfer.core.errors import (
    PeerLostError,
    QueueFullError,
    RequestTimeoutError,
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
from shipinfer.server.remote_instance import (
    IngressLane,
    RemoteInstance,
    ResultReader,
    RingIngress,
)


def _name(tag: str) -> str:
    return f"shipinfer-test-{tag}-{uuid.uuid4().hex[:10]}"


class FakeModel:
    """Serves every request on ONE long-lived worker thread, like a real instance.

    The single worker is the property that matters (round 1's review): a completion callback
    that blocks on a ring blocks *this* thread, and the next request never settles — a fake
    that spawned a thread per request could never see that.
    """

    def __init__(self, *, fail: bool = False, delay_s: float = 0.0) -> None:
        self.fail = fail
        self.delay_s = delay_s
        self.seen: list[str] = []
        self.completed = 0
        self.depth = 0
        self.admitted = 0
        self.dispatches = 0
        self.rejections = 0
        self._work: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def admit(self, request: InferenceRequest) -> WorkItem:
        # First-entry work happens exactly once per request; the counter is the round-3
        # regression (a retry must never re-admit).
        self.admitted += 1
        self.seen.append(request.context.camera_id)
        if self.fail:
            raise ValueError("engine refused the batch")
        return WorkItem(request, ResponseFuture(request))

    def dispatch(self, item: WorkItem) -> bool:
        self.dispatches += 1
        self._work.put((item.request, item.future))
        return True

    def reject(self) -> None:
        self.rejections += 1

    def close(self) -> None:
        self._work.put(None)
        self._worker.join(timeout=2)

    def _run(self) -> None:
        while True:
            item = self._work.get()
            if item is None:
                return
            request, future = item
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
            self.completed += 1
            future.set_result(response)  # the done callback runs HERE, on the worker


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
        [
            IngressLane(
                submitter="A",
                inbound=request_ring_owner,
                results=result_ring_writer,
                admit=model.admit,
                dispatch=model.dispatch,
                reject=model.reject,
                load=lambda: (model.depth, 500.0),
            )
        ],
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
        model.close()
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
    """A reply is held through a burst and dropped only for a corpse — and the worker thread
    that settled the future never waits on a ring either way."""

    def _lane(self, results_slots: int = 1, patience_s: float = 0.15):
        layout = RingLayout(slots=4, slot_bytes=64 * 1024)
        inbound_owner = SharedRing.create(_name("inb"), layout, owner="B")
        inbound_writer = SharedRing.open(inbound_owner.name, layout)
        results_owner = SharedRing.create(
            _name("res"), RingLayout(slots=results_slots, slot_bytes=64 * 1024), owner="A"
        )
        results_writer = SharedRing.open(results_owner.name, results_owner.layout)
        model = FakeModel()
        lane = IngressLane(
            submitter="A",
            inbound=inbound_owner,
            results=results_writer,
            admit=model.admit,
            dispatch=model.dispatch,
            reject=model.reject,
            load=lambda: (0, 0.0),
        )
        ingress = RingIngress([lane], stamp_every_s=0.05, result_patience_s=patience_s)
        return ingress, model, inbound_writer, results_owner, results_writer

    def _submit(self, writer, camera: str, frame: int) -> None:
        index = writer.claim(timeout_s=1.0)
        wire_request = _request(camera, frame)
        from shipinfer.server import remote_wire

        remote_wire.encode_request(wire_request, writer.payload(index))
        writer.publish(index)

    def test_a_closed_result_ring_means_the_submitter_is_gone(self) -> None:
        ingress, model, inbound_writer, results_owner, results_writer = self._lane()
        results_owner.close()  # the submitter left; its reader failed its futures already
        self._submit(inbound_writer, "cam", 1)
        ingress.start()
        deadline = time.monotonic() + 5.0
        while ingress.dropped < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ingress.dropped == 1, "counted, not raised"
        ingress.stop()
        ingress.join(timeout=2)
        model.close()
        inbound_writer.close()
        results_writer.close()

    def test_a_full_result_ring_never_blocks_the_worker_and_expires_with_a_count(self) -> None:
        """Round 1's central scenario: the result ring is full, and the GPU worker thread —
        the one that settles futures — must keep serving anyway. The reply waits on the
        sweeper's queue and is dropped with a count when its patience runs out."""
        ingress, model, inbound_writer, results_owner, results_writer = self._lane(
            results_slots=1, patience_s=0.15
        )
        blocker = results_writer.claim(timeout_s=1.0)  # the one result slot, held throughout
        self._submit(inbound_writer, "cam", 1)
        self._submit(inbound_writer, "cam", 2)
        ingress.start()
        settle_deadline = time.monotonic() + 2.0
        while model.completed < 2 and time.monotonic() < settle_deadline:
            time.sleep(0.005)
        assert model.completed == 2, (
            "the single worker settled BOTH futures while the result ring was full — "
            "a blocking reply would have wedged it after the first"
        )
        drop_deadline = time.monotonic() + 5.0
        while ingress.dropped < 2 and time.monotonic() < drop_deadline:
            time.sleep(0.01)
        assert ingress.dropped == 2, "both replies expired with a count"
        results_writer.abandon(blocker)
        ingress.stop()
        ingress.join(timeout=2)
        model.close()
        inbound_writer.close()
        results_owner.close()
        results_writer.close()


class TestPendingEntriesExpire:
    def test_a_live_owner_that_never_answers_does_not_pin_the_work_item(self) -> None:
        """The owner is alive (fresh heartbeats) but dropped one reply: without a deadline
        the pending entry pins the request's arrays for the process's life."""
        layout = RingLayout(slots=2, slot_bytes=8192)
        submit_owner = SharedRing.create(_name("hb"), layout, owner="B")
        submit_writer = SharedRing.open(submit_owner.name, layout)
        reader = ResultReader(lost_after_s=10.0, pending_timeout_s=0.1)
        request = _request("quay-9", 4)
        item = WorkItem(request, ResponseFuture(request))
        reader.expect(item, "B")
        reader.watch("B", submit_writer)
        submit_owner.stamp(depth=0, ewma_latency_us=1.0)  # alive, just silent
        reader.start()
        try:
            with pytest.raises(RequestTimeoutError, match="never answered"):
                item.future.result(timeout=3.0)
            assert reader.pending == 0
        finally:
            reader.stop()
            reader.join(timeout=2)
            submit_writer.close()
            submit_owner.close()


class TestASaturatedOwnerPushesBack:
    class SaturatedThenWilling:
        """Admits normally; refuses the first N dispatch attempts, then accepts."""

        def __init__(self, refusals: int) -> None:
            self.refusals = refusals
            self.dispatch_attempts = 0
            self.inner = FakeModel()

        def admit(self, request: InferenceRequest) -> WorkItem:
            return self.inner.admit(request)

        def dispatch(self, item: WorkItem) -> bool:
            self.dispatch_attempts += 1
            if self.refusals > 0:
                self.refusals -= 1
                return False
            return self.inner.dispatch(item)

        def reject(self) -> None:
            self.inner.reject()

    def _tier(
        self,
        model,
        *,
        slots: int = 4,
        patience_s: float = 5.0,
        backoff_s: float = 0.005,
    ):
        layout = RingLayout(slots=slots, slot_bytes=64 * 1024)
        inbound_owner = SharedRing.create(_name("sat"), layout, owner="B")
        inbound_writer = SharedRing.open(inbound_owner.name, layout)
        results_owner = SharedRing.create(_name("satr"), layout, owner="A")
        results_writer = SharedRing.open(results_owner.name, layout)
        reader = ResultReader(lost_after_s=5.0)
        reader.add_result_ring("B", results_owner)
        proxy = RemoteInstance(
            owner="B", model_name="m", submit=inbound_writer, reader=reader, lost_after_s=5.0
        )
        ingress = RingIngress(
            [
                IngressLane(
                    submitter="A",
                    inbound=inbound_owner,
                    results=results_writer,
                    admit=model.admit,
                    dispatch=model.dispatch,
                    reject=model.reject,
                    load=lambda: (0, 0.0),
                )
            ],
            stamp_every_s=0.05,
            result_patience_s=patience_s,
            retry_backoff_s=backoff_s,
        )
        return ingress, reader, proxy, (inbound_writer, results_owner, results_writer)

    def _teardown(self, ingress, reader, model, rings) -> None:
        reader.stop()
        ingress.stop()
        ingress.join(timeout=2)
        reader.join(timeout=2)
        model.inner.close()
        for ring in rings:  # the inbound owner is closed by the ingress itself
            ring.close()

    def test_a_refused_request_is_retried_and_succeeds_not_failed_as_text(self) -> None:
        """Round 2: the owner's model-queue saturation must not leave the process as a text
        failure — the request waits with its slot held and runs when the queue drains."""
        model = self.SaturatedThenWilling(refusals=3)
        ingress, reader, proxy, rings = self._tier(model)
        ingress.start()
        reader.start()
        try:
            request = _request("quay-5", 11)
            future = ResponseFuture(request)
            proxy.enqueue(WorkItem(request, future))
            response = future.result(timeout=5.0)
            assert response.context.frame_id == 11, "retried until the queue drained — not text"
            assert ingress.failed == 0
            assert model.inner.admitted == 1, "retries are dispatch-only, never re-admission"
        finally:
            self._teardown(ingress, reader, model, rings)

    def test_a_saturated_lane_is_probed_not_spun_and_admitted_once(self) -> None:
        """Round 3: a still-refusing lane must not pin a core or re-enter first-entry work.

        Attempts over the window are bounded by window / retry_backoff — a spin makes
        tens of thousands, and each one would re-run validation and bump the counters.
        """
        model = self.SaturatedThenWilling(refusals=10_000_000)
        ingress, reader, proxy, rings = self._tier(model, backoff_s=0.01)
        ingress.start()
        reader.start()
        try:
            request = _request("quay-7", 3)
            proxy.enqueue(WorkItem(request, ResponseFuture(request)))
            time.sleep(0.3)
            assert model.inner.admitted == 1, "admission is first-entry work, once"
            # 0.3 s at a 10 ms probe is ~30 attempts plus the first; give scheduling slack.
            assert model.dispatch_attempts <= 60, model.dispatch_attempts
            assert ingress.failed == 0
        finally:
            self._teardown(ingress, reader, model, rings)

    def test_persistent_saturation_fills_the_ring_and_the_submitter_spills(self) -> None:
        """While the owner stays saturated the slots stay claimed, the ring fills, and the
        submitter's next enqueue raises RingFullError — the dispatcher's spill signal."""
        model = self.SaturatedThenWilling(refusals=10_000_000)
        ingress, reader, proxy, rings = self._tier(model, slots=2, patience_s=30.0)
        ingress.start()
        reader.start()
        try:
            deadline = time.monotonic() + 5.0
            spilled = False
            submitted = 0
            while time.monotonic() < deadline:
                request = _request("quay-6", submitted)
                try:
                    proxy.enqueue(WorkItem(request, ResponseFuture(request)))
                    submitted += 1
                except RingFullError:
                    spilled = True
                    break
                time.sleep(0.01)
            assert spilled, "the held slots filled the ring; the submitter got its spill signal"
            assert ingress.failed == 0, "nothing was failed as text while merely saturated"
        finally:
            self._teardown(ingress, reader, model, rings)


class TestTheProxySeesItsOwnBacklog:
    def test_depth_counts_published_but_untaken_requests(self) -> None:
        """Round 3: without the ring backlog a burst reads the stamped depth for every
        request and herds onto one peer while a local instance idles."""
        layout = RingLayout(slots=4, slot_bytes=64 * 1024)
        owner = SharedRing.create(_name("bl"), layout, owner="B")
        writer = SharedRing.open(owner.name, layout)
        reader = ResultReader(lost_after_s=5.0)
        proxy = RemoteInstance(
            owner="B", model_name="m", submit=writer, reader=reader, lost_after_s=5.0
        )
        try:
            owner.stamp(depth=1, ewma_latency_us=10.0)
            assert proxy.depth == 1
            items = []
            for frame in (1, 2):
                request = _request("quay-8", frame)
                item = WorkItem(request, ResponseFuture(request))
                proxy.enqueue(item)
                items.append(item)
            assert proxy.depth == 3, "the stamped depth plus the two in flight"
            for item in items:  # settled in any way, the charge comes off this peer
                item.fail(ServerStateError("test settles it"))
            assert proxy.depth == 1, "settled requests are no longer charged"
        finally:
            writer.close()
            owner.close()


class TestTheReaderFailsWhatItStrands:
    def test_stop_fails_pending_futures_with_their_tags(self) -> None:
        """Round 3 / ADR-002: engine.stop() must not leave a caller riding out its own
        timeout — a stranded future fails at reader shutdown, tag intact."""
        reader = ResultReader(lost_after_s=10.0)
        request = _request("quay-2", 9)
        item = WorkItem(request, ResponseFuture(request))
        reader.start()
        try:
            reader.expect(item, "B")
        finally:
            reader.stop()
            reader.join(timeout=2)
        with pytest.raises(ServerStateError, match="quay-2/9"):
            item.future.result(timeout=1.0)
        assert reader.pending == 0


class TestAWireRefusalSpills:
    def test_an_unencodable_request_lands_on_a_local_instance(self) -> None:
        """Round 4: a camera_id the wire cannot carry is this *candidate* refusing, not the
        frame failing — the spill loop tries the next instance and the ring stays untouched.
        Without the WireRefusedError(QueueFullError) shape this was a per-camera outage."""
        layout = RingLayout(slots=4, slot_bytes=64 * 1024)
        owner_ring = SharedRing.create(_name("wref"), layout, owner="B")
        writer = SharedRing.open(owner_ring.name, layout)
        reader = ResultReader(lost_after_s=5.0)
        proxy = RemoteInstance(
            owner="B", model_name="m", submit=writer, reader=reader, lost_after_s=5.0
        )
        owner_ring.stamp(depth=0, ewma_latency_us=1.0)  # ready and shallowest: tried first
        try:
            deep = FakeLocal(depth=100, full=True)  # the burst filled the local queue
            roomy = FakeLocal(depth=50, full=False)  # the instance that should get the frame
            dispatcher = Dispatcher(
                model_name="m",
                instances=[deep, proxy, roomy],
                policy=LocalityAwareSpilloverPolicy(spill_threshold=4),
            )
            request = _request("q" * 70, 5)  # 70 bytes: _fixed refuses at 64
            item = WorkItem(request, ResponseFuture(request))
            dispatcher.dispatch(item, lambda instance, work: instance.enqueue(work))
            assert [i.request.context.frame_id for i in roomy.enqueued] == [5]
            assert writer.depth == 0, "the claimed slot was abandoned, not published"
            assert proxy.depth == 0, "nothing is charged against the peer"
        finally:
            writer.close()
            owner_ring.close()


class TestTheSweepStampIsPerModel:
    def test_lanes_sharing_a_load_compute_it_once_and_both_are_stamped(self) -> None:
        """Round 4: `_stamp` was O(lanes x instances) per call on the tier's hottest
        thread. Lanes of one model share one load callable, computed once per sweep."""
        layout = RingLayout(slots=2, slot_bytes=8192)
        ring_a = SharedRing.create(_name("sa"), layout, owner="B")
        ring_b = SharedRing.create(_name("sb"), layout, owner="B")
        calls: list[int] = []

        def load() -> tuple[int, float]:
            calls.append(1)
            return (2, 5.0)

        def lane(ring: SharedRing) -> IngressLane:
            return IngressLane(
                submitter="A",
                inbound=ring,
                results=ring,  # never written in this test
                admit=lambda request: (_ for _ in ()).throw(AssertionError("not served")),
                dispatch=lambda item: True,
                reject=lambda: None,
                load=load,
            )

        ingress = RingIngress([lane(ring_a), lane(ring_b)], stamp_every_s=10.0)
        try:
            ingress._stamp(force=True)
            assert len(calls) == 1, "one model, one depth computation, two lanes stamped"
            assert ring_a.load_signal()[0] == 2 and ring_b.load_signal()[0] == 2
        finally:
            ring_a.close()
            ring_b.close()


class TestTheGiveUpArrivesTyped:
    def test_past_the_cap_the_submitter_gets_a_queue_full_error(self) -> None:
        """Round 4: the wire carries a status code, so the owner's saturation give-up
        arrives as the QueueFullError it is — not a generic ServerStateError."""
        model = TestASaturatedOwnerPushesBack.SaturatedThenWilling(refusals=10_000_000)
        helper = TestASaturatedOwnerPushesBack()
        ingress, reader, proxy, rings = helper._tier(model, patience_s=0.05, backoff_s=0.01)
        ingress.start()
        reader.start()
        try:
            request = _request("quay-3", 7)
            future = ResponseFuture(request)
            proxy.enqueue(WorkItem(request, future))
            with pytest.raises(QueueFullError, match="at A's peer"):
                future.result(timeout=5.0)
            assert ingress.failed == 1
            assert model.inner.rejections == 1, "the give-up counted exactly once"
        finally:
            helper._teardown(ingress, reader, model, rings)
