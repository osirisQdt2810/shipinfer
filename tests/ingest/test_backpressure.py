"""The inherited bug, as a regression test at the ingest boundary.

``references/bitbucket-subfaceid/docs/flow.md`` records the symptom in the operator's own
words: *"camera đông người được nhận diện đầy đủ, camera vắng người thỉnh thoảng bị miss"* —
the crowded cameras were fully processed and the quiet ones intermittently lost. The cause
was one shared 1000-slot buffer that evicted the **globally oldest** entry when full, so a
camera producing 30 detections a frame pushed out the frames of a camera producing 2.

Two separable claims, one class each:

1. **the actor's** claim — a refused frame is dropped and *counted against its camera*,
   which is only possible in the one component that knows which camera a frame came from;
2. **the queue's** claim — under overload the greediest camera loses a frame and the quiet
   one keeps its place.

The second needs a sink that maps frames onto queued requests. That mapping is dispatch
policy and belongs to ``pipeline`` (see :mod:`shipinfer.ingest.sink`), so the version here
is a **test double that documents the contract** — deliberately the smallest thing that can
carry the assertion, and the thing a reviewer of ``pipeline``'s real adapter should compare
against.
"""

from __future__ import annotations

import time

import pytest

from shipinfer.core.request import (
    InferenceRequest,
    Priority,
    RequestContext,
    ResponseFuture,
)
from shipinfer.core.settings import OverflowPolicy
from shipinfer.core.types import Tensor
from shipinfer.ingest import BoundedSink, CameraActor, Frame
from shipinfer.scheduling.queues import FairPriorityQueue
from shipinfer.scheduling.work import WorkItem, summarise_fairness

from .conftest import synthetic_image

pytestmark = pytest.mark.timeout(20)

CAPACITY = 8


class WorkQueueSink:
    """Publishes frames into a ``RequestQueue`` as inference requests.

    A stand-in for the adapter ``pipeline`` owns. It needs no translation layer: the two
    exceptions ``RequestQueue.put`` raises — ``QueueFullError`` and
    ``RequestCancelledError`` — are already the two the ``FrameSink`` contract names, and
    both already live in :mod:`shipinfer.core.errors`.

    Per-camera policy is resolved by ``camera_id`` rather than carried on the frame: a
    ``Frame`` is data and a priority is policy, and mixing the two is how the decode path
    ends up knowing about scheduler lanes.
    """

    def __init__(
        self, queue, *, model="ship_detector", input_name="images", deadline_ms=0, cameras=()
    ):
        self.queue = queue
        self.model = model
        self.input_name = input_name
        self.deadline_ns = deadline_ms * 1_000_000
        self.priority = {camera.camera_id: camera.priority for camera in cameras}

    def put(self, frame: Frame) -> None:
        request = InferenceRequest(
            model_name=self.model,
            inputs={self.input_name: Tensor.from_numpy(frame.as_batch())},
            context=frame.context,
            priority=self.priority.get(frame.camera_id, Priority.NORMAL),
            deadline_ns=frame.captured_ns + self.deadline_ns if self.deadline_ns else 0,
        )
        self.queue.put(WorkItem(request, ResponseFuture(request)))


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


def _run_camera(camera, count, sink, settings, scripted_factory):
    """Run one camera to completion against ``sink`` and return its actor."""
    factory, _ = scripted_factory(
        script=[synthetic_image(i) for i in range(count)], finite=True
    )
    actor = CameraActor(
        camera,
        sink,
        settings=settings,
        source_factory=factory,
        sleep=lambda _: None,
    )
    actor.start()
    assert _wait_for(lambda: not actor.is_running), actor.health
    actor.stop()
    return actor


class TestActorDropsWhenTheSinkRefuses:
    """A refused frame is dropped at the edge and counted against the camera that sent it."""

    def test_a_refusing_sink_produces_counted_drops(
        self, make_camera, fast_settings, scripted_factory
    ):
        sink = BoundedSink(capacity=3, name="tiny")
        actor = _run_camera(make_camera("cam0"), 10, sink, fast_settings(), scripted_factory)

        assert sink.accepted == 3
        assert sink.refused == 7
        health = actor.health
        assert health.frames_read == 10
        assert health.frames_published == 3
        assert health.frames_dropped == 7
        assert actor.metrics.frames_dropped.value(camera="cam0", reason="sink_full") == 7

    def test_drops_are_attributed_to_the_camera_that_caused_them(
        self, make_camera, fast_settings, scripted_factory
    ):
        """The whole reason the drop decision lives in the actor rather than in the sink."""
        sink = BoundedSink(capacity=2, name="tiny")
        settings = fast_settings()
        quiet = _run_camera(make_camera("cam_quiet"), 2, sink, settings, scripted_factory)
        loud = _run_camera(make_camera("cam_loud"), 40, sink, settings, scripted_factory)

        assert (
            quiet.health.frames_dropped == 0
        ), "the quiet camera got in first and paid nothing"
        assert loud.health.frames_dropped == 40, "every refusal is charged to the flood"
        assert loud.metrics.frames_dropped.value(camera="cam_loud", reason="sink_full") == 40


class TestFairQueueEvictionThroughASink:
    """Under overload the greediest camera loses a frame, never the quietest.

    The quiet camera publishes **first**, so its frame is the globally oldest entry in the
    queue — which is precisely the case the previous system got wrong. A plain drop-oldest
    buffer fails these tests.
    """

    def test_overload_evicts_the_greediest_camera_not_the_oldest_frame(
        self, make_camera, fast_settings, scripted_factory
    ):
        queue = FairPriorityQueue(
            "ingest", capacity=CAPACITY, overflow=OverflowPolicy.DROP_OLDEST
        )
        sink = WorkQueueSink(queue)
        settings = fast_settings()

        _run_camera(make_camera("cam_quiet"), 1, sink, settings, scripted_factory)
        assert queue.depth == 1

        loud = _run_camera(make_camera("cam_loud"), 40, sink, settings, scripted_factory)

        assert queue.depth == CAPACITY
        counts = summarise_fairness(list(queue.close()))
        assert (
            counts.get("cam_quiet") == 1
        ), "the quiet camera's frame survived the flood; evicting it is the inherited bug"
        assert counts.get("cam_loud") == CAPACITY - 1
        assert loud.health.frames_read == 40
        assert loud.health.frames_dropped == 0, "eviction is the queue's choice, not a refusal"

    def test_a_skewed_fleet_keeps_every_quiet_camera_represented(
        self, make_camera, fast_settings, scripted_factory
    ):
        """8x skew, the shape of `shipinfer bench --skew 8`, driven through real actors."""
        queue = FairPriorityQueue("ingest", capacity=16, overflow=OverflowPolicy.DROP_OLDEST)
        sink = WorkQueueSink(queue)
        settings = fast_settings()

        for camera_id in ("cam1", "cam2", "cam3", "cam4"):
            _run_camera(make_camera(camera_id), 2, sink, settings, scripted_factory)
        _run_camera(make_camera("cam_loud"), 80, sink, settings, scripted_factory)

        counts = summarise_fairness(list(queue.close()))
        for camera_id in ("cam1", "cam2", "cam3", "cam4"):
            assert counts.get(camera_id, 0) >= 1, f"{camera_id} was starved: {counts}"
        assert counts["cam_loud"] <= 16 - 4

    def test_a_low_priority_flood_cannot_displace_a_critical_camera(
        self, make_camera, fast_settings, scripted_factory
    ):
        """Priority sits above fairness: the lane matters before the camera does (ADR-005)."""
        queue = FairPriorityQueue("ingest", capacity=4, overflow=OverflowPolicy.DROP_OLDEST)
        gate = make_camera("gate", priority=Priority.TRACKING_CRITICAL)
        dock = make_camera("dock", priority=Priority.BACKGROUND)
        sink = WorkQueueSink(queue, cameras=(gate, dock))
        settings = fast_settings()

        _run_camera(gate, 1, sink, settings, scripted_factory)
        _run_camera(dock, 30, sink, settings, scripted_factory)

        counts = summarise_fairness(list(queue.close()))
        assert counts.get("gate") == 1, counts

    def test_rejecting_is_the_default_and_reaches_the_producer(
        self, make_camera, fast_settings, scripted_factory
    ):
        """The default is honest refusal, not eviction: the producer learns it is being shed."""
        queue = FairPriorityQueue("ingest", capacity=3)  # OverflowPolicy.REJECT
        sink = WorkQueueSink(queue)
        actor = _run_camera(make_camera("cam0"), 10, sink, fast_settings(), scripted_factory)

        stats = queue.stats()
        assert stats.accepted == 3
        assert stats.rejected == 7
        assert stats.evicted == 0
        assert actor.health.frames_dropped == 7, "the refusal reached the actor and was counted"


class TestSinkAdapterContract:
    """What `pipeline`'s real adapter must do with a frame — asserted against the double.

    These are not tests of ingest. They exist so the contract documented in
    :mod:`shipinfer.ingest.sink` has an executable form, and so a reviewer of the production
    adapter has something concrete to compare against.
    """

    def test_the_tag_survives_the_mapping_untouched(
        self, make_camera, fast_settings, scripted_factory
    ):
        queue = FairPriorityQueue("ingest", capacity=8)
        _run_camera(
            make_camera("cam0"), 4, WorkQueueSink(queue), fast_settings(), scripted_factory
        )

        items = list(queue.close())
        assert [i.request.context.key for i in items] == [("cam0", i) for i in range(4)]
        assert all(isinstance(i.request.context, RequestContext) for i in items)

    def test_the_frame_becomes_a_batch_major_request(
        self, make_camera, fast_settings, scripted_factory
    ):
        queue = FairPriorityQueue("ingest", capacity=8)
        sink = WorkQueueSink(queue, model="ship_detector", input_name="images")
        _run_camera(make_camera("cam0"), 1, sink, fast_settings(), scripted_factory)

        request = next(iter(queue.close())).request
        assert request.model_name == "ship_detector"
        assert request.inputs["images"].shape == (1, 6, 8, 3)

    def test_a_frame_deadline_is_measured_from_capture(
        self, make_camera, fast_settings, scripted_factory
    ):
        """From capture, not from enqueue: the point is to discard a frame that is too late."""
        queue = FairPriorityQueue("ingest", capacity=8)
        sink = WorkQueueSink(queue, deadline_ms=250)
        _run_camera(make_camera("cam0"), 1, sink, fast_settings(), scripted_factory)

        request = next(iter(queue.close())).request
        assert request.deadline_ns == request.context.captured_ns + 250_000_000

    def test_camera_priority_selects_the_queue_lane(
        self, make_camera, fast_settings, scripted_factory
    ):
        queue = FairPriorityQueue("ingest", capacity=8)
        gate = make_camera("gate", priority=Priority.TRACKING_CRITICAL)
        _run_camera(
            gate, 1, WorkQueueSink(queue, cameras=(gate,)), fast_settings(), scripted_factory
        )

        assert next(iter(queue.close())).request.priority is Priority.TRACKING_CRITICAL
