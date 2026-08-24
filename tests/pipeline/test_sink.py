"""The ingest -> scheduler adapter, against the real :class:`QueueFrameSink`.

``tests/ingest/test_backpressure.py::TestSinkAdapterContract`` states the same four
properties against a test double, because ``ingest`` may not import the scheduler (ADR-011).
These are the same claims made of the production adapter, which is the point: the double
documents the contract and this file is the contract being kept.

The fifth class here is the one the double cannot make — that a refusal reaches the *camera
actor* untouched, so the drop can be charged to the camera that caused it.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.errors import QueueFullError, RequestCancelledError
from shipinfer.core.request import Priority, RequestContext
from shipinfer.core.settings import OverflowPolicy
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest.frame import FrameCounter
from shipinfer.ingest.sink import FrameSink
from shipinfer.pipeline.sink import QueueFrameSink, TaggedFrame
from shipinfer.scheduling.queues import FairPriorityQueue

pytestmark = pytest.mark.timeout(30)


def stamp(camera: str = "cam0", count: int = 1, size: tuple[int, int] = (6, 8)):
    """``count`` frames from one camera, tagged exactly as a camera actor would tag them."""
    counter = FrameCounter(camera)
    return [counter.stamp(np.zeros((*size, 3), dtype=np.uint8)) for _ in range(count)]


class TestTheAdapterSatisfiesTheProtocol:
    """Structural typing is the whole mechanism, so it is worth asserting."""

    def test_the_sink_satisfies_the_frame_sink_protocol(self):
        sink = QueueFrameSink(FairPriorityQueue("ingest", capacity=4))
        assert isinstance(sink, FrameSink)

    def test_an_ingest_frame_satisfies_the_tagged_frame_protocol(self):
        """``pipeline`` describes what it needs from a frame rather than importing it."""
        assert isinstance(stamp()[0], TaggedFrame)


class TestSinkAdapterContract:
    """The four claims ``shipinfer.ingest.sink``'s docstring makes about this adapter."""

    def test_the_tag_survives_the_mapping_untouched(self):
        queue = FairPriorityQueue("ingest", capacity=8)
        sink = QueueFrameSink(queue)
        for frame in stamp("cam0", 4):
            sink.put(frame)

        items = list(queue.close())
        assert [i.request.context.key for i in items] == [("cam0", i) for i in range(4)]
        assert all(isinstance(i.request.context, RequestContext) for i in items)

    def test_the_frame_becomes_a_batch_major_request(self):
        queue = FairPriorityQueue("ingest", capacity=8)
        sink = QueueFrameSink(
            queue, settings=IngestSettings(target_model="ship_detector", input_name="images")
        )
        sink.put(stamp("cam0", 1)[0])

        request = next(iter(queue.close())).request
        assert request.model_name == "ship_detector"
        assert request.inputs["images"].shape == (1, 6, 8, 3)

    def test_a_frame_deadline_is_measured_from_capture(self):
        """From capture, not from enqueue: a deeply queued frame must not look fresh."""
        queue = FairPriorityQueue("ingest", capacity=8)
        sink = QueueFrameSink(queue, settings=IngestSettings(frame_deadline_ms=250))
        sink.put(stamp("cam0", 1)[0])

        request = next(iter(queue.close())).request
        assert request.deadline_ns == request.context.captured_ns + 250_000_000

    def test_camera_priority_selects_the_queue_lane(self):
        """Resolved by camera id, because a frame is data and a priority is configuration."""
        queue = FairPriorityQueue("ingest", capacity=8)
        gate = CameraConfig(
            camera_id="gate", uri="rtsp://gate", priority=Priority.TRACKING_CRITICAL
        )
        sink = QueueFrameSink(queue, settings=IngestSettings(cameras=[gate]))
        sink.put(stamp("gate", 1)[0])

        assert next(iter(queue.close())).request.priority is Priority.TRACKING_CRITICAL


class TestPerCameraPolicy:
    """Policy is per camera and resolved once, not carried on a thousand frames a second."""

    def test_a_camera_added_at_runtime_gets_the_fleet_default(self):
        queue = FairPriorityQueue("ingest", capacity=4)
        known = CameraConfig(camera_id="known", uri="rtsp://a", priority=Priority.HIGH)
        sink = QueueFrameSink(
            queue, settings=IngestSettings(cameras=[known], target_model="ship_detector")
        )

        sink.put(stamp("new_camera", 1)[0])

        request = next(iter(queue.close())).request
        assert request.priority is Priority.NORMAL
        assert request.model_name == "ship_detector"
        # Memoised, so the discovery is paid once per camera rather than once per frame.
        assert sink.policies()["new_camera"] == ("ship_detector", Priority.NORMAL)

    def test_a_camera_can_override_the_target_model(self):
        queue = FairPriorityQueue("ingest", capacity=4)
        camera = CameraConfig(camera_id="thermal", uri="rtsp://t", model="thermal_detector")
        sink = QueueFrameSink(queue, settings=IngestSettings(cameras=[camera]))

        sink.put(stamp("thermal", 1)[0])

        assert next(iter(queue.close())).request.model_name == "thermal_detector"

    def test_priorities_reach_the_queue_as_lanes(self):
        """The lane is what a TRACKING_CRITICAL camera is buying, so drain order proves it."""
        queue = FairPriorityQueue("ingest", capacity=8)
        cameras = [
            CameraConfig(camera_id="quiet", uri="rtsp://q", priority=Priority.BACKGROUND),
            CameraConfig(camera_id="gate", uri="rtsp://g", priority=Priority.TRACKING_CRITICAL),
        ]
        sink = QueueFrameSink(queue, settings=IngestSettings(cameras=cameras))

        sink.put(stamp("quiet", 1)[0])
        sink.put(stamp("gate", 1)[0])

        from shipinfer.scheduling.queues import BatchWindow

        drained = queue.get_batch(BatchWindow(max_batch_size=2))
        assert [item.request.context.camera_id for item in drained] == ["gate", "quiet"]


class TestBackpressureReachesTheProducer:
    """A refusal is raised, not returned — and it reaches the camera actor untouched.

    This is the claim the ingest lane's double cannot make on its own, and it is the whole
    substance of ADR-005: the actor is the only component that knows which camera a frame came
    from, so it is the only one that can charge the drop to the camera that caused it.
    """

    def test_a_full_queue_raises_queue_full_error_with_depth_and_capacity(self):
        queue = FairPriorityQueue("ingest", capacity=2, overflow=OverflowPolicy.REJECT)
        sink = QueueFrameSink(queue)
        frames = stamp("cam0", 3)

        sink.put(frames[0])
        sink.put(frames[1])
        with pytest.raises(QueueFullError) as raised:
            sink.put(frames[2])

        assert raised.value.depth == 2
        assert raised.value.capacity == 2
        assert sink.accepted == 2

    def test_a_closed_queue_raises_request_cancelled(self):
        """The consumer is gone; the actor should finish rather than log one line per frame."""
        queue = FairPriorityQueue("ingest", capacity=2)
        sink = QueueFrameSink(queue)
        queue.close()

        with pytest.raises(RequestCancelledError):
            sink.put(stamp("cam0", 1)[0])

    def test_the_refusal_reaches_the_camera_actor_and_is_charged_to_its_camera(self):
        """End to end through the real actor: the drop is counted against ``loud``."""
        from shipinfer.ingest.camera.actor import CameraActor

        queue = FairPriorityQueue("ingest", capacity=2, overflow=OverflowPolicy.REJECT)
        sink = QueueFrameSink(queue)
        config = CameraConfig(camera_id="loud", uri="rtsp://loud")
        actor = CameraActor(config, sink, settings=IngestSettings(), sleep=lambda _s: None)

        for frame in stamp("loud", 5):
            actor._publish(frame)

        assert actor.health.frames_published == 2
        assert actor.health.frames_dropped == 3
        assert actor.metrics.frames_dropped.value(camera="loud", reason="sink_full") == 3
