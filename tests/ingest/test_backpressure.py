"""The inherited bug, as a regression test at the ingest boundary.

``references/bitbucket-subfaceid/docs/flow.md`` records the symptom in the operator's own
words: *"camera đông người được nhận diện đầy đủ, camera vắng người thỉnh thoảng bị miss"* —
the crowded cameras were fully processed and the quiet ones intermittently lost. The cause
was one shared 1000-slot buffer that evicted the **globally oldest** entry when full, so a
camera producing 30 detections a frame pushed out the frames of a camera producing 2.

These tests assert the opposite behaviour end to end from the ingest side: under overload,
the **greediest** camera loses a frame and the quiet one keeps its place. They would pass
trivially against a plain FIFO drop-oldest queue only if the quiet camera's frame had been
enqueued last — so it is enqueued *first*, which is exactly the case the old system got
wrong.
"""

from __future__ import annotations

import time

import pytest

from shipinfer.core.settings import OverflowPolicy
from shipinfer.scheduling.queues import FairPriorityQueue
from shipinfer.scheduling.work import summarise_fairness

from .conftest import synthetic_image

pytestmark = pytest.mark.timeout(20)

CAPACITY = 8


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


def _run_camera(camera_id, count, queue, settings, scripted_factory, make_camera):
    """Run one camera to completion and return its actor."""
    from shipinfer.ingest import CameraActor

    factory, _ = scripted_factory(
        script=[synthetic_image(i) for i in range(count)], finite=True
    )
    actor = CameraActor(
        make_camera(camera_id),
        queue,
        settings=settings,
        source_factory=factory,
        sleep=lambda _: None,
    )
    actor.start()
    assert _wait_for(lambda: not actor.is_running), actor.health
    actor.stop()
    return actor


class TestFairQueueEviction:
    """Under overload the greediest camera loses a frame, never the quietest."""

    def test_overload_evicts_the_greediest_camera_not_the_oldest_frame(
        self, make_camera, fast_settings, scripted_factory
    ):
        queue = FairPriorityQueue(
            "ingest", capacity=CAPACITY, overflow=OverflowPolicy.DROP_OLDEST
        )
        settings = fast_settings()

        # The quiet camera goes FIRST, so its frame is the globally oldest entry in the queue.
        # Under the previous system's rule it is therefore the first thing thrown away.
        _run_camera("cam_quiet", 1, queue, settings, scripted_factory, make_camera)
        assert queue.depth == 1

        # Then the flood: forty frames into eight slots.
        loud = _run_camera("cam_loud", 40, queue, settings, scripted_factory, make_camera)

        assert queue.depth == CAPACITY
        counts = summarise_fairness(list(queue.close()))
        assert (
            counts.get("cam_quiet") == 1
        ), "the quiet camera's frame survived the flood; evicting it is the inherited bug"
        assert counts.get("cam_loud") == CAPACITY - 1
        assert loud.health.frames_read == 40

    def test_a_skewed_fleet_keeps_every_quiet_camera_represented(
        self, make_camera, fast_settings, scripted_factory
    ):
        """8x skew, the same shape as `shipinfer bench --skew 8`, driven through real actors."""
        queue = FairPriorityQueue("ingest", capacity=16, overflow=OverflowPolicy.DROP_OLDEST)
        settings = fast_settings()

        for camera_id in ("cam1", "cam2", "cam3", "cam4"):
            _run_camera(camera_id, 2, queue, settings, scripted_factory, make_camera)
        _run_camera("cam_loud", 80, queue, settings, scripted_factory, make_camera)

        counts = summarise_fairness(list(queue.close()))
        for camera_id in ("cam1", "cam2", "cam3", "cam4"):
            assert counts.get(camera_id, 0) >= 1, f"{camera_id} was starved: {counts}"
        assert counts["cam_loud"] <= 16 - 4

    def test_a_low_priority_flood_cannot_displace_a_critical_camera(
        self, make_camera, fast_settings, scripted_factory
    ):
        """Priority sits above fairness: the lane matters before the camera does (ADR-005)."""
        from shipinfer.core.request import Priority
        from shipinfer.ingest import CameraActor

        queue = FairPriorityQueue("ingest", capacity=4, overflow=OverflowPolicy.DROP_OLDEST)
        settings = fast_settings()

        factory, _ = scripted_factory(script=[synthetic_image(0)], finite=True)
        critical = CameraActor(
            make_camera("gate", priority=Priority.TRACKING_CRITICAL),
            queue,
            settings=settings,
            source_factory=factory,
            sleep=lambda _: None,
        )
        critical.start()
        assert _wait_for(lambda: not critical.is_running)
        critical.stop()

        background = CameraActor(
            make_camera("dock", priority=Priority.BACKGROUND),
            queue,
            settings=settings,
            source_factory=scripted_factory(
                script=[synthetic_image(i) for i in range(30)], finite=True
            )[0],
            sleep=lambda _: None,
        )
        background.start()
        assert _wait_for(lambda: not background.is_running)
        background.stop()

        counts = summarise_fairness(list(queue.close()))
        assert counts.get("gate") == 1, counts


class TestHonestRejection:
    """The default is a counted refusal, so the producer learns it is being shed."""

    def test_rejecting_is_the_default_and_is_reported(
        self, make_camera, fast_settings, scripted_factory
    ):
        """The default is honest refusal, not eviction: the producer learns it is being shed."""
        queue = FairPriorityQueue("ingest", capacity=3)  # OverflowPolicy.REJECT
        actor = _run_camera("cam0", 10, queue, fast_settings(), scripted_factory, make_camera)

        stats = queue.stats()
        assert stats.accepted == 3
        assert stats.rejected == 7
        assert stats.evicted == 0
        assert actor.health.frames_dropped == 7
