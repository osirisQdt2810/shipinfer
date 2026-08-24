"""The camera actor: the tag, the reconnect policy, and shutting down without hanging.

Every test here is time-bounded on purpose. An ingest actor is a thread that sleeps and
blocks for a living, so the failure mode of a bug in it is a hung suite rather than a red
one — which is worse, because CI then just stops instead of telling anyone.

Two shapes recur:

* a ``finite`` scripted source, so the actor runs out of frames and exits on its own and the
  test can assert an *exact* count;
* a :class:`RecordingSleep` wired to ``actor.request_stop``, so the reconnect delays are
  asserted as a sequence in milliseconds of wall clock instead of minutes of real waiting.

The actor publishes into a :class:`~shipinfer.ingest.sink.BoundedSink` here. What the fair
queue does with those frames is the scheduler's business and is asserted in
``test_backpressure.py``, through the sink adapter that ``pipeline`` will own.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from shipinfer.core.errors import ServerStateError, SourceUnavailableError
from shipinfer.ingest import BoundedSink, CameraActor, CameraState, IngestMetrics

from .conftest import FRAME_COUNT, RecordingSleep, synthetic_image, tick

pytestmark = pytest.mark.timeout(20)


def _wait_for(predicate, timeout_s: float = 5.0, poll_s: float = 0.002) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def _actor(camera, sink, settings, factory, sleep=None):
    return CameraActor(
        camera,
        sink,
        settings=settings,
        source_factory=factory,
        sleep=sleep if sleep is not None else (lambda _: None),
    )


def _images(count: int) -> list:
    return [synthetic_image(i) for i in range(count)]


class TestFrameTagging:
    """Every frame carries `(camera_id, frame_id)`, and the sequence never restarts."""

    def test_frames_arrive_tagged_with_camera_and_frame_id(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        factory, _ = scripted_factory(script=_images(4), finite=True)
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory)
        actor.start()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()

        frames = sink.drain()
        tags = [(f.camera_id, f.frame_id) for f in frames]
        assert tags == [("cam0", 0), ("cam0", 1), ("cam0", 2), ("cam0", 3)]
        assert actor.state is CameraState.EXHAUSTED

    def test_the_frame_carries_the_image_and_both_clocks(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        """What leaves ingest is a Frame: no model name, no tensor, no future.

        Mapping one onto an inference request is the sink's job, and the sink belongs to
        whoever consumes the frames — see :mod:`shipinfer.ingest.sink`.
        """
        factory, _ = scripted_factory(script=_images(1), finite=True)
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory)
        actor.start()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()

        frame = sink.drain()[0]
        assert frame.image.shape == (6, 8, 3)
        assert frame.as_batch().shape == (1, 6, 8, 3), "batch-major for the consumer"
        assert np.shares_memory(frame.image, frame.as_batch()), "as_batch views, never copies"
        assert frame.captured_ns > 0
        assert frame.captured_unix_ns > 0
        assert frame.context.key == ("cam0", 0)

    def test_two_cameras_count_independently(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        settings = fast_settings()
        actors = []
        for camera_id in ("cam0", "cam1"):
            factory, _ = scripted_factory(script=_images(3), finite=True)
            actor = _actor(make_camera(camera_id), sink, settings, factory)
            actors.append(actor)
            actor.start()

        assert _wait_for(lambda: all(not a.is_running for a in actors))
        for actor in actors:
            actor.stop()

        per_camera: dict[str, list[int]] = {}
        for frame in sink.drain():
            per_camera.setdefault(frame.camera_id, []).append(frame.frame_id)
        assert per_camera == {"cam0": [0, 1, 2], "cam1": [0, 1, 2]}

    def test_frame_ids_keep_climbing_across_a_reconnect(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        """The reason the counter lives on the actor: a reconnect must not reissue frame 0."""
        script = [
            synthetic_image(0),
            synthetic_image(1),
            ConnectionResetError("stream dropped"),
            synthetic_image(2),
            synthetic_image(3),
        ]
        factory, created = scripted_factory(script=script, finite=True, shared_script=True)
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory)
        actor.start()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()

        assert [f.frame_id for f in sink.drain()] == [0, 1, 2, 3]
        assert len(created) == 2, "the failed read built a new source"
        assert created[0] is not created[1]

    def test_first_frame_id_survives_a_process_restart(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        factory, _ = scripted_factory(script=_images(2), finite=True)
        actor = _actor(
            make_camera("cam0", first_frame_id=5_000), sink, fast_settings(), factory
        )
        actor.start()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()
        assert [f.frame_id for f in sink.drain()] == [5_000, 5_001]


class TestReconnectBackoff:
    """A dropped camera is retried on a growing, capped schedule that a frame resets."""

    def test_connect_failures_back_off_geometrically_and_hit_the_cap(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        recorder = RecordingSleep(stop_after=6)
        factory, created = scripted_factory(open_failures=100)
        actor = _actor(
            make_camera("cam0"),
            sink,
            fast_settings(reconnect_initial_ms=100, reconnect_max_ms=800),
            factory,
            sleep=recorder,
        )
        recorder.on_stop = actor.request_stop
        actor.start()
        assert recorder.wait(), recorder.delays
        actor.stop()

        assert recorder.delays == [0.1, 0.2, 0.4, 0.8, 0.8, 0.8], recorder.delays
        assert len(created) == 6, "one new source per attempt"
        assert sink.depth == 0

    def test_a_failing_read_reconnects_with_the_same_backoff(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        recorder = RecordingSleep(stop_after=3)
        factory, _ = scripted_factory(script=[OSError("decoder died")])
        actor = _actor(
            make_camera("cam0"),
            sink,
            fast_settings(reconnect_initial_ms=50, reconnect_max_ms=200),
            factory,
            sleep=recorder,
        )
        recorder.on_stop = actor.request_stop
        actor.start()
        assert recorder.wait(), recorder.delays
        actor.stop()
        assert recorder.delays == [0.05, 0.1, 0.2], recorder.delays

    def test_one_good_frame_resets_the_backoff(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        """Recovery must be cheap: a camera that blips repeatedly must not creep up to the cap."""
        recorder = RecordingSleep(stop_after=4)
        factory, _ = scripted_factory(
            script=[synthetic_image(0), OSError("blip")] * 8, shared_script=True
        )
        actor = _actor(
            make_camera("cam0"),
            sink,
            fast_settings(reconnect_initial_ms=100, reconnect_max_ms=5_000),
            factory,
            sleep=recorder,
        )
        recorder.on_stop = actor.request_stop
        actor.start()
        assert recorder.wait(), recorder.delays
        actor.stop()

        assert recorder.delays == [0.1, 0.1, 0.1, 0.1], recorder.delays


class TestUnhealthyCamera:
    """A camera that opens but never delivers is unhealthy, and stays that way."""

    def test_a_source_that_never_delivers_becomes_unhealthy(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        """Opening successfully is not health. A stream that delivers nothing must page someone."""
        factory, created = scripted_factory(script=[None])
        actor = _actor(
            make_camera("cam0"),
            sink,
            fast_settings(empty_reads_before_reconnect=2, failures_before_unhealthy=3),
            factory,
            sleep=tick,
        )
        actor.start()
        assert _wait_for(lambda: actor.state is CameraState.UNHEALTHY), actor.health
        actor.stop()

        health = actor.health
        assert health.frames_read == 0
        assert health.empty_reads >= 4
        assert health.connect_failures >= 3
        assert len(created) >= 3, "it kept retrying rather than spinning on one dead source"
        assert "consecutive empty reads" in health.last_error

    def test_unhealthy_is_sticky_until_a_frame_clears_it(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        """A camera retrying every 30 s must not flap between UNHEALTHY and CONNECTING."""
        factory, _ = scripted_factory(open_failures=100)
        actor = _actor(
            make_camera("cam0"),
            sink,
            fast_settings(failures_before_unhealthy=2),
            factory,
            sleep=tick,
        )
        actor.start()
        assert _wait_for(lambda: actor.state is CameraState.UNHEALTHY)
        for _ in range(5):
            assert actor.state is CameraState.UNHEALTHY
            time.sleep(0.002)
        actor.stop()

    def test_a_missing_decode_runtime_is_not_retried(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        """No amount of waiting installs PyGObject; retrying only buries the log line."""
        recorder = RecordingSleep()
        factory, created = scripted_factory(
            open_error=SourceUnavailableError("gstreamer", "PyGObject is not importable")
        )
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory, sleep=recorder)
        actor.start()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()

        assert len(created) == 1, "one attempt, then it gave up"
        assert recorder.delays == []
        assert actor.state is CameraState.UNHEALTHY
        assert "SourceUnavailableError" in actor.health.last_error


class TestExhaustedSource:
    """A finite source finishing is not a fault to reconnect to."""

    def test_an_exhausted_replay_source_finishes_instead_of_reconnecting(
        self, make_camera, fast_settings, sink, frame_dir
    ):
        actor = CameraActor(
            make_camera("cam0", uri=str(frame_dir), source="replay", loop=False),
            sink,
            settings=fast_settings(),
            sleep=lambda _: None,
        )
        actor.start()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()

        assert actor.state is CameraState.EXHAUSTED
        assert len(sink.drain()) == FRAME_COUNT


class TestShutdown:
    """Stopping is idempotent, bounded, and never hangs."""

    def test_stop_is_idempotent_and_joining_a_stopped_actor_does_not_hang(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        factory, created = scripted_factory(script=_images(1), finite=True)
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory)
        actor.start()
        assert _wait_for(lambda: sink.depth >= 1)

        actor.stop()
        actor.stop()
        actor.stop(timeout_s=0.0)
        assert actor.is_running is False
        assert all(
            source.closes >= 1 for source in created
        ), "the source is released on the way out"

    def test_stopping_an_actor_that_was_never_started_is_a_no_op(
        self, make_camera, fast_settings, sink
    ):
        actor = CameraActor(make_camera("cam0"), sink, settings=fast_settings())
        actor.stop()
        assert actor.state is CameraState.STOPPED
        assert actor.is_running is False

    def test_an_actor_cannot_be_restarted(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        factory, _ = scripted_factory(script=[None])
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory, sleep=tick)
        actor.start()
        with pytest.raises(ServerStateError, match="already been started"):
            actor.start()
        actor.stop()
        with pytest.raises(ServerStateError):
            actor.start()

    def test_a_closing_sink_stops_the_actor_rather_than_spamming_the_log(
        self, make_camera, fast_settings, scripted_factory
    ):
        sink = BoundedSink(capacity=4, name="ingest")
        factory, _ = scripted_factory(script=[synthetic_image(0)])
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory, sleep=tick)
        actor.start()
        assert _wait_for(lambda: actor.health.frames_read >= 1)
        sink.close()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()
        assert actor.health.frames_dropped >= 1

    def test_the_context_manager_starts_and_stops(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        factory, _ = scripted_factory(script=_images(1), finite=True)
        with _actor(make_camera("cam0"), sink, fast_settings(), factory) as actor:
            assert _wait_for(lambda: sink.depth >= 1)
        assert actor.is_running is False

    def test_the_actor_survives_a_bug_in_the_queue(
        self, make_camera, fast_settings, scripted_factory
    ):
        """A broken consumer must degrade one camera, not kill its thread and go quiet."""
        sink = BoundedSink(capacity=8, name="exploding")

        def explode(_item):
            raise RuntimeError("boom")

        sink.put = explode  # type: ignore[method-assign]

        recorder = RecordingSleep(stop_after=2)
        factory, _ = scripted_factory(script=[synthetic_image(0)])
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory, sleep=recorder)
        recorder.on_stop = actor.request_stop
        actor.start()
        assert recorder.wait(), recorder.delays
        actor.stop()

        assert len(recorder.delays) == 2, "it backed off instead of hot-looping on the bug"
        assert "RuntimeError" in actor.health.last_error


class TestHealthAndMetrics:
    """What an operator can see: per-camera counters, drops, deadlines, priority."""

    def test_health_reports_what_the_operator_needs(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        factory, _ = scripted_factory(script=_images(3), finite=True)
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory)
        actor.start()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()

        health = actor.health
        assert health.camera_id == "cam0"
        assert health.frames_read == 3
        assert health.frames_published == 3
        assert health.frames_dropped == 0
        assert health.connects == 1
        assert health.last_frame_unix_ns > 0
        assert health.drop_ratio == 0.0
        assert set(health.as_dict()) >= {"camera_id", "state", "frames_read", "fps"}

    def test_metrics_are_labelled_per_camera(
        self, make_camera, fast_settings, scripted_factory, sink
    ):
        metrics = IngestMetrics()
        settings = fast_settings()
        for camera_id, count in (("cam0", 4), ("cam1", 1)):
            factory, _ = scripted_factory(script=_images(count), finite=True)
            actor = CameraActor(
                make_camera(camera_id),
                sink,
                settings=settings,
                metrics=metrics,
                source_factory=factory,
                sleep=lambda _: None,
            )
            actor.start()
            assert _wait_for(lambda a=actor: not a.is_running)
            actor.stop()

        assert metrics.frames_total.value(camera="cam0") == 4
        assert metrics.frames_total.value(camera="cam1") == 1
        assert metrics.frames_published.value(camera="cam0") == 4
        assert metrics.reconnects_total.value(camera="cam0") == 1
        assert metrics.connect_failures_total.value(camera="cam0") == 0

    def test_a_full_queue_is_counted_not_swallowed(
        self, make_camera, fast_settings, scripted_factory
    ):
        """Backpressure has to be visible; ADR-005's point is that a drop is countable."""
        sink = BoundedSink(capacity=2, name="tiny")
        factory, _ = scripted_factory(script=_images(20), finite=True)
        actor = _actor(make_camera("cam0"), sink, fast_settings(), factory)
        actor.start()
        assert _wait_for(lambda: not actor.is_running)
        actor.stop()

        health = actor.health
        assert health.frames_read == 20
        assert health.frames_published == 2, "capacity 2 means two frames land"
        assert health.frames_dropped == 18
        assert health.drop_ratio == pytest.approx(0.9)
        assert actor.metrics.frames_dropped.value(camera="cam0", reason="sink_full") == 18
