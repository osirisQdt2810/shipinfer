"""The FrameSink protocol and the two sinks this package ships.

Neither of these is the production path — that is a `RequestQueue`-backed adapter in
`pipeline/`. What is worth pinning here is the **contract**: a sink that cannot take a frame
raises, with an error type the actor knows how to act on, because an RTSP camera cannot be
backpressured and somebody has to decide to drop.
"""

from __future__ import annotations

import threading

import pytest

from shipinfer.core.errors import QueueFullError, RequestCancelledError
from shipinfer.ingest import BoundedSink, CountingSink, FrameCounter, FrameSink

from .conftest import synthetic_image


def _frames(camera_id: str, count: int, counter: FrameCounter | None = None):
    counter = counter or FrameCounter(camera_id)
    return [counter.stamp(synthetic_image(i)) for i in range(count)]


class TestProtocolConformance:
    """Both shipped sinks satisfy the protocol the actor is typed against."""

    @pytest.mark.parametrize("sink", [CountingSink(), BoundedSink(4)])
    def test_a_shipped_sink_is_a_frame_sink(self, sink):
        assert isinstance(sink, FrameSink)

    def test_the_protocol_needs_exactly_one_method(self):
        """A one-method seam is why `ingest` needs nothing from the scheduler."""

        class Minimal:
            def put(self, frame):
                pass

        assert isinstance(Minimal(), FrameSink)


class TestCountingSink:
    """The measurement harness: never refuses, keeps nothing, counts per camera."""

    def test_it_counts_per_camera_and_never_refuses(self):
        sink = CountingSink()
        for frame in _frames("cam0", 500):
            sink.put(frame)
        for frame in _frames("cam1", 3):
            sink.put(frame)

        assert sink.total == 503
        assert sink.counts() == {"cam0": 500, "cam1": 3}
        assert len(sink) == 503

    def test_it_keeps_no_frames_by_default(self):
        """A 60-second run at 1000 fps must not accumulate 60 000 frames of memory."""
        sink = CountingSink()
        sink.put(_frames("cam0", 1)[0])
        assert sink.latest("cam0") is None

    def test_keep_last_holds_one_frame_per_camera(self):
        sink = CountingSink(keep_last=True)
        frames = _frames("cam0", 3)
        for frame in frames:
            sink.put(frame)
        assert sink.latest("cam0") is frames[-1]
        assert sink.latest("cam9") is None

    def test_it_is_safe_to_share_across_camera_threads(self):
        """One sink, many actors: that is how it will be used, so count it that way."""
        sink = CountingSink()

        def publish(camera_id: str) -> None:
            for frame in _frames(camera_id, 200):
                sink.put(frame)

        threads = [
            threading.Thread(target=publish, args=(f"cam{i}",), daemon=True) for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert sink.total == 1_600
        assert sink.counts() == {f"cam{i}": 200 for i in range(8)}


class TestBoundedSink:
    """Refuses at a known depth, so a producer's reaction to "no" can be tested."""

    def test_it_accepts_up_to_capacity_then_refuses(self):
        sink = BoundedSink(capacity=3, name="tiny")
        frames = _frames("cam0", 5)
        for frame in frames[:3]:
            sink.put(frame)

        with pytest.raises(QueueFullError) as excinfo:
            sink.put(frames[3])

        error = excinfo.value
        assert error.queue_name == "tiny"
        assert (error.depth, error.capacity) == (3, 3), "the numbers an operator needs"
        assert sink.accepted == 3
        assert sink.refused == 1
        assert sink.depth == 3

    def test_draining_makes_room_again(self):
        sink = BoundedSink(capacity=2)
        frames = _frames("cam0", 4)
        sink.put(frames[0])
        sink.put(frames[1])
        assert sink.drain() == frames[:2]
        assert sink.depth == 0
        sink.put(frames[2])
        assert sink.accepted == 3

    def test_a_closed_sink_reports_cancellation_not_fullness(self):
        """The actor treats the two differently: one drops a frame, the other stops."""
        sink = BoundedSink(capacity=8)
        sink.close()
        sink.close()
        assert sink.is_closed is True
        with pytest.raises(RequestCancelledError, match="closed"):
            sink.put(_frames("cam0", 1)[0])

    def test_counts_are_per_camera(self):
        sink = BoundedSink(capacity=16)
        for frame in _frames("cam0", 3):
            sink.put(frame)
        for frame in _frames("cam1", 1):
            sink.put(frame)
        assert sink.counts() == {"cam0": 3, "cam1": 1}

    def test_a_zero_capacity_sink_is_rejected(self):
        with pytest.raises(ValueError, match="capacity"):
            BoundedSink(capacity=0)
