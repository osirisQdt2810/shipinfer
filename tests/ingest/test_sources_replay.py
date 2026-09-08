"""The replay source: the backend that makes the rest of the ingest plane testable."""

from __future__ import annotations

import threading
import time
from unittest import mock

import numpy as np
import pytest

from shipinfer.core.errors import SourceOpenError
from shipinfer.ingest import FrameCounter
from shipinfer.ingest.sources.replay import ReplaySource

from .conftest import FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH


class TestReplayPlayback:
    """A file or frame directory is delivered in order, at the requested rate."""

    def test_a_frame_directory_replays_in_order(self, make_camera, frame_dir):
        camera = make_camera("cam0", uri=str(frame_dir), source="replay", fps=0.0, loop=False)
        with ReplaySource(camera) as source:
            assert (source.height, source.width) == (FRAME_HEIGHT, FRAME_WIDTH)
            frames = [source.read() for _ in range(FRAME_COUNT)]

        assert [f.frame_id for f in frames] == list(range(FRAME_COUNT))
        assert all(f.camera_id == "cam0" for f in frames)
        # The synthetic images encode their index in channel 0, so an out-of-order replay shows.
        assert [int(f.image[0, 0, 0]) for f in frames] == list(range(1, FRAME_COUNT + 1))

    def test_a_video_file_replays(self, make_camera, video_file):
        camera = make_camera("cam0", uri=str(video_file), source="replay", loop=False)
        with ReplaySource(camera) as source:
            assert source.width == FRAME_WIDTH
            assert source.fps > 0
            first = source.read()
        assert first is not None
        assert first.image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)

    def test_scaling_is_honoured_and_reported(self, make_camera, frame_dir):
        camera = make_camera(
            uri=str(frame_dir), source="replay", width=32, height=16, loop=False
        )
        with ReplaySource(camera) as source:
            assert (source.height, source.width) == (16, 32)
            assert source.read().image.shape == (16, 32, 3)

    def test_pacing_is_applied_at_the_target_rate(self, make_camera, frame_dir):
        """A 200 fps replay of 6 frames must not take a second; it must also not be instant."""
        import time

        camera = make_camera(uri=str(frame_dir), source="replay", fps=200.0, loop=False)
        with ReplaySource(camera) as source:
            started = time.perf_counter()
            for _ in range(FRAME_COUNT):
                source.read()
            elapsed = time.perf_counter() - started
        assert 0.01 < elapsed < 1.0, elapsed


class TestReplayHonoursAStop:
    """The pace wait is the other place a Python source could sit through a stop.

    `PY-SOURCE-HAS-NO-STOP-SIGNAL` closed the GStreamer read; #165's review found this one
    still open, and it is the same seam: `csrc/.../sources/replay.cpp` passes
    `stop().wait_for(...)` into `pacer_.wait` and caps it at one `read_timeout_s`.
    """

    def test_a_stop_ends_the_pace_wait_instead_of_sleeping_it_out(self, make_camera, frame_dir):
        # 1 fps, so an un-interruptible wait would sit here for a second per frame.
        camera = make_camera(uri=str(frame_dir), source="replay", fps=1.0, loop=False)
        stop = threading.Event()
        with ReplaySource(camera, stop=stop) as source:
            assert source.read() is not None, "the first frame is due immediately"
            stop.set()
            started = time.perf_counter()
            assert source._do_read() is None, "a stop ends the wait with no frame"
            elapsed = time.perf_counter() - started
        assert elapsed < 0.2, f"it returned in {elapsed:.3f}s rather than waiting out 1 s"

    def test_a_frame_period_over_the_read_timeout_is_an_empty_read(
        self, make_camera, frame_dir
    ):
        """`replay.cpp`'s own branch: waiting the budget and then answering with a frame the
        actor asked for `due_s` ago is worse than answering empty."""
        camera = make_camera(uri=str(frame_dir), source="replay", fps=1.0, loop=False)
        stop = threading.Event()
        with ReplaySource(camera, stop=stop) as source:
            assert source.read() is not None
            # A read timeout well under the 1 s frame period.
            with mock.patch.object(type(source), "read_timeout_s", property(lambda self: 0.05)):
                started = time.perf_counter()
                assert source._do_read() is None, "over budget answers empty"
                elapsed = time.perf_counter() - started
        assert 0.03 < elapsed < 0.5, f"it waited {elapsed:.3f}s, not the ~0.05 s budget"

    def test_with_no_stop_event_the_pacer_still_paces(self, make_camera, frame_dir):
        """The additive half: nothing is required to pass a stop, and pacing is unchanged."""
        camera = make_camera(uri=str(frame_dir), source="replay", fps=200.0, loop=False)
        with ReplaySource(camera) as source:
            started = time.perf_counter()
            for _ in range(FRAME_COUNT):
                source.read()
            elapsed = time.perf_counter() - started
        assert 0.01 < elapsed < 1.0, elapsed


class TestEndOfInput:
    """End of a finite source is not a fault; looping is what a stress test wants."""

    def test_a_finite_source_reports_exhausted_rather_than_failing(
        self, make_camera, frame_dir
    ):
        camera = make_camera(uri=str(frame_dir), source="replay", loop=False)
        with ReplaySource(camera) as source:
            for _ in range(FRAME_COUNT):
                assert source.read() is not None
            assert source.is_exhausted is False
            assert source.read() is None
            assert source.is_exhausted is True
            # Still None, still not an error: a finished file is not a fault.
            assert source.read() is None

    def test_looping_keeps_going_and_keeps_counting(self, make_camera, frame_dir):
        camera = make_camera(uri=str(frame_dir), source="replay", loop=True)
        with ReplaySource(camera) as source:
            frames = [source.read() for _ in range(FRAME_COUNT * 2 + 1)]
        assert source.is_exhausted is False
        assert [f.frame_id for f in frames] == list(range(FRAME_COUNT * 2 + 1))
        # The images repeat; the frame ids do not. That is the invariant downstream depends on.
        assert np.array_equal(frames[0].image, frames[FRAME_COUNT].image)


class TestFrameTagging:
    """The frame id belongs to the caller's counter, not to the source."""

    def test_the_counter_is_the_actors_not_the_sources(self, make_camera, frame_dir):
        """A reconnect replaces the source; the tag sequence must not restart (ADR-002)."""
        camera = make_camera(uri=str(frame_dir), source="replay", loop=False)
        counter = FrameCounter(camera.camera_id)

        with ReplaySource(camera, counter) as first:
            ids = [first.read().frame_id for _ in range(3)]
        with ReplaySource(camera, counter) as second:
            ids += [second.read().frame_id for _ in range(3)]

        assert ids == [0, 1, 2, 3, 4, 5]

    def test_first_frame_id_offsets_the_sequence(self, make_camera, frame_dir):
        camera = make_camera(
            uri=str(frame_dir), source="replay", first_frame_id=1000, loop=False
        )
        with ReplaySource(camera) as source:
            assert [source.read().frame_id for _ in range(3)] == [1000, 1001, 1002]


class TestReplayFailures:
    """A bad path is a typed error naming the camera, not a bare OSError."""

    def test_a_missing_path_is_a_typed_error(self, make_camera, tmp_path):
        camera = make_camera(uri=str(tmp_path / "nope.mp4"), source="replay")
        with pytest.raises(SourceOpenError, match="does not exist"):
            ReplaySource(camera).open()

    def test_an_empty_directory_is_a_typed_error(self, make_camera, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SourceOpenError, match="no images"):
            ReplaySource(make_camera(uri=str(empty), source="replay")).open()


class TestReplayLifecycle:
    """Open and close are idempotent, and replay never claims hardware decode."""

    def test_open_and_close_are_idempotent(self, make_camera, frame_dir):
        source = ReplaySource(make_camera(uri=str(frame_dir), source="replay"))
        source.open()
        source.open()
        assert source.is_open is True
        source.close()
        source.close()
        assert source.is_open is False

    def test_replay_never_claims_hardware_decode(self, make_camera, frame_dir):
        camera = make_camera(uri=str(frame_dir), source="replay", hwaccel=True)
        assert ReplaySource(camera).hwaccel is False
