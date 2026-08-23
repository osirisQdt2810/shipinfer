"""Fixtures for the ingest tier — all of which run with no camera, no GPU and no GStreamer.

Two doubles carry most of the weight:

* :class:`ScriptedSource` — a :class:`FrameSource` driven by a list. It is how "returns None
  forever", "raises on the third read" and "fails to open twice, then works" become one-line
  test setups instead of three bespoke fakes.
* :class:`RecordingSleep` — records every delay the actor asks for, so a test asserts the
  *sequence* of reconnect delays rather than merely that a retry happened, and does it in
  milliseconds of wall clock.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.frame import FrameCounter
from shipinfer.scheduling.queues import BatchWindow, FairPriorityQueue

FRAME_COUNT = 6
FRAME_HEIGHT = 6
FRAME_WIDTH = 8


def synthetic_image(index: int, *, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH):
    """A tiny BGR image whose content encodes its index, so a reorder is visible."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = index + 1
    return image


# -- doubles -------------------------------------------------------------------------------


class ScriptedSource(FrameSource):
    """A source that replays a script of reads.

    Each script entry is an image, ``None`` (nothing available yet), or an exception instance
    (raised). What happens at the end of the script is the interesting knob:

    * default — the last entry repeats forever, so ``script=[None]`` is "opens fine, never
      delivers": the most common real failure of an RTSP camera and the one the actor must
      eventually report as unhealthy.
    * ``finite=True`` — the source reports :attr:`is_exhausted`, so the actor finishes on its
      own and a test can assert an *exact* frame count instead of racing a reconnect.
    * ``cursor`` — a shared read index, so the script continues across the reconnect that
      replaces the source. Without it every new source restarts at entry 0.
    """

    name = "scripted"

    def __init__(
        self,
        config: CameraConfig,
        counter: FrameCounter | None = None,
        *,
        settings: IngestSettings | None = None,
        script: Sequence[Any] | None = None,
        finite: bool = False,
        cursor: list[int] | None = None,
        open_failures: int = 0,
        open_error: BaseException | None = None,
        size: tuple[int, int] = (FRAME_HEIGHT, FRAME_WIDTH),
    ) -> None:
        super().__init__(config, counter, settings=settings)
        self.script = list(script if script is not None else [synthetic_image(0)])
        self.finite = finite
        self.cursor = cursor if cursor is not None else [0]
        self.open_failures = open_failures
        self.open_error = open_error
        self.size = size
        self.opens = 0
        self.closes = 0
        self.reads = 0

    @property
    def is_exhausted(self) -> bool:
        return self.finite and self.cursor[0] >= len(self.script)

    def _do_open(self) -> None:
        self.opens += 1
        if self.open_error is not None:
            raise self.open_error
        if self.opens <= self.open_failures:
            raise ConnectionRefusedError(f"synthetic open failure #{self.opens}")
        self._set_format(self.size[0], self.size[1], self.config.fps or 20.0)

    def _do_read(self):
        self.reads += 1
        index = self.cursor[0]
        self.cursor[0] = index + 1
        if index >= len(self.script):
            if self.finite:
                return None
            index = len(self.script) - 1
        entry = self.script[index]
        if isinstance(entry, BaseException):
            raise entry
        return entry

    def _do_close(self) -> None:
        self.closes += 1


class RecordingSleep:
    """A ``sleep`` that records instead of sleeping, and signals when it has seen enough.

    The :meth:`wait` barrier is what makes these tests deterministic rather than merely
    usually-green: ``actor.start(); actor.stop()`` is a race, because ``stop`` sets the flag
    the loop is checking. A test waits for the delay sequence it asked for, *then* stops.

    Args:
        stop_after: once this many sleeps have been recorded, call ``on_stop`` and release
            :meth:`wait`.
        on_stop: normally ``actor.request_stop`` — assigned after construction, because the
            actor needs the sleep first.
    """

    def __init__(self, *, stop_after: int | None = None) -> None:
        self.delays: list[float] = []
        self.stop_after = stop_after
        self.on_stop: Callable[[], None] | None = None
        self.reached = threading.Event()

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        if self.stop_after is not None and len(self.delays) >= self.stop_after:
            if self.on_stop is not None:
                self.on_stop()
            self.reached.set()

    def wait(self, timeout_s: float = 5.0) -> bool:
        """Block until ``stop_after`` delays have been recorded."""
        return self.reached.wait(timeout_s)

    @property
    def total(self) -> float:
        return sum(self.delays)


def tick(_seconds: float) -> None:
    """A stand-in for ``sleep`` that keeps a retry loop fast but not *hot*.

    A no-op sleep turns the actor's backoff into a spin: at a few hundred thousand
    iterations a second it emits millions of log records, which pytest holds in memory for
    the duration of the test. Bounding the loop at roughly 1 kHz keeps such a test in
    milliseconds and its memory in kilobytes, and it is still three orders of magnitude
    faster than the real backoff. Use it wherever the actor is *not* expected to terminate
    on its own.
    """
    time.sleep(0.001)


class FakeClock:
    """A monotonic clock a test advances by hand, for the pacer's arithmetic."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


# -- fixtures ------------------------------------------------------------------------------


@pytest.fixture()
def make_camera():
    """Build a :class:`CameraConfig` with test-friendly defaults."""

    def _make(
        camera_id: str = "cam0", uri: str = "rtsp://camera/stream", **kwargs
    ) -> CameraConfig:
        return CameraConfig(camera_id=camera_id, uri=uri, **kwargs)

    return _make


@pytest.fixture()
def fast_settings():
    """Ingest settings tuned so a test is not waiting on real timeouts."""

    def _make(**kwargs) -> IngestSettings:
        defaults = {
            "target_model": "ship_detector",
            "input_name": "images",
            "read_timeout_ms": 50,
            "open_timeout_ms": 50,
            "empty_read_sleep_ms": 0,
            "empty_reads_before_reconnect": 2,
            "reconnect_initial_ms": 100,
            "reconnect_max_ms": 1_000,
            "reconnect_factor": 2.0,
            "reconnect_jitter": 0.0,
            "failures_before_unhealthy": 2,
        }
        defaults.update(kwargs)
        return IngestSettings(**defaults)

    return _make


@pytest.fixture()
def scripted_factory():
    """A ``source_factory`` that hands out :class:`ScriptedSource` instances.

    Returns ``(factory, created)`` — ``created`` accumulates every source built, which is how
    a test proves a reconnect made a *new* source while the frame counter carried on.
    """

    def _make(*, shared_script: bool = False, **source_kwargs):
        created: list[ScriptedSource] = []
        cursor: list[int] | None = [0] if shared_script else None

        def factory(config: CameraConfig, counter: FrameCounter) -> ScriptedSource:
            source = ScriptedSource(config, counter, cursor=cursor, **source_kwargs)
            created.append(source)
            return source

        return factory, created

    return _make


@pytest.fixture()
def queue():
    """A small fair queue, so overflow behaviour is reachable in a test."""
    return FairPriorityQueue("ingest", capacity=8)


@pytest.fixture()
def drain():
    """Pop everything currently queued, without blocking on an empty queue."""

    def _drain(target: FairPriorityQueue, limit: int = 512) -> list:
        items: list = []
        while target.depth and len(items) < limit:
            items.extend(target.get_batch(BatchWindow(max_batch_size=min(32, limit))))
        return items

    return _drain


@pytest.fixture()
def frame_dir(tmp_path: Path) -> Path:
    """A directory of PNG frames — the replay fixture with no codec dependency."""
    cv2 = pytest.importorskip("cv2", reason="writing the replay fixture needs OpenCV")
    directory = tmp_path / "frames"
    directory.mkdir()
    for index in range(FRAME_COUNT):
        assert cv2.imwrite(str(directory / f"{index:04d}.png"), synthetic_image(index))
    return directory


@pytest.fixture()
def video_file(tmp_path: Path) -> Path:
    """A tiny video container. Skipped when this OpenCV build has no usable writer."""
    cv2 = pytest.importorskip("cv2", reason="writing the replay fixture needs OpenCV")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("this OpenCV build has no MJPG writer")
    for index in range(FRAME_COUNT):
        writer.write(synthetic_image(index))
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("the OpenCV writer produced no file")
    return path
