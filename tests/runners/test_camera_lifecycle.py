"""Cameras on the in-process runner: added, read, dropped fairly, removed and released.

The file that says arch.md §5① and §5② meet: an ingest actor per camera, publishing through
:class:`~shipinfer.runners.frames.ChainFrameSink` into the runner's bounded per-camera lane,
out through the chain and into a sink a test can read. It runs **offline** — no GPU, no
GStreamer, no camera — because every element is a mock and the frame source is injected.

Two doubles carry it, and both are local on purpose:

* :class:`ScriptedSource` is a four-line cousin of ``tests/ingest/conftest.py``'s. The
  ``tests/`` directories are not packages, so there is nothing to import it from without
  inventing a shared test package; ``tests/runners/test_inprocess.py`` says the same about
  its copy of the mock chain. This one carries only what a runner test needs — a finite
  script and a shared cursor — and deliberately not the reconnect-scripting the ingest tier's
  version exists for.
* :class:`RecordingQueue` is the configured fair queue with one extra list, which is how a
  per-camera priority *band* becomes an assertion rather than an inference from ordering.

One test needs a real decoder, and it is the one that proves the ``decode: {impl: replay}``
name resolves to a real ingest source rather than to a fake this file supplies. It writes
PNGs and is skipped where OpenCV is not installed; everything else is green everywhere, which
is the ratio the offline tier is for.
"""

from __future__ import annotations

import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, ServerStateError
from shipinfer.core.request import Priority
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.ingest import CameraConfig
from shipinfer.core.types import Tensor
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.camera.health import CameraState, IngestSummary
from shipinfer.ingest.frame import FrameCounter
from shipinfer.launch.control import CameraSpec
from shipinfer.runners.inprocess import _NO_INGEST, InprocessRunner
from shipinfer.scheduling.queues import FairPriorityQueue
from shipinfer.scheduling.work import WorkItem
from shipinfer.topology import ChainItem, ChainSpec, ElementKind, Topology
from shipinfer.topology.elements.mock import MockDetect, MockOutput
from shipinfer.topology.registry import registry_for

#: A straight line whose head is a real decode element, so the runner resolves a real ingest
#: source name from it. ``mock`` detect accepts ``bgr@cpu`` as its second choice, which is
#: what the host-memory head cap negotiates down to.
CHAIN = """
name: replayed
elements:
  decode: {impl: __DECODE__}
  detect: {impl: __DETECT__, model: ship_detector}
  output: {impl: mock}
"""

#: Two decode roots that cannot agree on what enters the chain: one host-memory, one the
#: mock's device handle. Every root sees the same submitted frame, so the runner refuses.
TWO_HEADS = """
name: two_heads
elements:
  decode_a: {impl: replay}
  decode_b: {impl: mock, after: []}
  detect:   {impl: mock, model: ship_detector, after: [decode_a, decode_b]}
  output:   {impl: mock}
"""

HEIGHT, WIDTH = 4, 6


def image(index: int) -> np.ndarray:
    """A tiny BGR frame whose first channel encodes its index, so a reorder is visible."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = (index + 1) % 256
    return frame


# -- doubles -------------------------------------------------------------------------------


class ScriptedSource(FrameSource):
    """A source that hands out a fixed list of images and then reports itself exhausted.

    ``finite`` is the knob that makes a test terminate on its own: an exhausted source is not
    a fault, so the actor finishes instead of reconnecting and the frame count becomes an
    equality rather than a lower bound (``ingest/camera/actor.py::_on_empty_read``).
    """

    name: ClassVar[str] = "scripted"

    def __init__(
        self,
        config: CameraConfig,
        counter: FrameCounter | None = None,
        *,
        settings: Any = None,
        frames: int = 3,
        finite: bool = True,
    ) -> None:
        super().__init__(config, counter, settings=settings)
        self.frames = frames
        self.finite = finite
        self.index = 0

    @property
    def is_exhausted(self) -> bool:
        return self.finite and self.index >= self.frames

    def _do_open(self) -> None:
        self._set_format(HEIGHT, WIDTH, self.config.fps or 20.0)

    def _do_read(self) -> np.ndarray | None:
        index = self.index
        self.index += 1
        if index >= self.frames:
            return None
        return image(index)

    def _do_close(self) -> None:
        return None


def scripted(
    frames: int = 3, finite: bool = True, per_camera: dict[str, tuple[int, bool]] | None = None
):
    """A ``source_factory`` handing every camera its own :class:`ScriptedSource`.

    ``per_camera`` is what makes a loud camera and a quiet one one fixture: a fleet is only
    interesting when its cameras differ, and that is exactly the case ADR-005 is about.
    """

    def factory(config: CameraConfig, counter: FrameCounter) -> ScriptedSource:
        count, ends = (per_camera or {}).get(config.camera_id, (frames, finite))
        return ScriptedSource(config, counter, frames=count, finite=ends)

    return factory


@registry_for(ElementKind.DETECT).register("camera-gate")
class GateDetect(MockDetect):
    """A detector that parks the only worker until the test releases it.

    Makes backpressure deterministic: with the worker held here and a lane of one, "the next
    frame found no room" is a fact rather than a race against a poll loop.
    """

    def __init__(self, name: str, params: Any = None, *, model: str | None = None) -> None:
        super().__init__(name, params, model=model)
        self.entered = threading.Event()
        self.release = threading.Event()

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.entered.set()
        self.release.wait(10.0)
        return super()._do_process(item)


class RecordingQueue(FairPriorityQueue):
    """The configured fair queue, plus the band each item was admitted into."""

    def __init__(self, name: str, capacity: int, **options: Any) -> None:
        super().__init__(name, capacity, **options)
        self.bands: list[tuple[str, Priority]] = []

    def put(self, item: WorkItem) -> None:
        super().put(item)
        self.bands.append((item.request.context.camera_id, item.request.priority))

    def band_of(self, camera_id: str) -> set[Priority]:
        return {band for camera, band in self.bands if camera == camera_id}


# -- helpers -------------------------------------------------------------------------------


def load(*, decode: str = "replay", detect: str = "mock") -> Topology:
    text = textwrap.dedent(CHAIN).replace("__DECODE__", decode).replace("__DETECT__", detect)
    return Topology.from_spec(ChainSpec.from_yaml(text))


def settings(**kwargs: Any) -> ServerSettings:
    """Deployment settings with the two sections this runner reads."""
    pipeline = {"workers": 1, "queue_capacity": 64, **kwargs.pop("pipeline", {})}
    ingest = {
        "read_timeout_ms": 50,
        "open_timeout_ms": 50,
        "empty_read_sleep_ms": 0,
        "empty_reads_before_reconnect": 2,
        "reconnect_initial_ms": 10,
        "reconnect_max_ms": 50,
        "reconnect_jitter": 0.0,
        **kwargs.pop("ingest", {}),
    }
    return ServerSettings(pipeline=pipeline, ingest=ingest, **kwargs)


def sink(chain: Topology) -> MockOutput:
    element = chain.node("output").element
    assert isinstance(element, MockOutput)
    return element


def metric(runner: InprocessRunner, name: str) -> Any:
    """One metric handle off the runner's registry, by name."""
    for handle in runner.metrics.registry.collect():
        if handle.name == name:
            return handle
    raise AssertionError(f"{name} is not on the runner's registry")


def until(predicate, timeout_s: float = 10.0, poll_s: float = 0.005) -> bool:
    """Poll a condition, so a test asserts the outcome rather than a sleep length."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


@pytest.fixture()
def runner_over() -> Iterator:
    """Build, start and always stop a runner — a leaked actor thread outlives the test."""
    made: list[InprocessRunner] = []

    def _make(chain: Topology, **kwargs: Any) -> InprocessRunner:
        runner = InprocessRunner(chain, **kwargs)
        made.append(runner)
        runner.start()
        return runner

    yield _make
    for runner in made:
        runner.stop(timeout_s=5.0)


# -- the happy path -------------------------------------------------------------------------


class TestAFrameTravelsFromACameraToTheSink:
    def test_every_frame_reaches_the_output_carrying_its_own_tag(self, runner_over) -> None:
        """The end-to-end claim of B1, over an injected source so it is green everywhere.

        Three frames, three items at the sink, with ``frame_id`` counting from zero — the
        ``(camera_id, frame_id)`` tag ADR-002 says must survive every hand-over, asserted at
        the far end of a chain the frame crossed through a queue and a worker thread.
        """
        chain = load()
        runner = runner_over(chain, settings=settings(), source_factory=scripted(frames=3))

        runner.add_camera(CameraSpec("cam-a", "injected://a", 0.0))

        assert until(lambda: len(sink(chain).emitted) == 3), sink(chain).emitted
        assert [item.key for item in sink(chain).emitted] == [
            ("cam-a", 0),
            ("cam-a", 1),
            ("cam-a", 2),
        ]

    def test_the_payload_is_a_tensor_of_the_decoded_frame(self, runner_over) -> None:
        """A ``pool`` element refuses any other payload by type, so this is not cosmetic.

        ``(1, H, W, C)`` and not ``(H, W, C)``: every request in this system is batch-major
        even at batch one, which is what lets the assembler stack frames without a special
        case (``topology/elements/pool.py``).
        """
        chain = load()
        runner = runner_over(chain, settings=settings(), source_factory=scripted(frames=1))

        runner.add_camera(CameraSpec("cam-a", "injected://a"))

        assert until(lambda: sink(chain).emitted)
        payload = sink(chain).emitted[0].payload
        assert isinstance(payload, Tensor)
        assert payload.shape == (1, HEIGHT, WIDTH, 3)

    def test_the_head_cap_on_the_item_is_the_one_the_loader_negotiated(
        self, runner_over
    ) -> None:
        """``bgr@cpu`` because that is what the *edge* carries, not what a mock stamps."""
        chain = load()
        runner = runner_over(chain, settings=settings(), source_factory=scripted(frames=1))
        recorded: list[str] = []
        original = chain.node("decode").element.process

        def spy(item: ChainItem) -> ChainItem | None:
            recorded.append(str(item.caps))
            return original(item)

        chain.node("decode").element.process = spy  # type: ignore[method-assign]
        runner.add_camera(CameraSpec("cam-a", "injected://a"))

        assert until(lambda: recorded)
        assert recorded[0] == "bgr@cpu"

    def test_a_real_replay_source_decodes_a_directory_of_png_frames(
        self, tmp_path: Path, runner_over
    ) -> None:
        """The one test with a real decoder in it: ``impl: replay`` must resolve to one.

        Everything else here injects a source, which proves the lifecycle and proves nothing
        about the name in the chain file. This is the half that does — the runner puts
        ``source="replay"`` on the camera config and ``ingest/registry.py`` has to find a
        :class:`~shipinfer.ingest.sources.replay.ReplaySource` behind it.

        ``fps`` is set because the replay source paces itself to it: at ``0.0`` a looping
        directory delivers as fast as the CPU allows, which is a hot loop rather than a test.
        """
        cv2 = pytest.importorskip("cv2", reason="writing the PNG fixture needs OpenCV")
        frames = tmp_path / "frames"
        frames.mkdir()
        for index in range(4):
            assert cv2.imwrite(str(frames / f"{index:04d}.png"), image(index))

        chain = load()
        runner = runner_over(chain, settings=settings())
        runner.add_camera(CameraSpec("cam-file", str(frames), 50.0))

        assert until(lambda: len(sink(chain).emitted) >= 4), sink(chain).emitted
        assert {item.key[0] for item in sink(chain).emitted} == {"cam-file"}
        emitted = sink(chain).emitted[0]
        assert isinstance(emitted.payload, Tensor)
        assert emitted.payload.shape == (1, HEIGHT, WIDTH, 3)

    def test_a_camera_configured_in_the_settings_tree_starts_with_the_runner(
        self, runner_over
    ) -> None:
        """``ingest.cameras`` is the fleet; ``add_camera`` is how one arrives later.

        Both doors, one manager: a runner that only honoured the RPC would leave a configured
        deployment silently camera-less.
        """
        chain = load()
        runner = runner_over(
            chain,
            settings=settings(
                ingest={"cameras": [{"camera_id": "cam-cfg", "uri": "injected://cfg"}]}
            ),
            source_factory=scripted(frames=2),
        )

        assert runner.cameras == ("cam-cfg",)
        assert until(lambda: len(sink(chain).emitted) == 2)


# -- fairness and attribution ---------------------------------------------------------------


class TestTheLoudCameraPaysForItsOwnFlood:
    def test_the_drop_is_charged_to_the_camera_that_flooded(self, runner_over) -> None:
        """ADR-005 in one assertion: the greedy camera loses frames, the quiet one does not.

        Deterministic by construction rather than by timing. One worker, parked inside the
        gate on the quiet camera's only frame; a lane of one, holding the loud camera's first
        frame; everything the loud camera produces after that finds no room. The previous
        generation's shared evict-oldest buffer would have thrown away the quiet camera's
        frame to make room for the loud one's — which is the bug this project exists to fix.

        Both counters are asserted, because one dropped frame is deliberately counted twice
        and the pair answers two different questions (``runners/frames.py``): the admission
        door's ledger for the shard, and the camera's own read-against-``frames_read`` ratio.
        """
        chain = load(detect="camera-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = runner_over(
            chain,
            settings=settings(pipeline={"queue_capacity": 1}),
            # The quiet camera offers exactly one frame and then finishes; the loud one never
            # stops. Anything else and "quiet lost nothing" would be luck.
            source_factory=scripted(per_camera={"quiet": (1, True), "loud": (10_000, False)}),
        )

        runner.add_camera(CameraSpec("quiet", "injected://quiet"))
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        runner.add_camera(CameraSpec("loud", "injected://loud"))

        assert until(lambda: runner.metrics.items_dropped.value(camera="loud") >= 2)
        assert runner.metrics.items_dropped.value(camera="quiet") == 0

        cameras = runner.health()["cameras"]
        assert cameras["loud"]["frames_dropped"] >= 2
        assert cameras["quiet"]["frames_dropped"] == 0
        ingest_drops = metric(runner, "shipinfer_ingest_frames_dropped_total")
        assert ingest_drops.value(camera="loud", reason="sink_full") >= 2
        assert ingest_drops.value(camera="quiet", reason="sink_full") == 0

        gate.release.set()

    def test_the_ingest_metrics_share_the_runners_registry(self, runner_over) -> None:
        """One exporter, both halves of a lost frame. Two registries is two dashboards."""
        chain = load()
        runner = runner_over(chain, settings=settings(), source_factory=scripted(frames=1))
        runner.add_camera(CameraSpec("cam-a", "injected://a"))

        assert until(lambda: sink(chain).emitted)
        registry = runner.metrics.registry
        assert "shipinfer_ingest_frames_total" in registry
        assert "shipinfer_runner_items_accepted_total" in registry


class TestThePriorityBandComesFromTheCameraConfig:
    def test_a_configured_camera_is_admitted_into_its_own_band(self) -> None:
        """``priority:`` on a camera used to apply to nothing at all.

        Every item was admitted at the default, so a camera watching a restricted area queued
        behind an idle one — the one customisation ADR-005 says a generic inference server
        cannot express, configured and then ignored.
        """
        chain = load()
        queue = RecordingQueue("recording", 64)
        runner = InprocessRunner(
            chain,
            settings=settings(
                ingest={
                    "cameras": [
                        {
                            "camera_id": "cam-hot",
                            "uri": "injected://hot",
                            "priority": Priority.TRACKING_CRITICAL,
                        },
                        {"camera_id": "cam-cold", "uri": "injected://cold"},
                    ]
                }
            ),
            queue=queue,
            source_factory=scripted(frames=2),
        )
        runner.start()
        try:
            assert until(lambda: len(sink(chain).emitted) == 4), sink(chain).emitted
        finally:
            runner.stop(timeout_s=5.0)

        assert queue.band_of("cam-hot") == {Priority.TRACKING_CRITICAL}
        assert queue.band_of("cam-cold") == {Priority.NORMAL}

    def test_a_camera_nobody_configured_gets_the_default_rather_than_a_refusal(
        self,
    ) -> None:
        """A camera added at runtime is normal — a site gains cameras during commissioning."""
        chain = load()
        queue = RecordingQueue("recording", 64)
        runner = InprocessRunner(
            chain, settings=settings(), queue=queue, source_factory=scripted(frames=1)
        )
        runner.start()
        try:
            runner.add_camera(CameraSpec("cam-new", "injected://new"))
            assert until(lambda: queue.bands)
        finally:
            runner.stop(timeout_s=5.0)

        assert queue.band_of("cam-new") == {Priority.NORMAL}

    def test_the_critical_band_survives_the_falsy_zero(self) -> None:
        """``Priority.TRACKING_CRITICAL`` is ``0``, so ``get(...) or default`` demotes it.

        The bug is invisible in every ordinary test — every other band is truthy — and it
        silently demotes the one camera whose priority was the reason for having priorities.
        """
        chain = load()
        runner = InprocessRunner(chain, settings=settings())
        runner._priorities["cam-x"] = Priority.TRACKING_CRITICAL

        assert runner._priority_for("cam-x") is Priority.TRACKING_CRITICAL


# -- the control plane's three methods -------------------------------------------------------


class TestAddRemoveAndDrain:
    def test_adding_before_start_is_refused_rather_than_implicitly_starting(self) -> None:
        """The chain is not open, so the first frame would meet an element's own refusal."""
        runner = InprocessRunner(load(), settings=settings())

        with pytest.raises(ServerStateError, match="before start"):
            runner.add_camera(CameraSpec("cam-a", "injected://a"))

    def test_a_duplicate_id_is_refused_and_the_first_camera_keeps_running(
        self, runner_over
    ) -> None:
        """Two actors on one stream would be two frame counters minting duplicate tags."""
        chain = load()
        runner = runner_over(
            chain, settings=settings(), source_factory=scripted(frames=64, finite=False)
        )
        runner.add_camera(CameraSpec("cam-a", "injected://a"))

        with pytest.raises(ConfigurationError, match="already running"):
            runner.add_camera(CameraSpec("cam-a", "injected://again"))

        assert runner.cameras == ("cam-a",)

    def test_removing_an_unknown_camera_names_what_is_running(self, runner_over) -> None:
        """A typo in an operator's call gets an answer, not a silent no-op."""
        runner = runner_over(
            load(), settings=settings(), source_factory=scripted(frames=64, finite=False)
        )
        runner.add_camera(CameraSpec("cam-a", "injected://a"))

        with pytest.raises(ConfigurationError, match="cam-b"):
            runner.remove_camera("cam-b")

    def test_removing_an_unknown_camera_on_a_runner_with_none_says_the_same_thing(
        self, runner_over
    ) -> None:
        """One answer for one fact: a caller that had to tell these apart writes an if/elif."""
        runner = runner_over(load(), settings=settings())

        with pytest.raises(ConfigurationError, match="not running"):
            runner.remove_camera("cam-a")

    def test_a_removed_camera_stops_cleanly_and_leaves_the_set(self, runner_over) -> None:
        runner = runner_over(
            load(), settings=settings(), source_factory=scripted(frames=64, finite=False)
        )
        runner.add_camera(CameraSpec("cam-a", "injected://a"))
        runner.add_camera(CameraSpec("cam-b", "injected://b"))

        assert runner.remove_camera("cam-a", timeout_s=5.0) is True
        assert runner.cameras == ("cam-b",)

    def test_draining_a_runner_with_no_cameras_is_clean_and_zero(self, runner_over) -> None:
        """``0`` is the honest answer for "nothing had to be abandoned", either way."""
        runner = runner_over(load(), settings=settings())

        assert runner.drain(timeout_s=1.0) == 0

    def test_a_drain_releases_every_camera_and_leaves_the_chain_running(
        self, runner_over
    ) -> None:
        """A drain empties a shard without tearing it down — the workers stay up."""
        chain = load()
        runner = runner_over(
            chain, settings=settings(), source_factory=scripted(frames=64, finite=False)
        )
        runner.add_camera(CameraSpec("cam-a", "injected://a"))
        assert until(lambda: sink(chain).emitted)

        assert runner.drain(timeout_s=5.0) == 0

        assert runner.cameras == ()
        assert runner.is_running
        assert runner.health()["workers"]["alive"] == 1

    def test_a_camera_may_be_added_after_a_drain(self, runner_over) -> None:
        """Whether it *should* be is the shard servicer's refusal, not this runner's."""
        chain = load()
        runner = runner_over(chain, settings=settings(), source_factory=scripted(frames=2))
        runner.add_camera(CameraSpec("cam-a", "injected://a"))
        runner.drain(timeout_s=5.0)

        runner.add_camera(CameraSpec("cam-b", "injected://b"))

        assert until(
            lambda: any(key[0] == "cam-b" for key in [i.key for i in sink(chain).emitted])
        )


# -- what health and stats say ---------------------------------------------------------------


class TestHealthAndStatsCarryTheCameras:
    def test_health_names_every_camera_and_its_state(self, runner_over) -> None:
        """Keyed by id and carrying the ingest plane's own snapshot verbatim.

        The count is never the interesting answer: which camera is CONNECTING, which is
        EXHAUSTED at the end of its file and which is dropping frames is.
        """
        chain = load()
        runner = runner_over(chain, settings=settings(), source_factory=scripted(frames=2))
        runner.add_camera(CameraSpec("cam-a", "injected://a"))
        assert until(lambda: len(sink(chain).emitted) == 2)

        cameras = runner.health()["cameras"]

        assert set(cameras) == {"cam-a"}
        assert cameras["cam-a"]["camera_id"] == "cam-a"
        assert cameras["cam-a"]["frames_read"] == 2
        assert cameras["cam-a"]["frames_published"] == 2

    def test_a_runner_with_no_cameras_reports_an_empty_map_not_a_missing_key(self) -> None:
        """A missing key is a runner lying about what it is.

        ``runners/service.py`` derives the shard state from this map, so a runner that
        manages cameras and reports no key answers ``ready`` forever — never ``running`` —
        however many cameras it is actually reading.
        """
        runner = InprocessRunner(load(), settings=settings())

        assert runner.health()["cameras"] == {}
        assert InprocessRunner.manages_cameras is True

    def test_an_exhausted_source_is_visible_in_health(self, runner_over) -> None:
        """A finite file finishing is not a fault, and an operator has to be able to see it."""
        chain = load()
        runner = runner_over(chain, settings=settings(), source_factory=scripted(frames=2))
        runner.add_camera(CameraSpec("cam-a", "injected://a"))

        assert until(
            lambda: runner.health()["cameras"].get("cam-a", {}).get("state")
            == CameraState.EXHAUSTED.value
        ), runner.health()["cameras"]

    def test_stats_carry_the_producers_side_of_the_ledger(self, runner_over) -> None:
        chain = load()
        runner = runner_over(chain, settings=settings(), source_factory=scripted(frames=3))
        runner.add_camera(CameraSpec("cam-a", "injected://a"))
        assert until(lambda: len(sink(chain).emitted) == 3)

        stats = runner.stats()

        assert stats["ingest"]["frames_read"] == 3
        assert stats["ingest"]["frames_published"] == 3
        assert stats["items"]["accepted"] == 3

    def test_the_no_camera_placeholder_has_the_ingest_summary_shape(self) -> None:
        """The copy that keeps ``import shipinfer.runners`` free of a decode runtime.

        ``_NO_INGEST`` is written out rather than taken from :class:`IngestSummary`, so this
        is the check that stops the two drifting: a field added to the summary and not here
        would make a runner's ``stats()`` change shape the moment it got its first camera.
        """
        summary = IngestSummary(
            cameras=0,
            streaming=0,
            unhealthy=0,
            total_fps=0.0,
            frames_read=0,
            frames_published=0,
            frames_dropped=0,
        )

        assert dict(_NO_INGEST) == summary.as_dict()

    def test_a_runner_with_no_cameras_reports_the_placeholder(self) -> None:
        runner = InprocessRunner(load(), settings=settings())

        assert runner.stats()["ingest"] == dict(_NO_INGEST)


# -- shutdown ---------------------------------------------------------------------------------


class TestStopReleasesTheCameras:
    def test_no_actor_thread_outlives_the_stop(self) -> None:
        """A daemon thread left reading is a producer publishing into a closed queue.

        It also outlives the *test*, which is how one leaked camera makes a later, unrelated
        test flaky — so this asserts on the process's thread list rather than on the manager.
        """
        chain = load()
        runner = InprocessRunner(
            chain, settings=settings(), source_factory=scripted(frames=64, finite=False)
        )
        runner.start()
        runner.add_camera(CameraSpec("cam-a", "injected://a"))
        assert until(lambda: sink(chain).emitted)

        runner.stop(timeout_s=5.0)

        assert until(
            lambda: not [t for t in threading.enumerate() if t.name.startswith("ingest-cam-a")]
        ), [t.name for t in threading.enumerate()]
        assert runner.cameras == ()

    def test_the_stop_leaves_no_unresolved_future(self) -> None:
        """Every admitted frame gets a typed outcome, including the ones caught mid-flight.

        The sink discards the future it is handed on purpose, so nothing here is *waiting* on
        one — but the runner still promises one, and a frame that vanishes with no outcome at
        all is the failure ADR-005 exists to prevent. Recorded on the way past and checked
        afterwards.
        """
        chain = load(detect="camera-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        futures: list[Any] = []
        runner = InprocessRunner(
            chain, settings=settings(), source_factory=scripted(frames=64, finite=False)
        )
        submit = runner._do_submit

        def recording(item: ChainItem):
            future = submit(item)
            futures.append(future)
            return future

        runner._do_submit = recording  # type: ignore[method-assign]
        runner.start()
        try:
            runner.add_camera(CameraSpec("cam-a", "injected://a"))
            assert until(lambda: len(futures) >= 4)
        finally:
            gate.release.set()
            runner.stop(timeout_s=5.0)

        assert futures, "no frame was ever admitted"
        assert until(lambda: all(future.done() for future in futures)), [
            index for index, future in enumerate(futures) if not future.done()
        ]

    def test_a_restart_reads_the_configured_cameras_again(self) -> None:
        """The manager is dropped at the stop, not reused.

        A decoder abandoned at a shutdown deadline is parked inside a read, not gone, and it
        still holds a sink bound to the stopped cycle. Handing that manager back would make
        the stale actor a live producer into the new cycle's queue — the ingest-side spelling
        of the bug ``_work`` documents on the worker side.
        """
        chain = load()
        runner = InprocessRunner(
            chain,
            settings=settings(
                ingest={"cameras": [{"camera_id": "cam-cfg", "uri": "injected://cfg"}]}
            ),
            source_factory=scripted(frames=1),
        )
        runner.start()
        assert until(lambda: len(sink(chain).emitted) == 1)
        runner.stop(timeout_s=5.0)
        assert runner.cameras == ()

        runner.start()
        try:
            assert until(lambda: len(sink(chain).emitted) == 2)
            assert runner.cameras == ("cam-cfg",)
        finally:
            runner.stop(timeout_s=5.0)


# -- what the chain has to say --------------------------------------------------------------


class TestTheChainDecidesTheHead:
    def test_two_decode_roots_that_disagree_are_refused_at_start(self) -> None:
        """One manager publishes one item and every root sees it, so there is no answer."""
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(TWO_HEADS)))
        runner = InprocessRunner(chain, settings=settings())

        with pytest.raises(ConfigurationError, match="disagree about what enters"):
            runner.start()

        assert not runner.is_running

    def test_the_decode_impl_decides_which_ingest_source_a_camera_gets(self) -> None:
        """``decode: {impl: pyav}`` puts ``source="pyav"`` on every camera config.

        Asserted on the config the runner builds rather than by opening a decoder, which is
        what keeps this test offline — resolving the *name* is the runner's half, and
        ``ingest/registry.py`` owns the other.
        """
        runner = InprocessRunner(load(decode="pyav"), settings=settings())

        config = runner._camera_config(CameraSpec("cam-a", "rtsp://camera/live", 20.0))

        assert config.source == "pyav"
        assert config.camera_id == "cam-a"
        assert config.uri == "rtsp://camera/live"
        assert config.fps == 20.0

    def test_a_params_source_outranks_the_class_default(self) -> None:
        """A slot's own configuration beats its class's, as everywhere else in the chain."""
        text = textwrap.dedent(CHAIN).replace(
            "{impl: __DECODE__}", "{impl: replay, params: {source: pyav}}"
        )
        chain = Topology.from_spec(ChainSpec.from_yaml(text.replace("__DETECT__", "mock")))
        runner = InprocessRunner(chain, settings=settings())

        assert runner._camera_config(CameraSpec("cam-a", "x")).source == "pyav"

    def test_the_head_cap_is_read_from_the_edge_and_not_from_produces(self) -> None:
        chain = load()
        runner = InprocessRunner(chain, settings=settings())

        assert str(runner._head().caps) == str(chain.edges[0].caps) == "bgr@cpu"
