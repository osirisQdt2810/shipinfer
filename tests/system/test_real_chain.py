"""The real chain, end to end: a video file in, perception events out. No mock anywhere.

The system test rule (V148): a feature is exercised on the pipeline it will actually run —
``decode -> detect -> track -> output``, real OpenCV decode, a real TensorRT engine, real
ByteTrack, a real sink — not on a chain of mocks that proves only that the wiring compiles.

Driven **in process** rather than through ``shipinfer run``, which blocks until interrupted:
the composition below is that command's (``cli/commands/run.py``), reusing its own helpers so
the two cannot drift, with a bounded wait and a deterministic stop in place of its ``_wait``.

One worker, on purpose: the tracker's ordering guard refuses a frame that lost a race between
workers and files ``track`` in ``missing_stages``, which the id assertions would have to be
weakened to tolerate. The engine, the footage and the submodule are all gitignored or
vendored, so each is a **named** skip carrying the command that supplies it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from importlib.util import find_spec
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from tests.support.devices import a_test_device

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "model_repository" / "ship_detector" / "1" / "model.plan"

#: The documented default footage: a real RTSP capture of people, in the gitignored reference
#: checkout. ``SHIPINFER_SYSTEM_VIDEO`` overrides it with any video file or directory of
#: frames — the two things ``ingest/sources/replay.py`` opens — holding at least
#: :data:`EVENTS_WANTED` frames.
DEFAULT_VIDEO = (
    REPO
    / "references/gitea-body-face-joint-yolo/test_imgs/111206"
    / "vlc-record-2024-01-18-16h12m36s-rtsp___172.23.111.206_554_main-.mp4"
)
VIDEO = Path(os.environ.get("SHIPINFER_SYSTEM_VIDEO") or DEFAULT_VIDEO)


#: How many frames the chain must publish before the run is stopped. Eight is enough for the
#: tracker to confirm a track (it withholds one until it has earned confirmation) and to show
#: the id holding across several frames, while keeping the run inside a test's patience.
EVENTS_WANTED = 8

#: The chain under test. Every ``impl:`` here is a production implementation; the assertion
#: that it stayed that way is :class:`TestNoMockTookPart`. ``flush_every: 0`` so the file is
#: readable while the chain is still running, which is what the bounded wait polls.
CHAIN = """
name: real_chain
elements:
  decode: {{impl: replay}}
  detect: {{impl: pool, model: ship_detector}}
  track:  {{impl: shipvision, per: camera}}
  output: {{impl: jsonlines, params: {{path: {path}, flush_every: 0}}}}
"""

#: Applied to every test here rather than to each class: the two classes differ in what they
#: assert, not in what they need. ``timeout`` overrides the suite's 120 s, which a TensorRT
#: load plus a paced replay does not fit inside.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.timeout(600),
    pytest.mark.skipif(
        not PLAN.is_file(),
        reason=f"no engine at {PLAN}; build one with `python scripts/build_engines.py`",
    ),
    pytest.mark.skipif(
        not VIDEO.exists(),
        reason=f"no footage at {VIDEO}; check out the reference repositories under "
        f"references/, or point SHIPINFER_SYSTEM_VIDEO at a video file or frame directory",
    ),
    pytest.mark.skipif(
        find_spec("shipvision") is None,
        reason="the shipvision submodule is not importable; run "
        "`git submodule update --init 3rdparty/shipvision` and put it on PYTHONPATH",
    ),
]


@dataclass(frozen=True)
class Source:
    """What OpenCV says the input is — the independent oracle the events must match."""

    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class RealRun:
    """One execution of the real chain, read back after the GPU has been given up."""

    events: list[dict[str, Any]]
    chain: Any
    platform: str
    source: Source


def source_truth(path: Path) -> Source:
    """Open the input with OpenCV directly and report its geometry and rate.

    An oracle rather than a second implementation: cv2 is what ``ReplaySource`` reads these
    numbers from, so asking it here says "the chain propagated the source's truth" without
    re-deriving that truth from our own code. A directory of frames has no container rate, so
    it takes the source's documented fallback — imported, not spelled ``25`` here.
    """
    import cv2

    from shipinfer.ingest.sources.replay import _FALLBACK_FPS, FRAME_SUFFIXES

    if path.is_dir():
        frames = sorted(p for p in path.iterdir() if p.suffix.lower() in FRAME_SUFFIXES)
        image = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        return Source(int(image.shape[1]), int(image.shape[0]), _FALLBACK_FPS)
    capture = cv2.VideoCapture(str(path))
    try:
        return Source(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)) or _FALLBACK_FPS,
        )
    finally:
        capture.release()


def read_events(path: Path) -> list[dict[str, Any]]:
    """The sink's file as parsed objects, refusing a line that is not JSON."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def wait_for(path: Path, wanted: int, timeout_s: float = 120.0) -> None:
    """Block until the sink has written ``wanted`` complete lines, or say how few arrived.

    Newlines are counted rather than the file parsed: a line longer than the stream's buffer
    can be half-written when this looks, and a ``JSONDecodeError`` from the poll would report
    a bug in the sink that is not there. A failure here is the chain not producing, which is
    what this file exists to catch, so it raises rather than leaving the assertions an empty
    file to read.
    """
    deadline = time.monotonic() + timeout_s
    complete = 0
    while time.monotonic() < deadline:
        complete = path.read_text(encoding="utf-8").count("\n") if path.is_file() else 0
        if complete >= wanted:
            return
        time.sleep(0.1)
    raise AssertionError(
        f"the chain published {complete} event(s) in {timeout_s:.0f}s, wanted {wanted}; "
        f"input {VIDEO} must hold at least that many frames"
    )


def run_the_chain(out: Path) -> RealRun:
    """Build the real chain the way ``shipinfer run`` does, drive it, and give the GPU back.

    The imports are function-local because this module is collected in the offline tier too —
    where the ``gpu`` marker deselects it — and ``shipinfer.engine`` pulls in torch.
    """
    from shipinfer.cli.commands.run import (
        cameras_from_inputs,
        image_ops_are_needed,
        model_pool_is_needed,
        place_cameras,
    )
    from shipinfer.core.settings import ServerSettings
    from shipinfer.engine import InferenceServer
    from shipinfer.runners import build_runner
    from shipinfer.runtime.ops import get_thread_local_image_ops
    from shipinfer.topology import ChainSpec, Topology

    text = CHAIN.format(path=out)
    chain = Topology.from_spec(ChainSpec.from_yaml(text, name="real_chain"))
    # The repository's own `ship_detector/config.yaml` and its own engine, loaded alone: the
    # chain names one model, and the other three would cost VRAM and load time for nothing.
    settings = ServerSettings(
        model_repository=REPO / "model_repository",
        load_all_models=False,
        startup_models=["ship_detector"],
        devices={"visible_gpus": [a_test_device()]},
        pipeline={"workers": 1},
    )
    assert model_pool_is_needed("inprocess", chain), "a `pool` detector needs a model pool"
    assert image_ops_are_needed("inprocess", chain), "a `pool` detector needs image ops"
    engine = InferenceServer(settings)
    try:
        ops = get_thread_local_image_ops(
            settings.execution.provider,
            devices=tuple(engine.devices.visible_gpus),
            device_manager=engine.devices,
            memory=engine.memory,
        )
        runner = build_runner(
            "inprocess", chain, settings, chain_yaml=text, models=engine, ops=ops
        )
        engine.start()
        platform = engine.model("ship_detector").artifact.config.platform
        runner.start()
        try:
            place_cameras(runner, cameras_from_inputs([str(VIDEO)], loop=False))
            wait_for(out, EVENTS_WANTED)
        finally:
            runner.stop(timeout_s=15.0)
    finally:
        # A crash must not be what frees the device (CLAUDE.md's hygiene rule): the engine's
        # own teardown releases the contexts, and the cache is dropped behind it so a shared
        # box gets the allocator's reservation back too.
        engine.stop()
        release_cuda()
    return RealRun(read_events(out), chain, str(platform), source_truth(VIDEO))


def release_cuda() -> None:
    """Hand the caching allocator's reservation back, if torch is even importable."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - hygiene must never mask the real failure
        pass


def detections(event: dict[str, Any]) -> list[tuple[str, float, list[float]]]:
    """Every detection in one event, whatever class it landed in.

    Ships and people are parallel arrays in the v1-descended payload, so a test that read only
    ``det_id_vec`` would pass on footage full of ships by asserting nothing at all.
    """
    people = zip(
        event["det_id_vec"], event["det_body_score_vec"], event["body_bbox_vec"], strict=True
    )
    ships = zip(
        event["ship_det_id_vec"],
        event["det_ship_score_vec"],
        event["ship_bbox_vec"],
        strict=True,
    )
    return [(str(i), float(s), list(b)) for i, s, b in (*people, *ships)]


def track_ids(event: dict[str, Any]) -> set[int]:
    """The published track ids in one event; empty while the tracker is still warming up."""
    both = (*event["body_track_id_vec"], *event["ship_track_id_vec"])
    return {int(i) for i in both if i is not None}


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> RealRun:
    """One run of the real chain, shared by every assertion in this module.

    Module-scoped because a TensorRT engine load per test method would turn seven assertions
    into seven engine loads. The run is torn down before any of them read it, so nothing here
    holds a device while the file is being asserted on.
    """
    return run_the_chain(tmp_path_factory.mktemp("real_chain") / "events.jsonl")


class TestTheRealChainProducesRealEvents:
    def test_every_frame_becomes_one_jsonl_event_of_the_current_schema(self, run) -> None:
        """The sink's file parses, and the frames arrive once each, in order, under one id."""
        from shipinfer.core.events import SCHEMA_VERSION

        assert len(run.events) >= EVENTS_WANTED
        assert [e["image_id"] for e in run.events] == list(range(len(run.events)))
        assert {e["camera_id"] for e in run.events} == {"cam-000"}
        assert {e["schema_version"] for e in run.events} == {SCHEMA_VERSION}
        assert {e["type"] for e in run.events} == {"Det2MOT"}

    def test_no_stage_was_skipped_and_every_event_is_timed(self, run) -> None:
        """A complete frame, from a chain where every element ran on it.

        ``missing_stages`` is the field that distinguishes "nothing was there" from "the
        tracker never saw it", so a run where it is empty everywhere is the strong claim.
        """
        assert [e["missing_stages"] for e in run.events] == [[] for _ in run.events]
        assert {e["reason"] for e in run.events} == {"complete"}
        assert not any(e["partial"] for e in run.events)
        assert all(e["latency_us"] > 0 for e in run.events)
        assert all(e["emitted_unix_ns"] > e["captured_unix_ns"] > 0 for e in run.events)

    def test_the_detector_found_real_objects_with_plausible_scores(self, run) -> None:
        """The shape of the truth, not a golden: a re-encode moves the last decimal.

        Boxes are asserted to lie inside the *source* frame, which is what says the letterbox
        was undone against the original pixels rather than left in 640x640 network space.
        """
        for event in run.events:
            rows = detections(event)
            assert len(rows) >= 2, f"frame {event['image_id']} found {len(rows)} object(s)"
            assert len({det_id for det_id, _, _ in rows}) == len(rows)
            assert all(0.0 < score < 1.0 for _, score, _ in rows)
            assert max(score for _, score, _ in rows) > 0.8
            for _, _, (x1, y1, x2, y2) in rows:
                assert 0 <= x1 < x2 <= run.source.width
                assert 0 <= y1 < y2 <= run.source.height

    def test_the_tracker_published_ids_that_hold_across_frames(self, run) -> None:
        """Real tracking, which is the property a per-frame detector cannot fake.

        Stability is asserted over *consecutive* frames rather than over the whole run: the
        tracker withholds a track until it is confirmed, so the first frames legitimately
        carry no id at all.
        """
        published = [track_ids(event) for event in run.events]
        assert any(published), "no detection was ever given a track id"
        pairs = pairwise(published)
        held = [before for before, after in pairs if before and before == after]
        assert len(held) >= 3, published

    def test_a_track_id_and_its_state_are_published_together(self, run) -> None:
        """Never an id with no state, or a state with no id — a consumer joins on both."""
        for event in run.events:
            states = (*event["body_track_state_vec"], *event["ship_track_state_vec"])
            ids = (*event["body_track_id_vec"], *event["ship_track_id_vec"])
            assert [s is None for s in states] == [i is None for i in ids]

    def test_the_event_carries_the_source_s_own_geometry_and_rate(self, run) -> None:
        """Width, height and fps are the input's, carried the whole way to the sink."""
        assert {e["img_width"] for e in run.events} == {run.source.width}
        assert {e["img_height"] for e in run.events} == {run.source.height}
        assert {e["img_fps"] for e in run.events} == {round(run.source.fps)}


class TestNoMockTookPart:
    def test_every_slot_resolved_to_its_production_implementation(self, run) -> None:
        """The guard on the whole file: swap a mock back in and this goes red.

        Exact classes rather than a name match, because ``impl:`` is a registry key and a
        double registered under ``pool`` would satisfy the string.
        """
        from shipinfer.topology.elements.decode import ReplayDecode
        from shipinfer.topology.elements.output import JsonLinesOutput
        from shipinfer.topology.elements.pool import PoolDetect
        from shipinfer.topology.elements.track import ShipvisionTrack

        expected = {
            "decode": ReplayDecode,
            "detect": PoolDetect,
            "track": ShipvisionTrack,
            "output": JsonLinesOutput,
        }
        resolved = {node.name: type(node.element) for node in run.chain}
        assert resolved == expected, run.chain.describe()

    def test_no_element_came_from_the_mock_module_and_the_model_is_a_real_engine(
        self, run
    ) -> None:
        """The two doors a mock gets in by: the element registry, and the model's platform."""
        modules = {type(node.element).__module__ for node in run.chain}
        assert not any(module.endswith(".mock") for module in modules), modules
        assert run.platform == "tensorrt"
        assert {e["sub_id"] for e in run.events} == {"shard-0"}
