"""A chain walked by a real runner, ending in a file of schema-v4 events.

The end-to-end this slice exists for. Everything else about the ``output`` element is a unit
test over one item; this is the one that says a *frame* entering an in-process runner comes
out the far end as one JSON line with the identity the stateful tier assigned — through the
fair lane, the worker threads and the real ingest actor, none of which any unit test touches.

Two runs, and both are needed:

* :class:`TestAChainOfDoublesWritesEvents` uses stand-ins for ``track`` and ``mtmc`` and
  therefore runs on a checkout with **no submodule**, which is what CI has. What it pins is
  the *contract between the elements*: the shapes C4 and C6 file are the shapes the sink
  reads, and a change to either goes red here.
* :class:`TestTheRealStatefulTierWritesEvents` runs the actual shipvision elements. What it
  pins is that those shapes are the ones really produced — a doubles-only test would agree
  with a mistake in the double forever.
"""

from __future__ import annotations

import json
import textwrap
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.ingest import CameraConfig
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.frame import FrameCounter
from shipinfer.launch.control import CameraSpec
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.topology import ChainSpec, Topology
from shipinfer.topology import bridge as bridge_module
from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.elements.detections import Detections
from shipinfer.topology.registry import registry_for

pytestmark = [pytest.mark.timeout(120)]

needs_shipvision = pytest.mark.skipif(
    not bridge_module.shipvision_available(),
    reason="shipvision is not importable; the submodule is not checked out",
)

HEIGHT, WIDTH = 400, 400

#: One ship and one person per frame, far apart, so an IoU attribution has one right answer.
BOXES = np.array([[10.0, 10.0, 110.0, 210.0], [300.0, 300.0, 340.0, 380.0]], dtype=np.float32)


# -- doubles -----------------------------------------------------------------------------


@registry_for(ElementKind.DETECT).register("events-detect")
class EventsDetect(Element):
    """A detector that files what an ``output`` element reads: rows *and* the frame extent.

    ``mock`` files a one-row ``Detections`` with a ``frame_hw`` of ``(100, 100)`` that no
    source ever produced. This one files two rows of two classes at the source's real size,
    which is what makes the ship/person split in the emitted event mean something.
    """

    kind: ClassVar[ElementKind] = ElementKind.DETECT
    accepts: ClassVar[tuple[str, ...]] = ("bgr@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("bgr@cpu",)

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem:
        return item.derive(
            detections=Detections(
                boxes=BOXES,
                scores=np.array([0.9, 0.8], dtype=np.float32),
                class_ids=np.array([8, 0], dtype=np.int32),
                labels=("ship", "person"),
            ),
            frame_hw=(HEIGHT, WIDTH),
        )


@registry_for(ElementKind.EMBED).register("events-embed")
class EventsEmbed(Element):
    """An embedder filing one vector per detection row, keyed by row as a crop element does."""

    kind: ClassVar[ElementKind] = ElementKind.EMBED
    accepts: ClassVar[tuple[str, ...]] = ("bgr@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("bgr@cpu",)

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem:
        rows = len(item.meta["detections"])
        return item.derive(
            vectors={
                index: np.array([1.0, float(index)], dtype=np.float32) for index in range(rows)
            }
        )


@dataclass
class PublishedTrack:
    """What the ``output`` element reads off a track. Two fields is the whole contract."""

    track_id: int
    state: str = "confirmed"


@registry_for(ElementKind.TRACK).register("events-track")
class EventsTrack(Element):
    """A tracker filing C4's shape: publishable tracks, and the row each came from.

    The track objects are a two-field namespace on purpose — the element under test reads
    ``track_id`` and ``state`` and nothing else, and a double that carried more would suggest
    otherwise.
    """

    kind: ClassVar[ElementKind] = ElementKind.TRACK
    accepts: ClassVar[tuple[str, ...]] = ("bgr@cpu", "meta@cpu")
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem:
        rows = range(len(item.meta["detections"]))
        # Track ids that are not row indices, so a test cannot pass by confusing the two.
        tracks = [PublishedTrack(100 + index) for index in rows]
        return item.derive(
            caps=self.output_caps[0],
            payload=None,
            tracks=tracks,
            track_rows=tuple(rows),
        )


@registry_for(ElementKind.MTMC).register("events-mtmc")
class EventsMtmc(Element):
    """A cross-camera tier filing C6's shape: one global id per track, aligned with them."""

    kind: ClassVar[ElementKind] = ElementKind.MTMC
    accepts: ClassVar[tuple[str, ...]] = ("meta@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem:
        tracks = item.meta.get("tracks", ())
        return item.derive(global_ids=[900 + index for index in range(len(tracks))])


@registry_for(ElementKind.TRACK).register("events-track-per-camera")
class EventsTrackPerCamera(EventsTrack):
    """The same tracker, with ids that differ per camera — as a real one's do.

    ``shipvision`` hands out track ids from one process-wide counter precisely so two
    cameras' tracklets can meet downstream without colliding, and the cross-camera property
    worth publishing is "two different track ids, one ``global_id``". A double whose
    ids matched across cameras could not show that.
    """

    def _do_process(self, item: ChainItem) -> ChainItem:
        rows = range(len(item.meta["detections"]))
        offset = 100 if item.context.camera_id == "cam-a" else 200
        tracks = [PublishedTrack(offset + index) for index in rows]
        return item.derive(
            caps=self.output_caps[0],
            payload=None,
            tracks=tracks,
            track_rows=tuple(rows),
        )


@registry_for(ElementKind.MTMC).register("events-mtmc-shared")
class EventsMtmcShared(Element):
    """A cross-camera tier that actually *merges*: one global id per object, fleet-wide.

    The real assigner clusters appearance vectors inside a synchronised instant, so whether a
    given run produces a merge is a timing-and-tuning question — which is why the deterministic
    version of that property is pinned at the chain item in
    ``tests/topology/test_mtmc_element.py``. What is *not* covered there is the last hop: that
    a merge, once made, survives the fan-out onto detection rows and the JSON encoding and
    arrives at a consumer as one id under two camera ids. This double supplies the merge (by
    class, since the frames are identical) so that hop can be asserted on the published bytes.
    """

    kind: ClassVar[ElementKind] = ElementKind.MTMC
    accepts: ClassVar[tuple[str, ...]] = ("meta@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)

    #: Fleet-wide identity per object class. Camera-independent, which is the whole point.
    IDS: ClassVar[dict[str, int]] = {"ship": 900, "person": 901}

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem:
        labels = item.meta["detections"].labels
        rows = item.meta["track_rows"]
        return item.derive(global_ids=[self.IDS[labels[row]] for row in rows])


class ScriptedSource(FrameSource):
    """A source that hands out a fixed number of frames and then reports itself exhausted."""

    name: ClassVar[str] = "scripted-events"

    def __init__(
        self,
        config: CameraConfig,
        counter: FrameCounter | None = None,
        *,
        settings: Any = None,
        frames: int = 3,
    ) -> None:
        super().__init__(config, counter, settings=settings)
        self.frames = frames
        self.index = 0

    @property
    def is_exhausted(self) -> bool:
        return self.index >= self.frames

    def _do_open(self) -> None:
        self._set_format(HEIGHT, WIDTH, self.config.fps or 20.0)

    def _do_read(self) -> np.ndarray | None:
        index = self.index
        self.index += 1
        if index >= self.frames:
            return None
        return np.full((HEIGHT, WIDTH, 3), index + 1, dtype=np.uint8)

    def _do_close(self) -> None:
        return None


def scripted(frames: int = 3):
    def factory(config: CameraConfig, counter: FrameCounter) -> ScriptedSource:
        return ScriptedSource(config, counter, frames=frames)

    return factory


def until(predicate, timeout_s: float = 20.0, poll_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def events_in(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture()
def runner() -> Iterator:
    made: list[InprocessRunner] = []

    def _make(chain: Topology, **kwargs: Any) -> InprocessRunner:
        built = InprocessRunner(chain, **kwargs)
        made.append(built)
        built.start()
        return built

    yield _make
    for built in made:
        built.stop(timeout_s=10.0)


def settings(workers: int = 2) -> ServerSettings:
    return ServerSettings(
        pipeline={"workers": workers, "queue_capacity": 64},
        ingest={"read_timeout_ms": 50, "open_timeout_ms": 50, "empty_read_sleep_ms": 0},
    )


def chain_for(path: Path, *, track: str, mtmc: str, extra: str = "") -> Topology:
    return Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(f"""
                name: events
                elements:
                  decode: {{impl: replay}}
                  detect: {{impl: events-detect}}
                  embed:  {{impl: events-embed}}
                  track:  {{impl: {track}{extra}}}
                  mtmc:   {{impl: {mtmc}, params: {{group: quay, cameras: [cam-a, cam-b]}}}}
                  output: {{impl: jsonlines, params: {{path: "{path}", flush_every: 0}}}}
                """)))


class TestAChainOfDoublesWritesEvents:
    """Frames in, one v4 line out per frame, with the identity on the right object."""

    def test_one_event_per_frame_reaches_the_file(self, runner, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        chain = chain_for(path, track="events-track", mtmc="events-mtmc")
        started = runner(chain, settings=settings(), source_factory=scripted(frames=3))
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))

        assert until(lambda: len(events_in(path)) == 3), events_in(path)
        assert {event["image_id"] for event in events_in(path)} == {0, 1, 2}
        assert {event["camera_id"] for event in events_in(path)} == {"cam-a"}

    def test_the_event_is_schema_v4_with_the_identity_on_each_object(
        self, runner, tmp_path: Path
    ) -> None:
        """The whole fan-out in one assertion set: rows, track ids, global ids, embeddings.

        ``ship`` is row 0 and ``person`` is row 1, and the two are published into *different*
        parallel arrays — so an attribution that slipped by one row would put the person's
        track id in ``ship_track_id_vec`` and this would go red rather than merely change a
        number.
        """
        path = tmp_path / "events.jsonl"
        chain = chain_for(path, track="events-track", mtmc="events-mtmc")
        started = runner(chain, settings=settings(), source_factory=scripted(frames=2))
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))

        assert until(lambda: len(events_in(path)) == 2), events_in(path)
        event = events_in(path)[0]

        assert event["schema_version"] == 4
        assert event["type"] == "Det2MOT"
        assert event["ship_det_id_vec"] == [f"cam-a_{event['image_id']}_0"]
        assert event["det_id_vec"] == [f"cam-a_{event['image_id']}_1"]
        assert event["ship_track_id_vec"] == [100]
        assert event["body_track_id_vec"] == [101]
        assert event["ship_global_id_vec"] == [900]
        assert event["body_global_id_vec"] == [901]
        assert event["ship_feature_vec"] == [[1.0, 0.0]]
        assert event["body_feature_vec"] == [[1.0, 1.0]]
        assert (event["img_width"], event["img_height"]) == (WIDTH, HEIGHT)
        assert event["partial"] is False

    def test_the_event_carries_the_cameras_negotiated_rate(
        self, runner, tmp_path: Path
    ) -> None:
        """V148's first real run shipped ``img_fps: 0`` in every event: no element can
        discover the rate, so nothing filed it. The frame sink now stamps ``meta["fps"]``
        from the actor's connected source and ``output`` passes it to the builder. The
        scripted source negotiates 20 fps (``conftest``), so 0 here means the plumbing
        broke somewhere between the actor and the event."""
        path = tmp_path / "events.jsonl"
        chain = chain_for(path, track="events-track", mtmc="events-mtmc")
        started = runner(chain, settings=settings(), source_factory=scripted(frames=2))
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))

        assert until(lambda: len(events_in(path)) == 2), events_in(path)
        assert {event["img_fps"] for event in events_in(path)} == {20}

    def test_two_cameras_never_share_a_line(self, runner, tmp_path: Path) -> None:
        """The tag is the whole basis of reassembly, and a sink is where it is finally read."""
        path = tmp_path / "events.jsonl"
        chain = chain_for(path, track="events-track", mtmc="events-mtmc")
        started = runner(chain, settings=settings(workers=4), source_factory=scripted(frames=2))
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))
        started.add_camera(CameraSpec("cam-b", "injected://b", 0.0))

        assert until(lambda: len(events_in(path)) == 4), events_in(path)
        tags = {(event["camera_id"], event["image_id"]) for event in events_in(path)}
        assert tags == {(camera, frame) for camera in ("cam-a", "cam-b") for frame in (0, 1)}

    def test_one_object_seen_by_two_cameras_is_published_under_one_global_id(
        self, runner, tmp_path: Path
    ) -> None:
        """v4's headline claim, checked on the bytes a consumer reads.

        ``track_id`` answers "which object is this within this camera" and ``global_id``
        answers "which object is this across the fleet" — so the *same* object seen by two
        cameras must arrive as two different track ids under one global id. Every stage
        between the merge and the file can break that: the fan-out puts the id on a detection
        row, the encoder writes it into a parallel array, and a row-order slip would publish
        the ship's fleet identity on the person.

        The merge itself comes from a double, deliberately. Whether the real assigner merges a
        given pair depends on whether their instant closed and on the clustering threshold —
        timing and tuning, pinned deterministically at the chain item in
        ``tests/topology/test_mtmc_element.py::test_near_identical_embeddings_share_one_global_id``.
        What this test owns is the hop that one cannot see: merge in, published event out.
        """
        path = tmp_path / "events.jsonl"
        chain = chain_for(path, track="events-track-per-camera", mtmc="events-mtmc-shared")
        started = runner(chain, settings=settings(workers=4), source_factory=scripted(frames=2))
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))
        started.add_camera(CameraSpec("cam-b", "injected://b", 0.0))

        assert until(lambda: len(events_in(path)) == 4), events_in(path)
        by_camera: dict[str, dict[str, Any]] = {}
        for event in events_in(path):
            by_camera.setdefault(event["camera_id"], event)
        a, b = by_camera["cam-a"], by_camera["cam-b"]

        assert a["ship_track_id_vec"] == [100] and b["ship_track_id_vec"] == [200]
        assert a["body_track_id_vec"] == [101] and b["body_track_id_vec"] == [201]
        assert a["ship_global_id_vec"] == b["ship_global_id_vec"] == [900]
        assert a["body_global_id_vec"] == b["body_global_id_vec"] == [901]
        assert (
            a["ship_global_id_vec"] != a["body_global_id_vec"]
        ), "two objects in one frame merged into one fleet identity"


@needs_shipvision
class TestTheRealStatefulTierWritesEvents:
    """The same walk with the actual ``track`` and ``mtmc`` elements behind it."""

    def test_a_real_tracklet_reaches_the_file_on_the_object_it_belongs_to(
        self, runner, tmp_path: Path
    ) -> None:
        """The identity comes from shipvision and lands on the row the attribution chose.

        Nothing asserts a literal track id: ids come from one process-wide counter, so the
        number depends on how many tests ran first. What is asserted is that the *ship* row
        carries one and the *person* row carries a different one — which is the property an
        event consumer depends on and the one an off-by-one attribution breaks.
        """
        path = tmp_path / "events.jsonl"
        chain = chain_for(
            path,
            track="shipvision",
            mtmc="events-mtmc",
            extra=", params: {options: {min_hits: 1, max_age: 3}}",
        )
        started = runner(chain, settings=settings(), source_factory=scripted(frames=3))
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))

        assert until(lambda: len(events_in(path)) == 3), events_in(path)
        tracked = [event for event in events_in(path) if event["ship_track_id_vec"] != [None]]
        assert tracked, "no frame carried a tracklet"
        event = tracked[0]
        assert event["schema_version"] == 4
        assert len(event["ship_track_id_vec"]) == 1
        assert len(event["body_track_id_vec"]) == 1
        assert event["ship_track_id_vec"] != event["body_track_id_vec"]
        assert event["ship_track_state_vec"] == ["confirmed"]

    def test_a_global_id_never_rides_on_an_object_with_no_tracklet(
        self, runner, tmp_path: Path
    ) -> None:
        """Both `mtmc` elements are real, and the fan-out is checked by its own invariant.

        Not "every frame carries ids": whether an instant closes, and whether the assigner has
        seen a track often enough to identify it, are timing and tuning questions that
        `tests/topology/test_mtmc_element.py` pins deterministically at the chain item. What is
        true of *every* run is structural — a global id is an identity for a **tracklet**, so
        an object with no `track_id` cannot have one, and an array a row short would put one
        object's identity on another. Both of those are corruptions with no symptom downstream,
        and both are checked here on the published bytes.
        """
        path = tmp_path / "events.jsonl"
        chain = chain_for(
            path,
            track="shipvision",
            mtmc="shipvision",
            extra=", params: {options: {min_hits: 1, max_age: 3}}",
        )
        started = runner(chain, settings=settings(workers=4), source_factory=scripted(frames=2))
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))
        started.add_camera(CameraSpec("cam-b", "injected://b", 0.0))

        assert until(lambda: len(events_in(path)) == 4), events_in(path)

        for event in events_in(path):
            assert event["schema_version"] == 4
            for prefix in ("ship", "body"):
                globals_ = event[f"{prefix}_global_id_vec"]
                tracks = event[f"{prefix}_track_id_vec"]
                boxes = event[f"{prefix}_bbox_vec"]
                assert len(globals_) == len(tracks) == len(boxes)
                for global_id, track_id in zip(globals_, tracks, strict=True):
                    assert global_id is None or track_id is not None, (
                        "a global id is an identity for a tracklet, so an object with no "
                        "track cannot carry one"
                    )
