"""The ``mtmc`` element: one identity across cameras, and the gaps it is honest about.

Green **with or without** ``3rdparty/shipvision``. CI deliberately does not check the
submodule out, so the classes that drive a real cross-camera tracker skip and the classes
that assert the *contract* — the caps, the registration, the refusal at ``open()`` — run
everywhere. The absence is **arranged** rather than assumed, exactly as
``tests/topology/test_bridge.py`` and ``test_track_element.py`` do it.

The tracker is the real one wherever it is available, and the tracks handed to it are real
``shipvision`` ``Track`` objects, because the failures under test are all about *evidence*:
two views of one object have to become one id, two objects must not, and two tracks in one
camera must never merge whatever their embeddings say. A fake associator would prove this
element can call a method.

Nothing asserts a *literal* global id. ``GlobalIdAssigner`` counts from zero per tracker
instance, so the id a test sees depends on nothing useful. What is asserted is the property:
shared or not shared, present or ``None``.

The synchronisation itself is not tested here — that is
``tests/topology/test_barrier.py``, over the pure class, with no submodule in sight.
What this file adds is everything that needs a tracker: the association, the scatter, and the
element's own refusals.
"""

from __future__ import annotations

import logging
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, ValidationError
from shipinfer.core.metrics import MetricsRegistry
from shipinfer.core.request import RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.ingest import CameraConfig
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.frame import FrameCounter
from shipinfer.launch.control import CameraSpec
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.topology import ChainSpec, Topology
from shipinfer.topology import bridge as bridge_module
from shipinfer.topology.barrier import WaiterBudget
from shipinfer.topology.base import (
    CameraGroup,
    ChainItem,
    Element,
    ElementContext,
    ElementKind,
)
from shipinfer.topology.caps import Caps
from shipinfer.topology.elements.detections import Detections
from shipinfer.topology.elements.mtmc import (
    DEFAULT_ALGORITHM,
    DEFAULT_CLUSTERER,
    DEFAULT_MATRIX_BUILDER,
    DEFAULT_SYNC_WINDOW_MS,
    MISSED_UNASSIGNABLE,
    MISSED_WOULD_STARVE,
    MISSING_TRACKS,
    ShipvisionMtmc,
    parse_group,
)
from shipinfer.topology.registry import create_element, registry_for

pytestmark = [pytest.mark.timeout(60)]

needs_shipvision = pytest.mark.skipif(
    not bridge_module.shipvision_available(),
    reason="shipvision.mtmc is not importable; the submodule is not checked out",
)

#: A tracker that will associate on the first instant it sees a track. ``min_hits`` defaults
#: to 3 — three consecutive qualifying frames before a track may be identified — and a test
#: that spent three instants establishing trust would be testing the gate, which shipvision
#: tests itself.
EAGER: dict[str, Any] = {"min_hits": 1}

HEIGHT, WIDTH = 400, 400

#: Wide enough that no instant here closes on the window unless a test means it to.
WIDE_MS = 30_000.0

#: A plausible unix capture clock for the items these tests build, in nanoseconds. Every
#: ``instant=`` in this file is an offset from it, because a *zero* ``captured_unix_ns`` is a
#: refusal (``TestAFrameWithoutGlobalIdsSaysSo``) and not a usable default.
EPOCH_NS = 1_700_000_000_000_000_000

CHAIN = """
name: associated
elements:
  decode: {impl: replay}
  detect: {impl: framed-detect}
  track:  {impl: shipvision, params: {algorithm: bytetrack, options: {min_hits: 1, max_age: 3}}}
  mtmc:   {impl: shipvision, params: {group: quay, cameras: [cam-a, cam-b], options: {min_hits: 1}}}
  output: {impl: none, params: {keep_last: 16}}
"""


# -- doubles ---------------------------------------------------------------------------------


@registry_for(ElementKind.DETECT).register("framed-detect")
class FramedDetect(Element):
    """A detector that files what the stateful tier actually reads: boxes **and frame size**.

    A detector that filed ``meta["detections"]`` and no ``meta["frame_hw"]`` would be fine
    for ``track`` (it passes 0x0 through to a tracker that does not use it) and is not fine
    here: ``CameraTracks`` refuses a zero frame size on purpose, because the height gate, the
    truncated-box test and the homography's domain would all be silently wrong. A real
    ``pool`` detector files both (``elements/pool.py``). This is that shape, invented.
    """

    kind: ClassVar[ElementKind] = ElementKind.DETECT
    accepts: ClassVar[tuple[str, ...]] = ("bgr@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("bgr@cpu",)

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem:
        # `vectors` as well, one row per detection, because cross-camera identity is decided
        # on appearance: `GlobalIdAssigner` refuses a track with no embedding, and a chain
        # with no embedder in it would exercise that refusal rather than the association.
        return item.derive(
            detections=Detections(
                boxes=np.array([[10.0, 10.0, 110.0, 310.0]], dtype=np.float32),
                scores=np.array([0.9], dtype=np.float32),
                class_ids=np.array([0], dtype=np.int32),
                labels=("ship",),
            ),
            vectors=np.array([SAME_A]),
            frame_hw=(HEIGHT, WIDTH),
        )


class ScriptedSource(FrameSource):
    """A source that hands out a fixed number of frames and then reports itself exhausted."""

    name: ClassVar[str] = "scripted-mtmc"

    def __init__(
        self,
        config: CameraConfig,
        counter: FrameCounter | None = None,
        *,
        settings: Any = None,
        frames: int = 4,
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


def scripted(frames: int = 4):
    def factory(config: CameraConfig, counter: FrameCounter) -> ScriptedSource:
        return ScriptedSource(config, counter, frames=frames)

    return factory


# -- helpers -----------------------------------------------------------------------------------


def unit(*components: float) -> np.ndarray:
    """A unit appearance vector, so two of them differ by direction and nothing else."""
    vector = np.zeros(8, dtype=np.float32)
    vector[: len(components)] = components
    return vector / np.linalg.norm(vector)


#: Two vectors a re-ID model would call the same object, and one it would not.
SAME_A = unit(1.0, 0.0)
SAME_B = unit(0.99, 0.01)
OTHER = unit(0.0, 1.0)


def track(
    track_id: int,
    camera: str,
    frame: int,
    box: tuple[float, float, float, float],
    embedding: np.ndarray | None,
) -> Any:
    """One ``shipvision.types.Track`` — the object the ``track`` element files."""
    types = bridge_module.load_types()
    return types.Track(
        track_id=track_id,
        box=np.array(box, dtype=np.float32),
        tag=types.FrameTag(camera_id=camera, frame_id=frame, timestamp=frame * 0.05),
        embedding=embedding,
    )


#: A box tall enough to clear the height gate (a ninth of the frame) with room to spare.
TALL = (10.0, 10.0, 110.0, 310.0)
BESIDE = (150.0, 10.0, 250.0, 310.0)
#: A box far too short to identify: the height gate holds it back and it comes back ``None``.
SHORT = (10.0, 10.0, 40.0, 30.0)


def item(
    camera: str = "cam-a",
    frame: int = 0,
    *,
    instant: float = 0.0,
    tracks: Any = None,
    frame_hw: Any = (HEIGHT, WIDTH),
    **meta: Any,
) -> ChainItem:
    """One chain item on the metadata plane, as ``track`` hands it on.

    ``instant`` is an offset in seconds from :data:`EPOCH_NS`, not an absolute capture time.
    It has to be offset from *something* real: ``captured_unix_ns`` defaults to ``0`` and the
    element refuses that, because a source that never stamps the clock would put every frame
    of every camera into one instant.
    """
    payload = {"tracks": tracks} if tracks is not None else {}
    if frame_hw is not None:
        payload["frame_hw"] = frame_hw
    return ChainItem(
        context=RequestContext(
            camera_id=camera,
            frame_id=frame,
            captured_unix_ns=EPOCH_NS + int(instant * 1e9),
        ),
        caps=Caps.parse("meta@cpu"),
        payload=None,
        meta={**payload, **meta},
    )


def opened(
    params: dict[str, Any] | None = None,
    *,
    workers: int | None = 4,
    registry: MetricsRegistry | None = None,
    name: str = "mtmc",
    budget: WaiterBudget | None = None,
) -> ShipvisionMtmc:
    """A ``ShipvisionMtmc`` over a real tracker, wide window, eager gate."""
    declared: dict[str, Any] = {
        "sync_window_ms": WIDE_MS,
        "options": EAGER,
        **(params or {}),
    }
    element = create_element(ElementKind.MTMC, "shipvision", name, declared)
    element.open(ElementContext(workers=workers, metrics=registry, waiter_budget=budget))
    return element  # type: ignore[return-value]


def until(predicate, timeout_s: float = 10.0, poll_s: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def value(registry: MetricsRegistry, name: str, **labels: str) -> float:
    for handle in registry.collect():
        if handle.name == name:
            return handle.value(**labels)  # type: ignore[attr-defined]
    return 0.0


class Submitter(threading.Thread):
    """One pipeline worker walking one frame into the element."""

    def __init__(self, element: ShipvisionMtmc, chain_item: ChainItem) -> None:
        super().__init__(daemon=True)
        self._element = element
        self._item = chain_item
        self.emitted: ChainItem | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.emitted = self._element.process(self._item)
        except BaseException as exc:
            self.error = exc


@pytest.fixture()
def element() -> Iterator[ShipvisionMtmc]:
    built = opened()
    try:
        yield built
    finally:
        built.close()


@pytest.fixture()
def pair(element) -> ShipvisionMtmc:
    """An element whose group is two announced cameras, so an instant waits for both.

    Without the announcements the barrier's live set is empty and the first frame closes its
    own instant alone (:meth:`InstantBarrier.camera_added` says why that is the right
    degenerate behaviour). Every test about *association* needs a group of two, and it has to
    be arranged.
    """
    element.camera_added("cam-a")
    element.camera_added("cam-b")
    return element


@pytest.fixture()
def metrics() -> MetricsRegistry:
    return MetricsRegistry()


# -- the contract, with or without the submodule ------------------------------------------------


class TestMtmcCaps:
    """What the loader reads off the class, before anything is opened."""

    def test_it_is_registered_under_its_kind_and_name(self) -> None:
        assert "shipvision" in registry_for(ElementKind.MTMC).names()
        assert registry_for(ElementKind.MTMC).get("shipvision") is ShipvisionMtmc

    def test_the_caps_are_exactly_these_strings(self) -> None:
        assert ShipvisionMtmc.accepts == ("meta@cpu",)
        assert ShipvisionMtmc.produces == ("meta@cpu",)

    def test_it_is_not_a_sink_and_needs_nothing_from_the_process(self) -> None:
        built = create_element(ElementKind.MTMC, "shipvision", "mtmc")
        assert not built.is_sink
        assert (
            ShipvisionMtmc.requires_model_name,
            ShipvisionMtmc.needs_model,
            ShipvisionMtmc.needs_image_ops,
        ) == (False, False, False)

    def test_the_defaults_are_the_documented_ones(self) -> None:
        built = create_element(ElementKind.MTMC, "shipvision", "mtmc")
        assert (DEFAULT_ALGORITHM, DEFAULT_MATRIX_BUILDER, DEFAULT_CLUSTERER) == (
            "cluster",
            "gated",
            "agglomerative",
        )
        assert built._window_s == pytest.approx(DEFAULT_SYNC_WINDOW_MS / 1e3)

    def test_the_group_defaults_to_the_slot_name(self) -> None:
        assert create_element(ElementKind.MTMC, "shipvision", "quay_west").group == "quay_west"
        assert (
            create_element(ElementKind.MTMC, "shipvision", "mtmc", {"group": "quay"}).group
            == "quay"
        )


class TestParamsAreRefusedAtConstruction:
    """A chain file's typo stops the loader, not frame 40 000."""

    def test_a_camera_roster_that_is_a_string_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="must be a list of camera ids"):
            create_element(ElementKind.MTMC, "shipvision", "mtmc", {"cameras": "cam-a"})

    def test_a_non_positive_window_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match=r"sync_window_ms.*must be positive"):
            create_element(ElementKind.MTMC, "shipvision", "mtmc", {"sync_window_ms": 0})

    def test_a_window_that_is_not_a_number_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match=r"sync_window_ms.*must be a number"):
            create_element(ElementKind.MTMC, "shipvision", "mtmc", {"sync_window_ms": "soon"})

    def test_options_must_be_a_mapping(self) -> None:
        with pytest.raises(ConfigurationError, match=r"options.*must be a mapping"):
            create_element(ElementKind.MTMC, "shipvision", "mtmc", {"options": ["min_hits"]})

    def test_calibration_must_be_a_mapping(self) -> None:
        with pytest.raises(ConfigurationError, match=r"calibration.*must be a mapping"):
            create_element(ElementKind.MTMC, "shipvision", "mtmc", {"calibration": []})

    def test_parse_group_is_the_one_parser_and_camera_group_is_what_it_publishes(
        self,
    ) -> None:
        assert parse_group({"group": "quay", "cameras": ["a", "b"]}, where="x") == (
            "quay",
            ("a", "b"),
        )
        assert parse_group({}, where="x") == ("", ())


class TestTheGroupIsDeclaredToTheRunnerAndNotParsedTwice:
    """``Element.camera_group()`` is what a runner asks, and it needs no submodule and no
    ``open()`` — the runner asks when it is built, long before a camera or a tracker exists.

    The hook is what keeps the launcher free of an ``ElementKind.MTMC`` test, an import of
    this module, and a second parse of ``params: cameras:`` (ADR-017 §2). A second
    cross-camera element would be a method override rather than an ``elif`` in ``runners/``.
    """

    def test_a_declared_roster_is_published_as_a_camera_group(self) -> None:
        built = create_element(
            ElementKind.MTMC, "shipvision", "mtmc", {"group": "quay", "cameras": ["q-0", "q-1"]}
        )

        assert built.camera_group() == CameraGroup("quay", ("q-0", "q-1"))

    def test_the_group_name_falls_back_to_the_slot_name(self) -> None:
        built = create_element(ElementKind.MTMC, "shipvision", "berth", {"cameras": ["q-0"]})

        assert built.camera_group() == CameraGroup("berth", ("q-0",))

    def test_no_roster_is_no_group_and_not_an_empty_one(self) -> None:
        """ "The chain did not say" and "the chain grouped nothing" are different facts: only
        the first one lets the runner keep balancing by load."""
        assert create_element(ElementKind.MTMC, "shipvision", "mtmc", {}).camera_group() is None
        assert (
            create_element(
                ElementKind.MTMC, "shipvision", "mtmc", {"group": "quay"}
            ).camera_group()
            is None
        )

    def test_an_ordinary_element_declares_no_group(self) -> None:
        """The ABC's default, which is what makes the runner's walk kind-free."""
        assert (
            create_element(ElementKind.TRACK, "shipvision", "track", {}).camera_group() is None
        )
        assert FramedDetect("detect").camera_group() is None


class TestWithoutTheSubmodule:
    """A host that never checked ``3rdparty/shipvision`` out, arranged rather than assumed."""

    @pytest.fixture()
    def masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in list(sys.modules):
            if name == "shipvision" or name.startswith("shipvision."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, "shipvision", None)
        bridge_module.load_mtmc.cache_clear()
        bridge_module.load_types.cache_clear()
        monkeypatch.setattr(bridge_module.load_mtmc, "cache_clear", lambda: None, raising=False)
        yield
        bridge_module.load_mtmc.cache_clear()
        bridge_module.load_types.cache_clear()

    def test_the_chain_still_builds_the_element(self, masked: None) -> None:
        """The caps have to be readable on a laptop; that is what validates a chain there."""
        built = create_element(ElementKind.MTMC, "shipvision", "mtmc")
        assert built.accepts == ("meta@cpu",)

    def test_open_refuses_with_the_command_that_fixes_it(self, masked: None) -> None:
        built = create_element(ElementKind.MTMC, "shipvision", "mtmc")

        with pytest.raises(ConfigurationError) as raised:
            built.open(ElementContext(workers=4))

        message = str(raised.value)
        assert "shipvision.mtmc" in message
        assert "git submodule update --init 3rdparty/shipvision" in message
        assert "pip install -e 3rdparty/shipvision" in message

    def test_a_failed_open_leaves_the_element_closed(self, masked: None) -> None:
        built = create_element(ElementKind.MTMC, "shipvision", "mtmc")
        with pytest.raises(ConfigurationError):
            built.open(ElementContext())
        assert not built.is_open
        assert built.barrier is None


# -- association --------------------------------------------------------------------------------


@needs_shipvision
class TestOneObjectAcrossTwoCameras:
    def test_near_identical_embeddings_share_one_global_id(self, pair) -> None:
        """The property the whole element exists for: two views, one identity."""
        first = Submitter(pair, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)]))
        first.start()
        assert until(lambda: pair.barrier.waiters == 1)

        second = pair.process(item("cam-b", 0, tracks=[track(2, "cam-b", 0, TALL, SAME_B)]))
        first.join(10.0)

        assert first.error is None
        assert first.emitted is not None
        left = first.emitted.meta["global_ids"]
        right = second.meta["global_ids"]
        assert left == right, "two views of one object were given different global ids"
        assert left[0] is not None

    def test_two_distinct_objects_get_two_ids(self, pair) -> None:
        first = Submitter(pair, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)]))
        first.start()
        assert until(lambda: pair.barrier.waiters == 1)

        second = pair.process(item("cam-b", 0, tracks=[track(2, "cam-b", 0, TALL, OTHER)]))
        first.join(10.0)

        assert first.emitted.meta["global_ids"][0] != second.meta["global_ids"][0]
        assert None not in first.emitted.meta["global_ids"] + second.meta["global_ids"]

    def test_two_tracks_in_one_camera_never_merge(self, element) -> None:
        """Enforced in shipvision and asserted here, because the failure is invisible: MTMC
        silently becomes a within-camera deduplicator, every count falls, and every metric
        improves."""
        emitted = element.process(
            item(
                "cam-a",
                0,
                tracks=[
                    track(1, "cam-a", 0, TALL, SAME_A),
                    track(2, "cam-a", 0, BESIDE, SAME_B),
                ],
            )
        )

        ids = emitted.meta["global_ids"]
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_a_gated_track_comes_back_as_none_and_is_still_returned(self, element) -> None:
        """``None`` and not ``-1``, present and not omitted: "we have no identity for this"
        and "this track did not exist" are different things on an operator's screen."""
        emitted = element.process(
            item(
                "cam-a",
                0,
                tracks=[track(1, "cam-a", 0, TALL, SAME_A), track(2, "cam-a", 0, SHORT, OTHER)],
            )
        )

        ids = emitted.meta["global_ids"]
        assert len(ids) == 2
        assert ids[0] is not None
        assert ids[1] is None

    def test_the_scatter_is_keyed_and_not_positional(self, pair) -> None:
        """The reassembly bug this element is most exposed to, and the one that looks fine.

        ``FrameTrackCluster`` flattens the group into one observation list, so camera B's rows
        are at offsets 1 and 2 of an answer whose first entry belongs to camera A. A positional
        scatter hands B camera A's id and looks entirely plausible.
        """
        first = Submitter(pair, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)]))
        first.start()
        assert until(lambda: pair.barrier.waiters == 1)

        second = pair.process(
            item(
                "cam-b",
                0,
                tracks=[
                    track(2, "cam-b", 0, TALL, OTHER),
                    track(3, "cam-b", 0, BESIDE, unit(0.0, 0.0, 1.0)),
                ],
            )
        )
        first.join(10.0)

        alone = first.emitted.meta["global_ids"]
        pair = second.meta["global_ids"]
        assert len(alone) == 1 and len(pair) == 2
        # Three distinct objects, three distinct ids, and camera B's two are not camera A's.
        assert len({alone[0], pair[0], pair[1]}) == 3

    def test_a_group_of_one_camera_gives_each_track_its_own_id(self, element) -> None:
        """A one-camera group is a legitimate deployment: no cross-camera merge, stable ids."""
        element.camera_added("cam-a")

        emitted = element.process(
            item(
                "cam-a",
                0,
                tracks=[
                    track(1, "cam-a", 0, TALL, SAME_A),
                    track(2, "cam-a", 0, BESIDE, OTHER),
                ],
            )
        )

        ids = emitted.meta["global_ids"]
        assert len(set(ids)) == 2
        assert None not in ids

    def test_global_ids_are_a_list_aligned_with_this_items_tracks(self, element) -> None:
        """The shape a ``track`` element publishes and an ``output`` element serialises."""
        tracks = [track(1, "cam-a", 0, TALL, SAME_A), track(2, "cam-a", 0, BESIDE, OTHER)]
        emitted = element.process(item("cam-a", 0, tracks=tracks))

        assert isinstance(emitted.meta["global_ids"], list)
        assert len(emitted.meta["global_ids"]) == len(tracks)

    def test_an_instant_with_no_tracks_anywhere_still_associates(self, element) -> None:
        """Empty is not missing. A camera with nothing visible reported, and said so."""
        emitted = element.process(item("cam-a", 0, tracks=[]))

        assert emitted.meta["global_ids"] == []
        assert "mtmc" not in emitted.meta.get("missing_stages", ())


@needs_shipvision
class TestThePlaneAndTheTag:
    def test_the_item_keeps_its_caps_its_payload_and_its_tag(self, element) -> None:
        """``mtmc`` adds metadata to a frame already on the metadata plane. Only ``track``
        changes plane, and it has already done so."""
        emitted = element.process(
            item("cam-7", 42, tracks=[track(1, "cam-7", 42, TALL, SAME_A)])
        )

        assert emitted.context is not None
        assert emitted.key == ("cam-7", 42)
        assert str(emitted.caps) == "meta@cpu"
        assert emitted.payload is None

    def test_the_other_elements_metadata_is_carried_forward(self, element) -> None:
        emitted = element.process(
            item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)], vectors=["v"])
        )

        assert emitted.meta["vectors"] == ["v"]
        assert "tracks" in emitted.meta


# -- the gaps ------------------------------------------------------------------------------------


@needs_shipvision
class TestAFrameWithoutGlobalIdsSaysSo:
    def test_a_frame_the_tracker_never_answered_for_is_emitted_with_a_gap(
        self, metrics
    ) -> None:
        built = opened(registry=metrics)
        try:
            emitted = built.process(item("cam-a", 0, tracks=None))
        finally:
            built.close()

        assert emitted.meta["missing_stages"] == ("mtmc",)
        assert "global_ids" not in emitted.meta
        assert value(metrics, "shipinfer_mtmc_frames_missing_total", reason=MISSING_TRACKS) == 1

    def test_the_gap_is_appended_and_never_replaces_an_earlier_one(self, element) -> None:
        emitted = element.process(item("cam-a", 0, tracks=None, missing_stages=("track",)))

        assert emitted.meta["missing_stages"] == ("track", "mtmc")

    def test_a_frame_that_would_starve_the_last_worker_is_emitted_immediately(
        self, metrics
    ) -> None:
        """The never-starve guard, reached through the element rather than the barrier."""
        built = opened(workers=1, registry=metrics)
        try:
            started = time.monotonic()
            built.camera_added("cam-a")
            built.camera_added("cam-b")
            emitted = built.process(
                item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)])
            )
            elapsed = time.monotonic() - started
        finally:
            built.close()

        assert elapsed < 1.0, "a single-worker runner waited for another camera"
        assert emitted.meta["missing_stages"] == ("mtmc",)
        assert value(metrics, "shipinfer_mtmc_would_starve_total") == 1
        assert (
            value(metrics, "shipinfer_mtmc_frames_missing_total", reason=MISSED_WOULD_STARVE)
            == 1
        )

    def test_a_track_with_no_embedding_is_a_gap_and_not_a_dead_frame(self, metrics) -> None:
        """``GlobalIdAssigner`` refuses to identify a track with no appearance vector.

        That is a *data* condition — an embedder that was spilled, a crop that produced
        nothing, or a chain with no embedder in front of ``track`` at all — so the frame is
        emitted with its boxes and its per-camera track ids and an honest gap. Letting the
        library's refusal out would fail the item's future and stop the walk, which costs the
        frame its whole event for something that is true of the appearance vectors alone.
        """
        built = opened(registry=metrics)
        try:
            emitted = built.process(item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, None)]))
        finally:
            built.close()

        assert emitted.meta["missing_stages"] == ("mtmc",)
        assert "global_ids" not in emitted.meta
        assert (
            value(metrics, "shipinfer_mtmc_frames_missing_total", reason=MISSED_UNASSIGNABLE)
            == 1
        )

    def test_an_unassignable_instant_releases_the_frames_waiting_on_it(self, pair) -> None:
        """The waiters get the same gap rather than sitting out the window."""
        waiting = Submitter(pair, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)]))
        waiting.start()
        assert until(lambda: pair.barrier.waiters == 1)

        emitted = pair.process(item("cam-b", 0, tracks=[track(2, "cam-b", 0, TALL, None)]))
        waiting.join(10.0)

        assert not waiting.is_alive()
        assert emitted.meta["missing_stages"] == ("mtmc",)
        assert waiting.emitted.meta["missing_stages"] == ("mtmc",)

    def test_a_tracks_value_of_the_wrong_type_is_a_loud_refusal(self, element) -> None:
        with pytest.raises(ValidationError, match="meta\\['tracks'\\]"):
            element.process(item("cam-a", 0, tracks={"output0": object()}))

    def test_a_missing_frame_size_is_a_loud_refusal(self, element) -> None:
        """Not a gap: it is identical on every frame of a mis-wired chain, and a zero would
        make the height gate, the truncated-box test and the homography all silently wrong."""
        with pytest.raises(ValidationError, match="frame_hw"):
            element.process(
                item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)], frame_hw=None)
            )

    def test_a_zero_frame_size_is_a_loud_refusal(self, element) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            element.process(
                item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)], frame_hw=(0, 0))
            )

    def test_a_zero_capture_clock_is_a_loud_refusal_like_a_zero_frame_size(
        self, element
    ) -> None:
        """``RequestContext.captured_unix_ns`` defaults to ``0``, so a source that never
        stamps it is indistinguishable from one that stamps the epoch — and either way every
        frame of every camera lands in one instant, which closes once and leaves the rest of
        the deployment ``late`` for the life of the process. Same class of mis-wiring as a
        zero ``frame_hw``, so it gets the same treatment rather than a per-frame gap that
        reads like clock skew."""
        unstamped = ChainItem(
            context=RequestContext(camera_id="cam-a", frame_id=0),
            caps=Caps.parse("meta@cpu"),
            payload=None,
            meta={"tracks": [track(1, "cam-a", 0, TALL, SAME_A)], "frame_hw": (HEIGHT, WIDTH)},
        )

        with pytest.raises(ValidationError, match="captured_unix_ns"):
            element.process(unstamped)

        assert element.barrier.open_instants == 0, "a refused frame still opened an instant"


# -- the never-starve guard across elements ------------------------------------------------------


@needs_shipvision
class TestTwoMtmcSlotsCannotParkEveryWorkerBetweenThem:
    """The guard is a **process** budget, not a per-element one.

    Two ``mtmc`` slots in one chain is a supported configuration: the loader takes an explicit
    ``kind:`` and this element's own camera gauge is labelled by element precisely because two
    independent groups can exist. Counting waiters per element let slot A admit ``workers - 1``
    and slot B, which had seen none, admit the last worker — every pipeline worker parked,
    neither barrier able to close on evidence, and the shard turned into a fixed window of
    latency with a stalled queue behind it. Bounded rather than a hang, which is what made it
    a stall dressed as a wait rather than a crash.
    """

    def test_the_last_worker_is_never_parked_by_the_second_slot(self) -> None:
        workers = 3
        budget = WaiterBudget(workers - 1)
        quay = opened(name="quay", workers=workers, budget=budget)
        gate = opened(name="gate", workers=workers, budget=budget)
        try:
            for element in (quay, gate):
                for camera in ("cam-a", "cam-b", "cam-c"):
                    element.camera_added(camera)
            parked = [
                Submitter(quay, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)])),
                Submitter(gate, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)])),
            ]
            for worker in parked:
                worker.start()
            assert until(lambda: budget.held == workers - 1)

            # The last worker of the process arrives at whichever slot; it must not park.
            started = time.monotonic()
            emitted = gate.process(
                item("cam-b", 0, tracks=[track(2, "cam-b", 0, TALL, SAME_A)])
            )
            elapsed = time.monotonic() - started

            assert elapsed < 1.0, "the second slot parked the last pipeline worker"
            assert emitted.meta["missing_stages"] == ("mtmc",)
            assert quay.barrier.waiters + gate.barrier.waiters == workers - 1
        finally:
            quay.close()
            gate.close()
            for worker in parked:
                worker.join(5.0)

    def test_the_runner_hands_every_element_the_same_budget(self) -> None:
        """One object per runner: two `element_context()` calls that built two budgets would
        put the bug back with the fix still in the file."""
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        built = InprocessRunner(
            chain,
            settings=ServerSettings(pipeline={"workers": 5, "queue_capacity": 8}),
        )

        first, second = built.element_context(), built.element_context()

        assert first.waiter_budget is second.waiter_budget
        assert first.waiter_budget is not None
        assert first.waiter_budget.permits == 4, "workers - 1, so one always drains its lane"


# -- the camera lifecycle ---------------------------------------------------------------------------


@needs_shipvision
class TestTheCameraLifecycle:
    def test_an_added_camera_is_waited_for(self, element) -> None:
        element.camera_added("cam-a")
        element.camera_added("cam-b")

        waiting = Submitter(
            element, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)])
        )
        waiting.start()

        assert until(lambda: element.barrier.waiters == 1)
        element.close()
        waiting.join(10.0)
        assert not waiting.is_alive()

    def test_a_removed_camera_no_longer_holds_an_instant_open(self, element) -> None:
        """The half that matters: the instant that is *already open* stops waiting too.

        Without it every instant for the rest of the process sits out the whole window for a
        camera that will never report again — 60 ms per frame, reported as a healthy chain.
        """
        element.camera_added("cam-a")
        element.camera_added("cam-b")
        waiting = Submitter(
            element, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)])
        )
        waiting.start()
        assert until(lambda: element.barrier.waiters == 1)

        started = time.monotonic()
        element.camera_removed("cam-b")
        waiting.join(10.0)

        assert not waiting.is_alive(), "the open instant still waited for the removed camera"
        assert time.monotonic() - started < 10.0
        assert waiting.error is None
        assert waiting.emitted.meta["global_ids"][0] is not None
        assert "mtmc" not in waiting.emitted.meta.get("missing_stages", ())

    def test_a_removed_camera_is_gone_from_the_live_set(self, element) -> None:
        element.camera_added("cam-a")
        element.camera_added("cam-b")
        assert element.barrier.live == frozenset({"cam-a", "cam-b"})

        element.camera_removed("cam-b")

        assert element.barrier.live == frozenset({"cam-a"})

    def test_the_hooks_are_safe_before_open_and_after_close(self) -> None:
        built = create_element(ElementKind.MTMC, "shipvision", "mtmc")
        built.camera_added("cam-a")
        built.camera_removed("cam-a")
        built.open(ElementContext(workers=4))
        built.close()
        built.camera_added("cam-a")
        built.camera_removed("cam-a")

    def test_the_live_camera_gauge_follows_the_hooks(self, metrics) -> None:
        built = opened(registry=metrics)
        try:
            built.camera_added("cam-a")
            built.camera_added("cam-b")
            assert value(metrics, "shipinfer_mtmc_cameras", element="mtmc") == 2
            built.camera_removed("cam-a")
            assert value(metrics, "shipinfer_mtmc_cameras", element="mtmc") == 1
        finally:
            built.close()


@needs_shipvision
class TestAGroupItsWorkersCannotCoverIsSaidOutLoud:
    """The never-starve guard is honest, bounded and counted — and at the shipped default of
    four workers it answers half of an eight-camera group without anybody being told.

    Coverage is ``min(1, workers / group_size)``: an instant closes on the *last* camera's
    frame and every earlier frame of it is parked in a worker until then, so four workers
    answer four frames of every instant and emit the other four with a gap. That is the same
    class of silent shortfall as the bucket key this element used to have, moved from the
    window to the worker count, and a counter nobody is scraping yet is not a warning.
    """

    def _warnings(self, caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_declared_roster_the_workers_cannot_cover_is_warned_about_at_open(
        self, caplog
    ) -> None:
        roster = [f"cam-{index}" for index in range(8)]
        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.mtmc"):
            built = opened({"group": "quay", "cameras": roster}, workers=4)
        built.close()

        warned = self._warnings(caplog)
        assert len(warned) == 1, warned
        assert "8 cameras" in warned[0], "it names the group"
        assert "4 frame(s) of each instant can be answered" in warned[0], "and the coverage"
        assert "pipeline.workers" in warned[0] and "at least 8" in warned[0], "and the fix"

    def test_a_declared_roster_says_it_again_when_the_cameras_actually_arrive(
        self, caplog
    ) -> None:
        """Two lines in a process lifetime, and deliberately two: the first says the declared
        configuration cannot work, the second says it has started happening and with how many
        cameras. The ramp from an empty shard always passes through a covered count, so a
        latch set at ``open()`` would be cleared by the ramp anyway."""
        roster = [f"cam-{index}" for index in range(8)]
        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.mtmc"):
            built = opened({"group": "quay", "cameras": roster}, workers=4)
            try:
                for camera in roster:
                    built.camera_added(camera)
            finally:
                built.close()

        warned = self._warnings(caplog)
        assert len(warned) == 2, warned
        assert "8 cameras (its declared roster)" in warned[0]
        assert "5 cameras (live on this shard)" in warned[1], "the crossing, once"

    def test_a_group_the_workers_do_cover_says_nothing(self, caplog) -> None:
        roster = [f"cam-{index}" for index in range(8)]
        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.mtmc"):
            built = opened({"group": "quay", "cameras": roster}, workers=8)
        built.close()

        assert self._warnings(caplog) == [], "eight workers answer an eight-camera instant"

    def test_a_live_set_that_grows_past_the_budget_warns_once_at_each_crossing(
        self, caplog
    ) -> None:
        """Cameras are added by API at run time, so a chain that declared nothing — or
        declared four and was given six — crosses the line hours after ``open()``. Once per
        crossing and not once per announcement: this runs on the lifecycle thread."""
        built = opened(workers=4)  # no roster: nothing to warn about at open
        try:
            with caplog.at_level(logging.WARNING, logger="shipinfer.topology.mtmc"):
                for index in range(4):
                    built.camera_added(f"cam-{index}")
                assert self._warnings(caplog) == [], "four workers cover four cameras"

                built.camera_added("cam-4")
                built.camera_added("cam-5")
                after_crossing = self._warnings(caplog)

                built.camera_removed("cam-4")
                built.camera_removed("cam-5")
                built.camera_added("cam-6")
                built.camera_added("cam-7")
                after_recrossing = self._warnings(caplog)
        finally:
            built.close()

        assert len(after_crossing) == 1, after_crossing
        assert "5 cameras" in after_crossing[0], "the crossing, not every camera after it"
        assert len(after_recrossing) == 2, "back under the budget and over it again is news"

    def test_no_worker_count_but_a_budget_says_the_barrier_still_waits(self, caplog) -> None:
        """The element used to log "it will emit every frame immediately" whichever way the
        barrier was built, and a supplied budget wins over an absent worker count."""
        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.mtmc"):
            built = opened(workers=None, budget=WaiterBudget(2))
        try:
            assert built.barrier.budget.permits == 2
            warned = self._warnings(caplog)
            assert len(warned) == 1, warned
            assert "will wait" in warned[0] and "2 permit(s)" in warned[0]
        finally:
            built.close()

    def test_no_worker_count_and_no_budget_does_say_it_never_waits(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="shipinfer.topology.mtmc"):
            built = opened(workers=None)
        try:
            warned = self._warnings(caplog)
            assert len(warned) == 1, warned
            assert "emit every frame immediately" in warned[0]
        finally:
            built.close()


@needs_shipvision
class TestClosingTheElement:
    def test_close_releases_every_waiting_worker(self, element) -> None:
        """``stop()`` joins the workers before it closes the chain, so a barrier that only
        resolved on its own window would hold a shutdown for a whole window per frame."""
        element.camera_added("cam-a")
        element.camera_added("cam-b")
        waiting = Submitter(
            element, item("cam-a", 0, tracks=[track(1, "cam-a", 0, TALL, SAME_A)])
        )
        waiting.start()
        assert until(lambda: element.barrier.waiters == 1)

        element.close()
        waiting.join(10.0)

        assert not waiting.is_alive()
        assert waiting.emitted.meta["missing_stages"] == ("mtmc",)

    def test_reopening_starts_a_fresh_barrier_and_a_fresh_identity_space(self, element) -> None:
        element.camera_added("cam-a")
        first = element.barrier
        element.close()
        element.open(ElementContext(workers=4))

        assert element.barrier is not first
        assert element.barrier.live == frozenset()


# -- calibration ------------------------------------------------------------------------------------


@needs_shipvision
class TestCalibration:
    def test_no_calibration_is_an_appearance_only_deployment_not_a_failure(self) -> None:
        built = opened()
        try:
            assert built.is_open
        finally:
            built.close()

    def test_homographies_reach_the_matcher(self) -> None:
        built = opened(
            {
                "calibration": {
                    "cam-a": {
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "camera_width": WIDTH,
                        "camera_height": HEIGHT,
                    }
                }
            }
        )
        try:
            plane = built._tracker.builder.ground_plane
            assert plane is not None
            assert plane.has("cam-a")
        finally:
            built.close()

    def test_a_singular_homography_is_refused_at_open_naming_the_camera(self) -> None:
        built = create_element(
            ElementKind.MTMC,
            "shipvision",
            "mtmc",
            {"calibration": {"cam-a": {"matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}}},
        )

        with pytest.raises(ConfigurationError, match="cam-a"):
            built.open(ElementContext(workers=4))

    def test_a_calibration_entry_that_is_not_a_mapping_is_refused(self) -> None:
        built = create_element(
            ElementKind.MTMC, "shipvision", "mtmc", {"calibration": {"cam-a": [1, 2, 3]}}
        )

        with pytest.raises(ConfigurationError, match="cam-a"):
            built.open(ElementContext(workers=4))


# -- the chain ----------------------------------------------------------------------------------------


class TestTheChainNegotiatesMetaEndToEnd:
    """``detect -> track -> mtmc -> output``, as a loader question. No submodule needed."""

    def test_the_chain_loads_and_every_edge_after_track_is_meta_at_cpu(self) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))

        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in chain.edges}
        assert caps[("detect", "track")] == "bgr@cpu"
        assert caps[("track", "mtmc")] == "meta@cpu"
        assert caps[("mtmc", "output")] == "meta@cpu"

    def test_the_slot_resolves_to_this_implementation(self) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        node = chain.node("mtmc")

        assert node.kind is ElementKind.MTMC
        assert node.element.impl == "shipvision"
        assert node.element.group == "quay"
        assert node.element.roster == ("cam-a", "cam-b")

    def test_an_mtmc_slot_needs_no_model(self) -> None:
        """Cross-camera association is an algorithm over metadata, not a repository model."""
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        assert chain.node("mtmc").element.model is None


@needs_shipvision
class TestMtmcOverTheRunner:
    """The element inside a real chain, walked by real workers off a real fair lane."""

    @pytest.fixture()
    def runner(self) -> Iterator:
        made: list[InprocessRunner] = []

        def _make(chain: Topology, **kwargs: Any) -> InprocessRunner:
            built = InprocessRunner(chain, **kwargs)
            made.append(built)
            built.start()
            return built

        yield _make
        for built in made:
            built.stop(timeout_s=5.0)

    def _settings(self, workers: int) -> ServerSettings:
        return ServerSettings(
            pipeline={"workers": workers, "queue_capacity": 64},
            ingest={"read_timeout_ms": 50, "open_timeout_ms": 50, "empty_read_sleep_ms": 0},
        )

    def test_two_cameras_reach_the_sink_with_global_ids_or_an_honest_gap(self, runner) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        started = runner(
            chain, settings=self._settings(workers=4), source_factory=scripted(frames=3)
        )
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))
        started.add_camera(CameraSpec("cam-b", "injected://b", 0.0))

        published = chain.node("output").element._sink
        assert until(lambda: published.emitted == 6), published.stats()

        events = published.events()
        assert {(event.camera_id, event.frame_id) for event in events} == {
            (camera, frame) for camera in ("cam-a", "cam-b") for frame in range(3)
        }
        for event in events:
            has_ids = any(record.global_id is not None for record in event.objects)
            gapped = "mtmc" in event.missing_stages
            assert has_ids != gapped, "a frame must carry ids or say it has none"

    def test_the_element_learns_its_cameras_from_the_runners_lifecycle(self, runner) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        started = runner(
            chain, settings=self._settings(workers=4), source_factory=scripted(frames=2)
        )
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))
        started.add_camera(CameraSpec("cam-b", "injected://b", 0.0))
        node = chain.node("mtmc").element

        assert node.barrier.live == frozenset({"cam-a", "cam-b"})

        assert until(lambda: chain.node("output").element._sink.emitted == 4)
        started.remove_camera("cam-b")

        assert node.barrier.live == frozenset({"cam-a"})

    def test_the_element_counts_on_the_runners_own_registry(self, runner) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        started = runner(
            chain, settings=self._settings(workers=4), source_factory=scripted(frames=2)
        )
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))
        started.add_camera(CameraSpec("cam-b", "injected://b", 0.0))

        assert until(lambda: chain.node("output").element._sink.emitted == 4)

        assert value(started.metrics.registry, "shipinfer_mtmc_cameras", element="mtmc") == 2

    def test_a_stop_does_not_sleep_through_the_barrier(self, runner) -> None:
        """Every wait is bounded and ``close_all`` releases the rest, so a shutdown with a
        camera missing costs the deadline it was given and not one window per parked worker."""
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        started = runner(
            chain, settings=self._settings(workers=4), source_factory=scripted(frames=2)
        )
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))
        started.add_camera(CameraSpec("cam-b", "injected://b", 0.0))
        assert until(lambda: chain.node("output").element._sink.emitted >= 2)

        began = time.monotonic()
        started.stop(timeout_s=5.0)

        assert time.monotonic() - began < 5.0
