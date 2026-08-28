"""The ``track`` element: per-camera identities, and the refusals that must not fail a frame.

Green **with or without** ``3rdparty/shipvision``, which is the shape every test file that
touches the submodule has to have: CI deliberately does not check it out (``.claude/CLAUDE.md``),
so the classes that drive a real tracker skip and the classes that assert the *contract* —
caps, registration, the refusal at ``open()`` — run everywhere. The absence is **arranged**
rather than assumed, exactly as ``tests/topology/test_bridge.py`` does it: ``None`` in
``sys.modules`` is what CPython's import machinery reads as "known not importable", which is
the same ``ImportError`` a missing submodule raises and the same one the bridge catches.

The tracker itself is the real one wherever it is available. A fake tracker would prove that
this element can call a method, which is not the failure being guarded against — the failures
here are an out-of-order frame changing which identities exist, a removed camera's shard
leaking, and a re-added camera being refused forever.

Nothing asserts a *literal* track id. ``shipvision`` hands out ids from one process-wide
counter, deliberately, so two cameras' tracklets can meet downstream without colliding — the
id a test sees therefore depends on how many tests ran before it. What is asserted is the
property: one object keeps one id, two cameras never share one, and a gap past ``max_age``
starts a new one.
"""

from __future__ import annotations

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
from shipinfer.core.settings.pipeline import TrackingSettings
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.frame import FrameCounter
from shipinfer.launch.control import CameraSpec
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.topology import ChainSpec, Topology
from shipinfer.topology import bridge as bridge_module
from shipinfer.topology.base import ChainItem, ElementContext, ElementKind
from shipinfer.topology.caps import Caps
from shipinfer.topology.elements.detections import Detections
from shipinfer.topology.elements.mock import MockOutput
from shipinfer.topology.elements.track import (
    DEFAULT_ALGORITHM,
    DEFAULT_REGRESSION_RESET,
    ShipvisionTrack,
)
from shipinfer.topology.registry import create_element, registry_for

pytestmark = [pytest.mark.timeout(60)]

#: Everything that drives a real tracker. Per class rather than per module, because the caps,
#: the registration and the missing-submodule refusal are the half that has to be checked on a
#: checkout with nothing to check out.
needs_shipvision = pytest.mark.skipif(
    not bridge_module.shipvision_available(),
    reason="shipvision.mot is not importable; the submodule is not checked out",
)

#: A tracker that publishes on the first hit and forgets after three missed frames. Both are
#: away from the defaults on purpose: a ten-frame test must not spend three of them waiting
#: for ``min_hits``, and a disappearance has to be observable inside a test's patience.
FAST: dict[str, Any] = {"min_hits": 1, "max_age": 3}

#: The chain the negotiation tests load. ``mock-cpu`` is the host-memory detector, so the edge
#: into ``track`` is ``bgr@cpu`` — today's chain end to end — and the edge out of it is
#: ``meta@cpu``, which is the plane change this element exists to be.
CHAIN = """
name: tracked
elements:
  decode: {impl: replay}
  detect: {impl: mock-cpu}
  track:  {impl: shipvision, params: {algorithm: bytetrack, options: {min_hits: 1, max_age: 3}}}
  output: {impl: mock}
"""

HEIGHT, WIDTH = 4, 6


# -- doubles ---------------------------------------------------------------------------------


class ScriptedSource(FrameSource):
    """A source that hands out a fixed number of frames and then reports itself exhausted.

    A four-line cousin of the one in ``tests/runners/test_camera_lifecycle.py``. The ``tests/``
    directories are not packages, so there is nothing to import it from without inventing a
    shared test package, and that file says the same about its own copy.
    """

    name: ClassVar[str] = "scripted"

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


# -- helpers ---------------------------------------------------------------------------------


def detections(*boxes: tuple[float, float, float, float], label: str = "ship") -> Detections:
    """A frame's detections, one row per box, all of one class."""
    if not boxes:
        return Detections.empty()
    return Detections(
        boxes=np.array(boxes, dtype=np.float32),
        scores=np.full(len(boxes), 0.9, dtype=np.float32),
        class_ids=np.zeros(len(boxes), dtype=np.int32),
        labels=(label,) * len(boxes),
    )


def item(
    camera: str = "cam-a",
    frame: int = 0,
    *,
    caps: str = "bgr@cpu",
    payload: object = "frame-handle",
    **meta: Any,
) -> ChainItem:
    """One chain item with the tag, the payload and the metadata a tracker reads."""
    return ChainItem(
        context=RequestContext(
            camera_id=camera, frame_id=frame, captured_unix_ns=frame * 50_000_000
        ),
        caps=Caps.parse(caps),
        payload=payload,
        meta=dict(meta),
    )


def track_ids(emitted: ChainItem) -> list[int]:
    return sorted(t.track_id for t in emitted.meta["tracks"])


def opened(params: dict[str, Any] | None = None, registry: MetricsRegistry | None = None):
    """A ``ShipvisionTrack`` opened over a real tracker, and always closed."""
    element = create_element(
        ElementKind.TRACK, "shipvision", "track", {"options": FAST, **(params or {})}
    )
    element.open(ElementContext(metrics=registry))
    return element


def until(predicate, timeout_s: float = 10.0, poll_s: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


@pytest.fixture()
def element() -> Iterator[ShipvisionTrack]:
    built = opened()
    try:
        yield built  # type: ignore[misc]
    finally:
        built.close()


@pytest.fixture()
def metrics() -> MetricsRegistry:
    return MetricsRegistry()


def value(registry: MetricsRegistry, name: str, **labels: str) -> float:
    for handle in registry.collect():
        if handle.name == name:
            return handle.value(**labels)  # type: ignore[attr-defined]
    return 0.0


# -- the contract, with or without the submodule ---------------------------------------------


class TestTrackCaps:
    """What the loader reads off the class, before anything is opened.

    Every assertion here has to hold on a host with no submodule, because
    :meth:`Topology.from_spec` instantiates every element in a chain to read exactly these —
    which is what makes a chain file validatable on a laptop.
    """

    def test_it_is_registered_under_its_kind_and_name(self) -> None:
        assert "shipvision" in registry_for(ElementKind.TRACK).names()
        assert registry_for(ElementKind.TRACK).get("shipvision") is ShipvisionTrack

    def test_the_caps_are_exactly_these_strings(self) -> None:
        """``meta@cpu`` first is the preference: a tracker wants boxes, never pixels.

        The other two are the same chain before and after phase D — ``bgr@cpu`` today,
        ``nv12@gpu`` once the decoder lands frames in VRAM — and in both cases this element is
        where the chain leaves the frame behind, which is why it lists them at all.
        """
        assert ShipvisionTrack.accepts == ("meta@cpu", "bgr@cpu", "nv12@gpu")
        assert ShipvisionTrack.produces == ("meta@cpu",)

    def test_it_is_not_a_sink(self) -> None:
        assert ShipvisionTrack("track").is_sink is False

    def test_it_declares_no_model_and_no_image_ops(self) -> None:
        """Three answers rather than three omissions.

        A MOT algorithm is not a repository model, so the chain names no ``model:`` (the
        loader's question) and the process that builds the runner must not build an
        ``InferenceServer`` for it (the pool's question). And it reads boxes rather than
        pixels, so it never letterboxes and must not make ``shipinfer run`` resolve an image-ops
        implementation out of ``runtime.ops`` — which would put torch behind a chain that has no
        use for it.
        """
        assert ShipvisionTrack.requires_model_name is False
        assert ShipvisionTrack.needs_model is False
        assert ShipvisionTrack.needs_image_ops is False

    def test_it_can_be_built_and_its_params_validated_with_no_tracker(self) -> None:
        """Construction is cheap and hardware-free, which is what the loader relies on."""
        built = create_element(
            ElementKind.TRACK, "shipvision", "track", {"algorithm": "ocsort"}
        )
        assert built.name == "track"
        assert built.is_open is False

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"options": [1, 2]}, "options"),
            ({"classes": "ship"}, "classes"),
            ({"regression_reset": -1}, "regression_reset"),
            ({"regression_reset": "soon"}, "regression_reset"),
        ],
    )
    def test_a_malformed_param_stops_the_deploy_at_construction(
        self, params: dict[str, Any], match: str
    ) -> None:
        """At construction and not on the first frame: the loader builds every element."""
        with pytest.raises(ConfigurationError, match=match):
            create_element(ElementKind.TRACK, "shipvision", "track", params)

    def test_the_default_algorithm_is_the_settings_tree_default(self) -> None:
        """``topology`` may not read the settings tree, so the literal is duplicated — and
        this is what keeps the two from drifting into two different defaults."""
        assert TrackingSettings().algorithm == DEFAULT_ALGORITHM


class TestAChainCanNameIt:
    """``Topology.from_spec`` accepts a ``track`` slot behind a detector, on any host."""

    def test_the_chain_loads_and_negotiates_bgr_in_and_meta_out(self) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))

        edges = {(edge.producer, edge.consumer): str(edge.caps) for edge in chain.edges}
        assert edges[("detect", "track")] == "bgr@cpu"
        assert edges[("track", "output")] == "meta@cpu"

    def test_the_slot_needs_no_model_line(self) -> None:
        """A gallery-free, pool-free element: naming a ``model:`` for it would be a lie the
        loader used to demand (ADR-017, amended in C2)."""
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))

        node = chain.node("track")
        assert node.element.model is None
        assert node.element.needs_model is False


class TestWithoutTheSubmodule:
    """A host that never checked ``3rdparty/shipvision`` out, arranged rather than assumed."""

    @pytest.fixture(autouse=True)
    def masked(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        for loader in (
            bridge_module.load_mot,
            bridge_module.load_mtmc,
            bridge_module.load_reid,
            bridge_module.load_types,
        ):
            loader.cache_clear()
        for name in list(sys.modules):
            if name == "shipvision" or name.startswith("shipvision."):
                monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "shipvision", None)
        yield
        monkeypatch.undo()
        for loader in (
            bridge_module.load_mot,
            bridge_module.load_mtmc,
            bridge_module.load_reid,
            bridge_module.load_types,
        ):
            loader.cache_clear()

    def test_open_refuses_with_a_typed_error_naming_the_submodule_fix(self) -> None:
        """``ConfigurationError`` at ``open()``, not ``ImportError`` at import.

        Two different problems with two different fixes: "unknown element" means the chain
        names something that does not exist, and this means the chain is right and the host is
        incomplete. So the implementation is still *listed*, the chain still *loads*, and the
        refusal carries the command.
        """
        built = create_element(ElementKind.TRACK, "shipvision", "track", {})

        with pytest.raises(ConfigurationError) as caught:
            built.open(ElementContext())

        message = str(caught.value)
        assert "shipvision" in message
        assert "git submodule update --init 3rdparty/shipvision" in message
        assert "pip install -e 3rdparty/shipvision" in message

    def test_a_refused_open_leaves_the_element_closed(self) -> None:
        """The ABC unwinds a partial open; a chain that reopened it must not skip ``_do_open``."""
        built = create_element(ElementKind.TRACK, "shipvision", "track", {})

        with pytest.raises(ConfigurationError):
            built.open(ElementContext())

        assert built.is_open is False
        assert built.context is None

    def test_the_chain_still_loads_so_the_failure_names_the_host_not_the_yaml(self) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))

        assert chain.node("track").element.impl == "shipvision"


# -- the real tracker ------------------------------------------------------------------------


@needs_shipvision
class TestDeterministicTracking:
    def test_one_box_walking_across_ten_frames_keeps_one_id(self, element) -> None:
        """The property the element exists for, and the one a per-frame detector cannot give.

        Ten frames, one object moving twenty pixels a frame. The ids are not asserted
        literally — ``shipvision`` mints them from a process-wide counter — but there must be
        exactly one of them across the whole run.
        """
        seen = set()
        for frame in range(10):
            x = 20.0 * frame
            emitted = element.process(
                item(frame=frame, detections=detections((x, 50.0, x + 40, 120.0)))
            )
            assert len(emitted.meta["tracks"]) == 1
            seen.update(track_ids(emitted))

        assert len(seen) == 1, f"one object produced {len(seen)} identities: {seen}"

    def test_two_separated_boxes_are_two_stable_ids(self, element) -> None:
        for frame in range(6):
            x = 10.0 * frame
            emitted = element.process(
                item(
                    frame=frame,
                    detections=detections(
                        (x, 10.0, x + 30, 60.0), (x + 300, 10.0, x + 330, 60.0)
                    ),
                )
            )
            if frame == 0:
                first = track_ids(emitted)
            assert len(emitted.meta["tracks"]) == 2

        assert len(set(first)) == 2
        assert track_ids(emitted) == first, "two objects swapped or re-minted their ids"

    def test_an_object_gone_longer_than_max_age_comes_back_as_a_new_id(self, element) -> None:
        """A track that dies must not be resurrected under its old identity.

        ``max_age`` is 3 here, so five empty frames is comfortably past it. Reappearing under
        the old id would tell the cross-camera tier that one object was continuously present
        through a gap in which nothing was seen.
        """
        before = track_ids(
            element.process(item(frame=0, detections=detections((0, 0, 40, 40))))
        )
        for frame in range(1, 6):
            element.process(item(frame=frame, detections=detections()))
        after = track_ids(element.process(item(frame=6, detections=detections((0, 0, 40, 40)))))

        assert before and after
        assert set(before).isdisjoint(after), "a track outlived max_age and kept its id"


@needs_shipvision
class TestPerCameraIsolation:
    """The failure with no symptom: two cameras on one tracker report real objects nowhere."""

    def test_each_camera_gets_its_own_tracker(self, element) -> None:
        element.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))
        element.process(item("cam-b", 0, detections=detections((0, 0, 40, 40))))

        shard = element._shard
        assert set(shard.cameras) == {"cam-a", "cam-b"}
        assert shard.tracker_for("cam-a") is not shard.tracker_for("cam-b")

    def test_two_cameras_never_share_an_identity(self, element) -> None:
        a = track_ids(element.process(item("cam-a", 0, detections=detections((0, 0, 40, 40)))))
        b = track_ids(element.process(item("cam-b", 0, detections=detections((0, 0, 40, 40)))))

        assert set(a).isdisjoint(b), "two cameras' identical boxes got one identity"

    def test_camera_b_is_untouched_by_camera_as_ordering_failure(self, element) -> None:
        """One camera's stream position is one camera's. A shared high-water mark would refuse
        camera B's frame 0 the moment camera A reached frame 5, which reads downstream as a
        camera that stopped producing objects."""
        for frame in range(6):
            element.process(item("cam-a", frame, detections=detections((0, 0, 40, 40))))

        emitted = element.process(item("cam-b", 0, detections=detections((0, 0, 40, 40))))

        assert "missing_stages" not in emitted.meta
        assert len(emitted.meta["tracks"]) == 1


@needs_shipvision
class TestOutOfOrderFrames:
    """A frame that lost the race is emitted without ids — never failed, never absorbed."""

    def test_a_late_frame_is_emitted_with_track_in_missing_stages(self, element) -> None:
        """Feeding a tracker a frame it has already passed double-ages every track and
        double-counts the hit that promotes one, so a replayed frame silently changes which
        identities exist. It is refused — and the *frame* still leaves the element, with its
        boxes intact, because the runner fails an item's future on anything raised out of an
        element and a frame with an honest gap is worth more than no frame at all."""
        element.process(item(frame=5, detections=detections((0, 0, 40, 40))))

        emitted = element.process(item(frame=3, detections=detections((0, 0, 40, 40))))

        assert emitted is not None
        assert emitted.meta["missing_stages"] == ("track",)
        assert "tracks" not in emitted.meta
        assert emitted.key == ("cam-a", 3), "the tag did not survive the refusal"

    def test_the_refusal_is_counted_per_camera_and_by_reason(self, metrics) -> None:
        built = opened(registry=metrics)
        try:
            built.process(item(frame=5, detections=detections((0, 0, 40, 40))))
            built.process(item(frame=3, detections=detections((0, 0, 40, 40))))
        finally:
            built.close()

        assert value(metrics, "shipinfer_track_frames_out_of_order_total", camera="cam-a") == 1
        assert (
            value(metrics, "shipinfer_track_frames_untracked_total", reason="out_of_order") == 1
        )

    def test_the_camera_keeps_tracking_after_a_refusal(self, element) -> None:
        """The refusal costs one frame, not the stream. `stats()` is where an operator reads
        whether the rate is material — and the fix for a high one is upstream."""
        element.process(item(frame=5, detections=detections((0, 0, 40, 40))))
        element.process(item(frame=3, detections=detections((0, 0, 40, 40))))

        emitted = element.process(item(frame=6, detections=detections((0, 0, 40, 40))))

        assert len(emitted.meta["tracks"]) == 1
        assert element._shard.stats()["out_of_order"] == 1

    def test_a_regression_smaller_than_the_threshold_is_not_a_reset(self, element) -> None:
        """A reorder is bounded by the worker count; a restart is not. Confusing the first for
        the second would restart every identity on the camera because two frames raced."""
        element.process(item(frame=10, detections=detections((0, 0, 40, 40))))
        element.process(item(frame=8, detections=detections((0, 0, 40, 40))))

        assert element._shard.stats()["implicit_resets"] == 0


@needs_shipvision
class TestTrackLifecycle:
    def test_camera_removed_drops_the_shard(self, element) -> None:
        """The leak this hook exists to close: without it a removed camera's tracker, its
        Kalman state and its lock live for the process's life."""
        element.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))
        element.process(item("cam-b", 0, detections=detections((0, 0, 40, 40))))

        element.camera_removed("cam-a")

        assert element._shard.cameras == ("cam-b",)

    def test_camera_removed_for_an_unknown_camera_is_a_no_op(self, element) -> None:
        """It fires for every element on the runner, most of which never saw the camera."""
        element.camera_removed("cam-never")

        assert element._shard.cameras == ()

    def test_the_tracker_count_gauge_follows_the_table(self, metrics) -> None:
        built = opened(registry=metrics)
        try:
            built.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))
            built.process(item("cam-b", 0, detections=detections((0, 0, 40, 40))))
            assert value(metrics, "shipinfer_track_cameras", element="track") == 2

            built.camera_removed("cam-a")
            assert value(metrics, "shipinfer_track_cameras", element="track") == 1
        finally:
            built.close()

    def test_a_re_added_camera_is_accepted_at_frame_zero(self, element) -> None:
        """ADR-018 names remove + add as the one recovery for a lost camera, and a re-added
        camera's ingest actor mints a fresh ``FrameCounter``. Without the reset its frame 0 is
        below the previous run's high-water mark and every frame is refused, forever."""
        for frame in range(6):
            element.process(item("cam-a", frame, detections=detections((0, 0, 40, 40))))

        element.camera_added("cam-a")
        emitted = element.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))

        assert "missing_stages" not in emitted.meta
        assert len(emitted.meta["tracks"]) == 1

    def test_a_re_added_camera_does_not_continue_its_old_identities(self, element) -> None:
        before = track_ids(
            element.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))
        )

        element.camera_added("cam-a")
        after = track_ids(
            element.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))
        )

        assert set(before).isdisjoint(after), "the reset kept the tracks it was meant to forget"

    def test_camera_added_builds_no_tracker_for_a_camera_that_has_none(self, element) -> None:
        """It fires for every camera on the shard. Minting a Kalman filter for the forty-nine
        that are not on this element — on the thread holding the runner's lifecycle lock — is
        work for nothing."""
        element.camera_added("cam-never")

        assert element._shard.cameras == ()

    def test_an_unannounced_regression_past_the_threshold_is_an_implicit_reset(
        self, metrics
    ) -> None:
        """An ingest actor that restarted with nobody calling remove + add. Refusing those
        means refusing every frame of that camera until the process ends, which is the state
        ADR-018's recovery exists for and which ``shipinfer run`` leaves nobody to trigger."""
        built = opened(registry=metrics)
        try:
            built.process(item("cam-a", 500, detections=detections((0, 0, 40, 40))))

            emitted = built.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))

            assert "missing_stages" not in emitted.meta
            assert len(emitted.meta["tracks"]) == 1
            assert built._shard.stats()["implicit_resets"] == 1
            assert value(metrics, "shipinfer_track_implicit_resets_total", camera="cam-a") == 1
        finally:
            built.close()

    def test_one_frame_short_of_the_threshold_is_still_a_reorder(self, metrics) -> None:
        """The low side of the boundary, at exactly ``regression_reset - 1``.

        The band between "a regression of 2" and "a regression of 500" was free: `>=` could
        become `>`, or the threshold could be scaled by any factor, and every other test
        stayed green. Getting this line wrong is asymmetric — one frame too low and a single
        reordered frame restarts every identity on the camera, which is the failure the
        threshold exists to prevent — so the pin is *at* the boundary rather than near it.
        """
        built = opened(registry=metrics)
        try:
            built.process(item("cam-a", 100, detections=detections((0, 0, 40, 40))))

            emitted = built.process(
                item(
                    "cam-a",
                    100 - (DEFAULT_REGRESSION_RESET - 1),
                    detections=detections((0, 0, 40, 40)),
                )
            )

            assert emitted.meta["missing_stages"] == ("track",)
            assert "tracks" not in emitted.meta
            assert built._shard.stats()["out_of_order"] == 1
            assert built._shard.stats()["implicit_resets"] == 0
            assert value(metrics, "shipinfer_track_implicit_resets_total", camera="cam-a") == 0
        finally:
            built.close()

    def test_a_regression_of_exactly_the_threshold_is_a_restart(self, metrics) -> None:
        """The high side, at exactly ``regression_reset``: the comparison is ``>=``, so this
        frame is the first one read as a restarted stream rather than as a reordering. It is
        tracked, it is not counted as out of order, and the reset is counted once."""
        built = opened(registry=metrics)
        try:
            built.process(item("cam-a", 100, detections=detections((0, 0, 40, 40))))

            emitted = built.process(
                item(
                    "cam-a",
                    100 - DEFAULT_REGRESSION_RESET,
                    detections=detections((0, 0, 40, 40)),
                )
            )

            assert "missing_stages" not in emitted.meta
            assert len(emitted.meta["tracks"]) == 1
            assert built._shard.stats()["implicit_resets"] == 1
            assert built._shard.stats()["out_of_order"] == 0
            assert value(metrics, "shipinfer_track_implicit_resets_total", camera="cam-a") == 1
        finally:
            built.close()

    def test_regression_reset_zero_refuses_every_regression(self, metrics) -> None:
        """The old behaviour, for a deployment that would rather see the frames stop than see
        every identity under a camera restart."""
        built = opened({"regression_reset": 0}, registry=metrics)
        try:
            built.process(item("cam-a", 500, detections=detections((0, 0, 40, 40))))

            emitted = built.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))

            assert emitted.meta["missing_stages"] == ("track",)
            assert built._shard.stats()["implicit_resets"] == 0
        finally:
            built.close()

    def test_the_threshold_default_is_above_a_reorder_and_below_a_restart(self) -> None:
        assert DEFAULT_REGRESSION_RESET == 64

    def test_a_stream_restarting_inside_the_window_is_refused_then_continues_old_ids(
        self, element
    ) -> None:
        """The window :data:`DEFAULT_REGRESSION_RESET` names, pinned so the docstring's caveat
        is a measurement and not a guess.

        A camera that ran to frame 39 and restarts at 0 regresses by only 39 — real, but too
        small to call a restart. So every frame of the new stream up to the old high-water mark
        is refused, and the first frame past it continues the **old** identities across the
        discontinuity with no reset counted. Strictly better than refusing the camera for the
        process's life, and strictly worse than the announced remove + add.
        """
        long_lived = opened({"options": {"min_hits": 1, "max_age": 300}})
        try:
            for frame in range(40):
                emitted = long_lived.process(
                    item("cam-a", frame, detections=detections((0, 0, 40, 40)))
                )
            before = track_ids(emitted)

            refused = [
                long_lived.process(item("cam-a", frame, detections=detections((0, 0, 40, 40))))
                for frame in range(40)
            ]
            after_the_cut = long_lived.process(
                item("cam-a", 40, detections=detections((0, 0, 40, 40)))
            )

            assert all("tracks" not in e.meta for e in refused), "the whole restart is refused"
            assert long_lived._shard.stats()["implicit_resets"] == 0
            assert track_ids(after_the_cut) == before, "the old identity crossed the cut"
        finally:
            long_lived.close()

    def test_close_forgets_every_camera(self, element) -> None:
        """A reopened chain must not continue ids from before a gap it cannot see."""
        element.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))

        element.close()

        assert element._shard is None


@needs_shipvision
class TestEmptyFrame:
    def test_an_empty_frame_still_advances_the_tracker(self, element) -> None:
        """Ageing is how a track dies. An element that treated an empty frame as nothing to do
        would keep every departed object alive forever and the pool would grow for as long as
        the process ran."""
        element.process(item(frame=0, detections=detections((0, 0, 40, 40))))
        assert element._shard.stats()["tracks"] == 1

        for frame in range(1, 8):
            emitted = element.process(item(frame=frame, detections=detections()))
            assert emitted.meta["tracks"] == []
            assert "missing_stages" not in emitted.meta

        assert element._shard.stats()["tracks"] == 0, "an unseen track was never aged out"

    def test_a_frame_the_detector_never_answered_for_is_not_an_empty_frame(
        self, element, metrics
    ) -> None:
        """ "No objects" and "no answer" are different events, and only the first should age a
        track. A missing key is the second."""
        built = opened(registry=metrics)
        try:
            emitted = built.process(item(frame=0))

            assert emitted.meta["missing_stages"] == ("track",)
            assert "tracks" not in emitted.meta
            assert built._shard.cameras == (), "a frame with no detections built a tracker"
            assert (
                value(metrics, "shipinfer_track_frames_untracked_total", reason="no_detections")
                == 1
            )
        finally:
            built.close()

    def test_missing_stages_accumulates_rather_than_replacing(self, element) -> None:
        """Another element's absence must survive this one's."""
        emitted = element.process(item(frame=0, missing_stages=("segment",)))

        assert emitted.meta["missing_stages"] == ("segment", "track")


@needs_shipvision
class TestTracksAreAttributedToDetectionRows:
    """``meta["track_rows"]``: which detection each published track came from.

    The mapping an ``output`` element cannot recover for itself. A track's box is the filtered
    estimate, so only the element that ran the tracker holds the detections, the tracks and the
    solver at once; a sink that redid this would be a second, quieter tracker. Every way of
    getting it wrong publishes a track id on the wrong object, which has no symptom downstream
    beyond two cameras disagreeing about what they are looking at.
    """

    def test_one_row_index_per_published_track(self, element) -> None:
        emitted = element.process(item(detections=detections((0, 0, 40, 40))))

        rows = emitted.meta["track_rows"]
        assert len(rows) == len(emitted.meta["tracks"])
        assert rows == (0,)

    def test_a_track_is_attributed_to_the_box_it_actually_overlaps(self, element) -> None:
        """Two objects far apart, so there is one right answer and the test can name it."""
        emitted = element.process(
            item(detections=detections((0, 0, 40, 40), (300, 300, 340, 340)))
        )

        tracks = emitted.meta["tracks"]
        rows = emitted.meta["track_rows"]
        by_row = dict(zip(rows, tracks, strict=True))
        assert set(rows) == {0, 1}
        assert by_row[0].box[0] < 100
        assert by_row[1].box[0] > 200

    def test_a_classes_slot_maps_back_to_the_frames_own_row_numbering(self) -> None:
        """A slot that tracks only ships must not report the *subset's* index.

        The bug this pins is off by however many rows the slot skipped: the tracker sees rows
        [1] of a two-row frame as its row 0, and an event built on that number puts the ship's
        identity on the person.
        """
        built = opened({"classes": ["ship"]})
        try:
            emitted = built.process(
                item(
                    detections=Detections(
                        boxes=np.array(
                            [[300, 300, 340, 340], [0, 0, 40, 40]], dtype=np.float32
                        ),
                        scores=np.full(2, 0.9, dtype=np.float32),
                        class_ids=np.array([0, 8], dtype=np.int32),
                        labels=("person", "ship"),
                    )
                )
            )
        finally:
            built.close()

        assert emitted.meta["track_rows"] == (1,), "the ship is row 1 of the frame"

    def test_a_frame_with_no_detections_attributes_nothing(self, element) -> None:
        emitted = element.process(item(detections=Detections.empty()))

        assert emitted.meta["track_rows"] == ()

    def test_an_untracked_frame_files_no_attribution(self, element) -> None:
        """No tracks, no rows: the two keys are one answer and never half of one."""
        emitted = element.process(item())

        assert "track_rows" not in emitted.meta
        assert emitted.meta["missing_stages"] == ("track",)

    def test_the_threshold_refuses_an_answer_rather_than_tuning_one(self) -> None:
        """``attribution_iou: 1.0`` demands a perfect overlap, so nothing is attributed.

        Not a realistic setting — it is the knob turned to its refusing end, which is the only
        way to assert that a track that matches nothing gets ``-1`` instead of its nearest box.
        """
        built = opened({"attribution_iou": 1.0})
        try:
            built.process(item(detections=detections((0, 0, 40, 40))))
            emitted = built.process(item(frame=1, detections=detections((10, 10, 50, 50))))
        finally:
            built.close()

        assert set(emitted.meta["track_rows"]) <= {-1}

    @pytest.mark.parametrize("declared", ["not-a-number", 0, 1.5, -0.2])
    def test_an_impossible_threshold_is_refused_at_load(self, declared: Any) -> None:
        with pytest.raises(ConfigurationError, match="attribution_iou"):
            create_element(
                ElementKind.TRACK, "shipvision", "track", {"attribution_iou": declared}
            )


@needs_shipvision
class TestTrackPlane:
    """The plane change: this is where the chain stops carrying pixels."""

    def test_the_output_is_labelled_meta_cpu(self, element) -> None:
        emitted = element.process(item(caps="nv12@gpu", detections=detections((0, 0, 40, 40))))

        assert str(emitted.caps) == "meta@cpu"

    def test_the_payload_is_dropped_with_the_label(self, element) -> None:
        """Leaving a device handle on an item labelled ``meta@cpu`` is a relabelling: it tells
        the rest of the chain that VRAM is host metadata, which is the laundering
        ``_substitute_donor`` refuses and the download arch.md section 8 exists to make
        visible."""
        emitted = element.process(
            item(
                caps="nv12@gpu", payload="device-handle", detections=detections((0, 0, 40, 40))
            )
        )

        assert emitted.payload is None

    def test_an_untracked_frame_changes_plane_too(self, element) -> None:
        """Otherwise a refusal would hand the next element an item whose cap and payload
        disagree, which is worse than the refusal."""
        element.process(item(frame=5, detections=detections((0, 0, 40, 40))))

        emitted = element.process(
            item(frame=3, caps="nv12@gpu", payload="device-handle", detections=detections())
        )

        assert str(emitted.caps) == "meta@cpu"
        assert emitted.payload is None

    def test_the_published_tracks_are_not_rewritten_by_the_next_frame(self, element) -> None:
        """A consumer may hold a frame's tracks past the camera's next frame — the
        cross-camera tier buffers exactly this — and the pool mutates its own ``Track``
        objects in place, so aliasing them would make a buffered run read as the last frame's
        state on every entry.

        Both pool backends make it safe and for different reasons (``TrackPool.output`` copies
        with ``dataclasses.replace``; the native pool decodes fresh objects off a fresh array
        every frame), so what is asserted here is the **property**, on whichever backend this
        host actually built.
        """
        first = element.process(item(frame=0, detections=detections((0, 0, 40, 40))))
        held = first.meta["tracks"]
        assert held, "nothing was published, so nothing was pinned"
        before = [
            (t.track_id, np.asarray(t.box, dtype=np.float64).tolist(), t.time_since_update)
            for t in held
        ]

        element.process(item(frame=1, detections=detections((7, 7, 47, 47))))

        after = [
            (t.track_id, np.asarray(t.box, dtype=np.float64).tolist(), t.time_since_update)
            for t in held
        ]
        assert after == before, "the next frame rewrote a track a consumer was still holding"

    def test_frame_hw_rides_along(self, element) -> None:
        """The boxes are in the source frame's pixels and the payload that could have said how
        big that is has just been dropped. ``mtmc`` refuses a non-positive extent."""
        emitted = element.process(
            item(detections=detections((0, 0, 40, 40)), frame_hw=(1080, 1920))
        )

        assert emitted.meta["frame_hw"] == (1080, 1920)


@needs_shipvision
class TestWhatItReadsOffTheItem:
    def test_the_classes_filter_selects_which_rows_are_tracked(self) -> None:
        built = opened({"classes": ["ship"]})
        try:
            emitted = built.process(
                item(
                    detections=Detections(
                        boxes=np.array(
                            [[0, 0, 40, 40], [200, 200, 260, 260]], dtype=np.float32
                        ),
                        scores=np.array([0.9, 0.9], dtype=np.float32),
                        class_ids=np.array([0, 1], dtype=np.int32),
                        labels=("ship", "person"),
                    )
                )
            )

            assert len(emitted.meta["tracks"]) == 1
        finally:
            built.close()

    def test_a_per_row_vectors_array_is_attached_as_appearance(self, element) -> None:
        """The vector is what carries an identity through the frames where geometry alone is
        ambiguous, so it must reach the tracker rather than be quietly dropped."""
        emitted = element.process(
            item(
                detections=detections((0, 0, 40, 40)),
                vectors=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        )

        embedding = emitted.meta["tracks"][0].embedding
        assert embedding is not None
        assert pytest.approx(1.0) == float(np.linalg.norm(embedding))

    def test_vectors_keyed_by_detection_index_are_accepted(self, element) -> None:
        emitted = element.process(
            item(
                detections=detections((0, 0, 40, 40)),
                vectors={0: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)},
            )
        )

        assert emitted.meta["tracks"][0].embedding is not None

    def test_vectors_keyed_outside_the_detection_rows_are_refused(self, element) -> None:
        """An off-by-N scatter-back. Without the coverage check every row silently gets
        ``embedding=None`` and the chain reads as healthy, which is the same silence the
        sequence form's length check already refuses, arriving through the other door."""
        with pytest.raises(ValidationError, match="name no detection"):
            element.process(
                item(
                    detections=detections((0, 0, 40, 40)),
                    vectors={5: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)},
                )
            )

    def test_vectors_covering_only_some_rows_stay_legal(self, element) -> None:
        """Partial coverage is what a chain with one re-ID model produces: only the person
        rows are embedded. The uncovered rows track on motion alone, which is the documented
        `embedding=None` path and not a refusal."""
        emitted = element.process(
            item(
                detections=Detections(
                    boxes=np.array([[0, 0, 40, 40], [200, 200, 260, 260]], dtype=np.float32),
                    scores=np.array([0.9, 0.9], dtype=np.float32),
                    class_ids=np.array([0, 1], dtype=np.int32),
                    labels=("ship", "person"),
                ),
                vectors={1: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)},
            )
        )

        embeddings = [t.embedding is not None for t in emitted.meta["tracks"]]
        assert sorted(embeddings) == [False, True]

    def test_an_empty_vectors_mapping_covers_nothing_and_is_not_an_off_by_n(
        self, element
    ) -> None:
        """The ordinary frame once a crop element is in the chain: ``embed_person`` sees three
        ships and covers none of them, so it files ``{}``
        (:class:`~shipinfer.topology.elements.pool._PoolCropElement`). Zero keys index nothing
        because there was nothing to index, which is not the same as keys that index nothing —
        and refusing it would fail a whole camera for being unremarkable."""
        emitted = element.process(item(detections=detections((0, 0, 40, 40)), vectors={}))

        assert len(emitted.meta["tracks"]) == 1
        assert emitted.meta["tracks"][0].embedding is None

    def test_vectors_that_cannot_be_attributed_to_rows_are_refused(self, element) -> None:
        """A re-ID model's raw output tensors under their own names are not an attribution.
        Ignoring them would be a measurable accuracy loss reported as a healthy chain."""
        with pytest.raises(ValidationError, match="vectors"):
            element.process(
                item(
                    detections=detections((0, 0, 40, 40)),
                    vectors=np.zeros((3, 4), dtype=np.float32),
                )
            )

    def test_raw_model_outputs_under_detections_are_refused_by_type(self, element) -> None:
        with pytest.raises(ValidationError, match="detections"):
            element.process(item(detections={"output0": object()}))

    def test_an_item_with_no_frame_id_is_refused_in_shipinfers_own_vocabulary(
        self, element
    ) -> None:
        """``RequestContext``'s default ``frame_id`` is ``-1``, and ``FrameTag`` refuses a
        negative one — with ``shipvision.errors.ConfigurationError``. This is the single
        hand-over where the element speaks another library's vocabulary, so it is the one
        place a caller could be handed somebody else's exception class; the check happens on
        this side of the seam."""
        untagged = ChainItem(
            context=RequestContext(camera_id="cam-a"),
            caps=Caps.parse("bgr@cpu"),
            payload="frame-handle",
            meta={"detections": detections((0, 0, 40, 40))},
        )

        with pytest.raises(ValidationError, match="frame_id -1"):
            element.process(untagged)

    def test_the_frame_tag_reaches_the_tracker_intact(self, element) -> None:
        """``(camera_id, frame_id)`` rides untouched from ingest to the last element (ADR-002),
        and here it crosses into another library's vocabulary — the one hand-over where a
        converter can lose it and nothing downstream can tell."""
        emitted = element.process(item("cam-7", 42, detections=detections((0, 0, 40, 40))))

        tag = emitted.meta["tracks"][0].tag
        assert (tag.camera_id, tag.frame_id) == ("cam-7", 42)
        assert tag.timestamp == pytest.approx(42 * 0.05)


@needs_shipvision
class TestUnderThreads:
    def test_each_camera_keeps_its_own_lock(self, element) -> None:
        """Fifty cameras keep fifty independent locks. One lock for the table would make the
        stateful tier a serial stage in a design whose whole point is that it is not — and the
        work under a camera's lock is a Kalman predict plus a Hungarian solve over at most
        ``max_detections`` boxes, which is tens of microseconds."""
        element.process(item("cam-a", 0, detections=detections((0, 0, 40, 40))))
        element.process(item("cam-b", 0, detections=detections((0, 0, 40, 40))))
        shard = element._shard

        assert shard._cameras["cam-a"].lock is not shard._cameras["cam-b"].lock

    def test_two_workers_on_one_camera_produce_one_ordered_stream(self, element) -> None:
        """Whichever frame checks second is refused, and neither is corrupted. Both outcomes
        are emitted, so the item count is exact rather than a lower bound."""
        results: list[ChainItem] = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def walk(frame: int) -> None:
            barrier.wait()
            emitted = element.process(item(frame=frame, detections=detections((0, 0, 40, 40))))
            with lock:
                results.append(emitted)

        threads = [threading.Thread(target=walk, args=(frame,)) for frame in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert len(results) == 2
        tracked = [r for r in results if "tracks" in r.meta]
        refused = [r for r in results if "tracks" not in r.meta]
        # Which frame checked first is the race, so the count is read off the shard's own
        # counter rather than asserted as a lower bound: `>= 1` cannot be false here, because
        # only the frame that checks *second* can lose.
        out_of_order = element._shard.stats()["out_of_order"]
        assert out_of_order in (0, 1), "a frame was refused that did not lose a race"
        assert len(tracked) == 2 - out_of_order
        assert [r.key[1] for r in refused] == ([] if out_of_order == 0 else [1])


# -- over the runner --------------------------------------------------------------------------


@needs_shipvision
class TestTrackOverTheRunner:
    """The element inside a real chain, walked by a real worker off a real fair lane.

    Offline: every frame comes from an injected source and no element needs a model. What this
    adds over the unit tests is the two things they cannot see — that the runner's metrics
    registry is what the element counts on, and that ``(camera_id, frame_id)`` survives the
    queue, the worker thread and the plane change on the way to the sink.
    """

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

    def test_every_frame_reaches_the_sink_with_its_tag_and_its_tracks(self, runner) -> None:
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        started = runner(
            chain,
            settings=ServerSettings(
                pipeline={"workers": 1, "queue_capacity": 64},
                ingest={"read_timeout_ms": 50, "open_timeout_ms": 50, "empty_read_sleep_ms": 0},
            ),
            source_factory=scripted(frames=4),
        )
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))

        sink = chain.node("output").element
        assert isinstance(sink, MockOutput)
        assert until(lambda: len(sink.emitted) == 4), sink.emitted

        assert [emitted.key for emitted in sink.emitted] == [
            ("cam-a", frame) for frame in range(4)
        ]
        for emitted in sink.emitted:
            assert str(emitted.caps) == "meta@cpu"
            assert emitted.payload is None
            assert "tracks" in emitted.meta

    def test_the_element_counts_on_the_runners_own_registry(self, runner) -> None:
        """A metric on a registry no exporter reads is worse than an absent one, because it
        reads as evidence. The runner hands its registry over on ``ElementContext``."""
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        started = runner(
            chain,
            settings=ServerSettings(
                pipeline={"workers": 1, "queue_capacity": 64},
                ingest={"read_timeout_ms": 50, "open_timeout_ms": 50, "empty_read_sleep_ms": 0},
            ),
            source_factory=scripted(frames=2),
        )
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))

        sink = chain.node("output").element
        assert until(lambda: len(sink.emitted) == 2)

        assert value(started.metrics.registry, "shipinfer_track_cameras", element="track") == 1

    def test_removing_the_camera_drops_its_tracker(self, runner) -> None:
        """The lifecycle hook wired in C2, reaching a stateful element for the first time."""
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        started = runner(
            chain,
            settings=ServerSettings(
                pipeline={"workers": 1, "queue_capacity": 64},
                ingest={"read_timeout_ms": 50, "open_timeout_ms": 50, "empty_read_sleep_ms": 0},
            ),
            source_factory=scripted(frames=2),
        )
        started.add_camera(CameraSpec("cam-a", "injected://a", 0.0))
        sink = chain.node("output").element
        assert until(lambda: len(sink.emitted) == 2)
        element = chain.node("track").element
        assert element._shard.cameras == ("cam-a",)

        started.remove_camera("cam-a")

        assert element._shard.cameras == ()


class TestTheAttributionArithmeticWithoutASolver:
    """The ``keep``-remap, pinned on a checkout with no submodule.

    Everything else in this file that touches ``track_rows`` drives a **real** tracker and
    therefore skips under the mask — which left the arithmetic that turns a solver's column
    number back into the frame's own row number with no offline coverage at all, and that
    arithmetic is exactly the off-by-N this family keeps catching. The solver is not what
    needs pinning here (it is ``shipvision``'s, and it is tested there); what needs pinning is
    what this element does with its answer.

    So the two handles ``open()`` resolves are injected instead: a fake ``iou_matrix`` and a
    fake ``associate`` that returns the matches the test names. No ``open()``, no submodule,
    no tracker — and a wrong remap fails here rather than only on a developer's machine.
    """

    class Track:
        """The one attribute ``_attribute`` reads off a published track."""

        def __init__(self, box: tuple[float, float, float, float]) -> None:
            self.box = np.array(box, dtype=np.float32)

    def build(self, matches, **params) -> tuple[Any, dict[str, Any]]:
        """A ``track`` element with both solver handles faked, and the call it recorded."""
        element = ShipvisionTrack("track", dict(params))
        seen: dict[str, Any] = {}

        def iou_matrix(tracks: np.ndarray, candidates: np.ndarray) -> np.ndarray:
            seen["tracks"] = tracks
            seen["candidates"] = candidates
            return np.zeros((tracks.shape[0], candidates.shape[0]), dtype=np.float64)

        def associate(cost: np.ndarray, max_cost: float):
            seen["cost"] = cost
            seen["max_cost"] = max_cost
            return matches, (), ()

        element._iou_matrix = iou_matrix
        element._associate = associate
        return element, seen

    def test_a_column_of_a_full_frame_is_that_frames_row(self) -> None:
        element, _ = self.build([(0, 1), (1, 0)])

        rows = element._attribute(
            detections((0, 0, 10, 10), (100, 100, 110, 110)),
            range(2),
            [self.Track((0, 0, 10, 10)), self.Track((100, 100, 110, 110))],
        )

        assert rows == (1, 0)

    def test_a_column_of_a_filtered_subset_is_remapped_to_the_frames_row(self) -> None:
        """The bug: a slot with ``classes:`` tracks rows [1, 3] and the solver calls them 0
        and 1. Publishing those numbers puts one object's identity on another's box, and it
        has no symptom until two cameras disagree about what they are looking at."""
        element, _ = self.build([(0, 0), (1, 1)], classes=["ship"])

        rows = element._attribute(
            detections((0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50), (60, 60, 70, 70)),
            (1, 3),
            [self.Track((20, 20, 30, 30)), self.Track((60, 60, 70, 70))],
        )

        assert rows == (1, 3)

    def test_only_the_kept_rows_are_offered_to_the_solver(self) -> None:
        """A filtered slot must not be able to attribute a track to a row it never saw."""
        element, seen = self.build([], classes=["ship"])

        element._attribute(
            detections((0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50), (60, 60, 70, 70)),
            (1, 3),
            [self.Track((20, 20, 30, 30))],
        )

        assert seen["candidates"].tolist() == [[20, 20, 30, 30], [60, 60, 70, 70]]

    def test_a_track_the_solver_matched_to_nothing_stays_minus_one(self) -> None:
        """``-1`` is ordinary — a coasting track is not a detection and has no row to ride."""
        element, _ = self.build([(1, 0)])

        rows = element._attribute(
            detections((0, 0, 10, 10)),
            range(1),
            [self.Track((500, 500, 510, 510)), self.Track((0, 0, 10, 10))],
        )

        assert rows == (-1, 0)

    def test_the_answer_is_one_entry_per_track_always(self) -> None:
        """``output`` reads ``tracks`` and ``track_rows`` in step and refuses a length gap."""
        element, _ = self.build([])

        assert element._attribute(detections(), range(0), [self.Track((0, 0, 1, 1))]) == (-1,)
        assert element._attribute(detections((0, 0, 1, 1)), range(1), []) == ()

    def test_the_solver_is_given_a_cost_and_the_configured_threshold(self) -> None:
        """``1 - IoU``, capped at ``1 - attribution_iou``: the element converts once, here."""
        element, seen = self.build([], attribution_iou=0.4)

        element._attribute(detections((0, 0, 10, 10)), range(1), [self.Track((0, 0, 10, 10))])

        assert seen["cost"].tolist() == [[1.0]]
        assert seen["max_cost"] == pytest.approx(0.6)
