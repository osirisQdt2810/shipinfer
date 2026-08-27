"""The decode elements: three registry entries, one head cap, and nothing with a thread in it.

Offline throughout, and that is a property of the family rather than of the test: a decode
element names an ingest source and never builds one, so nothing here needs GStreamer, PyAV,
OpenCV or a camera. The test that the *runner* turns those names into running cameras is
``tests/runners/test_camera_lifecycle.py``.
"""

from __future__ import annotations

import textwrap

import pytest

from shipinfer.core.errors import UnknownElementImplError
from shipinfer.core.request import RequestContext
from shipinfer.topology import Caps, ChainItem, ChainSpec, ElementKind, Topology
from shipinfer.topology.elements.decode import (
    GStreamerDecode,
    PyAvDecode,
    ReplayDecode,
    _IngestDecode,
)
from shipinfer.topology.registry import create_element, registry_for

CHAIN = """
name: decoded
elements:
  decode: {impl: __IMPL__}
  detect: {impl: mock, model: ship_detector}
  output: {impl: mock}
"""


def load(impl: str) -> Topology:
    return Topology.from_spec(
        ChainSpec.from_yaml(textwrap.dedent(CHAIN).replace("__IMPL__", impl))
    )


def item(caps: str = "bgr@cpu") -> ChainItem:
    return ChainItem(
        RequestContext(camera_id="cam-0", frame_id=7, captured_ns=11),
        Caps.parse(caps),
        payload="a frame",
        meta={"boxes": [1]},
    )


class TestEachOneNamesAnIngestSource:
    """The whole content of an implementation is the source it names — so assert that."""

    @pytest.mark.parametrize(
        ("impl", "source"),
        [("replay", "replay"), ("gstreamer", "gstreamer"), ("pyav", "pyav")],
    )
    def test_the_registered_name_resolves_to_a_class_naming_that_source(
        self, impl: str, source: str
    ) -> None:
        element = create_element(ElementKind.DECODE, impl, "decode")

        assert isinstance(element, _IngestDecode)
        assert element.source == source
        assert element.impl == impl

    @pytest.mark.parametrize(("alias", "canonical"), [("file", "replay"), ("gst", "gstreamer")])
    def test_the_aliases_reach_the_same_class(self, alias: str, canonical: str) -> None:
        registry = registry_for(ElementKind.DECODE)

        assert registry.get(alias) is registry.get(canonical)

    def test_every_source_it_names_is_one_ingest_actually_registers(self) -> None:
        """The one way this family can be wrong: a name that resolves to no decoder.

        A typo here is invisible until a camera is added at runtime, where it surfaces as an
        unknown video source rather than as a chain that will not load. Importing ``ingest``
        costs a decode-free registry walk (``ingest/registry.py``) and is exactly what the
        runner does when it starts a camera.
        """
        from shipinfer.ingest import SOURCES

        for cls in (ReplayDecode, GStreamerDecode, PyAvDecode):
            assert cls.source in SOURCES, (cls.__name__, cls.source, SOURCES.names())


class TestTheHeadCap:
    @pytest.mark.parametrize("impl", ["replay", "gstreamer", "pyav"])
    def test_all_three_declare_host_memory_bgr(self, impl: str) -> None:
        """Every source shipping today delivers an HWC BGR image in ordinary RAM.

        Declaring ``nv12@gpu`` would be the lie that matters: the loader would wire a
        device-path detector behind it and every frame would arrive in the wrong memory.
        """
        chain = load(impl)

        assert [str(cap) for cap in chain.node("decode").element.output_caps] == ["bgr@cpu"]
        assert str(chain.edges[0].caps) == "bgr@cpu"

    def test_it_takes_nothing_so_it_can_be_a_root(self) -> None:
        assert load("replay").node("decode").element.accepts == ()

    def test_gstreamer_gpu_is_still_unregistered(self) -> None:
        """Phase D registers it, with ``nv12@gpu`` and a DataPool behind it (arch.md §3).

        Registering it here with a ``bgr@cpu`` cap would be worse than not having it: the
        production chain would load, and every frame would take a device-to-host copy nobody
        asked for.
        """
        with pytest.raises(UnknownElementImplError) as caught:
            load("gstreamer-gpu")

        assert caught.value.impl == "gstreamer-gpu"


class TestAtWalkTimeItIsAPassThrough:
    def test_it_hands_the_item_on_with_its_tag_payload_and_caps_untouched(self) -> None:
        """The frame is already in the item — the runner's sink put it there."""
        element = ReplayDecode("decode")
        element.open()
        incoming = item()

        outgoing = element.process(incoming)

        assert outgoing is not None
        assert outgoing.context is incoming.context
        assert outgoing.payload == "a frame"
        assert outgoing.meta == {"boxes": [1]}

    def test_it_does_not_restamp_the_cap_with_its_own_first_produces(self) -> None:
        """A cap belongs to an edge, not to an element (``topology/chain.py::Edge``).

        Driven with a cap that is deliberately *not* ``bgr@cpu``: an element that stamped
        ``output_caps[0]`` would pass an assertion made with the matching one and would still
        be overwriting a negotiated per-edge decision with a per-element guess.
        """
        element = ReplayDecode("decode")
        element.open()

        outgoing = element.process(item("tensor@gpu"))

        assert outgoing is not None
        assert str(outgoing.caps) == "tensor@gpu"

    def test_open_and_close_acquire_nothing(self) -> None:
        """No camera, no thread, no runtime: the cameras are the runner's (arch.md §5①)."""
        element = PyAvDecode("decode")

        element.open()
        assert element.is_open
        element.close()
        element.close()

        assert not element.is_open
