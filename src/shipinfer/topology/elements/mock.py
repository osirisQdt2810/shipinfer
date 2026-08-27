"""Hardware-free elements, one per kind — the default in tests, like ``backends/mock.py``.

These exist so that the *chain* can be tested without a chain of real things. A mock element
declares the caps its real counterpart declares (``nv12@gpu`` on the device path,
``meta@cpu`` once the frame is behind us), counts its lifecycle calls, and adds one
metadata key. That is enough to pin every property the loader and the runners promise:
ordering, branch conditions, caps refusals, the tag surviving to the sink, idempotent
open/close.

**The cap convention, because it is not obvious.** An element that *reads* the frame and
adds metadata declares the frame's cap on both sides — a detector consumes ``nv12@gpu`` and
hands the same ``nv12@gpu`` on, because the next element still needs the pixels. Only
``track``, ``mtmc`` and ``output`` drop to the metadata plane (``meta@cpu``): by then the
boxes, vectors and ids are the payload and the frame is not needed. Getting this wrong in a
real element is how a chain ends up copying every frame to host memory to run a CPU tracker.

Nothing here does any work. ``MockDetect`` invents one box; it does not pretend to detect.
A mock that produced plausible numbers would eventually be trusted for something.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.registry import registry_for

__all__ = [
    "MockCpuDetect",
    "MockDecode",
    "MockDetect",
    "MockEmbed",
    "MockMtmc",
    "MockOutput",
    "MockRecognize",
    "MockSegment",
    "MockTrack",
]


class _Mock(Element):
    """Shared behaviour: count the lifecycle, forward the item, add one metadata key.

    The counters are what a lifecycle test asserts on. They are plain ints and not
    thread-safe, which is correct for the offline tier and stated here so nobody reaches for
    them from a multi-threaded runner test.
    """

    def __init__(self, name: str, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(name, params)
        self.opens = 0
        self.processes = 0
        self.closes = 0

    def _do_open(self, context: ElementContext) -> None:
        self.opens += 1

    def _do_close(self) -> None:
        self.closes += 1

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        """The metadata this element contributes. Overridden per kind."""
        return {}

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.processes += 1
        # `derive`, never a fresh ChainItem: that is what carries the (camera_id, frame_id)
        # tag forward, and a mock that cut the corner would make the tests agree with a
        # mistake a real element must not make.
        return item.derive(caps=self.output_caps[0], **self._meta(item))


@registry_for(ElementKind.DECODE).register("mock")
class MockDecode(_Mock):
    """A decoder that invents a frame handle instead of opening a camera."""

    kind: ClassVar[ElementKind] = ElementKind.DECODE
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.processes += 1
        camera, frame = item.key
        # Stands in for a device-resident frame handle. A string, so a test that leaks it
        # into a comparison fails loudly rather than comparing two opaque objects.
        return item.derive(
            caps=self.output_caps[0],
            payload=f"frame:{camera}:{frame}",
            frame_id=frame,
        )


@registry_for(ElementKind.DETECT).register("mock")
class MockDetect(_Mock):
    """A detector on the device path, with a host fallback it never prefers."""

    kind: ClassVar[ElementKind] = ElementKind.DETECT
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "bgr@cpu")
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        return {"boxes": [(0, 0, 10, 10)], "class": self.params.get("class", "ship")}


@registry_for(ElementKind.DETECT).register("mock-cpu")
class MockCpuDetect(_Mock):
    """A host-memory detector — the one that must **not** load behind a GPU decoder.

    Exists for exactly one test: a chain wiring ``nv12@gpu`` into ``bgr@cpu`` is refused
    instead of quietly growing a device-to-host copy per frame (arch.md §8).
    """

    kind: ClassVar[ElementKind] = ElementKind.DETECT
    accepts: ClassVar[tuple[str, ...]] = ("bgr@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("bgr@cpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        return {"boxes": [(0, 0, 10, 10)], "class": "ship"}


@registry_for(ElementKind.SEGMENT).register("mock")
class MockSegment(_Mock):
    """A segmenter: reads the frame, adds masks, hands the frame on."""

    kind: ClassVar[ElementKind] = ElementKind.SEGMENT
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu",)
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        return {"masks": [[0, 1]]}


@registry_for(ElementKind.EMBED).register("mock")
class MockEmbed(_Mock):
    """An embedder, which will take a pre-cropped tensor if one is offered."""

    kind: ClassVar[ElementKind] = ElementKind.EMBED
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "tensor@gpu")
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        return {"vectors": [[0.0, 1.0]]}


@registry_for(ElementKind.RECOGNIZE).register("mock")
class MockRecognize(_Mock):
    """A recogniser: turns vectors into identities."""

    kind: ClassVar[ElementKind] = ElementKind.RECOGNIZE
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu",)
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        return {"identities": ["ship-1"]}


@registry_for(ElementKind.TRACK).register("mock")
class MockTrack(_Mock):
    """A tracker: the step where the chain leaves the frame behind.

    Accepts either plane — a tracker fed by a detector still sees ``nv12@gpu``, one fed by
    another metadata element sees ``meta@cpu`` — and produces metadata only, because a MOT
    algorithm needs boxes and not pixels (arch.md §5⑥).
    """

    kind: ClassVar[ElementKind] = ElementKind.TRACK
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "meta@cpu")
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        return {"tracks": [1]}


@registry_for(ElementKind.MTMC).register("mock")
class MockMtmc(_Mock):
    """Cross-camera association: metadata in, metadata out."""

    kind: ClassVar[ElementKind] = ElementKind.MTMC
    accepts: ClassVar[tuple[str, ...]] = ("meta@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        return {"global_ids": ["g-1"]}


@registry_for(ElementKind.OUTPUT).register("mock")
class MockOutput(_Mock):
    """A sink that keeps what it was given, so a test can read the far end of a chain.

    Accepts ``*@*``: a sink is the one place a wildcard is honest, because whatever the
    chain ends up producing is what gets emitted.
    """

    kind: ClassVar[ElementKind] = ElementKind.OUTPUT
    accepts: ClassVar[tuple[str, ...]] = ("*@*",)

    def __init__(self, name: str, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(name, params)
        self.emitted: list[ChainItem] = []

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.processes += 1
        self.emitted.append(item)
        # None because the item is *consumed*, not because anything failed — the distinction
        # `Element.process` documents.
        return None
