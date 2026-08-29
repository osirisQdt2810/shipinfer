"""Hardware-free elements, one per kind — what chain tests load where no model is needed.

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

**What a mock does copy is the *type*.** ``MockDetect`` files its invented box under
``meta["detections"]`` as a real :class:`~shipinfer.topology.elements.detections.Detections`,
beside the older ``meta["boxes"]`` list the chain and runner tests read. That is what lets the
stateful elements phase C adds — ``track`` first — be tested end to end on a mock chain with no
model repository, against the parallel-array shape a ``pool`` detector actually decodes into
rather than a stand-in for it.

**No mock declares ``requires_model_name`` or ``needs_model``**, and that is the honest
answer rather than an omission: ``MockDetect`` invents a box and resolves nothing against a
model pool, so a chain of these needs neither a ``model:`` in front of it (the loader's
question) nor an ``InferenceServer`` behind it (the pool's). A test that needs an element
which *does* declare one declares it on a double of its own
(``tests/runners/test_inprocess.py``), which is where a stand-in for a `pool` element belongs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind, RowIndexed
from shipinfer.topology.elements.detections import DecodeParams, Detections
from shipinfer.topology.registry import registry_for

__all__ = [
    "MockAnyDecode",
    "MockCpuDetect",
    "MockCpuOutput",
    "MockDecode",
    "MockDetect",
    "MockEmbed",
    "MockLazyOutput",
    "MockMtmc",
    "MockOutput",
    "MockPassthrough",
    "MockRecognize",
    "MockSegment",
    "MockTrack",
    "invented_detections",
]


class _Mock(Element):
    """Shared behaviour: count the lifecycle, forward the item, add one metadata key.

    The counters are what a lifecycle test asserts on. They are plain ints and not
    thread-safe, which is correct for the offline tier and stated here so nobody reaches for
    them from a multi-threaded runner test.
    """

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
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
        #
        # `output_caps[0]` and not the negotiated `Edge.caps` -- and a real element must not
        # copy that. The cap on an item is the cap of the edge it is travelling, which the
        # loader decided per pair (`Edge`, and `Topology.edges` is where a runner reads it);
        # an element with two `produces` and two consumers hands a different cap to each,
        # so its own first declaration is not the answer. Stamping it here is valid only
        # because no mock declares two `produces`, so the first is the only one. Resolving
        # the edge properly means telling the element which edge it is on, and that is the
        # runner's job -- not something to smuggle into `ChainItem` from here.
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


#: The default class table, inverted. Built from :class:`DecodeParams` rather than written
#: out, because the two disagreeing is exactly the bug this replaces: ``invented_detections``
#: filed ``class_ids=[0]`` under the label ``"ship"`` while the table maps 0 to ``person`` and
#: 8 to ``ship``. A mock whose id contradicts its own label is worse than an obviously fake
#: one — it is the first thing a tracker test copies, and a class-conditional chain step
#: (``when: class == ship``) reads the id.
_CLASS_IDS: dict[str, int] = {
    label: class_id for class_id, label in DecodeParams().class_labels.items()
}

#: What a label the default table has no id for is filed as. Deliberately not ``0``: ``0`` is
#: ``person``, so reusing it would put an invented class under a real one's number.
_UNKNOWN_CLASS_ID = -1


def invented_detections(label: str) -> Detections:
    """One 10x10 box of ``label``, in the shape a real detector produces.

    The same :class:`~shipinfer.topology.elements.detections.Detections` a ``pool`` detector
    decodes into ``meta["detections"]``, so a chain of mocks exercises the *type* the stateful
    elements behind it consume rather than a stand-in for it. That matters more here than
    anywhere else in this module: ``track`` reads the parallel arrays, and a mock that filed a
    list of tuples would let a tracker element pass its tests against a shape no deployment
    ever produces.

    The numbers are still invented and still say so — one box at the origin, score 0.9. A mock
    that produced plausible detections would eventually be trusted for something.

    The **class id agrees with the label**, which is the one thing here that is not free to be
    arbitrary: a real decode derives one from the other through
    :attr:`DecodeParams.class_labels`, so a mock filing them independently produces a
    ``Detections`` no decoder could ever emit.
    """
    return Detections(
        boxes=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([_CLASS_IDS.get(label, _UNKNOWN_CLASS_ID)], dtype=np.int32),
        labels=(label,),
    )


@registry_for(ElementKind.DETECT).register("mock")
class MockDetect(_Mock):
    """A detector on the device path, with a host fallback it never prefers."""

    kind: ClassVar[ElementKind] = ElementKind.DETECT
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "bgr@cpu")
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        label = str(self.params.get("class", "ship"))
        return {
            "boxes": [(0, 0, 10, 10)],
            "detections": invented_detections(label),
            "frame_hw": (100, 100),
            "class": label,
        }


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
        return {
            "boxes": [(0, 0, 10, 10)],
            "detections": invented_detections("ship"),
            "frame_hw": (100, 100),
            "class": "ship",
        }


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
    """A recogniser: turns vectors into identities.

    Files the shape the real one files —
    :class:`~shipinfer.topology.elements.recognize.GalleryRecognize` publishes
    ``{detection row index: (identity, similarity)}`` — for the same reason
    :func:`invented_detections` builds a real ``Detections``: a mock that filed a list would
    let everything downstream of a chain of mocks pass its tests against a shape no
    deployment ever produces, and this key in particular is one the runner's fan-in merges
    across branches: C8a made ``InprocessRunner._merge_meta`` union
    :class:`~shipinfer.topology.base.RowIndexed` metadata row by row, which is a merge a
    positional list cannot take part in at all.

    The single row is row 0, which is the one detection :func:`invented_detections` invents,
    and the similarity is a flat 1.0 — invented, and obviously so.
    """

    kind: ClassVar[ElementKind] = ElementKind.RECOGNIZE
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu",)
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        # `RowIndexed` for the reason `GalleryRecognize` files one: the fan-in unions it.
        return {"identities": RowIndexed({0: ("ship-1", 1.0)})}


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


@registry_for(ElementKind.TRACK).register("mock-passthrough")
class MockPassthrough(_Mock):
    """A cap-transparent element: ``*@*`` in, ``*@*`` out. The evasion that must not work.

    A real element shaped like this is ordinary — a tee, a filter, a tracker that annotates
    whatever it is handed — and declaring the wildcard on both sides is the honest way to say
    "I do not touch the pixels". It is also the hole a per-edge caps check leaves open: an
    ``nv12@gpu`` producer matches ``*@*``, and ``*@*`` matches a ``bgr@cpu`` consumer, so the
    device-to-host download arch.md §8 refuses would reappear in the middle of the chain with
    both edges reported valid.

    It does not work, and this class exists so a test can prove it: the loader resolves the
    wildcard from the element's negotiated *input* cap before negotiating its output, so this
    element is ``nv12@gpu`` on both sides behind a GPU decoder and is refused in front of a
    host-only sink.
    """

    kind: ClassVar[ElementKind] = ElementKind.TRACK
    accepts: ClassVar[tuple[str, ...]] = ("*@*",)
    produces: ClassVar[tuple[str, ...]] = ("*@*",)

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.processes += 1
        # No `caps=`: the cap it hands on is the cap it was given. That is what
        # cap-transparent means at runtime, and it is what the loader assumed at load time
        # when it resolved this element's `*@*` from its inbound edge.
        return item.derive(tracks=[1])


@registry_for(ElementKind.DECODE).register("mock-any")
class MockAnyDecode(_Mock):
    """A decoder that will not say what it produces. The root a chain cannot resolve.

    Exists for one test. A wildcard ``produces`` is resolved from what arrives at the
    element, and nothing arrives at a root -- so a root declaring ``*@*`` leaves every cap
    downstream of it unknown, and the loader refuses it rather than stamping ``*@*`` on the
    first edge and calling the chain validated.
    """

    kind: ClassVar[ElementKind] = ElementKind.DECODE
    produces: ClassVar[tuple[str, ...]] = ("*@*",)


@registry_for(ElementKind.MTMC).register("mock")
class MockMtmc(_Mock):
    """Cross-camera association: metadata in, metadata out."""

    kind: ClassVar[ElementKind] = ElementKind.MTMC
    accepts: ClassVar[tuple[str, ...]] = ("meta@cpu",)
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)

    def _meta(self, item: ChainItem) -> dict[str, Any]:
        return {"global_ids": ["g-1"]}


@registry_for(ElementKind.OUTPUT).register("mock-cpu")
class MockCpuOutput(_Mock):
    """A sink that can only read host memory - a Kafka producer, a JSON writer, a database.

    The realistic counterpart of :class:`MockPassthrough`: most real sinks *are* host-only,
    because they serialise. Exists so a test can put one behind a GPU element and check the
    chain is refused rather than quietly growing a download per frame.
    """

    kind: ClassVar[ElementKind] = ElementKind.OUTPUT
    accepts: ClassVar[tuple[str, ...]] = ("bgr@cpu", "meta@cpu")

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        self.emitted: list[ChainItem] = []

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.processes += 1
        self.emitted.append(item)
        return None


class MockLazyOutput(_Mock):
    """A sink deliberately in **no** registry: the lazy-registration target.

    Undecorated, and that is the whole point. ``register_lazy`` names a dotted path so that
    the module need not be imported at registration time; a class that registered itself
    eagerly could not stand in for one whose import is impossible without its runtime
    (``pyds``). A test registers this lazily and builds it, which is the only way to walk the
    path a real DeepStream element will take -- including the fact that ``impl`` is set by
    :func:`~shipinfer.topology.registry.create_element` rather than by a decorator.
    """

    kind: ClassVar[ElementKind] = ElementKind.OUTPUT
    accepts: ClassVar[tuple[str, ...]] = ("*@*",)

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.processes += 1
        return None


@registry_for(ElementKind.OUTPUT).register("mock")
class MockOutput(_Mock):
    """A sink that keeps what it was given, so a test can read the far end of a chain.

    Accepts ``*@*``: a sink is the one place a wildcard is honest, because whatever the
    chain ends up producing is what gets emitted.
    """

    kind: ClassVar[ElementKind] = ElementKind.OUTPUT
    accepts: ClassVar[tuple[str, ...]] = ("*@*",)

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        self.emitted: list[ChainItem] = []

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.processes += 1
        self.emitted.append(item)
        # None because the item is *consumed*, not because anything failed — the distinction
        # `Element.process` documents.
        return None
