"""The chain loader: what it accepts, and — mostly — what it refuses.

This file is where the value of a declarative chain is cashed in. Every test below is a
deployment that would otherwise have started, run, and produced either nothing or the wrong
thing:

* a chain whose CPU detector sits behind a GPU decoder (a silent 3 GB/s download);
* a chain with no output element (every model runs, nothing is emitted);
* a chain whose ``after:`` has a typo (a branch that never runs);
* a segmenter spelled ``sement`` (a missing stage nobody notices).

The whole file runs offline, with no GPU and no driver, on the implementations a deployment
actually names: :class:`~shipinfer.topology.base.Element` constructors are required to be
hardware-free, so a chain of ``replay``/``pool``/``shipvision``/``none`` *loads* anywhere and
only ``open()`` needs a repository behind it.

The tests are named ``tests/topology/test_chain.py`` rather than ``test_topology.py``
because ``tests/server/test_topology.py`` already exists and pytest's default import mode
gives two same-named modules in non-package directories the same module name.
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml

from shipinfer.core.errors import (
    CapsMismatchError,
    ChainCycleError,
    ChainSpecError,
    ChainStructureError,
    ConditionSyntaxError,
    ConfigurationError,
    ServerStateError,
    TopologyError,
    UnknownElementError,
    UnknownElementImplError,
    UnknownElementKindError,
)
from shipinfer.core.request import RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.engine import InferenceServer
from shipinfer.runtime.ops import NumpyImageOps
from shipinfer.topology import (
    Caps,
    ChainItem,
    ChainSpec,
    Condition,
    Element,
    ElementContext,
    ElementKind,
    Topology,
    create_element,
    describe_elements,
    load_topology,
    registry_for,
)
from shipinfer.topology.elements.detections import Detections
from shipinfer.topology.elements.output import JsonLinesOutput, NullOutput, SinkOutput
from shipinfer.topology.elements.pool import PoolRecognize, PoolSegment
from tests.support.models import materialise

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``topology/ship_person.yaml``'s nine slots and its ``after:`` wiring -- including the two
#: lines that spell out the rejoin (``embed_person: after: detect`` and
#: ``track: after: [recognize, embed_person]``) -- on the implementations a host with no
#: accelerator can name today. It is a **branching** chain, not a straight line: two branches
#: split at ``detect`` and rejoin at ``track``.
#:
#: The rows each *embedder* and the *segmenter* cover are ``params: classes:`` and not
#: ``when:``, because that is the only spelling the loader accepts on a row-selecting element
#: (C8b, and P6-SEGMENT-CROP for the segmenter). ``decode`` is the stage that selects no rows.
#:
#: Written out rather than generated: a chain file is the thing under test, and a generated
#: one would test the generator. The cost of writing it out is that it can drift from the
#: real file, so ``TestTheProductionChainFile`` loads that file's own wiring and asserts it
#: resolves to the same thing -- the fixture being wrong is then a test failure rather than a
#: fixture that quietly proves something the deployment does not do.
CHAIN = textwrap.dedent("""
    name: ship_person_offline
    elements:
      decode:       {impl: replay}
      detect:       {impl: pool, model: ship_detector}
      segment:      {impl: pool, model: ship_segmenter, params: {classes: [ship]}}
      embed_ship:   {impl: pool, model: ship_embedder, params: {classes: [ship]}, after: segment}
      embed_person: {impl: pool, model: person_embedder, params: {classes: [person]}, after: detect}
      recognize:    {impl: shipvision, params: {classes: [ship]}, after: embed_ship}
      track:        {impl: shipvision, per: camera, after: [recognize, embed_person]}
      mtmc:         {impl: shipvision, scope: global}
      output:       {impl: none}
    """)


@registry_for(ElementKind.DECODE).register("chain-test-gpu")
class DeviceDecode(Element):
    """A decoder that leaves the frame in VRAM. Phase D's ``gstreamer-gpu``, in advance.

    Every shipped decode implementation delivers ``bgr@cpu`` (``elements/decode.py``), so
    there is nothing in the registry that can put a device cap on the head of a chain — and
    half the caps rules this file pins are about a chain that must not come back to host
    memory. Declared here rather than shipped: an operator cannot write this in a chain file,
    which is exactly the difference between a fixture and an implementation.
    """

    kind: ClassVar[ElementKind] = ElementKind.DECODE
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        return item.derive(caps=self.output_caps[0])


@registry_for(ElementKind.DECODE).register("chain-test-any")
class UnresolvableDecode(DeviceDecode):
    """A root that will not say what it produces. The chain the loader cannot resolve.

    A wildcard ``produces`` is filled in from what arrives at the element, and nothing arrives
    at a root — so this leaves every cap downstream unknown.
    """

    produces: ClassVar[tuple[str, ...]] = ("*@*",)


def load(text: str) -> Topology:
    """Build a topology from inline YAML, the way a chain file would be loaded."""
    return Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(text)))


def item(camera: str = "cam-1", frame: int = 7) -> ChainItem:
    return ChainItem(RequestContext(camera_id=camera, frame_id=frame), Caps.parse("bgr@cpu"))


def walk(chain: Topology, start: ChainItem) -> ChainItem:
    """Drive one item through every element, honouring each ``when:``, and return the emitted one.

    Stands in for the runner that does not exist yet, and implements the one semantics
    :meth:`ElementNode.admits` fixes: **skip and continue**. An element that does not admit
    the item is passed over and the *same* item goes on to its successors -- it is not
    dropped, and the walk does not stop.

    Walking the topological order as a single line is a simplification a real runner will
    not make (it will fan out at ``detect`` and fan in at ``track``), but it is exactly right
    for the question these tests ask: which elements see a given item. Each element on a
    branch is visited once, in an order the loader has already proved is legal.

    Raises:
        AssertionError: the chain emitted nothing, which for these chains is a broken walk
            rather than a property worth reporting per test.
    """
    for node in chain:
        node.element.open(ElementContext())
    try:
        current = start
        for node in chain:
            if not node.admits(current):
                continue
            result = node.element.process(current)
            if result is None:
                break
            current = result
        emitted = chain.node("output").element.emitted
        assert len(emitted) == 1, f"the sink emitted {len(emitted)} items, not one"
        return emitted[0]
    finally:
        for node in chain:
            node.element.close()


class TestRegistries:
    """One registry per kind, and at least one implementation of every kind."""

    @pytest.mark.parametrize("kind", list(ElementKind))
    def test_every_kind_has_a_registry_with_something_in_it(self, kind: ElementKind) -> None:
        """A kind nobody implements is a slot a chain file can name and never run.

        Deliberately does NOT resolve the entries. ``Registry.get()`` imports the target of a
        lazy registration, and lazy registration exists precisely for implementations whose
        module is expensive or absent -- ``kafka`` needs ``confluent_kafka``, and the next one
        may need TensorRT or GStreamer. Walking every dotted path here would make a bare
        ``pytest`` import them, which is how an offline suite quietly starts needing a driver
        (ADR-001). The kind is enforced where it is actually decided:
        :meth:`test_registering_a_class_under_the_wrong_kind_is_refused` for eager
        registration and
        :meth:`test_a_lazy_registration_of_the_wrong_kind_is_refused_at_creation` for lazy.
        """
        registry = registry_for(kind)

        assert registry.element_kind is kind
        assert list(registry.names()), "an empty registry cannot answer any chain"

    def test_registering_a_class_under_the_wrong_kind_is_refused(self) -> None:
        """The check that stops ``@DETECTORS.register`` from running a segmenter."""
        with pytest.raises(ConfigurationError, match="cannot be registered"):

            @registry_for(ElementKind.DETECT).register("wrong-kind")
            class Confused(Element):
                kind = ElementKind.SEGMENT

                def _do_open(self, context: ElementContext) -> None: ...

                def _do_process(self, item: ChainItem) -> ChainItem | None: ...

        assert "wrong-kind" not in registry_for(ElementKind.DETECT), "left half-registered"

    def test_one_impl_name_may_live_in_several_kinds(self) -> None:
        """The evidence for per-kind registries.

        ``pool`` is the default for all four model kinds and ``shipvision`` is a recogniser, a
        tracker *and* an MTMC implementation (arch.md §1) — three different classes under one
        name. A flat registry refuses a duplicate name, so it would force the kind into the
        name by convention.
        """
        pooled = {ElementKind.DETECT, ElementKind.SEGMENT, ElementKind.EMBED}
        shipvision = {ElementKind.RECOGNIZE, ElementKind.TRACK, ElementKind.MTMC}

        assert {registry_for(kind).get("pool").kind for kind in pooled} == pooled
        assert len({registry_for(kind).get("shipvision") for kind in shipvision}) == 3
        assert {registry_for(kind).get("shipvision").kind for kind in shipvision} == shipvision

    def test_an_unknown_impl_names_the_ones_that_exist(self) -> None:
        with pytest.raises(UnknownElementImplError) as caught:
            create_element(ElementKind.DETECT, "pol", "detect")

        assert "pol" in str(caught.value)
        assert "pool" in str(caught.value), "the message must list the available impls"
        assert caught.value.available == tuple(registry_for(ElementKind.DETECT).names())

    def test_an_unknown_kind_is_refused_by_name(self) -> None:
        with pytest.raises(UnknownElementKindError):
            registry_for("sement")

    def test_describe_elements_reports_every_kind(self) -> None:
        described = describe_elements()

        assert set(described) == {kind.value for kind in ElementKind}
        assert all(name and text for name, text in described["detect"]), "no description"

    def test_the_registered_name_lands_on_the_class(self) -> None:
        """``impl`` is set by the decorator, so the chain file and a log line agree.

        The canonical name and not the alias a chain happened to spell: ``jsonl`` and ``file``
        both resolve to ``jsonlines``, and a log line that echoed the alias would make two
        deployments of one element look like two elements.
        """
        registry = registry_for(ElementKind.OUTPUT)

        assert registry.get("jsonlines").impl == "jsonlines"
        assert registry.get("none").impl == "none"
        assert registry.get("jsonl").impl == "jsonlines"

    def test_a_lazy_registration_of_the_wrong_kind_is_refused_at_creation(self) -> None:
        """The hole a lazy entry opens, and where it is closed.

        ``register_lazy`` has no class to check when it is called — that is what makes it
        lazy — so the eager decorator's kind check cannot run. Without a second check, this
        registration would hand the chain loader a *tracker* as the chain's output sink, and
        the structure rules, which read ``node.kind``, would agree the chain ends properly:
        every model would run, the tracker would return metadata nobody emits, and no error
        would be raised anywhere. The check therefore lives in ``create_element``, the one
        place every element is built.
        """
        registry = registry_for(ElementKind.OUTPUT)
        # Registries are process-global, so a name added here is nameable by every test that
        # runs afterwards. Removed in `finally` rather than left: leaked state that only
        # works because pytest happens to run a class in declaration order is a test that
        # passes for a reason nobody chose.
        registry.register_lazy(
            "lazy-tracker",
            "shipinfer.topology.elements.track:ShipvisionTrack",
            description="a tracker smuggled into the output registry",
        )
        try:
            assert "lazy-tracker" in registry, "the registration itself cannot check"

            with pytest.raises(ConfigurationError, match="cannot be registered as a output"):
                create_element(ElementKind.OUTPUT, "lazy-tracker", "output")
        finally:
            registry._entries.pop("lazy-tracker", None)

    def test_the_shipped_lazy_registration_resolves_and_gets_its_name(self) -> None:
        """The other half, on the one implementation that really is registered lazily.

        ``kafka`` is a dotted path in ``elements/output.py`` because its sink imports
        ``confluent_kafka``, so nothing about it may be imported by ``shipinfer.topology``.
        Building it is what walks that path — and ``impl`` is set by ``create_element``,
        because no decorator ever ran on the class.
        """
        element = create_element(ElementKind.OUTPUT, "kafka", "output")

        assert element.kind is ElementKind.OUTPUT
        assert element.impl == "kafka", "no decorator ran, so the factory sets it"

    def test_one_class_under_two_lazy_names_is_refused_at_creation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``impl`` is a class attribute, so a second name renames the first's instances.

        The eager decorator writes ``impl`` once per class, but a lazy entry has nothing to
        write on until ``create_element`` builds it -- and two lazy names pointing at one
        class would have that factory rewrite the attribute under every element already
        built, some time after start-up. An element would then report an implementation the
        chain never asked for, in its logs and its metrics. Refused at the second creation
        instead, where the message can name both registrations.
        """
        module = types.ModuleType("shipinfer_test_two_lazy_names")

        class DoubleRegistered(Element):
            kind = ElementKind.OUTPUT
            accepts = ("*@*",)

            def _do_open(self, context: ElementContext) -> None: ...

            def _do_process(self, item: ChainItem) -> ChainItem | None: ...

        module.Sink = DoubleRegistered  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module.__name__, module)
        registry = registry_for(ElementKind.OUTPUT)
        for name in ("lazy-twice-a", "lazy-twice-b"):
            registry.register_lazy(
                name, f"{module.__name__}:Sink", description="one class, two names"
            )

        first = create_element(ElementKind.OUTPUT, "lazy-twice-a", "output")

        with pytest.raises(ConfigurationError, match="two implementation names"):
            create_element(ElementKind.OUTPUT, "lazy-twice-b", "output")
        assert first.impl == "lazy-twice-a", "the built element keeps its own name"

    def test_a_subclass_may_carry_its_own_lazy_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The false positive the rule above must not have: ``impl`` is inherited.

        A subclass of a registered element is ordinary -- ``nvinfer`` deriving from the
        ``pool`` detector -- and it inherits the base's ``impl``. Refusing on the *inherited*
        value would refuse every such subclass, so the check reads the class's own
        attribute.
        """
        module = types.ModuleType("shipinfer_test_subclass_lazy_name")

        class Derived(JsonLinesOutput):
            """A sink that is a subclass of an eagerly registered one."""

        module.Sink = Derived  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module.__name__, module)
        registry = registry_for(ElementKind.OUTPUT)
        registry.register_lazy(
            "lazy-derived", f"{module.__name__}:Sink", description="a subclass"
        )

        element = create_element(ElementKind.OUTPUT, "lazy-derived", "output")

        assert element.impl == "lazy-derived"
        assert JsonLinesOutput.impl == "jsonlines", "the base's own name must not be rewritten"


class TestLoadingAValidChain:
    def test_the_chain_loads_and_is_ordered(self) -> None:
        chain = load(CHAIN)

        assert chain.name == "ship_person_offline"
        assert len(chain) == 9
        order = [node.name for node in chain]
        for edge in chain.edges:
            assert order.index(edge.producer) < order.index(
                edge.consumer
            ), f"out of order: {edge}"

    def test_the_frame_is_carried_until_the_tracker_and_dropped_there(self) -> None:
        """The cap of every edge, which is the load-time evidence for the plane change.

        Everything up to ``track`` carries the frame; from ``track`` on it is ``meta@cpu``,
        because a MOT algorithm needs boxes and not pixels. A chain that dropped the pixels
        earlier — or carried them past the tracker to a sink that serialises — shows up right
        here. ``bgr@cpu`` because ``replay`` is a host-memory source; the same nine slots
        behind phase D's ``gstreamer-gpu`` read ``nv12@gpu`` for the first five edges, which
        is the wildcard on the ``pool`` elements being resolved rather than declared.
        """
        chain = load(CHAIN)
        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in chain.edges}

        assert caps[("decode", "detect")] == "bgr@cpu"
        assert caps[("detect", "segment")] == "bgr@cpu"
        assert caps[("recognize", "track")] == "bgr@cpu"
        assert caps[("track", "mtmc")] == "meta@cpu"
        assert caps[("mtmc", "output")] == "meta@cpu"

    def test_roots_and_sinks_are_the_decoder_and_the_output(self) -> None:
        chain = load(CHAIN)

        assert [node.name for node in chain.roots] == ["decode"]
        assert [node.name for node in chain.sinks] == ["output"]

    def test_the_default_predecessor_is_the_previously_declared_element(self) -> None:
        """The rule that lets a nine-step chain need no wiring at all."""
        chain = load("""
            elements:
              decode: {impl: replay}
              detect: {impl: pool, model: d}
              track:  {impl: shipvision}
              output: {impl: none}
            """)

        assert [node.name for node in chain.predecessors("detect")] == ["decode"]
        assert [node.name for node in chain.predecessors("track")] == ["detect"]
        assert [node.name for node in chain.predecessors("output")] == ["track"]

    def test_after_overrides_the_default_and_a_diamond_rejoins(self) -> None:
        """A branch and a fan-in, which is the whole reason ``after:`` exists.

        ``embed_person`` must follow ``detect``, not the ``segment`` declared before it, and
        both branches rejoin at ``track``.
        """
        chain = load(CHAIN)

        assert [node.name for node in chain.predecessors("embed_person")] == ["detect"]
        assert [node.name for node in chain.predecessors("embed_ship")] == ["segment"]
        assert sorted(node.name for node in chain.predecessors("track")) == [
            "embed_person",
            "recognize",
        ]
        assert sorted(node.name for node in chain.successors("detect")) == [
            "embed_person",
            "segment",
        ]

    def test_a_kind_is_inferred_from_the_slot_name(self) -> None:
        """``embed_ship`` is an embedder, which is what lets a chain hold two of them."""
        chain = load(CHAIN)

        assert chain.node("embed_ship").kind is ElementKind.EMBED
        assert chain.node("embed_person").kind is ElementKind.EMBED
        assert chain.node("decode").kind is ElementKind.DECODE

    def test_an_explicit_kind_overrides_the_slot_name(self) -> None:
        chain = load("""
            elements:
              camera: {impl: replay, kind: decode}
              emit:   {impl: none, kind: output}
            """)

        assert chain.node("camera").kind is ElementKind.DECODE

    def test_describe_prints_the_resolved_wiring(self) -> None:
        """The default-predecessor rule means the file does not state most of the edges."""
        described = load(CHAIN).describe()

        assert "decode -> detect [bgr@cpu]" in described
        assert "model=ship_detector" in described

    def test_describe_prints_a_frame_guard_where_one_is_written(self) -> None:
        """The production chain carries none, so this asks a chain that does."""
        described = load(TestConditions.GUARDED).describe()

        assert "when=class == ship" in described

    def test_describe_prints_which_rows_a_slot_selects(self) -> None:
        """`when=` without `classes=` answers half the question.

        Since C8b a slot picks objects with `params: {classes: [...]}` and frames with
        `when:`; a description that showed only the second let an operator read a chain and
        see no sign of the row filter that decides what the GPU is spent on.
        """
        described = load(
            "name: rows\nelements:\n"
            "  decode: {impl: replay}\n"
            "  detect: {impl: pool, model: d}\n"
            "  embed:  {impl: pool, model: e, params: {classes: [ship]}}\n"
            "  output: {impl: none}\n"
        ).describe()

        assert "classes=['ship']" in described
        assert "embed: embed/pool" in described

    def test_the_element_is_told_which_model_it_runs(self) -> None:
        """``model:`` reaches the element, not only the node's spec beside it.

        A ``pool`` element resolves that name against ``ElementContext.models`` at ``open``;
        it holds no reference to the node, so a name it is never handed is a name it cannot
        resolve. ``None`` for the four kinds that have no model, which is a kind's property
        and not a missing value -- the loader has already refused a model kind that names
        none.
        """
        chain = load(CHAIN)

        assert chain.node("detect").element.model == "ship_detector"
        assert chain.node("embed_ship").element.model == "ship_embedder"
        assert chain.node("embed_person").element.model == "person_embedder"
        assert chain.node("decode").element.model is None
        assert chain.node("track").element.model is None
        assert chain.node("output").element.model is None
        # `recognize` is `impl: shipvision`: a gallery element loads its own gallery and
        # names no repository model, which is the same "a kind's property, not a missing
        # value" the four above demonstrate.
        assert chain.node("recognize").element.model is None

    def test_the_runner_can_still_read_what_the_file_said(self) -> None:
        """``per``/``scope``/``params`` are the runner's business, so the spec is kept."""
        chain = load(CHAIN)

        assert chain.node("track").spec.per == "camera"
        assert chain.node("mtmc").spec.scope == "global"

    def test_a_wildcard_element_between_two_device_elements_loads(self) -> None:
        """Propagation resolves, it does not forbid. The other half of the §8 rule.

        The same ``*@*`` element that is refused in front of a host-only sink is fine between
        two device elements, and both of its edges come out ``nv12@gpu`` — not ``*@*``, which
        would be a validated chain nobody could read the caps of. ``pool`` is the shipped
        wildcard: it hands on whatever plane it was given.
        """
        chain = load("""
            elements:
              decode:  {impl: chain-test-gpu}
              segment: {impl: pool, model: s}
              detect:  {impl: pool, model: d, after: segment}
              track:   {impl: shipvision}
              output:  {impl: none}
            """)
        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in chain.edges}

        assert caps[("decode", "segment")] == "nv12@gpu"
        assert caps[("segment", "detect")] == "nv12@gpu"

    def test_a_cap_transparent_conditional_element_loads(self) -> None:
        """The bypass rule refuses a plane change, not every ``when:``.

        ``segment`` here consumes and produces ``nv12@gpu``, so the frame that skips it looks
        exactly like the frame that went through it and ``embed_ship`` cannot tell — which is
        the shape a branch condition is supposed to have.

        The guard is ``fps``, a fact ``runners/frames.py`` files, and not ``class``: this slot
        selects detection rows (P6-SEGMENT-CROP), so ``when: class == …`` on it is the
        refusal :class:`TestRowSelectionIsNotAFrameCondition` is about.
        """
        chain = load("""
            elements:
              decode:     {impl: chain-test-gpu}
              detect:     {impl: pool, model: d}
              segment:    {impl: pool, model: s, params: {classes: [ship]}, when: fps == 20}
              embed_ship: {impl: pool, model: e, params: {classes: [ship]}}
              track:      {impl: shipvision}
              output:     {impl: none}
            """)

        assert str(chain.node("segment").condition) == "fps == 20"
        assert all(str(edge.caps) == "nv12@gpu" for edge in chain.edges[:3])

    def test_loading_twice_gives_two_independent_chains(self) -> None:
        """Elements are stateful (a tracker holds per-camera state), so nothing is cached."""
        first, second = load(CHAIN), load(CHAIN)

        assert first.node("track").element is not second.node("track").element


class TestRefusals:
    def test_a_serialising_sink_behind_a_device_decoder_is_refused(self) -> None:
        """arch.md §8's rule: a chain that would silently download refuses to load.

        A sink serialises, so it reads host memory and says so — ``accepts: meta@cpu,
        bgr@cpu`` on every ``output`` implementation. Behind a decoder that leaves the frame
        in VRAM there is nothing to bridge the two, and the alternative to refusing is a
        device-to-host copy per frame that nothing in the chain file mentions.

        The message must name both sides and both cap lists — this is the error most likely
        to be read by someone who did not write the chain.
        """
        with pytest.raises(CapsMismatchError) as caught:
            load("""
                elements:
                  decode: {impl: chain-test-gpu}
                  output: {impl: none}
                """)

        message = str(caught.value)
        assert "'decode'" in message and "'output'" in message
        assert "nv12@gpu" in message and "bgr@cpu" in message
        assert "convert" in message, "the message should name the fix, not just the problem"
        assert caught.value.produced == ("nv12@gpu",)
        assert caught.value.accepted == ("meta@cpu", "bgr@cpu")

    def test_a_wildcard_element_cannot_launder_a_device_frame_to_a_host_sink(self) -> None:
        """The evasion a per-edge caps check leaves open, and the reason caps propagate.

        A ``pool`` element declares ``produces: *@*``, which is the honest way to say "I read
        the frame and hand it on untouched". Negotiated one edge at a time, both of its edges
        pass: ``nv12@gpu`` matches ``*@*``, and ``*@*`` matches ``bgr@cpu``. The chain would
        load, and the device-to-host download arch.md §8 exists to refuse would happen every
        frame in the middle of it.

        It is refused because the loader resolves the wildcard from what arrives at the
        element *before* negotiating its output, so by the time the sink is considered the
        segmenter is an ``nv12@gpu`` producer and nothing bridges the two memories.
        """
        with pytest.raises(CapsMismatchError) as caught:
            load("""
                elements:
                  decode:  {impl: chain-test-gpu}
                  segment: {impl: pool, model: s}
                  output:  {impl: none}
                """)

        assert caught.value.producer == "segment"
        assert caught.value.produced == ("nv12@gpu",), "the resolved cap, not the '*@*'"
        assert caught.value.consumer == "output"

    def test_a_root_must_say_what_it_produces(self) -> None:
        """A wildcard has nothing to resolve against when nothing precedes the element.

        The alternative is a chain whose first edge is stamped ``*@*`` and every edge after
        it unknowable — validated in name only.
        """
        with pytest.raises(ChainStructureError, match="must say what it produces"):
            load("""
                elements:
                  decode: {impl: chain-test-any}
                  output: {impl: none}
                """)

    def test_a_conditional_root_is_refused(self) -> None:
        """A ``when:`` on a root can never be true, so the chain would ingest nothing.

        Everything a condition can read is metadata an element wrote, and at ingest the
        metadata is empty; ``Condition.matches`` is false for a missing field, because
        absence is not evidence. So the decoder would be skipped for every frame of every
        camera while every process looked healthy -- exactly the silent failure the loader
        exists to refuse at start-up.
        """
        with pytest.raises(ChainStructureError, match="carries `when: class == ship`"):
            load("""
                elements:
                  decode: {impl: replay, when: class == ship}
                  detect: {impl: pool, model: d}
                  output: {impl: none}
                """)

    def test_a_wildcard_producer_fed_two_different_caps_is_refused(self) -> None:
        """A fan-in into a ``*@*`` element: the loader would have to pick one cap and guess.

        Fan-in edges carrying different caps is legal in general (each edge is negotiated on
        its own pair). It stops being legal when the element says "I hand on what I was
        given", because then there is no single answer to put on its outbound edge — and here the
        two roots hand it the same frame in two different memories.
        """
        with pytest.raises(ChainStructureError, match="wildcard format"):
            load("""
                elements:
                  decode_gpu: {impl: chain-test-gpu}
                  decode_cpu: {impl: replay, after: []}
                  segment:    {impl: pool, model: s, after: [decode_gpu, decode_cpu]}
                  track:      {impl: shipvision}
                  output:     {impl: none}
                """)

    def test_a_conditional_element_that_changes_the_plane_is_refused(self) -> None:
        """The bypass edge: what a ``when:`` element's absence exposes.

        A condition means skip-and-continue, so an item the tracker does not admit travels
        from ``decode`` straight to ``mtmc``. The declared edges are both fine —
        ``decode -> track`` is ``bgr@cpu``, ``track -> mtmc`` is ``meta@cpu`` — and the chain
        still cannot run, because on every frame the guard rejects, ``mtmc`` is handed pixels
        it cannot read. Checking only the declared edges would ship this.

        The guard is a frame-level fact and not ``class ==``: a tracker selects rows, so the
        loader refuses that spelling first (:class:`TestRowSelectionIsNotAFrameCondition`) and
        this test would never reach the bypass rule it is about.
        """
        with pytest.raises(CapsMismatchError) as caught:
            load("""
                elements:
                  decode: {impl: replay}
                  track:  {impl: shipvision, when: has_ship == true}
                  mtmc:   {impl: shipvision}
                  output: {impl: none}
                """)

        assert caught.value.skipped == "track"
        assert caught.value.producer == "decode"
        assert caught.value.consumer == "mtmc"
        assert "is skipped" in str(caught.value)

    def test_a_dangling_after_names_the_referrer_and_the_typo(self) -> None:
        with pytest.raises(UnknownElementError) as caught:
            load("""
                elements:
                  decode: {impl: replay}
                  detect: {impl: pool, model: d, after: decoed}
                  output: {impl: none}
                """)

        assert caught.value.referrer == "detect"
        assert caught.value.missing == "decoed"
        assert "decode" in str(caught.value)

    def test_a_cycle_is_reported_as_a_path_and_nothing_else(self) -> None:
        """The cycle itself, in flow order — not every element the sort could not place.

        ``mtmc`` and ``output`` hang off this cycle and can never be ordered either, so the
        set of elements with in-degree remaining is four names of which two are innocent.
        Reporting that set makes the reader find the loop; reporting the loop does not.
        """
        with pytest.raises(ChainCycleError) as caught:
            load("""
                elements:
                  decode: {impl: replay}
                  detect: {impl: pool, model: d, after: track}
                  track:  {impl: shipvision, after: detect}
                  mtmc:   {impl: shipvision}
                  output: {impl: none}
                """)

        assert caught.value.cycle == ("detect", "track")
        assert "detect -> track -> detect" in str(caught.value)

    def test_an_element_that_depends_on_itself_is_refused(self) -> None:
        with pytest.raises(ChainCycleError):
            load("""
                elements:
                  decode: {impl: replay}
                  detect: {impl: pool, model: d, after: detect}
                  output: {impl: none}
                """)

    @pytest.mark.parametrize(
        ("chain", "message", "why"),
        [
            (
                """
                elements:
                  decode: {impl: replay}
                """,
                "the chain has no output element",
                "a decoder alone emits nothing",
            ),
            (
                """
                elements:
                  decode: {impl: replay}
                  detect: {impl: pool, model: d}
                """,
                "the chain has no output element",
                "every model runs and nothing is emitted",
            ),
            (
                """
                elements:
                  decode: {impl: replay}
                  detect: {impl: pool, model: d}
                  track:  {impl: shipvision}
                """,
                "the chain has no output element",
                "the chain ends at a tracker, whose results go nowhere",
            ),
            (
                """
                elements:
                  detect: {impl: pool, model: d, after: []}
                  output: {impl: none}
                """,
                "must be a decode element",
                "a chain must start at a decoder",
            ),
            (
                """
                elements:
                  output: {impl: none, after: []}
                """,
                "must be a decode element",
                "an output sink with no decoder in front of it is not a chain",
            ),
            (
                """
                elements:
                  decode: {impl: replay}
                  output: {impl: none}
                  mtmc:   {impl: shipvision, after: output}
                """,
                "must be a sink",
                "an output element that hands something on is not the end",
            ),
        ],
    )
    def test_a_chain_needs_a_decode_root_and_an_output_sink(
        self, chain: str, message: str, why: str
    ) -> None:
        """Each case is matched on its *message*, which is what makes it discriminating.

        Every structure rule here raises the same ``ChainStructureError``, and several of the
        chains break more than one rule at once: the reverse-reachability check alone refuses
        a chain with no output element, so ``pytest.raises(ChainStructureError)`` on its own
        would stay green with the output rule deleted. The last case is the one that used to
        be answered with "the chain has no output sink" — true but useless, because there is
        an output element and the problem is what follows it.
        """
        with pytest.raises(ChainStructureError, match=message):
            load(chain)

    def test_a_branch_that_reaches_no_output_is_refused(self) -> None:
        """The orphan case: ``segment`` computes masks that nothing emits.

        Forward reachability would say nothing here — in a DAG every element is reachable
        from some root — so the check runs backwards from the output elements.
        """
        with pytest.raises(ChainStructureError, match="reach no output"):
            load("""
                elements:
                  decode:  {impl: replay}
                  detect:  {impl: pool, model: d}
                  output:  {impl: none}
                  segment: {impl: pool, model: s, after: detect}
                """)

    def test_an_element_that_resolves_a_repository_model_without_one_is_refused(self) -> None:
        """The requirement is the *implementation's*, so the chain must name a ``pool``.

        It used to be the kind's -- any ``recognize:`` needed a ``model:`` -- and
        ``impl: shipvision`` was refused here too. It is not any more, and deliberately: a
        gallery recogniser loads its own index and resolves nothing against the repository.
        ``tests/topology/test_model_requirement.py`` is where both halves of the new rule
        live; this is the refusal itself, in the file that owns the loader\'s.
        """
        with pytest.raises(ChainStructureError, match="model"):
            load("""
                elements:
                  decode: {impl: replay}
                  detect: {impl: pool}
                  output: {impl: none}
                """)

    def test_an_unknown_key_is_refused(self) -> None:
        """``extra="forbid"`` is the reason the schema is worth having."""
        with pytest.raises(ChainSpecError) as caught:
            load("""
                elements:
                  decode: {impl: replay, mdoel: ship_detector}
                  output: {impl: none}
                """)

        assert "mdoel" in str(caught.value)

    def test_a_misspelled_slot_name_is_refused(self) -> None:
        with pytest.raises(UnknownElementKindError) as caught:
            load("""
                elements:
                  decode: {impl: replay}
                  sement: {impl: pool, model: s}
                  output: {impl: none}
                """)

        assert caught.value.slot == "sement"
        assert "segment" in str(caught.value)

    def test_a_misspelled_explicit_kind_gets_the_same_error_as_a_misspelled_slot(
        self,
    ) -> None:
        """One mistake, one error type.

        A misspelled kind is the same mistake whether it is written as the slot name or as
        ``kind:``. Declaring ``kind`` as the enum in the schema would answer the second
        spelling with a pydantic ``ChainSpecError`` — a different type, a different message
        and a different place to look, for a typo the reader made once.
        """
        with pytest.raises(UnknownElementKindError) as caught:
            load("""
                elements:
                  camera: {impl: replay, kind: decdoe}
                  output: {impl: none}
                """)

        assert caught.value.slot == "decdoe"
        assert "decode" in str(caught.value)

    def test_a_non_string_top_level_key_is_refused_as_a_spec_error(self) -> None:
        """The schema is the only thing allowed to reject a document, however odd it is."""
        with pytest.raises(ChainSpecError):
            ChainSpec.from_yaml("1: elements\nelements: {decode: {impl: replay}}")

    def test_a_chain_with_no_elements_is_refused(self) -> None:
        with pytest.raises(ChainSpecError):
            load("elements: {}")

    def test_a_document_that_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(ChainSpecError, match="mapping"):
            ChainSpec.from_yaml("- decode\n- detect")

    def test_invalid_yaml_names_the_source(self) -> None:
        with pytest.raises(ChainSpecError, match=r"chain\.yaml"):
            ChainSpec.from_yaml("elements: {decode: {impl: replay}\n", source="chain.yaml")

    def test_a_missing_file_is_refused_with_its_path(self, tmp_path: Path) -> None:
        with pytest.raises(ChainSpecError, match=r"nope\.yaml"):
            load_topology(tmp_path / "nope.yaml")


class TestThePackageSurface:
    def test_a_lookup_miss_on_a_validated_chain_is_a_programming_error(self) -> None:
        """``KeyError``, not a typed configuration error.

        By the time a ``Topology`` exists, every name in the file has been resolved — so a
        miss here is a mistyped literal in a runner or a CLI, not something an operator can
        fix in YAML. Raising ``UnknownElementError`` would send them to edit a chain that is
        correct, and would let a caller's bug be caught by the same ``except TopologyError``
        that handles bad configuration.
        """
        chain = load(CHAIN)

        with pytest.raises(KeyError) as caught:
            chain.node("detetc")

        assert "detetc" in str(caught.value)
        assert "detect" in str(caught.value), "the message lists what there is"
        assert not isinstance(caught.value, TopologyError)

    def test_the_package_exports_classes_and_not_modules(self) -> None:
        """``__all__`` is the public vocabulary, and a module is not part of it.

        ``elements.pool`` and its siblings are imported by ``__init__`` for their
        registration side effect, which is necessary and is not the same as being an export:
        an implementation is reached through its registry, by the name a chain file uses.
        Exporting the module invites ``from shipinfer.topology import pool`` and a second way
        to get at an element that skips the seam — ``backends`` and ``ingest`` export the
        classes they mean and nothing else.
        """
        import shipinfer.topology as package

        exported = {name: getattr(package, name) for name in package.__all__}

        assert not [
            name for name, value in exported.items() if isinstance(value, types.ModuleType)
        ]
        assert "pool" in registry_for(ElementKind.DETECT), "the side effect still happened"


class TestConditions:
    #: A frame guard on a field the *test* files. `detect` is the whole-frame stage -- it
    #: selects no rows -- so the loader permits `class` there; the production chain carries no
    #: guard at all, because nothing in a real chain writes a frame-level `meta["class"]`
    #: (`docs/arch.md` §1).
    GUARDED = (
        "name: guarded\nelements:\n"
        "  decode:  {impl: replay}\n"
        "  detect:  {impl: pool, model: d, when: class == ship}\n"
        "  output:  {impl: none}\n"
    )

    def test_a_when_expression_is_parsed(self) -> None:
        chain = load(self.GUARDED)
        condition = chain.node("detect").condition

        assert condition == Condition("class", "==", "ship")
        assert str(condition) == "class == ship"

    @pytest.mark.parametrize("text", ["class = ship", "class", "== ship", "class ~ ship"])
    def test_a_malformed_when_is_refused(self, text: str) -> None:
        """``class = ship`` read as "always true" would run the heaviest model on every frame."""
        with pytest.raises(ConditionSyntaxError):
            Condition.parse(text)

    def test_a_condition_gates_an_element_on_metadata(self) -> None:
        chain = load(self.GUARDED)
        detect = chain.node("detect")

        assert detect.admits(item().derive(**{"class": "ship"}))
        assert not detect.admits(item().derive(**{"class": "person"}))

    def test_a_missing_field_satisfies_neither_operator(self) -> None:
        """Absence is not evidence: ``!=`` must not fire on a frame nobody classified."""
        assert not Condition.parse("class == ship").matches({})
        assert not Condition.parse("class != ship").matches({})

    def test_an_element_with_no_condition_sees_everything(self) -> None:
        assert load(CHAIN).node("detect").admits(item())


class CountedSink(NullOutput):
    """The real ``none`` sink, with its two lifecycle hooks counted.

    :class:`Element` exposes no counters and should not: nothing in production reads them.
    Idempotence is therefore asserted through a subclass that counts, over the real
    ``_do_open`` that builds a real sink and the real ``_do_close`` that closes it — so a
    hook that stopped being called fails here rather than being hidden by a stub.
    """

    def __init__(
        self,
        name: str,
        params: object = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)  # type: ignore[arg-type]
        self.opens = 0
        self.closes = 0

    def _do_open(self, context: ElementContext) -> None:
        super()._do_open(context)
        self.opens += 1

    def _do_close(self) -> None:
        super()._do_close()
        self.closes += 1


class TestElementLifecycle:
    def test_open_and_close_are_idempotent(self) -> None:
        element = CountedSink("output")

        element.open()
        element.open()
        assert element.is_open and element.opens == 1

        element.close()
        element.close()
        assert not element.is_open and element.closes == 1

    def test_closing_an_element_that_never_opened_does_nothing(self) -> None:
        element = CountedSink("output")

        element.close()

        assert element.closes == 0

    def test_processing_before_open_is_a_typed_refusal(self) -> None:
        """Not an implicit open: that is how a CUDA context lands on the wrong thread."""
        element = create_element(ElementKind.OUTPUT, "none", "output")

        with pytest.raises(ServerStateError, match="before open"):
            element.process(item())

    def test_a_failed_open_closes_what_it_managed_to_acquire(self) -> None:
        class Broken(CountedSink):
            def _do_open(self, context: ElementContext) -> None:
                super()._do_open(context)
                raise RuntimeError("no broker")

        element = Broken("output")

        with pytest.raises(RuntimeError, match="no broker"):
            element.open()

        assert not element.is_open
        assert element.closes == 1, "a half-open element must be unwound"

    def test_the_context_arrives_at_open_and_is_dropped_at_close(self) -> None:
        element = create_element(ElementKind.OUTPUT, "none", "output")
        context = ElementContext(shard_id=3)

        element.open(context)
        assert element.context is context

        element.close()
        assert element.context is None

    def test_params_reach_the_element(self) -> None:
        chain = load("""
            elements:
              decode: {impl: replay}
              output: {impl: none, params: {keep_last: 4}}
            """)

        assert chain.node("output").element.params == {"keep_last": 4}


#: One model, declared once, built to its own config by ``tests/support/models.py``. Small
#: on purpose: the walk is what is under test, not the arithmetic inside the module.
#: A YOLO-seg engine's two outputs at toy extents: ``(R, 6 + M)`` detection rows and an
#: ``(M, h, w)`` prototype bank, because ``PoolSegment`` folds the two into one area per crop.
_SEGMENTER = """
platform: pytorch
max_batch_size: 4
inputs: [{name: images, data_type: FP32, dims: [3, 16, 8]}]
outputs:
  - {name: output0, data_type: FP32, dims: [2, 8]}
  - {name: output1, data_type: FP32, dims: [2, 4, 2]}
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
"""

#: ``decode -> segment -> output`` on the three implementations that need nothing but a model
#: repository: a file source, a pool element, and a sink that counts. Every edge is
#: ``bgr@cpu``, which is what a host-memory chain looks like end to end.
WALKABLE = """
name: walkable
elements:
  decode:  {impl: replay}
  segment: {impl: pool, model: ship_segmenter, params: {input: images, classes: [ship]}}
  output:  {impl: none, params: {keep_last: 4}}
"""


class TestWalkingAChain:
    """The three elements driven for real, over a real model pool.

    The pool is an :class:`~shipinfer.engine.pool.InferenceServer` on the CPU with one
    TorchScript module in it, rather than a stand-in for one: a walk whose middle element
    never submitted anything would prove that the loader wired three objects together and
    nothing about the contract between them.
    """

    @pytest.fixture()
    def server(self, tmp_path: Path):
        root = tmp_path / "repo"
        (root / "ship_segmenter" / "1").mkdir(parents=True)
        (root / "ship_segmenter" / "config.yaml").write_text(_SEGMENTER.lstrip())
        materialise(root)
        settings = ServerSettings(
            model_repository=root,
            devices={"visible_gpus": []},
            execution={"warmup_iterations": 0},
        )
        with InferenceServer(settings) as running:
            yield running

    @staticmethod
    def frame(camera: str = "cam-42", frame_id: int = 1234, **meta: Any) -> ChainItem:
        """One host-resident frame with one ship on it, as ``PoolDetect`` hands it on.

        Real pixels and a real detection because the middle element crops: a segmenter is a
        row-selecting stage since P6-SEGMENT-CROP, so a frame with nothing on it would walk
        through submitting nothing and prove neither of this class's two properties.
        """
        filed: dict[str, Any] = {
            "detections": Detections(
                boxes=np.array([[2, 4, 30, 40]], dtype=np.float32),
                scores=np.array([0.9], dtype=np.float32),
                class_ids=np.array([8], dtype=np.int32),
                labels=("ship",),
            )
        }
        filed.update(meta)
        return ChainItem(
            RequestContext(camera_id=camera, frame_id=frame_id),
            Caps.parse("bgr@cpu"),
            payload=Tensor.from_numpy(
                np.arange(1 * 48 * 64 * 3, dtype=np.uint8).reshape(1, 48, 64, 3)
            ),
            meta=filed,
        )

    @staticmethod
    def context(server: Any) -> ElementContext:
        """The context a runner builds: a pool, and the ops the crop element refuses without."""
        return ElementContext(models=server, ops=NumpyImageOps())

    def test_the_tag_survives_every_element(self, server) -> None:
        """ADR-002, checked end to end: the ``(camera_id, frame_id)`` tag is never rewritten.

        Walked by hand rather than through a runner, because the property belongs to the
        elements: a runner that reassembled on arrival order would still pass its own tests
        with an element that rebuilt the tag here.
        """
        chain = load(WALKABLE)
        for node in chain:
            node.element.open(self.context(server))
        try:
            current = self.frame()
            start_key = current.key
            seen: list[str] = []
            for node in chain:
                result = node.element.process(current)
                seen.append(node.name)
                if result is None:
                    break
                current = result
                assert current.key == start_key, f"{node.name} rewrote the tag"

            published = chain.node("output").element.sink
            assert seen == ["decode", "segment", "output"]
            assert published.emitted == 1
            event = published.events()[0]
            assert (event.camera_id, event.frame_id) == start_key
        finally:
            for node in chain:
                node.element.close()

    def test_metadata_accumulates_along_the_chain(self, server) -> None:
        """An element *adds* to the metadata; it does not replace it.

        The frame arrives carrying what the decoder's sink stamped on it, and the segmenter's
        own key has to land beside that rather than instead of it — the difference between a
        chain whose stages compose and one where the last writer is the only writer.
        """
        chain = load(WALKABLE)
        decode, segment = chain.node("decode").element, chain.node("segment").element
        decode.open(ElementContext())
        segment.open(self.context(server))
        try:
            after_decode = decode.process(self.frame(frame_hw=(48, 64), fps=20.0))
            assert after_decode is not None
            after_segment = segment.process(after_decode)

            assert after_segment is not None
            assert after_segment.meta["frame_hw"] == (48, 64), "the frame size survives"
            assert set(after_segment.meta["masks"]) == {0}, "one area, on the ship's row"
        finally:
            decode.close()
            segment.close()

    def test_the_sink_consumes_the_item_rather_than_failing(self) -> None:
        """``None`` from a sink means "consumed", which is why failures raise instead."""
        sink = create_element(ElementKind.OUTPUT, "none", "output", {"keep_last": 2})
        sink.open()
        try:
            assert sink.process(self.frame()) is None
            assert sink.sink.emitted == 1
        finally:
            sink.close()


class TestTheDonorAtAFanIn:
    """Which predecessor donates payload and caps when two branches rejoin.

    Resolved by the loader and stored on the node, for exactly the reason
    :meth:`ElementNode.admits` lives there: the answer has to be the same in ``inprocess``,
    ``fleet`` and ``deepstream``, and three runners that each worked it out from the edges
    would eventually disagree about which branch donated a frame. The *merge* that consumes
    it stays in the runner (``tests/runners/test_walk.py``).
    """

    #: Two branches into one tracker, carrying **different** caps: ``detect`` hands the frame
    #: (``bgr@cpu``) and ``tap`` hands metadata (``meta@cpu``). ``after: [detect, tap]`` puts
    #: the *frame* predecessor first in declaration order on purpose — a tracker lists
    #: ``meta@cpu`` first in its ``accepts``, so the rule under test and the declaration order
    #: disagree, and with the two aligned the test would not know which one it proved.
    FAN_IN = """
        name: fan_in
        elements:
          decode: {impl: replay}
          detect: {impl: pool, model: ship_detector}
          tap:    {impl: shipvision, kind: track, after: detect}
          join:   {impl: shipvision, kind: track, after: [detect, tap]}
          output: {impl: none}
        """

    def test_the_donor_is_the_predecessor_whose_edge_carries_the_preferred_cap(self) -> None:
        """A payload is a frame handle or a tensor; half of one and half of another is not one.

        ``join`` is a tracker: it accepts ``meta@cpu`` before the two frame planes, so of its
        two predecessors the metadata branch donates — even though the frame branch is
        declared first.
        """
        chain = load(self.FAN_IN)

        node = chain.node("join")
        assert node.inputs == ("detect", "tap"), "the fixture's declaration order"
        assert node.donor == "tap", "the preferred cap wins over declaration order"

    def test_a_root_has_no_donor(self) -> None:
        chain = load(self.FAN_IN)

        assert chain.node("decode").donor is None
        assert chain.node("detect").donor == "decode", "a straight line donates from upstream"

    def test_a_fan_in_whose_edges_agree_takes_the_first_declared(self) -> None:
        """What a reader of the chain file would expect, and it must not depend on a hash."""
        chain = load("""
            name: agreeing_fan_in
            elements:
              decode: {impl: replay}
              detect: {impl: pool, model: d}
              left:   {impl: pool, kind: segment, model: s, after: detect,
                       params: {classes: [ship]}}
              right:  {impl: pool, kind: embed, model: e, after: detect}
              join:   {impl: pool, kind: segment, model: r, after: [right, left],
                       params: {classes: [person]}}
              output: {impl: none}
            """)

        assert chain.node("join").donor == "right"


class TestTheProductionChainFile:
    """``topology/ship_person.yaml`` is arch.md §1's chain, kept as a fixture until it runs."""

    PATH = REPO_ROOT / "topology" / "ship_person.yaml"

    @classmethod
    def substituted(cls) -> Topology:
        """The file with its one not-yet-landed spelling replaced, and nothing else touched.

        Parsed from the file rather than retyped, so this cannot drift from it. One
        substitution, and it names a phase rather than a convenience: ``decode: gstreamer-gpu``
        is phase D's zero-copy source and is not registered, so the loader cannot get past the
        first slot. Every other ``impl:``, every ``after:``, every ``params:`` is the file's,
        which is what makes the assertions below statements about the shipped chain.
        """
        raw = yaml.safe_load(cls.PATH.read_text(encoding="utf-8"))
        raw["elements"]["decode"]["impl"] = "replay"
        return Topology.from_spec(ChainSpec.model_validate(raw))

    def test_it_matches_the_schema(self) -> None:
        spec = ChainSpec.from_file(self.PATH)

        assert spec.name == "ship_person"
        assert list(spec.elements) == [
            "decode",
            "detect",
            "segment",
            "embed_ship",
            "embed_person",
            "recognize",
            "track",
            "mtmc",
            "output",
        ]
        assert spec.elements["segment"].when is None, "a guard nothing files is not written"
        assert spec.elements["embed_ship"].params["classes"] == ["ship"]
        assert spec.elements["embed_ship"].predecessors == ("segment",)

    def test_its_decoder_lands_in_a_later_phase(self) -> None:
        """It does not load yet, and the refusal says exactly which name is missing.

        When phase D registers ``gstreamer-gpu`` this test fails and should be replaced by
        one that loads the chain — that is the intended signal, not a maintenance cost.
        """
        with pytest.raises(UnknownElementImplError) as caught:
            Topology.from_file(self.PATH)

        assert caught.value.impl == "gstreamer-gpu"
        assert caught.value.kind == "decode"

    def test_every_row_selecting_slot_names_its_classes(self) -> None:
        """The conversion this file was waiting for, now done and pinned.

        It used to carry ``when: class == X`` on its three crop slots and ``impl: pool`` on
        ``recognize`` — the first refused since C8b because ``when:`` guards a frame while
        those elements select rows, the second because a ``pool`` recogniser files the model's
        raw response under a key ``output`` reads per row. Both were invisible only because
        the file stops at ``gstreamer-gpu`` first. This replaces the test that reported the
        gap: what it asserted is now what the file does.
        """
        chain = self.substituted()

        assert chain.node("embed_ship").element.declared_classes() == ("ship",)
        assert chain.node("embed_person").element.declared_classes() == ("person",)
        assert chain.node("recognize").element.declared_classes() == ("ship",)
        assert (
            chain.node("recognize").element.impl == "shipvision"
        ), "identity is a gallery query"
        assert all(node.condition is None for node in chain.nodes), "no guard nothing files"

    def test_its_own_wiring_is_a_valid_chain(self) -> None:
        """The file's *wiring* loads today, with only the phase-D decoder substituted.

        The test the file's header claims. Until phase D lands, the check above can only
        reach ``UnknownElementImplError`` — the loader stops at the first unregistered
        implementation and never gets to the structure or the caps — so the production chain
        could be wired wrong for four phases and every test would stay green. It was: this
        file used to fail with ``ChainStructureError: ['embed_person'] reach no output``,
        because ``track`` fell back to the declaration-order predecessor and picked up only
        ``recognize``.

        Substituting that one name and nothing else is the point. Anything more and this
        becomes a second fixture rather than a check on the first.
        """
        chain = self.substituted()

        assert len(chain) == 9
        assert [node.name for node in chain.sinks] == ["output"]
        assert sorted(node.name for node in chain.predecessors("track")) == [
            "embed_person",
            "recognize",
        ]
        assert [node.name for node in chain.predecessors("embed_person")] == ["detect"]

    def test_the_two_branches_cover_disjoint_rows(self) -> None:
        """The defect this chain shipped with: the ship branch ran on every person crop.

        ``when:`` is skip-and-continue, and it guards *one* element for a whole *frame*, so
        the guards on this file could never separate a frame's ships from its people at all:
        a frame holds both, nothing writes a frame-level ``meta['class']``, and the ship
        embedder and recogniser therefore either ran on every crop or on none. At the sizing
        in CLAUDE.md that is ~15 000 crops/s paying two extra model invocations, and the wrong
        answer is emitted downstream where nothing can tell it from a real one.

        Read through the row filter the loader demands: each branch names the labels it pays
        for, and the two sets do not meet. Deleting a ``classes:`` from either branch in the
        file fails this test.
        """
        chain = self.substituted()

        ships = set(chain.node("embed_ship").element.declared_classes())
        people = set(chain.node("embed_person").element.declared_classes())

        assert ships == {"ship"} and people == {"person"}
        assert not ships & people, "a crop paid for twice is a crop attributed twice"

    def test_every_row_selecting_stage_declares_its_rows(self) -> None:
        """And the whole-frame stage does not, which is the mirror of the conversion.

        ``detect`` submits one letterboxed frame and selects nothing, so a row filter on it
        would be as wrong as a frame guard on an embedder — and without this half, the test
        above would pass on a file that had simply had ``classes:`` sprayed across it.

        ``segment`` is on the other side of that line since P6-SEGMENT-CROP: without its
        ``classes:`` it runs the ship segmenter on every person crop at 640x640 — the defect
        the ship embedder shipped with, one stage earlier.
        """
        chain = self.substituted()
        selectors = {node.name for node in chain.nodes if node.element.selects_rows}

        assert selectors == {"segment", "embed_ship", "embed_person", "recognize", "track"}
        assert set(chain.node("segment").element.declared_classes()) == {"ship"}
        assert chain.node("detect").element.declared_classes() is None
        assert all(chain.node(name).condition is None for name in selectors)

    def test_the_inline_fixture_agrees_with_the_file(self) -> None:
        """``CHAIN`` resolves to exactly the wiring the production file resolves to.

        The fixture's docstring says it is the production chain's nine slots; this is what
        makes that sentence true. The drift worth catching is the fixture being *more*
        correct than the file it claims to copy — which is what happened, and is why
        ``CHAIN`` carried two ``after:`` clauses the file did not.

        Compared after validation rather than as text, because the default-predecessor rule
        means most of the wiring is not written in either place.
        """
        fixture = load(CHAIN)
        production = self.substituted()

        assert [node.name for node in fixture] == [node.name for node in production]
        assert [str(edge) for edge in fixture.edges] == [str(edge) for edge in production.edges]
        assert [str(node.condition) for node in fixture if node.condition] == [
            str(node.condition) for node in production if node.condition
        ]


class TestRowSelectionIsNotAFrameCondition:
    """``when:`` guards a frame, ``classes:`` selects rows, and the loader keeps them apart.

    The mistake is silent in both directions and expensive in both: a crop element guarded
    with ``when: class == ship`` is skipped on every frame — nothing writes a frame-level
    ``meta["class"]`` — so the ship embedder never runs and every ship's embedding is empty,
    with no counter saying anything is wrong. A ``classes: [vessel]`` in front of a detector
    that emits ``ship`` matches no row, which fails the same way one level down.
    """

    HEAD = "name: rows\nelements:\n  decode: {impl: replay}\n"

    def chain(self, *lines: str) -> Topology:
        return load(self.HEAD + "".join(f"  {line}\n" for line in lines))

    def test_a_row_selecting_element_may_not_carry_a_class_condition(self) -> None:
        with pytest.raises(ChainStructureError, match=r"classes: \[ship\]") as caught:
            self.chain(
                "detect: {impl: pool, model: d}",
                "embed:  {impl: pool, model: e, when: class == ship}",
                "output: {impl: none}",
            )

        assert "embed" in str(caught.value), "the message names the slot to edit"
        assert "FRAME" in str(caught.value)

    def test_the_same_holds_for_a_track_slot(self) -> None:
        """``track`` reads ``classes:`` too, so it is refused by the same declaration."""
        with pytest.raises(ChainStructureError, match="classes"):
            self.chain(
                "detect: {impl: pool, model: d}",
                "track:  {impl: shipvision, when: class == ship}",
                "output: {impl: none}",
            )

    def test_the_same_holds_for_a_recognize_slot(self) -> None:
        """The third reader of ``classes:``. ``ship_person.yaml`` carries exactly this
        spelling, and without the declaration on ``GalleryRecognize`` the chain loads and
        then publishes ``ship_id: null`` on every event for the life of the process."""
        with pytest.raises(ChainStructureError, match=r"classes: \[ship\]") as caught:
            self.chain(
                "detect:    {impl: pool, model: d}",
                "embed:     {impl: pool, model: e, params: {classes: [ship]}}",
                "recognize: {impl: shipvision, when: class == ship}",
                "output:    {impl: none}",
            )

        assert "recognize" in str(caught.value), "the message names the slot to edit"

    def test_the_refusal_reads_correctly_for_an_inequality(self) -> None:
        """``when: class != ship`` must not be told to write ``classes: [ship]`` — that is
        the inverse of what was asked, and a fix a reader can paste is the whole point."""
        with pytest.raises(ChainStructureError) as caught:
            self.chain(
                "detect: {impl: pool, model: d}",
                "embed:  {impl: pool, model: e, when: class != ship}",
                "output: {impl: none}",
            )

        message = str(caught.value)
        assert "classes: [ship]" not in message, "that would select exactly what was excluded"
        assert "every label except 'ship'" in message

    def test_the_fix_the_message_names_actually_loads(self) -> None:
        """A refusal that names a fix has to be checked against the fix, not only asserted."""
        chain = self.chain(
            "detect: {impl: pool, model: d}",
            "embed:  {impl: pool, model: e, params: {classes: [ship]}}",
            "output: {impl: none}",
        )

        assert chain.node("embed").element.declared_classes() == ("ship",)

    def test_a_frame_level_condition_is_untouched(self) -> None:
        """Only ``class`` is refused. ``when:`` keeps the job it is genuinely good at.

        The guard here is on ``fps`` because the runner actually files it
        (``runners/frames.py:144``): a spelling this suite blesses should be one that fires,
        not one that is false on every frame the way ``class`` is.
        """
        chain = self.chain(
            "detect: {impl: pool, model: d}",
            "embed:  {impl: pool, model: e, when: fps == 20}",
            "output: {impl: none}",
        )

        assert str(chain.node("embed").condition) == "fps == 20"

    def test_an_element_that_selects_no_rows_may_still_be_guarded_by_class(self) -> None:
        """The refusal is a declaration on the implementation, not a ban on a word.

        A ``detect`` slot is a whole-frame stage — it letterboxes one frame and reads no
        ``classes:`` — so the loader permits ``when: class == ship`` on one. Permitted is not
        recommended: nothing in a real chain writes a frame-level ``meta["class"]``, which is
        why ``topology/ship_person.yaml`` carries no guard at all
        (:meth:`TestTheProductionChainFile.test_every_row_selecting_stage_declares_its_rows`).
        What is under test here is the shape of the *rule* — a declaration on the
        implementation rather than a ban on a word.
        """
        chain = self.chain(
            "detect: {impl: pool, model: d, when: class == ship}",
            "output: {impl: none}",
        )

        assert str(chain.node("detect").condition) == "class == ship"


class TestAKeyReadPerRowMayNotBeFiledRaw:
    """``recognize: {impl: pool}`` in front of ``output`` publishes nothing, so it is refused.

    A pool element that submits whole frames files ``{output name: Tensor}`` under its key.
    ``output`` reads ``identities`` one entry per detection row, so the pair raises on the
    first frame and on every frame after it: the chain loads, every item's future fails, and
    the deployment's evidence that it works is a log nobody reads. The two halves are
    declarations, so the refusal reaches any future pairing of the same shape.
    """

    HEAD = "name: raw\nelements:\n  decode: {impl: replay}\n"

    def chain(self, *lines: str) -> Topology:
        return load(self.HEAD + "".join(f"  {line}\n" for line in lines))

    def test_a_pool_recognize_in_front_of_output_is_refused(self) -> None:
        with pytest.raises(ChainStructureError) as caught:
            self.chain(
                "detect:    {impl: pool, model: d}",
                "recognize: {impl: pool, model: r}",
                "output:    {impl: none}",
            )

        message = str(caught.value)
        assert "recognize" in message and "identities" in message
        assert "impl: shipvision" in message, "the refusal names the implementation that works"

    def test_the_implementation_the_message_names_loads(self) -> None:
        """The fix is checked against the loader, not asserted in prose."""
        chain = self.chain(
            "detect:    {impl: pool, model: d}",
            "recognize: {impl: shipvision}",
            "output:    {impl: none}",
        )

        assert chain.node("recognize").element.files_raw_response is False

    def test_a_pool_segment_is_untouched_and_now_for_two_reasons(self) -> None:
        """The rule is the declared pair, not "pool elements are suspect".

        ``masks`` is in no reader's ``reads_per_row`` — the half that held before
        P6-SEGMENT-CROP — and since it the segmenter scatters one area per detection row
        instead of filing its response raw, so neither half of the pair is met.
        """
        chain = self.chain(
            "detect:  {impl: pool, model: d}",
            "segment: {impl: pool, model: s}",
            "output:  {impl: none}",
        )

        assert chain.node("segment").element.files_raw_response is False

    def test_the_refusal_needs_both_halves_declared(self) -> None:
        """Not "pool elements are suspect": the reader's key and the producer's key have to be
        the same one, and the *producer* has to file raw. ``output`` reads all three keys
        since P6-SEGMENT-CROP, and the segment case above loads on the other half — it
        scatters one area per row rather than filing the engine's two outputs.
        """
        assert SinkOutput.reads_per_row == ("vectors", "identities", "masks")
        assert PoolRecognize.meta_key in SinkOutput.reads_per_row
        assert PoolRecognize.files_raw_response is True, "so the pair refuses"
        assert PoolSegment.meta_key in SinkOutput.reads_per_row
        assert PoolSegment.files_raw_response is False, "so the pair does not"


class TestClassesAreCheckedAgainstWhatTheDetectorEmits:
    """A branch that selects a label nobody detects is a dead branch that reports nothing."""

    HEAD = "name: labels\nelements:\n  decode: {impl: replay}\n"

    def chain(self, detect: str, embed: str) -> Topology:
        return load(f"{self.HEAD}  {detect}\n  {embed}\n  output: {{impl: none}}\n")

    LABELLED = (
        "detect: {impl: pool, model: d, "
        "params: {decode: {class_labels: {0: person, 8: ship}}}}"
    )

    def test_a_label_the_detector_never_emits_is_refused_naming_both_slots(self) -> None:
        with pytest.raises(ChainStructureError) as caught:
            self.chain(
                self.LABELLED, "embed: {impl: pool, model: e, params: {classes: [vessel]}}"
            )

        message = str(caught.value)
        assert "embed" in message and "detect" in message
        assert "vessel" in message and "ship" in message

    def test_a_recognize_slot_is_cross_checked_too(self) -> None:
        """The same typo on the third reader of ``classes:``, refused the same way."""
        with pytest.raises(ChainStructureError, match="vessel"):
            load(
                f"{self.HEAD}  {self.LABELLED}\n"
                "  embed: {impl: pool, model: e, params: {classes: [ship]}}\n"
                "  recognize: {impl: shipvision, params: {classes: [vessel]}}\n"
                "  output: {impl: none}\n"
            )

    def test_a_label_it_does_emit_loads(self) -> None:
        chain = self.chain(
            self.LABELLED, "embed: {impl: pool, model: e, params: {classes: [person]}}"
        )

        assert chain.node("embed").element.declared_classes() == ("person",)

    def test_nothing_is_checked_when_the_detector_declared_no_table(self) -> None:
        """A default table is a fallback, not a statement about this deployment's model.

        Refusing against it would invent a rule the chain never agreed to — and the default is
        two labels, so every deployment with a third would be refused for being ordinary.
        """
        chain = self.chain(
            "detect: {impl: pool, model: d}",
            "embed: {impl: pool, model: e, params: {classes: [dinghy]}}",
        )

        assert chain.node("embed").element.declared_classes() == ("dinghy",)

    def test_a_track_slots_classes_are_checked_too(self) -> None:
        """Every reader of ``classes:`` is checked, because every one of them fails the same."""
        with pytest.raises(ChainStructureError, match="vessel"):
            self.chain(
                self.LABELLED,
                "track: {impl: shipvision, params: {classes: [vessel]}}",
            )


class TestTheRunnableChainFile:
    """``topology/ship_person_cpu.yaml``: the sibling that loads on this host today."""

    PATH = REPO_ROOT / "topology" / "ship_person_cpu.yaml"

    def test_it_loads_with_no_substitutions(self) -> None:
        """The whole difference from ``ship_person.yaml``, whose decoder is phase D's.

        Loading is not running — the ``pool`` slots still need a repository and the shipvision
        slots still need the submodule — but every ``impl:`` in it resolves, which is what
        makes it the file to point ``shipinfer run`` at.
        """
        chain = Topology.from_file(self.PATH)

        assert chain.name == "ship_person_cpu"
        assert [node.name for node in chain.nodes][:2] == ["decode", "detect"]
        assert [node.name for node in chain.sinks] == ["output"]

    def test_the_chain_leaves_the_pixels_behind_at_track(self) -> None:
        """Host pixels to the tracker, metadata after it. The plane change, read off the file."""
        chain = Topology.from_file(self.PATH)
        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in chain.edges}

        assert caps[("decode", "detect")] == "bgr@cpu"
        assert caps[("embed_ship", "track")] == "bgr@cpu"
        assert caps[("track", "mtmc")] == "meta@cpu"
        assert caps[("mtmc", "output")] == "meta@cpu"

    def test_both_branches_rejoin_at_the_tracker(self) -> None:
        """``embed_person`` following ``detect`` and ``track`` following both is the wiring
        ``ship_person.yaml``'s header argues for at length; this file has to agree with it."""
        chain = Topology.from_file(self.PATH)

        assert [node.name for node in chain.predecessors("embed_person")] == ["detect"]
        assert sorted(node.name for node in chain.predecessors("track")) == [
            "embed_person",
            "embed_ship",
        ]

    def test_it_selects_rows_with_classes_and_guards_no_frame(self) -> None:
        """No ``when:`` anywhere: this file is the answer to the question the guards asked."""
        spec = ChainSpec.from_file(self.PATH)

        assert [slot for slot, declared in spec.elements.items() if declared.when] == []
        assert spec.elements["embed_ship"].params["classes"] == ["ship"]
        assert spec.elements["embed_person"].params["classes"] == ["person"]

    def test_the_declared_classes_agree_with_the_declared_detector(self) -> None:
        """The cross-check running on the shipped file, which is the point of shipping it."""
        chain = Topology.from_file(self.PATH)

        assert chain.node("detect").element.detection_labels() == ("person", "ship")
        assert chain.node("embed_ship").element.declared_classes() == ("ship",)

    def test_it_names_no_recognize_slot_yet(self) -> None:
        """Absent rather than commented in: a chain file is executable, and a half-wired
        branch that loads is worse than one that is not there. The follow-up adds one line."""
        assert "recognize" not in ChainSpec.from_file(self.PATH).elements
