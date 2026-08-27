"""The chain loader: what it accepts, and — mostly — what it refuses.

This file is where the value of a declarative chain is cashed in. Every test below is a
deployment that would otherwise have started, run, and produced either nothing or the wrong
thing:

* a chain whose CPU detector sits behind a GPU decoder (a silent 3 GB/s download);
* a chain with no output element (every model runs, nothing is emitted);
* a chain whose ``after:`` has a typo (a branch that never runs);
* a segmenter spelled ``sement`` (a missing stage nobody notices).

The whole file runs offline, with no GPU and no driver, because every element in it is a
mock and :class:`~shipinfer.topology.base.Element` constructors are required to be
hardware-free.

The tests are named ``tests/topology/test_chain.py`` rather than ``test_topology.py``
because ``tests/server/test_topology.py`` already exists and pytest's default import mode
gives two same-named modules in non-package directories the same module name.
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

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
from shipinfer.topology.elements.mock import MockDetect, MockOutput

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``topology/ship_person.yaml`` with every implementation replaced by its mock -- the same
#: nine slots, the same ``when:`` conditions -- including the ``class == ship`` repeated on
#: every element of the ship branch, because a condition guards one element only -- and the
#: same ``after:`` wiring, including the two lines that spell out the rejoin
#: (``embed_person: after: detect`` and ``track: after: [recognize, embed_person]``). It is
#: a **branching** chain, not a straight line: two branches split at ``detect`` and rejoin
#: at ``track``.
#:
#: Written out rather than generated: a chain file is the thing under test, and a generated
#: one would test the generator. The cost of writing it out is that it can drift from the
#: real file, so ``TestTheProductionChainFile`` loads that file's own wiring with the impls
#: substituted and asserts it is valid -- the fixture being wrong is then a test failure
#: rather than a fixture that quietly proves something the deployment does not do.
MOCK_CHAIN = textwrap.dedent("""
    name: mock_ship_person
    elements:
      decode:       {impl: mock}
      detect:       {impl: mock, model: ship_detector}
      segment:      {impl: mock, model: ship_segmenter, when: class == ship}
      embed_ship:   {impl: mock, model: ship_embedder, when: class == ship, after: segment}
      embed_person: {impl: mock, model: person_embedder, when: class == person, after: detect}
      recognize:    {impl: mock, model: ship_recognizer, when: class == ship, after: embed_ship}
      track:        {impl: mock, per: camera, after: [recognize, embed_person]}
      mtmc:         {impl: mock, scope: global}
      output:       {impl: mock}
    """)


def load(text: str) -> Topology:
    """Build a topology from inline YAML, the way a chain file would be loaded."""
    return Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(text)))


def item(camera: str = "cam-1", frame: int = 7) -> ChainItem:
    return ChainItem(RequestContext(camera_id=camera, frame_id=frame), Caps.parse("nv12@gpu"))


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
    """One registry per kind, and a hardware-free implementation of every kind."""

    @pytest.mark.parametrize("kind", list(ElementKind))
    def test_every_kind_has_a_registry_and_a_mock(self, kind: ElementKind) -> None:
        """A kind with no offline implementation is a kind the offline tier cannot test."""
        registry = registry_for(kind)

        assert registry.element_kind is kind
        assert "mock" in registry
        assert registry.get("mock").kind is kind

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

        ``pool`` is the default for all four model kinds and ``shipvision`` is both a
        tracker and an MTMC implementation (arch.md §1). A flat registry refuses a duplicate
        name, so it would force the kind into the name by convention.
        """
        classes = {registry_for(kind).get("mock") for kind in ElementKind}

        assert len(classes) == len(list(ElementKind))

    def test_an_unknown_impl_names_the_ones_that_exist(self) -> None:
        with pytest.raises(UnknownElementImplError) as caught:
            create_element(ElementKind.DETECT, "moc", "detect")

        assert "moc" in str(caught.value)
        assert "mock" in str(caught.value), "the message must list the available impls"
        assert caught.value.available == tuple(registry_for(ElementKind.DETECT).names())

    def test_an_unknown_kind_is_refused_by_name(self) -> None:
        with pytest.raises(UnknownElementKindError):
            registry_for("sement")

    def test_describe_elements_reports_every_kind(self) -> None:
        described = describe_elements()

        assert set(described) == {kind.value for kind in ElementKind}
        assert all(name and text for name, text in described["detect"]), "no description"

    def test_the_registered_name_lands_on_the_class(self) -> None:
        """``impl`` is set by the decorator, so the chain file and a log line agree."""
        assert registry_for(ElementKind.DETECT).get("mock").impl == "mock"
        assert registry_for(ElementKind.DETECT).get("mock-cpu").impl == "mock-cpu"

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
        registry.register_lazy(
            "mock-lazy-tracker",
            "shipinfer.topology.elements.mock:MockTrack",
            description="a tracker smuggled into the output registry",
        )

        assert "mock-lazy-tracker" in registry, "the registration itself cannot check"

        with pytest.raises(ConfigurationError, match="cannot be registered as a output"):
            create_element(ElementKind.OUTPUT, "mock-lazy-tracker", "output")

    def test_a_lazy_registration_of_the_right_kind_works_and_gets_its_name(self) -> None:
        """The other half: lazy is a legitimate style, and ``impl`` still lands."""
        registry = registry_for(ElementKind.OUTPUT)
        registry.register_lazy(
            "mock-lazy-output",
            "shipinfer.topology.elements.mock:MockLazyOutput",
            description="a sink registered lazily",
        )

        element = create_element(ElementKind.OUTPUT, "mock-lazy-output", "output")

        assert element.kind is ElementKind.OUTPUT
        assert element.impl == "mock-lazy-output", "no decorator ran, so the factory sets it"

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
        for name in ("mock-lazy-twice-a", "mock-lazy-twice-b"):
            registry.register_lazy(
                name, f"{module.__name__}:Sink", description="one class, two names"
            )

        first = create_element(ElementKind.OUTPUT, "mock-lazy-twice-a", "output")

        with pytest.raises(ConfigurationError, match="two implementation names"):
            create_element(ElementKind.OUTPUT, "mock-lazy-twice-b", "output")
        assert first.impl == "mock-lazy-twice-a", "the built element keeps its own name"

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

        class Derived(MockOutput):
            """A sink that is a subclass of an eagerly registered one."""

        module.Sink = Derived  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module.__name__, module)
        registry = registry_for(ElementKind.OUTPUT)
        registry.register_lazy(
            "mock-lazy-derived", f"{module.__name__}:Sink", description="a subclass"
        )

        element = create_element(ElementKind.OUTPUT, "mock-lazy-derived", "output")

        assert element.impl == "mock-lazy-derived"
        assert MockOutput.impl == "mock", "the base's own name must not be rewritten"


class TestLoadingAValidChain:
    def test_the_mock_chain_loads_and_is_ordered(self) -> None:
        chain = load(MOCK_CHAIN)

        assert chain.name == "mock_ship_person"
        assert len(chain) == 9
        order = [node.name for node in chain]
        for edge in chain.edges:
            assert order.index(edge.producer) < order.index(
                edge.consumer
            ), f"out of order: {edge}"

    def test_the_frame_stays_on_the_device_until_the_tracker(self) -> None:
        """The cap of every edge, which is the load-time evidence for the VRAM-first path.

        Everything up to ``track`` carries ``nv12@gpu``; from ``track`` on it is
        ``meta@cpu``, because a MOT algorithm needs boxes and not pixels. A chain that
        dropped to host memory earlier would show up right here.
        """
        chain = load(MOCK_CHAIN)
        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in chain.edges}

        assert caps[("decode", "detect")] == "nv12@gpu"
        assert caps[("detect", "segment")] == "nv12@gpu"
        assert caps[("recognize", "track")] == "nv12@gpu"
        assert caps[("track", "mtmc")] == "meta@cpu"
        assert caps[("mtmc", "output")] == "meta@cpu"

    def test_roots_and_sinks_are_the_decoder_and_the_output(self) -> None:
        chain = load(MOCK_CHAIN)

        assert [node.name for node in chain.roots] == ["decode"]
        assert [node.name for node in chain.sinks] == ["output"]

    def test_the_default_predecessor_is_the_previously_declared_element(self) -> None:
        """The rule that lets a nine-step chain need no wiring at all."""
        chain = load("""
            elements:
              decode: {impl: mock}
              detect: {impl: mock, model: d}
              track:  {impl: mock}
              output: {impl: mock}
            """)

        assert [node.name for node in chain.predecessors("detect")] == ["decode"]
        assert [node.name for node in chain.predecessors("track")] == ["detect"]
        assert [node.name for node in chain.predecessors("output")] == ["track"]

    def test_after_overrides_the_default_and_a_diamond_rejoins(self) -> None:
        """A branch and a fan-in, which is the whole reason ``after:`` exists.

        ``embed_person`` must follow ``detect``, not the ``segment`` declared before it, and
        both branches rejoin at ``track``.
        """
        chain = load(MOCK_CHAIN)

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
        chain = load(MOCK_CHAIN)

        assert chain.node("embed_ship").kind is ElementKind.EMBED
        assert chain.node("embed_person").kind is ElementKind.EMBED
        assert chain.node("decode").kind is ElementKind.DECODE

    def test_an_explicit_kind_overrides_the_slot_name(self) -> None:
        chain = load("""
            elements:
              camera: {impl: mock, kind: decode}
              emit:   {impl: mock, kind: output}
            """)

        assert chain.node("camera").kind is ElementKind.DECODE

    def test_describe_prints_the_resolved_wiring(self) -> None:
        """The default-predecessor rule means the file does not state most of the edges."""
        described = load(MOCK_CHAIN).describe()

        assert "decode -> detect [nv12@gpu]" in described
        assert "when=class == ship" in described
        assert "model=ship_detector" in described

    def test_the_element_is_told_which_model_it_runs(self) -> None:
        """``model:`` reaches the element, not only the node's spec beside it.

        A ``pool`` element resolves that name against ``ElementContext.models`` at ``open``;
        it holds no reference to the node, so a name it is never handed is a name it cannot
        resolve. ``None`` for the four kinds that have no model, which is a kind's property
        and not a missing value -- the loader has already refused a model kind that names
        none.
        """
        chain = load(MOCK_CHAIN)

        assert chain.node("detect").element.model == "ship_detector"
        assert chain.node("embed_ship").element.model == "ship_embedder"
        assert chain.node("embed_person").element.model == "person_embedder"
        assert chain.node("decode").element.model is None
        assert chain.node("track").element.model is None
        assert chain.node("output").element.model is None

    def test_the_runner_can_still_read_what_the_file_said(self) -> None:
        """``per``/``scope``/``params`` are the runner's business, so the spec is kept."""
        chain = load(MOCK_CHAIN)

        assert chain.node("track").spec.per == "camera"
        assert chain.node("mtmc").spec.scope == "global"

    def test_a_wildcard_passthrough_between_two_gpu_elements_loads(self) -> None:
        """Propagation resolves, it does not forbid. The other half of the §8 rule.

        The same ``*@*`` element that is refused in front of a host-only sink is fine between
        two device elements, and both of its edges come out ``nv12@gpu`` — not ``*@*``, which
        would be a validated chain nobody could read the caps of.
        """
        chain = load("""
            elements:
              decode: {impl: mock}
              track:  {impl: mock-passthrough}
              detect: {impl: mock, model: d, after: track}
              output: {impl: mock}
            """)
        caps = {(edge.producer, edge.consumer): str(edge.caps) for edge in chain.edges}

        assert caps[("decode", "track")] == "nv12@gpu"
        assert caps[("track", "detect")] == "nv12@gpu"

    def test_a_cap_transparent_conditional_element_loads(self) -> None:
        """The bypass rule refuses a plane change, not every ``when:``.

        ``segment`` here consumes and produces ``nv12@gpu``, so the frame that skips it looks
        exactly like the frame that went through it and ``embed_ship`` cannot tell — which is
        the shape a branch condition is supposed to have.
        """
        chain = load("""
            elements:
              decode:     {impl: mock}
              detect:     {impl: mock, model: d}
              segment:    {impl: mock, model: s, when: class == ship}
              embed_ship: {impl: mock, model: e}
              output:     {impl: mock}
            """)

        assert str(chain.node("segment").condition) == "class == ship"
        assert all(str(edge.caps) == "nv12@gpu" for edge in chain.edges[:3])

    def test_loading_twice_gives_two_independent_chains(self) -> None:
        """Elements are stateful (a tracker holds per-camera state), so nothing is cached."""
        first, second = load(MOCK_CHAIN), load(MOCK_CHAIN)

        assert first.node("track").element is not second.node("track").element


class TestRefusals:
    def test_a_cpu_detector_behind_a_gpu_decoder_is_refused(self) -> None:
        """arch.md §8's rule: a chain that would silently download refuses to load.

        The message must name both sides and both cap lists — this is the error most likely
        to be read by someone who did not write the chain.
        """
        with pytest.raises(CapsMismatchError) as caught:
            load("""
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock-cpu, model: d}
                  output: {impl: mock}
                """)

        message = str(caught.value)
        assert "'decode'" in message and "'detect'" in message
        assert "nv12@gpu" in message and "bgr@cpu" in message
        assert "convert" in message, "the message should name the fix, not just the problem"
        assert caught.value.produced == ("nv12@gpu",)
        assert caught.value.accepted == ("bgr@cpu",)

    def test_a_wildcard_passthrough_cannot_launder_a_gpu_frame_to_a_cpu_sink(self) -> None:
        """The evasion a per-edge caps check leaves open, and the reason caps propagate.

        ``mock-passthrough`` declares ``accepts: *@*`` and ``produces: *@*``, which is the
        honest way for a real tee or filter to say "I do not touch the pixels". Negotiated
        one edge at a time, both of its edges pass: ``nv12@gpu`` matches ``*@*``, and ``*@*``
        matches ``bgr@cpu``. The chain would load, and the device-to-host download arch.md §8
        exists to refuse would happen every frame in the middle of it.

        It is refused because the loader resolves the wildcard from what arrives at the
        element *before* negotiating its output, so by the time the sink is considered the
        passthrough is an ``nv12@gpu`` producer and nothing bridges the two memories.
        """
        with pytest.raises(CapsMismatchError) as caught:
            load("""
                elements:
                  decode: {impl: mock}
                  track:  {impl: mock-passthrough}
                  output: {impl: mock-cpu}
                """)

        assert caught.value.producer == "track"
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
                  decode: {impl: mock-any}
                  output: {impl: mock}
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
                  decode: {impl: mock, when: class == ship}
                  detect: {impl: mock, model: d}
                  output: {impl: mock}
                """)

    def test_a_wildcard_producer_fed_two_different_caps_is_refused(self) -> None:
        """A fan-in into a passthrough: the loader would have to pick one cap and guess.

        Fan-in edges carrying different caps is legal in general (each edge is negotiated on
        its own pair). It stops being legal when the element says "I hand on what I was
        given", because then there is no single answer to put on its outbound edge.
        """
        with pytest.raises(ChainStructureError, match="wildcard format"):
            load("""
                elements:
                  decode:     {impl: mock}
                  detect:     {impl: mock, model: d}
                  track:      {impl: mock, after: detect}
                  track_join: {impl: mock-passthrough, after: [detect, track]}
                  output:     {impl: mock}
                """)

    def test_a_conditional_element_that_changes_the_plane_is_refused(self) -> None:
        """The bypass edge: what a ``when:`` element's absence exposes.

        A condition means skip-and-continue, so an item the tracker does not admit travels
        from ``decode`` straight to ``mtmc``. The declared edges are both fine —
        ``decode -> track`` is ``nv12@gpu``, ``track -> mtmc`` is ``meta@cpu`` — and the
        chain still cannot run, because for every person frame ``mtmc`` is handed a device
        frame it cannot read. Checking only the declared edges would ship this.
        """
        with pytest.raises(CapsMismatchError) as caught:
            load("""
                elements:
                  decode: {impl: mock}
                  track:  {impl: mock, when: class == ship}
                  mtmc:   {impl: mock}
                  output: {impl: mock}
                """)

        assert caught.value.skipped == "track"
        assert caught.value.producer == "decode"
        assert caught.value.consumer == "mtmc"
        assert "is skipped" in str(caught.value)

    def test_a_dangling_after_names_the_referrer_and_the_typo(self) -> None:
        with pytest.raises(UnknownElementError) as caught:
            load("""
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock, model: d, after: decoed}
                  output: {impl: mock}
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
                  decode: {impl: mock}
                  detect: {impl: mock, model: d, after: track}
                  track:  {impl: mock, after: detect}
                  mtmc:   {impl: mock}
                  output: {impl: mock}
                """)

        assert caught.value.cycle == ("detect", "track")
        assert "detect -> track -> detect" in str(caught.value)

    def test_an_element_that_depends_on_itself_is_refused(self) -> None:
        with pytest.raises(ChainCycleError):
            load("""
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock, model: d, after: detect}
                  output: {impl: mock}
                """)

    @pytest.mark.parametrize(
        ("chain", "message", "why"),
        [
            (
                """
                elements:
                  decode: {impl: mock}
                """,
                "the chain has no output element",
                "a decoder alone emits nothing",
            ),
            (
                """
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock, model: d}
                """,
                "the chain has no output element",
                "every model runs and nothing is emitted",
            ),
            (
                """
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock, model: d}
                  track:  {impl: mock}
                """,
                "the chain has no output element",
                "the chain ends at a tracker, whose results go nowhere",
            ),
            (
                """
                elements:
                  detect: {impl: mock, model: d, after: []}
                  output: {impl: mock}
                """,
                "must be a decode element",
                "a chain must start at a decoder",
            ),
            (
                """
                elements:
                  output: {impl: mock, after: []}
                """,
                "must be a decode element",
                "an output sink with no decoder in front of it is not a chain",
            ),
            (
                """
                elements:
                  decode: {impl: mock}
                  output: {impl: mock}
                  mtmc:   {impl: mock, after: output}
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
                  decode:  {impl: mock}
                  detect:  {impl: mock, model: d}
                  output:  {impl: mock}
                  segment: {impl: mock, model: s, after: detect}
                """)

    def test_a_model_kind_without_a_model_is_refused(self) -> None:
        with pytest.raises(ChainStructureError, match="model"):
            load("""
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock}
                  output: {impl: mock}
                """)

    def test_an_unknown_key_is_refused(self) -> None:
        """``extra="forbid"`` is the reason the schema is worth having."""
        with pytest.raises(ChainSpecError) as caught:
            load("""
                elements:
                  decode: {impl: mock, mdoel: ship_detector}
                  output: {impl: mock}
                """)

        assert "mdoel" in str(caught.value)

    def test_a_misspelled_slot_name_is_refused(self) -> None:
        with pytest.raises(UnknownElementKindError) as caught:
            load("""
                elements:
                  decode: {impl: mock}
                  sement: {impl: mock, model: s}
                  output: {impl: mock}
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
                  camera: {impl: mock, kind: decdoe}
                  output: {impl: mock}
                """)

        assert caught.value.slot == "decdoe"
        assert "decode" in str(caught.value)

    def test_a_non_string_top_level_key_is_refused_as_a_spec_error(self) -> None:
        """The schema is the only thing allowed to reject a document, however odd it is."""
        with pytest.raises(ChainSpecError):
            ChainSpec.from_yaml("1: elements\nelements: {decode: {impl: mock}}")

    def test_a_chain_with_no_elements_is_refused(self) -> None:
        with pytest.raises(ChainSpecError):
            load("elements: {}")

    def test_a_document_that_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(ChainSpecError, match="mapping"):
            ChainSpec.from_yaml("- decode\n- detect")

    def test_invalid_yaml_names_the_source(self) -> None:
        with pytest.raises(ChainSpecError, match=r"chain\.yaml"):
            ChainSpec.from_yaml("elements: {decode: {impl: mock}\n", source="chain.yaml")

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
        chain = load(MOCK_CHAIN)

        with pytest.raises(KeyError) as caught:
            chain.node("detetc")

        assert "detetc" in str(caught.value)
        assert "detect" in str(caught.value), "the message lists what there is"
        assert not isinstance(caught.value, TopologyError)

    def test_the_package_exports_classes_and_not_modules(self) -> None:
        """``__all__`` is the public vocabulary, and a module is not part of it.

        ``elements.mock`` is imported by ``__init__`` for its registration side effect, which
        is necessary and is not the same as being an export: an implementation is reached
        through its registry, by the name a chain file uses. Exporting the module invites
        ``from shipinfer.topology import mock`` and a second way to get at an element that
        skips the seam — ``backends`` and ``ingest`` export the classes they mean and nothing
        else.
        """
        import shipinfer.topology as package

        exported = {name: getattr(package, name) for name in package.__all__}

        assert not [
            name for name, value in exported.items() if isinstance(value, types.ModuleType)
        ]
        assert "mock" in registry_for(ElementKind.DETECT), "the side effect still happened"


class TestConditions:
    def test_a_when_expression_is_parsed(self) -> None:
        chain = load(MOCK_CHAIN)
        condition = chain.node("segment").condition

        assert condition == Condition("class", "==", "ship")
        assert str(condition) == "class == ship"

    @pytest.mark.parametrize("text", ["class = ship", "class", "== ship", "class ~ ship"])
    def test_a_malformed_when_is_refused(self, text: str) -> None:
        """``class = ship`` read as "always true" would run the heaviest model on every frame."""
        with pytest.raises(ConditionSyntaxError):
            Condition.parse(text)

    def test_a_condition_gates_an_element_on_metadata(self) -> None:
        chain = load(MOCK_CHAIN)
        segment = chain.node("segment")

        assert segment.admits(item().derive(**{"class": "ship"}))
        assert not segment.admits(item().derive(**{"class": "person"}))

    def test_a_missing_field_satisfies_neither_operator(self) -> None:
        """Absence is not evidence: ``!=`` must not fire on a frame nobody classified."""
        assert not Condition.parse("class == ship").matches({})
        assert not Condition.parse("class != ship").matches({})

    def test_an_element_with_no_condition_sees_everything(self) -> None:
        assert load(MOCK_CHAIN).node("detect").admits(item())


class TestElementLifecycle:
    def test_open_and_close_are_idempotent(self) -> None:
        element = create_element(ElementKind.DETECT, "mock", "detect")

        element.open()
        element.open()
        assert element.is_open and element.opens == 1

        element.close()
        element.close()
        assert not element.is_open and element.closes == 1

    def test_closing_an_element_that_never_opened_does_nothing(self) -> None:
        element = create_element(ElementKind.DETECT, "mock", "detect")

        element.close()

        assert element.closes == 0

    def test_processing_before_open_is_a_typed_refusal(self) -> None:
        """Not an implicit open: that is how a CUDA context lands on the wrong thread."""
        element = create_element(ElementKind.DETECT, "mock", "detect")

        with pytest.raises(ServerStateError, match="before open"):
            element.process(item())

    def test_a_failed_open_closes_what_it_managed_to_acquire(self) -> None:
        class Broken(MockDetect):
            def _do_open(self, context: ElementContext) -> None:
                super()._do_open(context)
                raise RuntimeError("no camera")

        element = Broken("detect")

        with pytest.raises(RuntimeError, match="no camera"):
            element.open()

        assert not element.is_open
        assert element.closes == 1, "a half-open element must be unwound"

    def test_the_context_arrives_at_open_and_is_dropped_at_close(self) -> None:
        element = create_element(ElementKind.DETECT, "mock", "detect")
        context = ElementContext(shard_id=3)

        element.open(context)
        assert element.context is context

        element.close()
        assert element.context is None

    def test_params_reach_the_element(self) -> None:
        chain = load("""
            elements:
              decode: {impl: mock}
              detect: {impl: mock, model: d, params: {class: person}}
              output: {impl: mock}
            """)

        assert chain.node("detect").element.params == {"class": "person"}


class TestWalkingAChain:
    def test_the_tag_survives_every_element(self) -> None:
        """ADR-002, checked end to end: the ``(camera_id, frame_id)`` tag is never rewritten.

        Walking the chain by hand rather than through a runner — there is no runner yet, and
        the property belongs to the elements either way.
        """
        chain = load(MOCK_CHAIN)
        for node in chain:
            node.element.open(ElementContext())

        current = item("cam-42", 1234)
        start_key = current.key
        seen: list[str] = []
        for node in chain:
            if not node.admits(current):
                continue
            result = node.element.process(current)
            seen.append(node.name)
            if result is None:
                break
            current = result
            assert current.key == start_key, f"{node.name} rewrote the tag"

        sink = chain.node("output").element
        assert seen[-1] == "output"
        assert len(sink.emitted) == 1
        assert sink.emitted[0].key == start_key
        assert str(sink.emitted[0].caps) == "meta@cpu"
        for node in chain:
            node.element.close()

    def test_metadata_accumulates_along_the_chain(self) -> None:
        chain = load(MOCK_CHAIN)
        detect, segment = chain.node("detect").element, chain.node("segment").element
        detect.open()
        segment.open()

        after_detect = detect.process(item())
        assert after_detect is not None
        after_segment = segment.process(after_detect)

        assert after_segment is not None
        assert after_segment.meta["boxes"], "the detector's boxes must survive"
        assert after_segment.meta["masks"], "and the segmenter must have added its own"

    def test_the_sink_consumes_the_item_rather_than_failing(self) -> None:
        """``None`` from a sink means "consumed", which is why failures raise instead."""
        sink = create_element(ElementKind.OUTPUT, "mock", "output")
        sink.open()

        assert sink.process(item()) is None
        assert len(sink.emitted) == 1


class TestTheDonorAtAFanIn:
    """Which predecessor donates payload and caps when two branches rejoin.

    Resolved by the loader and stored on the node, for exactly the reason
    :meth:`ElementNode.admits` lives there: the answer has to be the same in ``inprocess``,
    ``fleet`` and ``deepstream``, and three runners that each worked it out from the edges
    would eventually disagree about which branch donated a frame. The *merge* that consumes
    it stays in the runner (``tests/runners/test_walk.py``).
    """

    #: Two branches into one tracker, carrying **different** caps: ``detect`` hands a frame
    #: (``nv12@gpu``) and ``tap`` hands metadata (``meta@cpu``). ``after: [tap, detect]``
    #: puts the metadata predecessor *first* in declaration order on purpose — the rule is
    #: about the negotiated cap, and with the two aligned the test would not know which one
    #: it proved.
    FAN_IN = """
        name: fan_in
        elements:
          decode: {impl: mock}
          detect: {impl: mock, model: ship_detector}
          tap:    {impl: mock, kind: track, after: detect}
          join:   {impl: mock, kind: track, after: [tap, detect]}
          output: {impl: mock}
        """

    def test_the_donor_is_the_predecessor_whose_edge_carries_the_preferred_cap(self) -> None:
        """A payload is a frame handle or a tensor; half of one and half of another is not one.

        ``join`` is a tracker: it accepts ``nv12@gpu`` before ``meta@cpu``, so of its two
        predecessors the one whose edge carries the frame donates — even though the metadata
        branch is declared first.
        """
        chain = load(self.FAN_IN)

        node = chain.node("join")
        assert node.inputs == ("tap", "detect"), "the fixture's declaration order"
        assert node.donor == "detect", "the preferred cap wins over declaration order"

    def test_a_root_has_no_donor(self) -> None:
        chain = load(self.FAN_IN)

        assert chain.node("decode").donor is None
        assert chain.node("detect").donor == "decode", "a straight line donates from upstream"

    def test_a_fan_in_whose_edges_agree_takes_the_first_declared(self) -> None:
        """What a reader of the chain file would expect, and it must not depend on a hash."""
        chain = load("""
            name: agreeing_fan_in
            elements:
              decode: {impl: mock}
              detect: {impl: mock, model: d}
              left:   {impl: mock, kind: segment, model: s, after: detect}
              right:  {impl: mock, kind: embed, model: e, after: detect}
              join:   {impl: mock, kind: recognize, model: r, after: [right, left]}
              output: {impl: mock}
            """)

        assert chain.node("join").donor == "right"


class TestTheProductionChainFile:
    """``topology/ship_person.yaml`` is arch.md §1's chain, kept as a fixture until it runs."""

    PATH = REPO_ROOT / "topology" / "ship_person.yaml"

    @classmethod
    def load_with_mocks(cls, *, detect_class: str | None = None) -> Topology:
        """The file, with every ``impl:`` replaced by ``mock`` and nothing else touched.

        Parsed from the file rather than retyped, so this cannot drift from it: the
        substitution is the *only* difference between what a deployment loads and what the
        offline tier can load today.

        Args:
            detect_class: what the mock detector should claim to have found, passed through
                ``params:``. It changes the *mock*, never the wiring -- ``MockDetect`` reads
                ``params["class"]`` and stamps it into the metadata, which is how a branch
                test chooses which branch of this chain ought to fire. Left out, the
                detector's own default (``ship``) applies.
        """
        raw = yaml.safe_load(cls.PATH.read_text(encoding="utf-8"))
        for declared in raw["elements"].values():
            declared["impl"] = "mock"
        if detect_class is not None:
            raw["elements"]["detect"]["params"] = {"class": detect_class}
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
        assert spec.elements["segment"].when == "class == ship"
        assert spec.elements["embed_ship"].predecessors == ("segment",)

    def test_its_implementations_land_in_a_later_phase(self) -> None:
        """It does not load yet, and the refusal says exactly which name is missing.

        When phase D registers ``gstreamer-gpu`` this test fails and should be replaced by
        one that loads the chain — that is the intended signal, not a maintenance cost.
        """
        with pytest.raises(UnknownElementImplError) as caught:
            Topology.from_file(self.PATH)

        assert caught.value.impl == "gstreamer-gpu"
        assert caught.value.kind == "decode"

    def test_its_own_wiring_is_a_valid_chain(self) -> None:
        """The file's *wiring* loads today, with only the ``impl:`` names substituted.

        The test the file's header claims. Until phase D lands, the check above can only
        reach ``UnknownElementImplError`` — the loader stops at the first unregistered
        implementation and never gets to the structure or the caps — so the production chain
        could be wired wrong for four phases and every test would stay green. It was: with
        the mocks swapped in, this file used to fail with
        ``ChainStructureError: ['embed_person'] reach no output``, because ``track`` fell
        back to the declaration-order predecessor and picked up only ``recognize``.

        Substituting ``impl:`` and nothing else is the point. Anything more and this becomes
        a second fixture rather than a check on the first.
        """
        chain = self.load_with_mocks()

        assert len(chain) == 9
        assert [node.name for node in chain.sinks] == ["output"]
        assert sorted(node.name for node in chain.predecessors("track")) == [
            "embed_person",
            "recognize",
        ]
        assert [node.name for node in chain.predecessors("embed_person")] == ["detect"]

    def test_a_person_detection_touches_no_ship_only_element(self) -> None:
        """The defect this chain shipped with: the ship branch ran on every person crop.

        ``when:`` is skip-and-continue, and it guards *one* element, so a condition on
        ``segment`` alone leaves ``embed_ship`` and ``recognize`` unguarded: a person was
        skipped past the segmenter and then handed to the ship embedder and the ship
        recogniser, which wrote ``meta["identities"]`` for a person. At the sizing in
        CLAUDE.md that is ~15 000 crops/s paying two extra model invocations, and the wrong
        answer is emitted downstream where nothing can tell it from a real one.

        Removing either ``when: class == ship`` from ``topology/ship_person.yaml`` fails
        this test.
        """
        chain = self.load_with_mocks(detect_class="person")

        emitted = walk(chain, item("cam-3", 11))

        for ship_only in ("segment", "embed_ship", "recognize"):
            assert (
                chain.node(ship_only).element.processes == 0
            ), f"{ship_only} ran on a person detection"
        assert chain.node("embed_person").element.processes == 1
        assert "identities" not in emitted.meta, "a person tracklet cannot carry a ship id"
        assert "masks" not in emitted.meta, "the ship segmenter left masks on a person"
        assert emitted.meta["class"] == "person"
        assert emitted.meta["vectors"], "the person embedding must still reach the sink"

    def test_a_ship_detection_runs_the_ship_branch_and_not_the_person_one(self) -> None:
        """The mirror: guarding the branch must not have turned it off.

        Without this half, the test above would also pass on a chain whose ``when:`` clauses
        are simply always false -- the ship models never running is the other way to have no
        ship identities.
        """
        chain = self.load_with_mocks(detect_class="ship")

        emitted = walk(chain, item("cam-3", 12))

        for ship_only in ("segment", "embed_ship", "recognize"):
            assert chain.node(ship_only).element.processes == 1, f"{ship_only} was skipped"
        assert chain.node("embed_person").element.processes == 0, "the person embedder ran"
        assert emitted.meta["identities"] == ["ship-1"]
        assert emitted.meta["masks"], "the segmenter's masks must reach the sink"

    def test_the_inline_fixture_agrees_with_the_file(self) -> None:
        """``MOCK_CHAIN`` resolves to exactly the wiring the production file resolves to.

        The fixture's docstring says it is the production chain with mocks; this is what
        makes that sentence true. The drift worth catching is the fixture being *more*
        correct than the file it claims to copy — which is what happened, and is why
        ``MOCK_CHAIN`` carried two ``after:`` clauses the file did not.

        Compared after validation rather than as text, because the default-predecessor rule
        means most of the wiring is not written in either place.
        """
        fixture = load(MOCK_CHAIN)
        production = self.load_with_mocks()

        assert [node.name for node in fixture] == [node.name for node in production]
        assert [str(edge) for edge in fixture.edges] == [str(edge) for edge in production.edges]
        assert [str(node.condition) for node in fixture if node.condition] == [
            str(node.condition) for node in production if node.condition
        ]
