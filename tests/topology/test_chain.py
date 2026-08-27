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

import textwrap
from pathlib import Path

import pytest

from shipinfer.core.errors import (
    CapsMismatchError,
    ChainCycleError,
    ChainSpecError,
    ChainStructureError,
    ConditionSyntaxError,
    ConfigurationError,
    ServerStateError,
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
from shipinfer.topology.elements.mock import MockDetect

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The straight-line chain of arch.md §1 with every implementation replaced by its mock.
#: Written out rather than generated: a chain file is the thing under test, and a generated
#: one would test the generator.
MOCK_CHAIN = textwrap.dedent("""
    name: mock_ship_person
    elements:
      decode:       {impl: mock}
      detect:       {impl: mock, model: ship_detector}
      segment:      {impl: mock, model: ship_segmenter, when: class == ship}
      embed_ship:   {impl: mock, model: ship_embedder, after: segment}
      embed_person: {impl: mock, model: person_embedder, when: class == person, after: detect}
      recognize:    {impl: mock, model: ship_recognizer, after: embed_ship}
      track:        {impl: mock, per: camera, after: [recognize, embed_person]}
      mtmc:         {impl: mock, scope: global}
      output:       {impl: mock}
    """)


def load(text: str) -> Topology:
    """Build a topology from inline YAML, the way a chain file would be loaded."""
    return Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(text)))


def item(camera: str = "cam-1", frame: int = 7) -> ChainItem:
    return ChainItem(RequestContext(camera_id=camera, frame_id=frame), Caps.parse("nv12@gpu"))


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

    def test_the_runner_can_still_read_what_the_file_said(self) -> None:
        """``per``/``scope``/``params`` are the runner's business, so the spec is kept."""
        chain = load(MOCK_CHAIN)

        assert chain.node("track").spec.per == "camera"
        assert chain.node("mtmc").spec.scope == "global"

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

    def test_a_cycle_is_refused_and_names_its_members(self) -> None:
        with pytest.raises(ChainCycleError) as caught:
            load("""
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock, model: d, after: track}
                  track:  {impl: mock, after: detect}
                  output: {impl: mock}
                """)

        assert set(caught.value.cycle) >= {"detect", "track"}

    def test_an_element_that_depends_on_itself_is_refused(self) -> None:
        with pytest.raises(ChainCycleError):
            load("""
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock, model: d, after: detect}
                  output: {impl: mock}
                """)

    @pytest.mark.parametrize(
        ("chain", "why"),
        [
            (
                """
                elements:
                  decode: {impl: mock}
                """,
                "a decoder alone emits nothing",
            ),
            (
                """
                elements:
                  decode: {impl: mock}
                  detect: {impl: mock, model: d}
                """,
                "every model runs and nothing is emitted",
            ),
            (
                """
                elements:
                  detect: {impl: mock, model: d, after: []}
                  output: {impl: mock}
                """,
                "a chain must start at a decoder",
            ),
        ],
    )
    def test_a_chain_needs_a_decode_root_and_an_output_sink(self, chain: str, why: str) -> None:
        with pytest.raises(ChainStructureError):
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


class TestTheProductionChainFile:
    """``topology/ship_person.yaml`` is arch.md §1's chain, kept as a fixture until it runs."""

    PATH = REPO_ROOT / "topology" / "ship_person.yaml"

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
