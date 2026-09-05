"""The resolved plan: what crosses to the C++ plane, and what both planes refuse.

The C++ half is `csrc/tests/test_plan_parity.cpp`, which reads the same committed goldens
and writes them back byte-identically. This half holds the *emitter* to those bytes -- so a
change to either writer names the plane that moved, rather than only failing the gate.

`TestBothPlanesRefuseTheSameText` is the table that matters most: a plan one plane reads and
the other rejects is the worst outcome this seam has, and no golden can express it.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.repository import ModelRepository
from shipinfer.repository.resolved import ModelRuntime, model_extents, model_runtimes
from shipinfer.topology import load_topology
from shipinfer.topology.plan import (
    PLAN_VERSION,
    PlanNode,
    PlanSyntaxError,
    ResolvedPlan,
    parse_plan,
    plan_text,
    resolve_plan,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "benchmarks" / "parity" / "golden" / "plans"
SCENARIOS = ROOT / "benchmarks" / "parity" / "scenarios" / "plans"
REPOSITORY = ROOT / "model_repository"

#: Each committed golden and the chain it was emitted from.
CHAINS = {
    "minimal": SCENARIOS / "minimal.yaml",
    "branching": SCENARIOS / "branching.yaml",
    "defaults": SCENARIOS / "defaults.yaml",
    "ship_person_cpu": ROOT / "topology" / "ship_person_cpu.yaml",
}


@pytest.fixture(scope="module")
def dims() -> dict[str, tuple[int, int]]:
    return model_extents(ModelRepository.load(REPOSITORY))


@pytest.fixture(scope="module")
def runtimes() -> dict[str, ModelRuntime]:
    return model_runtimes(ModelRepository.load(REPOSITORY))


def _plan(name: str, dims: dict[str, tuple[int, int]]) -> ResolvedPlan:
    """The shape of the chain alone — no runtimes, which is what most checks here are about."""
    return resolve_plan(load_topology(CHAINS[name]), dims=dims)


def _resolved(name: str) -> ResolvedPlan:
    """A plan resolved exactly as `benchmarks/parity/drive_plan.py` resolves it.

    Separate from `_plan` because the two halves of a byte compare must resolve the SAME way:
    a golden emitted with the runtimes and a test comparing one without them is a golden
    nothing checks.
    """
    models = ModelRepository.load(REPOSITORY)
    return resolve_plan(
        load_topology(CHAINS[name]),
        dims=model_extents(models),
        runtimes=model_runtimes(models),
    )


class TestTheCommittedGoldensAreWhatThisPlaneEmits:
    """The Python half of the byte compare, which no other seam has yet.

    The C++ gate failing tells you the two planes disagree; it does not tell you which one
    moved. This does, and it costs one chain load.
    """

    @pytest.mark.parametrize("name", sorted(CHAINS))
    def test_the_golden_is_reproduced_exactly(self, name: str) -> None:
        expected = (GOLDEN / f"{name}.plan").read_text(encoding="utf-8")

        assert plan_text(_resolved(name)) == expected, (
            f"the plan this code emits for {CHAINS[name].name} is not the committed golden. "
            f"If the change to the writer IS the decision, re-emit with "
            f"`python scripts/emit_parity_golden.py --kind plan --scenario {name} "
            f"--emit-golden --force` and say so in the PR"
        )

    @pytest.mark.parametrize("name", sorted(CHAINS))
    def test_the_golden_round_trips_through_the_reader(self, name: str) -> None:
        text = (GOLDEN / f"{name}.plan").read_text(encoding="utf-8")

        assert plan_text(parse_plan(text)) == text


class TestWhatTheChainResolvesTo:
    def test_the_label_table_comes_from_the_detector_slot(
        self, dims: dict[str, tuple[int, int]]
    ) -> None:
        """A ship is class **8** in this checkout, and that is why the plan carries a table.

        `cli/bench.cpp` hard-coded `{{0, "person"}, {1, "ship"}}` beside crop specs that used
        8, so it cropped the right rows and labelled every ship `unknown` in its events.
        """
        assert _plan("branching", dims).labels == {0: "person", 8: "ship"}

    def test_a_chain_with_no_detector_has_no_labels(
        self, dims: dict[str, tuple[int, int]]
    ) -> None:
        plan = _plan("minimal", dims)

        assert plan.labels == {} and plan.fields == {}
        assert [node.slot for node in plan.nodes] == ["decode", "output"]

    def test_a_declared_crop_wins_over_the_models(
        self, dims: dict[str, tuple[int, int]]
    ) -> None:
        """`params: {crop: {size: [512, 512]}}` on a segmenter whose config says 640."""
        plan = _plan("branching", dims)

        assert plan.node("segment").crop == (512, 512)
        assert dims["ship_segmenter"] == (640, 640), "the model really does say otherwise"

    def test_geometry_the_chain_omits_comes_from_the_repository(
        self, dims: dict[str, tuple[int, int]]
    ) -> None:
        plan = _plan("branching", dims)

        assert plan.node("embed_ship").crop == (256, 128), "tall, not square"
        assert plan.node("detect").letterbox == (640, 640)

    def test_the_runner_hints_and_the_frame_guard_are_carried(
        self, dims: dict[str, tuple[int, int]]
    ) -> None:
        plan = _plan("branching", dims)

        assert plan.node("track").per == "camera"
        assert plan.node("mtmc").scope == "global"
        assert plan.node("segment").when == "fps == 20"

    def test_row_selection_is_carried_per_slot(self, dims: dict[str, tuple[int, int]]) -> None:
        plan = _plan("branching", dims)

        assert plan.node("embed_ship").classes == ("ship",)
        assert plan.node("embed_person").classes == ("person",)
        # The segmenter declares one too, since #132's review: a `segment` slot with no
        # selection is EVERY row on the C++ plane, which is the ship segmenter on every
        # person crop at 640x640 and a `mask_area_px` filed on every person record.
        assert plan.node("segment").classes == ("ship",)
        assert plan.node("decode").classes is None, "and a decode declares none at all"

    def test_each_event_field_names_the_slots_that_fill_it(
        self, dims: dict[str, tuple[int, int]]
    ) -> None:
        """The C++ builder then appends `_out`, which is how a batch is keyed there."""
        plan = _plan("branching", dims)

        assert plan.fields == {
            "embedding": ("embed_person", "embed_ship"),
            "mask_area_px": ("segment",),
        }

    def test_edges_carry_the_cap_the_loader_negotiated(
        self, dims: dict[str, tuple[int, int]]
    ) -> None:
        plan = _plan("branching", dims)

        assert ("decode", "detect", "bgr@cpu") in plan.edges
        assert ("track", "mtmc", "meta@cpu") in plan.edges

    def test_a_model_extent_nobody_states_is_refused_by_name(self) -> None:
        """Refused, not defaulted: a crop at the wrong extent is answered with vectors that
        are wrong for every object on every camera, and nothing reports it."""
        chain = load_topology(CHAINS["branching"])

        with pytest.raises(ConfigurationError, match="cannot tell how big"):
            resolve_plan(chain, dims={})


class TestBothPlanesRefuseTheSameText:
    """The same table as `refuses_what_python_refuses()` in the C++ gate, line for line."""

    REFUSED = (
        ("", "no header at all"),
        ("node a decode replay\n", "a verb before the header"),
        ("plan 1 x\nplan 1 y\n", "a second header"),
        ("plan 2 x\n", "an unknown version"),
        ("plan 1 x\nmodel m\n", "an attribute before any node"),
        ("plan 1 x\nnode a b\n", "node with two arguments"),
        ("plan 1 x\nnode a b c\ncrop 256\n", "crop with one extent"),
        ("plan 1 x\nnode a b c\ncrop 0 128\n", "a crop that is not positive"),
        ("plan 1 x\nnode a b c\nscore nan\n", "a non-finite score"),
        ("plan 1 x\nnode a b c\nnonsense 1\n", "an unknown verb"),
        ("plan 1 x\nlabel eight ship\n", "a label id that is not an integer"),
        ("plan 1 x\nedge a b\n", "an edge with no cap"),
        ("plan 1 x\nfield embedding\n", "a field with no slot"),
        ("plan 1_0 x\n", "an integer `int()` accepts and `std::stoi` does not"),
        ("plan 1 x\nnode a b c\nnode a b c\n", "a second block for one slot"),
        ("plan 1 x\nlabel 8 ship\nlabel 8 vessel\n", "a second table row for one id"),
        ("plan 1 x\nnode a b c\nclasses ,ship\n", "an empty label in `classes`"),
        ("plan 1 x\nnode a b c\nscore inf\n", "`inf`, which the writer must never emit"),
        ("plan 1 x\nnode a b c\nclasses ship,\n", "a trailing comma, so an empty label"),
        ("plan 1 x\nfield embedding a\nfield embedding b\n", "a second `field` for one name"),
        ("plan 1 x\nfield embedding nosuch\n", "a `field` naming a slot no `node` declares"),
        ("plan 1 x\nnode a b c\nscore 0x10\n", "a hex float, which `stod` would accept"),
        ("plan 1 x\nlabel 99999999999999999999 ship\n", "an id too large for a C++ int"),
        ("plan 1 x\nnode a b c\nmax_detections -1\n", "`-1` for `no limit`, which is no cap"),
        ("plan 1 x\nnode a b c\nmax_detections 0\n", "and zero, for the same reason"),
        ("plan 1 x\nnode a b c\nscore 1e400\n", "an overflowing exponent, which is `inf`"),
        ("plan 1 x\nnode a b c\ninstances 0\n", "zero instances, which runs nothing"),
        ("plan 1 x\nnode a b c\ninstances -1\n", "and a negative count"),
        ("plan 1 x\nnode a b c\nqueue_delay_us -1\n", "a negative batch window"),
        ("plan 1 x\nnode a b c\ninstances two\n", "an instance count that is not a number"),
        ("plan 1 x\nnode a b c\nartefact a b\n", "an artefact path holding a space"),
    )
    ACCEPTED = (
        ("plan 1 -\n", "`-` is the empty chain name"),
        ("plan 1 x  # trailing comment\n", "a comment after a directive"),
        ("plan 1 ship person cpu\n", "a multi-word chain name is the rest of the line"),
        ("plan 1 x\nlabel 8 cargo ship\n", "and so is a multi-word label"),
        ("plan 1 x\nnode a b c\nclasses cargo ship,fishing vessel\n", "two labels, not four"),
        ("plan 1 x\nnode a b c\nclasses -\n", "`-` is a DECLARED empty selection"),
        ("plan 1 x\nnode a b c\nscore 1e-05\n", "an exponent-form threshold"),
        ("plan 1 x\nnode a b c\nscore 5e-324\n", "a subnormal, which `stod` used to refuse"),
        # The READER tolerates a collapsed value because it cannot know one was collapsed;
        # `_speakable` is what refuses to WRITE one, where the slot is still named. Stated
        # here so the asymmetry is deliberate rather than a hole -- and the C++ gate asserts
        # the same collapse on its side, for the same reason.
        ("plan 1 x\nnode a b c\nclasses cargo  ship\n", "a collapsed value reads as one word"),
        ("plan 1 x\nlabel 8 cargo\tship\n", "and a tab likewise, which the emitter refuses"),
        ("plan 1 x\nnode a b c\nqueue_delay_us 0\n", "no batch window, which is batching off"),
        ("plan 1 x\nnode a b c\ninstances 1\n", "the smallest count a slot can run"),
        ("plan 1 x\nnode a b c\nartefact m/1/model.plan\n", "a repository-relative artefact"),
    )

    @pytest.mark.parametrize("text,why", REFUSED, ids=[why for _, why in REFUSED])
    def test_refused(self, text: str, why: str) -> None:
        with pytest.raises(PlanSyntaxError):
            parse_plan(text)

    @pytest.mark.parametrize("text,why", ACCEPTED, ids=[why for _, why in ACCEPTED])
    def test_accepted(self, text: str, why: str) -> None:
        assert parse_plan(text).version == PLAN_VERSION

    def test_a_crop_that_is_not_positive_names_the_line(self) -> None:
        with pytest.raises(PlanSyntaxError, match=r"<string>:3"):
            parse_plan("plan 1 x\nnode a b c\ncrop 0 128\n")


class TestTheWriterIsDeterministic:
    def test_a_hand_built_plan_round_trips(self) -> None:
        plan = ResolvedPlan(
            name="",
            nodes=(
                PlanNode("d", "decode", "replay"),
                PlanNode(
                    "e",
                    "embed",
                    "pool",
                    model="m",
                    classes=("ship", "person"),
                    crop=(256, 128),
                    when="fps == 20",
                    per="camera",
                    scope="shard",
                    score_threshold=0.5,
                    max_detections=7,
                ),
            ),
            edges=(("d", "e", "bgr@cpu"),),
            labels={8: "ship"},
            fields={"embedding": ("e",)},
        )
        text = plan_text(plan)

        assert parse_plan(text) == plan, "the writer and the reader are inverses"
        assert plan_text(parse_plan(text)) == text


class TestTheFormatRefusesWhatItCannotCarry:
    """A value holding one of the format's own delimiters, refused where the slot is known.

    The golden cannot catch this: `classes cargo ship` is STABLE text -- write, read, write,
    identical -- and only its meaning changed, from one label to two. Multi-word labels are
    normal here (COCO's `traffic light`, a `cargo ship`), so the format carries them and
    refuses commas and `#` instead.
    """

    @staticmethod
    def _chain(tmp_path: Path, body: str) -> Path:
        path = tmp_path / "chain.yaml"
        path.write_text(textwrap.dedent(body))
        return path

    #: A detector whose own table names the multi-word labels below, because
    #: `_check_declared_classes` refuses a `classes:` naming a label nobody detects -- which
    #: is the loader working, and the reason these chains declare both halves.
    HEAD = """
        name: guarded
        elements:
          decode: {impl: replay}
          detect:
            impl: pool
            model: ship_detector
            params: {decode: {class_labels: {8: cargo ship, 9: fishing vessel}}}
    """

    def test_a_multi_word_class_survives_the_round_trip(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        """The case the review reproduced: space-delimited, it came back as two labels."""
        chain = self._chain(
            tmp_path,
            self.HEAD + """      embed_ship:
            impl: pool
            model: ship_embedder
            params: {classes: [cargo ship, fishing vessel]}
          output: {impl: jsonlines}
        """,
        )
        text = plan_text(resolve_plan(load_topology(chain), dims=dims))

        assert "classes cargo ship,fishing vessel" in text
        assert parse_plan(text).node("embed_ship").classes == ("cargo ship", "fishing vessel")

    def test_a_multi_word_label_and_chain_name_survive(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        chain = self._chain(
            tmp_path,
            """
            name: ship person cpu
            elements:
              decode: {impl: replay}
              detect:
                impl: pool
                model: ship_detector
                params: {decode: {class_labels: {0: person, 8: cargo ship}}}
              output: {impl: jsonlines}
            """,
        )
        text = plan_text(resolve_plan(load_topology(chain), dims=dims))
        back = parse_plan(text)

        assert back.name == "ship person cpu"
        assert back.labels == {0: "person", 8: "cargo ship"}

    @pytest.mark.parametrize(
        "params,what",
        [
            ("a,b", "a comma in a label, which is the `classes` delimiter"),
            ("a#b", "a `#` in a label, which starts a comment"),
        ],
        ids=["comma", "hash"],
    )
    def test_a_delimiter_inside_a_value_is_refused_by_name(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]], params: str, what: str
    ) -> None:
        chain = self._chain(
            tmp_path,
            f"""
            name: hostile
            elements:
              decode: {{impl: replay}}
              detect:
                impl: pool
                model: ship_detector
                params: {{decode: {{class_labels: {{0: "{params}"}}}}}}
              embed_ship:
                impl: pool
                model: ship_embedder
                params: {{classes: ["{params}"]}}
              output: {{impl: jsonlines}}
            """,
        )

        with pytest.raises(ConfigurationError, match="cannot be written to a plan"):
            resolve_plan(load_topology(chain), dims=dims)

    @pytest.mark.parametrize(
        "label,what",
        [
            ("", "an empty label, which emits `label 8 ` and reads as one argument"),
            ("ship ", "a padded label, which reads back as different text"),
            ("cargo  ship", "a repeated space, which `split()` collapses"),
            ("cargo\tship", "a tab, likewise"),
            ("ship\nnode injected detect pool", "a newline, which injects a NODE"),
        ],
        ids=["empty", "padded", "double-space", "tab", "newline"],
    )
    def test_an_empty_or_padded_value_is_refused(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]], label: str, what: str
    ) -> None:
        # `json.dumps`, not an f-string in quotes: a RAW newline inside a double-quoted YAML
        # scalar is folded to a space, so the test would pass on a value the guard allows.
        # JSON escaping is valid YAML double-quoted style, and `\n` survives as a newline.
        quoted = json.dumps(label)
        chain = self._chain(
            tmp_path,
            f"""
            name: padded
            elements:
              decode: {{impl: replay}}
              detect:
                impl: pool
                model: ship_detector
                params: {{decode: {{class_labels: {{8: {quoted}}}}}}}
              output: {{impl: jsonlines}}
            """,
        )

        with pytest.raises(ConfigurationError, match="cannot be written to a plan"):
            resolve_plan(load_topology(chain), dims=dims)

    def test_a_chain_called_dash_is_refused(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        """`-` is the writer's spelling for an unnamed chain, so it cannot also be a name."""
        chain = self._chain(
            tmp_path,
            """
            name: "-"
            elements:
              decode: {impl: replay}
              output: {impl: jsonlines}
            """,
        )

        with pytest.raises(ConfigurationError, match="unnamed chain"):
            resolve_plan(load_topology(chain), dims=dims)

    def test_a_non_positive_cap_is_refused_where_the_slot_is_named(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        """Nothing between a chain file and the decode loop refused `-1`: not the element's
        bare `int()`, not the unvalidated `DecodeParams`. So `resolve_plan` does, and the C++
        `detect_config` does -- because `static_cast<size_t>(-1)` is no bound at all there
        while `keep[: -1]` drops one row here."""
        chain = self._chain(
            tmp_path,
            """
            name: uncapped
            elements:
              decode: {impl: replay}
              detect:
                impl: pool
                model: ship_detector
                params: {decode: {max_detections: -1}}
              output: {impl: jsonlines}
            """,
        )

        with pytest.raises(ConfigurationError, match="a positive count"):
            resolve_plan(load_topology(chain), dims=dims)

    def test_a_non_finite_threshold_is_refused_rather_than_written(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        """`plan_text` used to emit `score inf`, which `parse_plan` then refused -- a plan
        `shipinfer plan` wrote and neither plane could read."""
        chain = self._chain(
            tmp_path,
            """
            name: infinite
            elements:
              decode: {impl: replay}
              detect:
                impl: pool
                model: ship_detector
                params: {decode: {score_threshold: .inf}}
              output: {impl: jsonlines}
            """,
        )

        with pytest.raises(ConfigurationError, match="finite numbers only"):
            resolve_plan(load_topology(chain), dims=dims)


class TestEveryDetectorsLabelsAreCarried:
    """Two detectors is a supported chain shape, and the plan carries ONE table.

    `_check_declared_classes` unions every detector's labels, so returning the first table
    dropped the second one's ids -- and the C++ event writer meeting an id its table does not
    know labels the row `unknown`, which is the defect ADR-020 cites as its motivation.
    """

    @staticmethod
    def _two(tmp_path: Path, second: str) -> Path:
        path = tmp_path / "two.yaml"
        path.write_text(textwrap.dedent(f"""
                name: two
                elements:
                  decode: {{impl: replay}}
                  detect:
                    impl: pool
                    model: ship_detector
                    params: {{decode: {{class_labels: {{8: ship}}}}}}
                  detect_person:
                    kind: detect
                    impl: pool
                    model: ship_detector
                    params: {{decode: {{class_labels: {second}}}}}
                    after: decode
                  output: {{impl: jsonlines, after: [detect, detect_person]}}
                """))
        return path

    def test_both_tables_are_merged(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        chain = self._two(tmp_path, "{0: person}")

        assert resolve_plan(load_topology(chain), dims=dims).labels == {0: "person", 8: "ship"}

    def test_two_detectors_disagreeing_on_an_id_is_refused_naming_both(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        chain = self._two(tmp_path, "{8: person}")

        with pytest.raises(ConfigurationError, match=r"'detect'.*'detect_person'"):
            resolve_plan(load_topology(chain), dims=dims)


class TestWhatTheChainWillDoAndNotWhatItWrote:
    """The plan carries EFFECTIVE decode settings, asked of the element.

    A detect slot that declares nothing still decodes with a table, a threshold and a cap
    (`elements/detections.py::DecodeParams`). A plan that omitted them made the other plane
    invent its own -- which is the hard-coded table ADR-020 exists to delete, so the omission
    undid the decision this format ships with.
    """

    SILENT = """
        name: silent
        elements:
          decode: {impl: replay}
          detect: {impl: pool, model: ship_detector}
          output: {impl: jsonlines}
    """

    def test_a_slot_that_declares_nothing_still_carries_its_defaults(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        path = tmp_path / "silent.yaml"
        path.write_text(textwrap.dedent(self.SILENT))
        plan = resolve_plan(load_topology(path), dims=dims)

        assert plan.labels == {
            0: "person",
            8: "ship",
        }, "the effective table, not the written one"
        assert plan.node("detect").score_threshold == 0.25
        assert plan.node("detect").max_detections == 100

    def test_the_element_is_the_source_and_not_the_params_dict(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        """`decode_parameters()` is the hook; re-reading `params` was a second interpretation
        of one setting, which `base.py` warns against for `detection_labels` already."""
        path = tmp_path / "silent.yaml"
        path.write_text(textwrap.dedent(self.SILENT))
        chain = load_topology(path)
        node = chain.node("detect")

        assert node.spec.params == {}, "the file really does say nothing"
        assert node.element.decode_parameters() is not None
        assert resolve_plan(chain, dims=dims).node("detect").score_threshold is not None

    def test_a_non_detect_slot_has_no_decode_settings(
        self, dims: dict[str, tuple[int, int]]
    ) -> None:
        plan = _plan("branching", dims)

        assert plan.node("embed_ship").score_threshold is None
        assert plan.node("embed_ship").max_detections is None

    def test_an_inverted_label_table_is_refused_and_typed(self, tmp_path: Path) -> None:
        """`{person: 0}` used to reach `int('person')` and raise a bare `ValueError` with no
        slot in it; the element's own resolution types it."""
        path = tmp_path / "inverted.yaml"
        path.write_text(textwrap.dedent("""
                name: inverted
                elements:
                  decode: {impl: replay}
                  detect:
                    impl: pool
                    model: ship_detector
                    params: {decode: {class_labels: {person: 0}}}
                  output: {impl: jsonlines}
                """))

        # At `resolve_plan`, because that is when the element resolves its decode params;
        # the loader builds the element without asking it to.
        with pytest.raises(ConfigurationError, match="detect element 'detect'"):
            resolve_plan(load_topology(path), dims={"ship_detector": (640, 640)})


class TestSelectNothingIsNotSelectEverything:
    """`classes: []` and no `classes:` at all used to emit byte-identical plans.

    `elements/detections.py` keeps them apart in a docstring naming this failure:
    "conflating the two would make a typo silently select everything -- at `track` a wrong
    answer, at an embedder a doubled GPU bill". The golden could not see it: both plans were
    the same text.
    """

    HEAD = """
        name: selective
        elements:
          decode: {impl: replay}
          detect:
            impl: pool
            model: ship_detector
            params: {decode: {class_labels: {0: person, 8: ship}}}
    """

    def _chain(self, tmp_path: Path, params: str) -> Path:
        path = tmp_path / "selective.yaml"
        path.write_text(
            textwrap.dedent(self.HEAD)
            + f"  embed_ship: {{impl: pool, model: ship_embedder{params}}}\n"
            + "  output: {impl: jsonlines}\n"
        )
        return path

    def test_a_declared_empty_selection_is_written_and_read_back(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        plan = resolve_plan(
            load_topology(self._chain(tmp_path, ", params: {classes: []}")), dims=dims
        )
        text = plan_text(plan)

        assert plan.node("embed_ship").classes == (), "declared, and empty"
        assert "classes -" in text
        assert parse_plan(text).node("embed_ship").classes == ()

    def test_no_selection_at_all_writes_no_line_and_reads_as_none(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        plan = resolve_plan(load_topology(self._chain(tmp_path, "")), dims=dims)
        text = plan_text(plan)

        assert plan.node("embed_ship").classes is None, "no selection declared"
        assert "classes" not in text
        assert parse_plan(text).node("embed_ship").classes is None

    def test_the_two_plans_are_not_the_same_text(
        self, tmp_path: Path, dims: dict[str, tuple[int, int]]
    ) -> None:
        """The whole point: they were identical, so nothing could tell them apart."""
        empty = plan_text(
            resolve_plan(
                load_topology(self._chain(tmp_path, ", params: {classes: []}")), dims=dims
            )
        )
        absent = plan_text(resolve_plan(load_topology(self._chain(tmp_path, "")), dims=dims))

        assert empty != absent
