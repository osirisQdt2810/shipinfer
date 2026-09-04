"""The resolved plan: what crosses to the C++ plane, and what both planes refuse.

The C++ half is `csrc/tests/test_plan_parity.cpp`, which reads the same committed goldens
and writes them back byte-identically. This half holds the *emitter* to those bytes -- so a
change to either writer names the plane that moved, rather than only failing the gate.

`TestBothPlanesRefuseTheSameText` is the table that matters most: a plan one plane reads and
the other rejects is the worst outcome this seam has, and no golden can express it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.repository import ModelRepository
from shipinfer.repository.extents import model_extents
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
    "ship_person_cpu": ROOT / "topology" / "ship_person_cpu.yaml",
}


@pytest.fixture(scope="module")
def dims() -> dict[str, tuple[int, int]]:
    return model_extents(ModelRepository.load(REPOSITORY))


def _plan(name: str, dims: dict[str, tuple[int, int]]) -> ResolvedPlan:
    return resolve_plan(load_topology(CHAINS[name]), dims=dims)


class TestTheCommittedGoldensAreWhatThisPlaneEmits:
    """The Python half of the byte compare, which no other seam has yet.

    The C++ gate failing tells you the two planes disagree; it does not tell you which one
    moved. This does, and it costs one chain load.
    """

    @pytest.mark.parametrize("name", sorted(CHAINS))
    def test_the_golden_is_reproduced_exactly(
        self, name: str, dims: dict[str, tuple[int, int]]
    ) -> None:
        expected = (GOLDEN / f"{name}.plan").read_text(encoding="utf-8")

        assert plan_text(_plan(name, dims)) == expected, (
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
        assert plan.node("segment").classes == ()

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
    )
    ACCEPTED = (
        ("plan 1 -\n", "`-` is the empty chain name"),
        ("plan 1 x  # trailing comment\n", "a comment after a directive"),
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
