"""Which batch fills a row two of them mention, and the goldens that pin it on both planes.

`build_records` takes its candidates "in priority order" and did not: it overwrote, so the
LAST batch to mention a row set the field. `records.h` justified having no class check with
"a batch only ever holds rows of ITS OWN class" -- true while every crop slot declared
`classes:`, and a resolved chain plan can now carry a slot with none, which means every row.

So the rule is FIRST candidate wins, on both planes, and it is asserted three ways: here, in
`csrc/tests/test_record_parity.cpp`, and by the committed goldens both build.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from benchmarks.parity.drive_records import (
    GOLDEN,
    SCENARIOS,
    batches_of,
    detections_of,
    load,
    render,
)
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.graph.state import build_records
from shipinfer.topology.elements.detections import Detections

SCENARIO_NAMES = ("first_candidate_wins", "scattered_frame")
CPP_GATE = Path(__file__).resolve().parents[2] / "csrc" / "tests" / "test_record_parity.cpp"


def _detections(count: int = 2) -> Detections:
    return Detections(
        boxes=np.zeros((count, 4), dtype=np.float32),
        scores=np.full(count, 0.5, dtype=np.float32),
        class_ids=np.zeros(count, dtype=np.int32),
        labels=("person",) * count,
    )


def _batch(name: str, indices: tuple[int, ...], rows: list[list[float]]) -> ObjectBatch:
    width = len(rows[0]) if rows else 0
    return ObjectBatch(
        name=name,
        class_name="",
        object_indices=indices,
        data=np.array(rows, dtype=np.float32).reshape(len(rows), width),
    )


class TestTheFirstCandidateWins:
    def test_a_contested_row_takes_the_first_batch(self) -> None:
        records = build_records(
            "cam0",
            7,
            _detections(),
            {
                "specific": _batch("specific", (0,), [[0.25, 0.5]]),
                "every_row": _batch("every_row", (0, 1), [[0.75, 0.875], [2.0, 3.0]]),
            },
            {"embedding": ("specific", "every_row")},
        )

        assert records[0].embedding == (0.25, 0.5), "the first candidate, not the last"

    def test_a_row_only_the_later_candidate_covers_is_still_filled(self) -> None:
        """First-wins is per ROW, not per batch: the second candidate is not skipped."""
        records = build_records(
            "cam0",
            7,
            _detections(),
            {
                "specific": _batch("specific", (0,), [[0.25, 0.5]]),
                "every_row": _batch("every_row", (0, 1), [[0.75, 0.875], [2.0, 3.0]]),
            },
            {"embedding": ("specific", "every_row")},
        )

        assert records[1].embedding == (2.0, 3.0)

    def test_reversing_the_order_reverses_the_answer(self) -> None:
        """The rule is the DECLARED order, which is what makes it a decision rather than
        an accident of which batch happens to be iterated last."""
        batches = {
            "specific": _batch("specific", (0,), [[0.25, 0.5]]),
            "every_row": _batch("every_row", (0, 1), [[0.75, 0.875], [2.0, 3.0]]),
        }
        records = build_records(
            "cam0", 7, _detections(), batches, {"embedding": ("every_row", "specific")}
        )

        assert records[0].embedding == (0.75, 0.875)

    def test_an_empty_batch_is_skipped_rather_than_claiming_the_row(self) -> None:
        records = build_records(
            "cam0",
            7,
            _detections(),
            {
                "empty": _batch("empty", (), []),
                "real": _batch("real", (0,), [[1.0, 2.0]]),
            },
            {"embedding": ("empty", "real")},
        )

        assert records[0].embedding == (1.0, 2.0), "an empty first candidate claims nothing"


class TestTheEdgesOfTheScatter:
    """What the byte compare cannot say, asserted on this plane as the C++ gate asserts it.

    An `unknown` record reaches neither `*_vec` -- `as_det2mot` partitions by `person` and
    `ship` -- so the golden is silent about the label itself. It catches a RELABEL, which is
    the expensive half, and this catches the rest.
    """

    def test_a_class_the_table_does_not_name_is_unknown(self) -> None:
        scenario = load("scattered_frame")
        records = build_records(
            scenario.camera,
            scenario.frame,
            detections_of(scenario),
            batches_of(scenario),
            dict(scenario.fields),
        )

        assert [record.class_name for record in records] == ["ship", "person", "unknown"]
        assert records[0].det_id == "cau-01_41_0", "`<camera>_<frame>_<index>`"

    def test_a_row_past_the_detection_list_fills_nothing(self) -> None:
        """`ship_embed_out` carries a row for index 9 and there are three detections."""
        scenario = load("scattered_frame")
        records = build_records(
            scenario.camera,
            scenario.frame,
            detections_of(scenario),
            batches_of(scenario),
            dict(scenario.fields),
        )

        assert records[0].embedding == (0.5, 0.875), "index 0 is filled"
        assert all(not record.embedding for record in records[1:]), "and nothing else is"


class TestTheCommittedGoldensAreWhatThisPlaneBuilds:
    """The Python half of the byte compare, so a divergence names the plane that moved.

    The same shape the chain-plan seam has: the C++ gate failing tells you the two planes
    disagree, and this tells you which one changed.
    """

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_the_golden_is_reproduced_exactly(self, name: str) -> None:
        expected = (GOLDEN / f"{name}.jsonl").read_text(encoding="ascii").strip()

        assert render(load(name)) == expected, (
            f"the event this plane builds for {name} is not the committed golden. If the "
            f"change to the builder IS the decision, re-emit with `python "
            f"scripts/emit_parity_golden.py --kind record --scenario {name} --emit-golden "
            f"--force` and say so in the PR"
        )

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_the_scenario_and_its_golden_both_exist(self, name: str) -> None:
        """Non-vacuity: a missing pair would make the check above pass by not running."""
        assert (SCENARIOS / f"{name}.scn").is_file()
        assert (GOLDEN / f"{name}.jsonl").is_file()

    def test_every_scenario_on_disk_is_in_both_gates(self) -> None:
        """A scenario nothing runs is a golden nobody compares.

        The event seam already asserts this (`benchmarks/tests/test_parity_events.py`), and
        the reason is that the list is hard-coded twice -- here and in the C++ gate -- so a
        third scenario plus its golden can be added and cross-plane compared by neither.
        """
        on_disk = {path.stem for path in SCENARIOS.glob("*.scn")}
        in_cpp = set(
            re.findall(
                r'"(\w+)"', re.search(r"kScenarios = \{([^}]*)\}", CPP_GATE.read_text())[1]
            )
        )

        assert on_disk == set(SCENARIO_NAMES) == in_cpp
        assert {
            path.stem for path in GOLDEN.glob("*.jsonl")
        } == on_disk, "every scenario has a golden and every golden has a scenario"

    def test_the_contested_row_is_visible_in_the_golden(self) -> None:
        """The golden alone cannot say WHICH batch won -- unless the two batches carry
        different numbers, which is what the scenario is built to do."""
        line = (GOLDEN / "first_candidate_wins.jsonl").read_text(encoding="ascii")

        assert '"ship_feature_vec":[[0.25,0.5]]' in line, "the first candidate's values"
        # Not `0.875` alone: that is also the ship's SCORE in this scenario. The whole
        # vector, which only the second candidate could have produced.
        assert "[[0.75,0.875]]" not in line, "and not the second candidate's"
