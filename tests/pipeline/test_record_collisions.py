"""What happens when two batches cover one detection, and the goldens for when none do.

`records.h` justified having no class check with "a batch only ever holds rows of ITS OWN
class" -- true while every crop slot declared `classes:`, and a resolved chain plan can carry
a slot with none, which is every row. `build_records` then OVERWROTE, so the last batch to
mention a row set the field.

The answer is the one the chain plane already had: a typed refusal. `PoolEmbed._scatter` and
`ChainWalk.inbound` raise on exactly this state, so `build_records` does too -- on both
planes, and asserted here, in `csrc/tests/test_record_parity.cpp`, and by a scenario that
deliberately has no golden.
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
from shipinfer.core.errors import InferenceError
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.graph.state import build_records
from shipinfer.topology.elements.detections import Detections

#: Scenarios with a golden, byte-compared by both planes.
GOLDEN_NAMES = ("scattered_frame",)
#: Scenarios with NO golden: a frame both planes must REFUSE, which is the half a byte
#: compare cannot express.
REFUSED_NAMES = ("contested_row_refused",)
SCENARIO_NAMES = GOLDEN_NAMES + REFUSED_NAMES
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


class TestAContestedRowIsRefused:
    """Two batches covering one detection is a chain-file error, not a tie to break.

    The chain plane decided this before this seam existed: `PoolEmbed._scatter` and
    `ChainWalk.inbound` raise when a second slot files a row an earlier one covered, and
    `tests/runners/test_walk.py` states why -- "there is no answer to 'which of these two
    vectors is this object's'. Silently keeping one would attach an appearance vector chosen
    by declaration order." `build_records` overwrote, then briefly took the first; it raises
    now, like its two neighbours and like the C++ builder.
    """

    def test_a_row_two_batches_cover_is_refused(self) -> None:
        with pytest.raises(InferenceError, match="two batches cover detection row 0"):
            build_records(
                "cam0",
                7,
                _detections(),
                {
                    "specific": _batch("specific", (0,), [[0.25, 0.5]]),
                    "every_row": _batch("every_row", (0, 1), [[0.75, 0.875], [2.0, 3.0]]),
                },
                {"embedding": ("specific", "every_row")},
            )

    def test_the_refusal_names_the_field_the_batch_and_the_fix(self) -> None:
        """An operator reading it has to find the two slots in the chain file."""
        with pytest.raises(InferenceError, match=r"'embedding'") as raised:
            build_records(
                "cam0",
                7,
                _detections(),
                {"a": _batch("a", (0,), [[1.0]]), "b": _batch("b", (0,), [[2.0]])},
                {"embedding": ("a", "b")},
            )

        assert "'b'" in str(raised.value), "the batch that hit the row"
        assert "classes" in str(raised.value), "and `params: classes:`, which is the fix"

    def test_two_batches_covering_DIFFERENT_rows_is_the_ordinary_case(self) -> None:
        """The refusal is about the ROW. Two candidates existing is normal -- that is how a
        ship's embedding comes from the ship embedder and a person's from the person one."""
        records = build_records(
            "cam0",
            7,
            _detections(),
            {
                "ships": _batch("ships", (0,), [[0.25, 0.5]]),
                "people": _batch("people", (1,), [[2.0, 3.0]]),
            },
            {"embedding": ("ships", "people")},
        )

        assert records[0].embedding == (0.25, 0.5)
        assert records[1].embedding == (2.0, 3.0)

    def test_an_empty_batch_does_not_claim_a_row(self) -> None:
        records = build_records(
            "cam0",
            7,
            _detections(),
            {"empty": _batch("empty", (), []), "real": _batch("real", (0,), [[1.0, 2.0]])},
            {"embedding": ("empty", "real")},
        )

        assert records[0].embedding == (1.0, 2.0)

    @pytest.mark.parametrize("name", REFUSED_NAMES)
    def test_the_scenario_written_for_it_is_refused(self, name: str) -> None:
        """The same scenario the C++ gate refuses, so the planes agree on the refusal and
        not only on the successful bytes."""
        scenario = load(name)

        with pytest.raises(InferenceError, match="two batches cover"):
            build_records(
                scenario.camera,
                scenario.frame,
                detections_of(scenario),
                batches_of(scenario),
                dict(scenario.fields),
            )


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

    @pytest.mark.parametrize("name", GOLDEN_NAMES)
    def test_the_golden_is_reproduced_exactly(self, name: str) -> None:
        expected = (GOLDEN / f"{name}.jsonl").read_text(encoding="ascii").strip()

        assert render(load(name)) == expected, (
            f"the event this plane builds for {name} is not the committed golden. If the "
            f"change to the builder IS the decision, re-emit with `python "
            f"scripts/emit_parity_golden.py --kind record --scenario {name} --emit-golden "
            f"--force` and say so in the PR"
        )

    @pytest.mark.parametrize("name", GOLDEN_NAMES)
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
        gate = CPP_GATE.read_text()
        # A missing list has to be a readable failure, not a `NoneType` subscript.
        found = {
            name: re.search(rf"{name} = \{{([^}}]*)\}}", gate)
            for name in ("kScenarios", "kRefused")
        }
        missing = sorted(name for name, match in found.items() if match is None)
        assert not missing, f"the C++ gate has no {missing} list any more; this scan needs it"
        in_cpp = {
            name: set(re.findall(r'"(\w+)"', match[1]))  # type: ignore[index]
            for name, match in found.items()
        }

        assert in_cpp["kScenarios"] == set(GOLDEN_NAMES), "the byte-compared scenarios"
        assert in_cpp["kRefused"] == set(REFUSED_NAMES), "and the refused ones"
        assert {path.stem for path in SCENARIOS.glob("*.scn")} == set(SCENARIO_NAMES)
        assert {path.stem for path in GOLDEN.glob("*.jsonl")} == set(
            GOLDEN_NAMES
        ), "every golden has a scenario, and a REFUSED scenario has none by design"

    def test_the_refused_scenario_has_no_golden(self) -> None:
        """Deliberate, and asserted so a later `--emit-golden --force` cannot quietly turn
        the refusal into a published event."""
        for name in REFUSED_NAMES:
            assert not (GOLDEN / f"{name}.jsonl").exists(), (
                f"{name} describes a frame both planes must REFUSE; a golden for it would "
                f"mean one of them published it"
            )
