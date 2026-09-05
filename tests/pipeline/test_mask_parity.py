"""The Python half of the mask byte compare, and the edges the golden cannot state.

The C++ gate failing says the two planes disagree about the fold; this says which one moved.
The seam exists because the RECORD gate cannot see a missing fold at all: its scenarios state
already-reduced `(N, 1)` rows, which is how the C++ plane went green for months while
publishing `output0[crop][0][0]` -- a box coordinate -- as `mask_area_px`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchmarks.parity.drive_masks import GOLDEN, SCENARIOS, load, render_masks
from benchmarks.parity.mask_scenario import arrays_of
from shipinfer.core.errors import InferenceError
from shipinfer.topology.elements.masks import InstanceMaskArea

#: Scenarios with a golden, byte-compared by both planes.
GOLDEN_NAMES = ("tiny_bank",)
#: Scenarios with NO golden: a response both planes must REFUSE.
REFUSED_NAMES = ("truncated_basis",)
SCENARIO_NAMES = GOLDEN_NAMES + REFUSED_NAMES
CPP_GATE = Path(__file__).resolve().parents[2] / "csrc" / "tests" / "test_mask_parity.cpp"


class TestTheCommittedGoldenIsWhatThisPlaneFolds:
    @pytest.mark.parametrize("name", GOLDEN_NAMES)
    def test_the_golden_is_reproduced_exactly(self, name: str) -> None:
        expected = (GOLDEN / f"{name}.txt").read_text(encoding="ascii")

        assert render_masks(load(name)) == expected, (
            f"the areas this plane folds for {name} are not the committed golden. If the "
            f"change to the fold IS the decision, re-emit with `python "
            f"scripts/emit_parity_golden.py --kind mask --scenario {name} --emit-golden "
            f"--force` and say so in the PR"
        )

    def test_every_scenario_on_disk_is_in_both_gates(self) -> None:
        """A scenario nothing runs is a golden nobody compares, and the list is hard-coded
        twice -- here and in the C++ gate -- so a third could be compared by neither."""
        gate = CPP_GATE.read_text()
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

        assert in_cpp["kScenarios"] == set(GOLDEN_NAMES)
        assert in_cpp["kRefused"] == set(REFUSED_NAMES)
        assert {path.stem for path in SCENARIOS.glob("*.scn")} == set(SCENARIO_NAMES)
        assert {path.stem for path in GOLDEN.glob("*.txt")} == set(GOLDEN_NAMES)

    def test_the_refused_scenario_has_no_golden(self) -> None:
        for name in REFUSED_NAMES:
            assert not (GOLDEN / f"{name}.txt").exists(), (
                f"{name} describes a response both planes must REFUSE; a golden for it would "
                f"mean one of them folded it"
            )


class TestTheEdgesTheGoldenCannotState:
    """Asserted on this plane exactly as `test_mask_parity.cpp` asserts them on the other."""

    def fold(self, name: str = "tiny_bank", **edits: float) -> InstanceMaskArea:
        scenario = load(name)
        return InstanceMaskArea(
            crop_hw=scenario.crop,
            score_threshold=float(edits.get("score_threshold", scenario.score_threshold)),
            mask_threshold=float(edits.get("mask_threshold", scenario.mask_threshold)),
        )

    def test_the_fold_names_the_quantity_and_not_a_model_output(self) -> None:
        areas = self.fold()(arrays_of(load("tiny_bank")))

        assert set(areas) == {"mask_area_px"}
        assert areas["mask_area_px"].shape == (4, 1), "one row per crop, one number wide"

    def test_a_crop_below_the_score_floor_reports_no_area(self) -> None:
        """Crop 1 scores 0.2 and its plane is entirely positive, so an unconditional argmax
        would report the WHOLE crop -- the plausible, wrong and silent answer."""
        areas = self.fold()(arrays_of(load("tiny_bank")))["mask_area_px"]

        assert areas[1][0] == 0.0
        assert areas[3][0] == 128.0, "and a crop whose mask fills it reports the crop"

    def test_without_the_floor_that_crop_reports_the_largest_area(self) -> None:
        """The non-vacuity of the check above: the floor is doing the work."""
        areas = self.fold(score_threshold=0.0)(arrays_of(load("tiny_bank")))["mask_area_px"]

        assert areas[1][0] == 128.0

    @pytest.mark.parametrize("name", REFUSED_NAMES)
    def test_the_refused_scenario_is_refused_here_too(self, name: str) -> None:
        with pytest.raises(InferenceError, match="mask coefficient"):
            render_masks(load(name))

    def test_an_output_of_the_wrong_rank_is_refused(self) -> None:
        """The Python plane's equivalent of the C++ `dims` guard: a numpy array carries its
        shape, so the failure is a rank the fold cannot read rather than a placeholder in it.
        Both planes refuse; neither folds a shape it was not given."""
        outputs = arrays_of(load("tiny_bank"))
        outputs["output1"] = outputs["output1"].reshape(4, -1)

        with pytest.raises(InferenceError, match=r"must be \(N, coeffs, h, w\)"):
            self.fold()(outputs)

    def test_a_missing_output_is_refused_by_name(self) -> None:
        outputs = arrays_of(load("tiny_bank"))
        del outputs["output1"]

        with pytest.raises(InferenceError, match="'output1' is missing"):
            self.fold()(outputs)
