"""The Python half of the tracking golden: is the committed file still what this plane says?

The C++ binary (`csrc/tests/test_tracking_parity.cpp`) prints its own per-row stream beside
this one, and where they differ that difference IS `tracker_options` in
`benchmarks/parity/known.py`. This side answers what that binary cannot: whether the GOLDEN
still matches the plane it was emitted from, so a submodule bump that moves a threshold is
caught here rather than showing up as "the port is wrong".

It also holds the same edge bracket the binary does, on this plane. A case that has drifted
off the tracker's gate still compares clean, which would make the measured count meaningless.

Skipped without the submodule, like its cluster sibling.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from benchmarks.parity.drive_tracking import GOLDEN, SCENARIOS, load, render_tracking

pytestmark = pytest.mark.skipif(
    find_spec("shipvision") is None,
    reason="the shipvision submodule is not importable; run "
    "`git submodule update --init 3rdparty/shipvision` and put it on PYTHONPATH",
)

NAMES = tuple(sorted(path.stem for path in SCENARIOS.glob("*.scn")))


class TestTheGoldenIsStillThisPlanesAnswer:
    def test_there_is_a_scenario_at_all(self) -> None:
        """A glob that finds nothing would make every test below vacuously green."""
        assert NAMES, f"no scenario files under {SCENARIOS}"

    @pytest.mark.parametrize("name", NAMES)
    def test_the_golden_is_reproduced_exactly(self, name: str) -> None:
        expected = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")

        assert render_tracking(load(name)) == expected, (
            f"this plane's per-row ids for {name} are not the committed golden. If the change "
            f"IS the decision, re-emit with `python scripts/emit_parity_golden.py --kind "
            f"tracking --scenario {name} --emit-golden --force` and say so in the PR"
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_the_ids_are_normalised_and_not_the_counter(self, name: str) -> None:
        """`shipvision` mints track ids from one PROCESS-WIDE counter, so a golden holding the
        literal number passes alone and fails in a suite. The labels restart at 1 per scenario,
        which is what makes this file reproducible -- and what a reader must not undo."""
        rendered = render_tracking(load(name))
        first_ids = [
            line.split("ids", 1)[1].split()
            for line in rendered.splitlines()
            if line.startswith("frame 0 ids")
        ]
        assert first_ids, "no scenario had a frame 0"
        for ids in first_ids:
            assert all(
                value in {"-", "1", "2"} for value in ids
            ), f"a first frame carries {ids}, which means the raw counter reached the golden"

    def test_the_case_still_brackets_the_tracker_gate(self) -> None:
        """`aspect_edge` continues its track and `aspect_break` re-mints, one 0.1px apart.

        Without this the scenarios can drift off the edge -- the tracker giving out earlier
        makes BOTH re-mint -- and the C++ binary still reports "0 differing" over a case that
        probes nothing. The measurement is only worth its line in the ledger while this holds.
        """
        rendered = render_tracking(load("attribution"))
        last = {}
        scenario = None
        for line in rendered.splitlines():
            if line.startswith("scenario "):
                scenario = line.split(" ", 1)[1]
            elif scenario and " ids" in line:
                last[scenario] = line.split(" ids", 1)[1].strip()

        assert last.get("steady") == "1", f"the control lost its one track: {last!r}"
        assert last.get("aspect_edge") == "1", (
            "aspect_edge no longer CONTINUES its track, so the narrowest width that keeps one "
            "has moved and this scenario is no longer on the edge. Re-measure before trusting "
            f"the count the C++ binary prints: {last!r}"
        )
        assert last.get("aspect_break") not in (None, "", "1"), (
            "aspect_break no longer RE-MINTS, so the pair no longer brackets the tracker's "
            f"gate and the case has stopped reaching the seam: {last!r}"
        )
