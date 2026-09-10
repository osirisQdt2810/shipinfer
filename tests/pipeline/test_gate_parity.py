"""The Python half of the gate golden: is the committed file still what the reference says?

The C++ binary (`csrc/tests/test_gate_parity.cpp`) is the real gate -- it is the port that
can drift. This one answers the other question, which that binary cannot: whether the GOLDEN
still matches the reference it was emitted from, so a submodule bump that changes what qualifies is
caught here rather than showing up as "the port is wrong".

Skipped without the submodule, and CI's `Tests (with the kernels' Python half)` job is where it
runs.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from benchmarks.parity.drive_gate import GOLDEN, SCENARIOS, load, render_gate

pytestmark = pytest.mark.skipif(
    find_spec("shipvision") is None,
    reason="the shipvision submodule is not importable; run "
    "`git submodule update --init 3rdparty/shipvision` and put it on PYTHONPATH",
)

#: Every scenario file with a golden. One today; the loop is what keeps the next one covered.
NAMES = tuple(sorted(path.stem for path in SCENARIOS.glob("*.txt")))


class TestTheGoldenIsStillTheReferencesAnswer:
    def test_there_is_a_scenario_at_all(self) -> None:
        """A glob that finds nothing would make every test below vacuously green."""
        assert NAMES, f"no scenario files under {SCENARIOS}"

    @pytest.mark.parametrize("name", NAMES)
    def test_the_golden_is_reproduced_exactly(self, name: str) -> None:
        expected = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")

        assert render_gate(load(name)) == expected, (
            f"the observations the reference admits for {name} are not the committed golden. "
            f"If a submodule bump IS the decision, re-emit with `python "
            f"scripts/emit_parity_golden.py --kind gate --scenario {name} --emit-golden "
            f"--force`, say so in the PR, and check the C++ gate in the same commit"
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_the_golden_names_the_command_that_emits_it(self, name: str) -> None:
        """The C++ gate asserts this too. Both, because a golden nobody can regenerate is one
        that gets hand-edited the first time it fails."""
        text = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")

        assert "emit_parity_golden.py --kind gate" in text


class TestTheCppGateReadsTheSameFiles:
    def test_the_binary_names_this_scenario_and_this_golden(self) -> None:
        """A path typo would leave the C++ gate reading a file nobody emits, and it would pass.
        Source-level because a Python test cannot run the binary."""
        source = (
            Path(__file__).resolve().parents[2] / "csrc" / "tests" / "test_gate_parity.cpp"
        ).read_text(encoding="utf-8")

        for name in NAMES:
            assert f"scenarios/gate/{name}.txt" in source
            assert f"golden/gate/{name}.txt" in source
