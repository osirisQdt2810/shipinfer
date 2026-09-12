"""The Python half of the gate golden: is the committed file still what the reference says?

The C++ binary (`csrc/tests/test_gate_parity.cpp`) is the real gate -- it is the port that
can drift. This one answers the other question, which that binary cannot: whether the GOLDEN
still matches the reference it was emitted from, so a submodule bump that changes what qualifies is
caught here rather than showing up as "the port is wrong".

Skipped without the submodule, and CI's `Tests (with the kernels' Python half)` job is where it
runs.
"""

from __future__ import annotations

import inspect
import re
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


class TestTheReferenceCommitsBeforeTheInstantIsApplied:
    """The tripwire for `MTMC-GATE-COMMITS-BEFORE-THE-GRAM-CAN-THROW`.

    `filter` installs its hit map before `MTMC.track` runs the gram, the clusterer and the
    assigner -- any of which can raise -- so an instant submitted twice advances every run
    twice. The C++ port does the same and `csrc/tests/test_mtmc_gate.cpp` pins it. Making one
    plane atomic alone would be the divergence, so the fix is upstream; this fails the day
    upstream makes it, which is the day to port the change rather than discover it.
    """

    def test_a_re_submitted_instant_advances_the_run_again(self) -> None:
        import numpy as np
        from shipvision.mtmc.frames import TrackKey, TrackObservation
        from shipvision.mtmc.gating import ObservationGate
        from shipvision.types import FrameTag, Track

        key = TrackKey(camera_id="cam0", track_id=1)
        track = Track(
            track_id=1,
            box=np.array([0.0, 0.0, 100.0, 300.0]),
            tag=FrameTag(camera_id="cam0", frame_id=0),
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        instant = [TrackObservation(key=key, track=track, frame_height=1080, frame_width=1920)]
        gate = ObservationGate(min_hits=3)

        gate.filter(instant)
        gate.filter(instant)

        assert len(gate.filter(instant)) == 1, (
            "the reference now withholds a re-submitted instant, which is the upstream fix "
            "`MTMC-GATE-COMMITS-BEFORE-THE-GRAM-CAN-THROW` asks for -- port it to "
            "`csrc/shipinfer/pipeline/mtmc/gate.cpp` and update the pin in test_mtmc_gate.cpp"
        )
        assert gate.hits(key) == 3


class TestBothPlanesDefaultToTheSameNumbers:
    """The golden pins BEHAVIOUR at the options a scenario names, so it cannot see a default.

    `max_absent_instants` is the one that matters and the one no scenario can reach: pinning it
    in the golden would take 33 near-identical instant lines for a single number. So the
    constants are compared directly, from the C++ header and the reference's own signature.
    """

    #: The C++ spelling of each default, and the reference keyword it has to equal.
    CONSTANTS = {
        "min_hits": "kDefaultMinHits",
        "min_height_fraction": "kDefaultMinHeightFraction",
        "max_absent_instants": "kDefaultMaxAbsentInstants",
    }

    @staticmethod
    def _cpp_default(name: str) -> str:
        source = (
            Path(__file__).resolve().parents[2]
            / "csrc"
            / "shipinfer"
            / "pipeline"
            / "mtmc"
            / "gate.h"
        ).read_text(encoding="utf-8")
        found = re.search(rf"constexpr \w+ {name} = ([^;]+);", source)
        assert found, f"{name} is not a constant in gate.h any more"
        return found.group(1).strip()

    @pytest.mark.parametrize("keyword,constant", sorted(CONSTANTS.items()))
    def test_the_cpp_constant_is_the_references_default(
        self, keyword: str, constant: str
    ) -> None:
        from shipvision.mtmc.gating import ObservationGate

        reference = inspect.signature(ObservationGate.__init__).parameters[keyword].default

        assert eval(self._cpp_default(constant)) == pytest.approx(reference), (
            f"the C++ gate defaults {keyword} to {self._cpp_default(constant)} and the "
            f"reference to {reference}; two defaults for one knob is how they drift"
        )


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
