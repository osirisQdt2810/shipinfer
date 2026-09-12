"""The Python half of the identity golden: is the committed file still what the reference says?

The C++ binary (`csrc/tests/test_identity_parity.cpp`) is the real gate -- it is the port that
can drift. This one answers the other question, which that binary cannot: whether the GOLDEN
still matches the reference it was emitted from, so a submodule bump that changes an id is
caught here rather than showing up as "the port is wrong".

Skipped without the submodule, and CI's `Tests (with the kernels' Python half)` job is where it
runs.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from benchmarks.parity.drive_identity import GOLDEN, SCENARIOS, load, render_identity

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

        assert render_identity(load(name)) == expected, (
            f"the ids the reference assigns for {name} are not the committed golden. If a "
            f"submodule bump IS the decision, re-emit with `python "
            f"scripts/emit_parity_golden.py --kind identity --scenario {name} --emit-golden "
            f"--force`, say so in the PR, and check the C++ gate in the same commit"
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_the_golden_names_the_command_that_emits_it(self, name: str) -> None:
        """The C++ gate asserts this too. Both, because a golden nobody can regenerate is one
        that gets hand-edited the first time it fails."""
        text = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")

        assert "emit_parity_golden.py --kind identity" in text


class TestAContestedCameraSlotHasOneHolder:
    """What `MTMC-ONE-CAMERA-TWICE-IN-A-CONTESTED-CLUSTER` was, now asserted the right way up.

    It used to be a tripwire: the reference adopted BOTH challengers and this file pinned that,
    so the day upstream fixed it something would notice. shipvision#17 is that day. The tests
    below are the same two instants with the assertions inverted, and they stay because the
    parity golden reaches the answer but not the reference's own `validate()`.
    """

    def observation(self, camera: str, track: int, x: float, y: float):
        import numpy as np
        from shipvision.mtmc.frames import TrackKey, TrackObservation
        from shipvision.types import FrameTag, Track

        return TrackObservation(
            key=TrackKey(camera_id=camera, track_id=track),
            track=Track(
                track_id=track,
                box=np.array([0.0, 0.0, 100.0, 300.0]),
                tag=FrameTag(camera_id=camera, frame_id=0),
                embedding=np.array([x, y], dtype=np.float32),
            ),
            frame_height=1080,
            frame_width=1920,
        )

    def _contested(self, validate: bool, second: float, third: float):
        from shipvision.mtmc.identity import GlobalIdAssigner

        look = self.observation
        assigner = GlobalIdAssigner(validate_every_step=validate)
        assigner.assign(
            [look("cam0", 1, 0, 1), look("cam1", 1, 1, 0), look("cam2", 1, 1, 0)], [0, 0, 0]
        )
        assigner.assign(
            [
                look("cam0", 1, 0, 1),
                look("cam1", 1, 1, 0),
                look("cam2", 1, 1, 0),
                look("cam0", 2, 1, second),
                look("cam0", 3, 1, third),
            ],
            [1, 0, 0, 0, 0],
        )
        return assigner

    @pytest.mark.parametrize(
        "second,third,holder", [(0.3, 0.8, 2), (0.8, 0.3, 3)], ids=["second", "third"]
    )
    def test_only_the_better_challenger_holds_the_slot(
        self, second: float, third: float, holder: int
    ) -> None:
        """Both directions. A fix that skipped the displaced track without still holding a
        contest would pass the first case and fail the second."""
        from shipvision.mtmc.frames import TrackKey

        # `validate_every_step=False` is production's default, and how this stayed silent.
        assigner = self._contested(False, second, third)

        target = assigner.owner_of(TrackKey(camera_id="cam1", track_id=1))
        from_cam0 = [key for key in assigner.members(target) if key.camera_id == "cam0"]

        assert from_cam0 == [TrackKey(camera_id="cam0", track_id=holder)], (
            f"a contested camera slot has one holder, and it is the better challenger. "
            f"{len(from_cam0)} here means the upstream fix has been reverted or lost in a "
            f"submodule bump; the C++ port in `csrc/.../mtmc/identity.cpp` carries the same "
            f"rule and `golden/identity/basic.txt` holds the answer"
        )

    def test_the_reference_validates_its_own_state_now(self) -> None:
        """The half no golden can carry: asked to check itself, the reference used to call the
        state it had just built a fault. `validate_every_step=True` is what the parity driver
        runs with, so this is also why the golden can hold these scenarios at all."""
        self._contested(True, 0.3, 0.8)  # raises TrackingError if the invariant is broken


class TestTheCppGateReadsTheSameFiles:
    def test_the_binary_names_this_scenario_and_this_golden(self) -> None:
        """A path typo would leave the C++ gate reading a file nobody emits, and it would pass.
        Source-level because a Python test cannot run the binary."""
        source = (
            Path(__file__).resolve().parents[2] / "csrc" / "tests" / "test_identity_parity.cpp"
        ).read_text(encoding="utf-8")

        for name in NAMES:
            assert f"scenarios/identity/{name}.txt" in source
            assert f"golden/identity/{name}.txt" in source
