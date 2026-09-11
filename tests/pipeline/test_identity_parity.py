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


class TestTheReferenceStillAdoptsTwoTracksFromOneCamera:
    """The tripwire for `MTMC-ONE-CAMERA-TWICE-IN-A-CONTESTED-CLUSTER`.

    A second challenger from one camera contests the ALREADY-DISPLACED incumbent, wins on the
    same evidence and is adopted too, because the contest re-reads `members_` while the loser
    leaves only in the deferred loop. The C++ port answers identically, so the fix starts
    upstream. **No scenario can hold this claim** (#219 round 2): `drive_identity.py` runs the
    reference with `validate_every_step=True`, where it throws instead of answering -- so the
    pin lived in a comment and nothing would notice the day upstream fixed it. This notices.
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

    def test_both_challengers_land_and_the_day_they_do_not_is_the_day_to_port_it(self) -> None:
        from shipvision.mtmc.frames import TrackKey
        from shipvision.mtmc.identity import GlobalIdAssigner

        look = self.observation
        # `validate_every_step=False` is production's default, and how this stays silent there.
        assigner = GlobalIdAssigner(validate_every_step=False)
        assigner.assign(
            [look("cam0", 1, 0, 1), look("cam1", 1, 1, 0), look("cam2", 1, 1, 0)], [0, 0, 0]
        )
        assigner.assign(
            [
                look("cam0", 1, 0, 1),
                look("cam1", 1, 1, 0),
                look("cam2", 1, 1, 0),
                look("cam0", 2, 1, 0),
                look("cam0", 3, 1, 0),
            ],
            [1, 0, 0, 0, 0],
        )

        target = assigner.owner_of(TrackKey(camera_id="cam1", track_id=1))
        members = assigner.members(target)
        from_cam0 = [key for key in members if key.camera_id == "cam0"]

        assert len(from_cam0) == 2, (
            f"the reference now adopts {len(from_cam0)} track(s) from one camera into a "
            f"contested cluster, so `MTMC-ONE-CAMERA-TWICE-IN-A-CONTESTED-CLUSTER` has been "
            f"fixed upstream: bump the submodule, port the same change to "
            f"`csrc/shipinfer/pipeline/mtmc/identity.cpp`, re-emit `golden/identity/basic.txt`, "
            f"and flip `two_challengers_from_one_camera_BOTH_land_and_the_reference_does_the_"
            f"same` in `csrc/tests/test_mtmc_identity.cpp` from 2 to 1"
        )

    def test_its_own_validation_refuses_the_state_it_built(self) -> None:
        """The other half of the same fact, and the reason no golden can carry it: asked to
        check itself, the reference calls this state a fault."""
        from shipvision.errors import TrackingError
        from shipvision.mtmc.identity import GlobalIdAssigner

        look = self.observation
        assigner = GlobalIdAssigner(validate_every_step=True)
        assigner.assign(
            [look("cam0", 1, 0, 1), look("cam1", 1, 1, 0), look("cam2", 1, 1, 0)], [0, 0, 0]
        )

        with pytest.raises(TrackingError, match="two tracks from one camera"):
            assigner.assign(
                [
                    look("cam0", 1, 0, 1),
                    look("cam1", 1, 1, 0),
                    look("cam2", 1, 1, 0),
                    look("cam0", 2, 1, 0),
                    look("cam0", 3, 1, 0),
                ],
                [1, 0, 0, 0, 0],
            )


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
