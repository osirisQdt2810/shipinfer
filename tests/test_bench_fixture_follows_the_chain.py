"""A chain that tracks gets footage a tracker can follow — decided from the plan, not a flag.

The default bench data is ten unrelated photographs served at N fps, so a camera's scene
changes completely every frame. Measured 11 Sep: the graph offered `mtmc` 0.36 observations per
frame against ~9.5 embedded rows and the age gate admitted nothing; on the pan fixture the same
chain offers 4.42 and admits 62% at the reference's own defaults. `scripts/run_cpp_bench.sh`
therefore reads the plan it just wrote and picks the fixture from it.

The predicate is taken FROM the script rather than restated here — a copy would keep passing
after the script's own pattern changed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_cpp_bench.sh"
GOLDEN = REPO_ROOT / "benchmarks" / "parity" / "golden" / "plans"


def pattern() -> str:
    """The script's own `grep -qE` pattern for a plan that tracks."""
    match = re.search(r'grep -qE "([^"]+)"', RUNNER.read_text(encoding="utf-8"))
    assert match, "the runner no longer greps the plan; this test would pass on anything"
    return match.group(1)


def selects_the_pan(plan: Path) -> bool:
    """Run the script's predicate, with grep, exactly as the script runs it."""
    done = subprocess.run(
        ["grep", "-qE", pattern(), str(plan)], check=False, capture_output=True
    )
    assert done.returncode in (0, 1), done.stderr.decode()
    return done.returncode == 0


@pytest.mark.parametrize("name", ["branching", "ship_person_cpu"])
def test_a_plan_that_tracks_gets_the_pan(name: str) -> None:
    assert selects_the_pan(GOLDEN / f"{name}.plan")


@pytest.mark.parametrize("name", ["minimal", "defaults", "detect_only"])
def test_a_plan_that_does_not_track_keeps_the_photographs(name: str) -> None:
    """Detection-only numbers stay comparable with their own history, which is the reason the
    switch is per chain rather than a new default for every run."""
    assert not selects_the_pan(GOLDEN / f"{name}.plan")


def test_a_plan_that_only_segments_keeps_the_photographs(tmp_path: Path) -> None:
    """No committed golden separates the two, and the distinction is the whole rule: a
    segmenter does not need temporal continuity and a tracker cannot work without it. Written
    here rather than looked for, because a pattern that matched every kind would pass on the
    goldens by coincidence -- none of the non-tracking ones has a segment node.
    """
    plan = tmp_path / "segment_only.plan"
    plan.write_text(
        "plan 3 segment-only\n"
        "node decode decode replay\n"
        "node detect detect pool\n"
        "node ship_segmenter segment pool\n"
        "node output output jsonlines\n",
        encoding="utf-8",
    )

    assert not selects_the_pan(plan)


def test_a_plan_with_a_tracker_and_nothing_else_gets_the_pan(tmp_path: Path) -> None:
    plan = tmp_path / "track_only.plan"
    plan.write_text(
        "plan 3 track-only\n"
        "node decode decode replay\n"
        "node detect detect pool\n"
        "node track track shipvision\n"
        "node output output jsonlines\n",
        encoding="utf-8",
    )

    assert selects_the_pan(plan)


@pytest.mark.parametrize("slot", ["ship-track", "ShipTrack"])
def test_the_slot_may_be_spelled_any_way_the_chain_spells_it(slot: str, tmp_path: Path) -> None:
    """The slot is an operator-chosen YAML key; the KIND is the closed enum `plan.py` writes.
    A predicate that constrains the slot refuses a legal chain -- silently, and on exactly the
    measurement this file exists to protect. The vocabulary already uses hyphens elsewhere
    (`impl: gstreamer-gpu`), so neither spelling is hypothetical.
    """
    plan = tmp_path / "hyphen.plan"
    plan.write_text(
        "plan 3 hyphen\n"
        "node decode decode replay\n"
        "node detect detect pool\n"
        f"node {slot} track shipvision\n",
        encoding="utf-8",
    )

    assert selects_the_pan(plan)


@pytest.mark.parametrize(
    "variable",
    [
        # The RTSP arm's, and the replay arm's. BOTH, because neither arm reads the other's
        # name: an operator who set the replay one and got the pan anyway would have no escape
        # hatch that works on the source they are running.
        "SHIPINFER_RTSP_PERSON_DATA",
        "SHIPINFER_RTSP_SHIP_DATA",
        "SHIPINFER_BENCH_PERSON_FRAMES",
        "SHIPINFER_BENCH_SHIP_FRAMES",
    ],
)
def test_an_explicit_fixture_always_wins(variable: str) -> None:
    """An operator who names the data gets it, tracking chain or not — the measurement that
    compares the two fixtures on one chain is only possible that way."""
    text = RUNNER.read_text(encoding="utf-8")
    guard = text[text.index("grep -qE") : text.index("export SHIPINFER_RTSP_PERSON_DATA")]

    assert f'[ -z "${{{variable}:-}}" ]' in guard


def test_a_half_written_fixture_is_never_served() -> None:
    """ "Is the directory non-empty" cannot tell a finished fixture from an interrupted one, and
    `rtsp_serve.py` then caches the short encode by directory name with no content check — so it
    would outlive even a manual `rm -rf` of the frames."""
    text = RUNNER.read_text(encoding="utf-8")
    block = text[text.index("for pair in") : text.index("export SHIPINFER_RTSP_PERSON_DATA")]

    assert '--out "$partial"' in block, "the generator writes under the served name"
    # `mv -T`, not `mv`: a plain move puts this lap INSIDE a directory another run finished
    # first, and the served name then holds `person/person.partial.123/...` -- which is the
    # half-written fixture this test is named for, arriving by a different road.
    assert 'mv -T "$partial" "$out"' in block, "nothing moves the finished fixture into place"
    assert 'rm -rf "$partial"' in block.split("mv -T")[1], "the loser of the race keeps its lap"


def test_the_generator_is_only_run_when_the_directory_is_empty() -> None:
    """`pan_frames` refuses a non-empty directory rather than mixing two laps, so the CALL has
    to sit inside the emptiness guard: hoisted out of it, every run after the first would abort
    on the fixture it was about to serve."""
    text = RUNNER.read_text(encoding="utf-8")
    block = text[text.index("for pair in") : text.index("export SHIPINFER_RTSP_PERSON_DATA")]
    guard = 'if [ -z "$(ls -A "$out" 2>/dev/null)" ]; then'
    assert guard in block

    guarded = block[block.index(guard) : block.index("\n    fi")]
    assert "make_pan_fixture.py" in guarded, (
        "the generator is invoked outside the emptiness guard, so a second run aborts on the "
        "fixture it was about to serve"
    )
