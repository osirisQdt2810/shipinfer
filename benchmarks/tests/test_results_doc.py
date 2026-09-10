"""The results page cites five things in the tree; a test says so when one of them moves.

`RESULTS.md` records what the harness measured, and its most load-bearing claim is a
NEGATIVE one -- that tracking and the fused kernels are in none of the numbers. A number
cannot be re-derived offline, but the citations can, and the day `track` reaches the C++
graph is the day the page stops being true rather than merely stale.

Offline by construction: this reads files, runs nothing, and touches no device.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "benchmarks" / "RESULTS.md"
GRAPH = REPO_ROOT / "csrc" / "shipinfer" / "pipeline" / "graph"


def test_the_readme_points_at_it() -> None:
    """A results page nobody is sent to is a file, not a deliverable."""
    readme = (REPO_ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    assert "RESULTS.md" in readme


@pytest.mark.parametrize("name", ["plan.cpp", "from_plan.cpp"])
def test_the_chain_measured_now_tracks_and_still_has_no_mtmc(name: str) -> None:
    """This test used to assert the OPPOSITE and failing was its purpose.

    It read "track not in body and mtmc not in body" behind `RESULTS.md`'s largest negative
    claim, and tracking landing broke it -- which is how the page got re-measured instead of
    going quiet. What is left of the claim is `mtmc`, so that is what it asserts, plus that
    the graph does reach a tracker by name.
    """
    body = (GRAPH / name).read_text(encoding="utf-8")

    assert "mtmc" not in body, f"{name} now builds an mtmc node; RESULTS.md still excludes it"
    if name == "from_plan.cpp":
        assert "create_associator" in body, (
            "the graph no longer asks for a tracker, so the page's tracking numbers describe "
            "a chain that is not being built"
        )


def test_the_binary_stamps_its_own_disclaimer() -> None:
    """The page quotes the bench's note rather than asserting the exclusion itself.

    The note is DERIVED now: "tracking" appears in it only when the run's own `stages` do not
    contain a track slot. A constant claim is the kind that keeps being printed after it stops
    being true, which is what happened to the tracking half of this sentence.
    """
    bench = (REPO_ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp").read_text(encoding="utf-8")

    assert "fused kernels are NOT in this measurement" in bench
    assert 'if (!tracked) out << ", and neither is tracking";' in bench
    # FROM THE PLAN, not from a scan of `stage_names`: those are SLOT names, so a chain whose
    # tracker is called `tap:` would run one and stamp "neither is tracking".
    assert "!planned.tracks.empty()));" in bench


def test_the_one_at_a_time_quote_is_the_harness_own_words() -> None:
    """`RESULTS.md` explains why the arms are interleaved by quoting `--systems`' help."""
    runner = (REPO_ROOT / "benchmarks" / "run_bench.py").read_text(encoding="utf-8")
    assert "run one at a time to keep the GPUs uncontended" in runner
    assert "run one at a time to keep the GPUs uncontended" in RESULTS.read_text(
        encoding="utf-8"
    )


def test_the_knob_the_page_credits_is_the_one_the_binary_reads() -> None:
    """The page's second-largest claim is now an env var away, so the spelling has to match.

    A page crediting `SHIPINFER_CUDA_BLOCKING_SYNC` while the binary gated on some other name
    would report 7.17x for a run that never had the flag -- which is the failure the harness's
    own refusal exists to prevent, one level up.
    """
    page = RESULTS.read_text(encoding="utf-8")
    bench = (REPO_ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp").read_text(encoding="utf-8")

    assert "SHIPINFER_CUDA_BLOCKING_SYNC=0" in page
    assert 'env_flag_unless_refused("SHIPINFER_CUDA_BLOCKING_SYNC")' in bench


def test_the_page_says_on_by_default_only_while_it_is() -> None:
    """Both figures on the page are labelled by which side of the default they are on, so a
    default that flips silently makes the verdict say the opposite of what it means.

    THIS TEST IS WHY THE PAGE MOVED. It was written in #208 with "off by default" and it
    failed the moment the default flipped, which is exactly what it was for -- the page and the
    code cannot drift apart on which figure is "as shipped".
    """
    page = RESULTS.read_text(encoding="utf-8")
    bench = (REPO_ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp").read_text(encoding="utf-8")

    assert "**on by default**" in page
    # The COUNT first, and over the WHOLE of `csrc/`: splitting on the gate marker inspects
    # only what follows it, and counting inside one file would miss the flag moving out of
    # `bench.cpp` -- `gpuSetDeviceFlags` is a `platform.h` alias any unit can reach.
    setters = [
        path
        for path in sorted((REPO_ROOT / "csrc").rglob("*"))
        if path.suffix in {".h", ".hpp", ".cuh", ".cpp", ".cu"}
        and "SetDeviceFlags" in path.read_text(encoding="utf-8")
        and path.name != "platform.h"
    ]
    assert setters == [REPO_ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp"], setters
    assert bench.count("SetDeviceFlags") == 1, (
        "more than one place sets a device flag, so the env gate is no longer the only way in "
        "and the page's `3.4x with =0` figure describes something unreachable"
    )
    guarded = bench.split('env_flag_unless_refused("SHIPINFER_CUDA_BLOCKING_SYNC")')[1]
    assert (
        "gpuSetDeviceFlags(gpuDeviceScheduleBlockingSync)" in guarded.split("std::printf")[0]
    ), (
        "the flag is set outside the env gate, so `=0` no longer refuses it and the page's "
        "`3.4x with =0` figure describes a configuration that cannot be reached"
    )
