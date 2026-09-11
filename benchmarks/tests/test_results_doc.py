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
def test_the_chain_measured_runs_the_whole_thing(name: str) -> None:
    """This test has now inverted TWICE, and failing was its purpose both times.

    It began as "track not in body and mtmc not in body" behind `RESULTS.md`'s largest
    negative claim. Tracking landing broke it, so the page was re-measured; `mtmc` landing
    broke what was left. There is no exclusion to guard any more, so what it guards instead is
    the positive: the graph asks for both by name, and the page carries the whole-chain
    numbers. A chain that quietly stopped building one of them would leave those numbers
    describing something else.
    """
    body = (GRAPH / name).read_text(encoding="utf-8")
    page = RESULTS.read_text(encoding="utf-8")

    if name == "from_plan.cpp":
        assert "create_associator" in body, (
            "the graph no longer asks for a tracker, so the page's tracking numbers describe "
            "a chain that is not being built"
        )
        assert "mtmc_runtime" in body or "MtmcRuntime" in body, (
            "the graph no longer builds the mtmc stage, so the whole-chain section is "
            "measuring a shorter chain than it says"
        )
    assert "tracked img/s" in page, (
        "the page must report the TRACKED rate and not only the accepted one: an untracked "
        "frame carries no ids and `mtmc` cannot associate it"
    )
    assert "track_frames_untracked" in page, (
        "and it must name the counter that separates the two, or the next reader quotes the "
        "accepted number as the chain's throughput -- which is what happened"
    )


def test_the_gate_the_page_measures_is_the_gate_a_chain_can_state() -> None:
    """The nine-arm table is a measurement of two thresholds, and it is only reproducible while
    those are the two a plan carries. A third one becoming settable is the day the table is
    incomplete rather than wrong, which is exactly when it should be reread."""
    page = RESULTS.read_text(encoding="utf-8")
    plan = (REPO_ROOT / "src" / "shipinfer" / "topology" / "plan.py").read_text(
        encoding="utf-8"
    )

    assert "min_hits" in page and "min_height_fraction" in page
    assert '_PLAN_OPTIONS = ("min_hits", "min_height_fraction")' in plan, (
        "RESULTS.md's gate table varies the two options a plan can carry; `_PLAN_OPTIONS` "
        "moved, so the table now describes a subset of the gate a chain can state"
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

    assert "SHIPINFER_CUDA_BLOCKING_SYNC=1" in page
    assert 'env_flag("SHIPINFER_CUDA_BLOCKING_SYNC")' in bench


def test_the_page_says_off_by_default_only_while_it_is() -> None:
    """Both figures on the page are labelled by which side of the default they are on, so a
    default that flips silently makes the verdict say the opposite of what it means."""
    page = RESULTS.read_text(encoding="utf-8")
    bench = (REPO_ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp").read_text(encoding="utf-8")

    assert "**off by default**" in page
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
        "and the page's `~3.4x default` figure describes something that does not ship"
    )
    guarded = bench.split('env_flag("SHIPINFER_CUDA_BLOCKING_SYNC")')[1]
    assert (
        "gpuSetDeviceFlags(gpuDeviceScheduleBlockingSync)" in guarded.split("std::printf")[0]
    ), (
        "the flag is set outside the env gate, so it is no longer off by default and the "
        "page's `~3.4x default` figure describes a configuration that no longer ships"
    )
