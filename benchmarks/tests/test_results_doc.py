"""The results page cites four things in the tree; a test says so when one of them moves.

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
def test_the_chain_measured_still_has_no_tracking(name: str) -> None:
    """`RESULTS.md` says "hand tracklets downstream" is in no number above, and this is the
    check behind that sentence. When tracking lands, this fails -- which is the point: the
    ratios were measured on a chain SHORTER than the deployed one, so adding the stage adds
    work on our side and every ratio has to be re-taken.
    """
    body = (GRAPH / name).read_text(encoding="utf-8")
    assert (
        "track" not in body and "mtmc" not in body
    ), f"{name} now builds a tracking node; benchmarks/RESULTS.md still claims it does not"


def test_the_binary_stamps_its_own_disclaimer() -> None:
    """The page quotes the bench's note rather than asserting the exclusion itself."""
    bench = (REPO_ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp").read_text(encoding="utf-8")
    assert "tracking and fused kernels are NOT in this " in bench


def test_the_one_at_a_time_quote_is_the_harness_own_words() -> None:
    """`RESULTS.md` explains why the arms are interleaved by quoting `--systems`' help."""
    runner = (REPO_ROOT / "benchmarks" / "run_bench.py").read_text(encoding="utf-8")
    assert "run one at a time to keep the GPUs uncontended" in runner
    assert "run one at a time to keep the GPUs uncontended" in RESULTS.read_text(
        encoding="utf-8"
    )
