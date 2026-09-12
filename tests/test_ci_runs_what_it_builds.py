"""Every C++ job in `cpp.yml` must RUN every binary its own build produced.

`CSRC-BENCH-UNCOMPILED` was a binary nobody compiled; its sibling is a binary CI compiles and
never runs. The lane job globbed `csrc/build/test_tracking_*`, so a lane binary named anything
else was built and skipped — which is why `test_cluster_parity` had to be renamed to be noticed
at all, and a convention nobody can see is not a guard.

Reads the workflow, runs nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cpp.yml"

#: The jobs that build C++ test binaries and are therefore expected to run them. `cpp-syntax`
#: compiles without linking and `cpp-gst-lane` runs one named binary with a pixel assertion, so
#: neither is in this rule; both are named here rather than excluded silently.
BUILDING_JOBS = ("cpp-offline", "cpp-shipvision-lane")


def jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def run_step(job: str) -> str:
    steps = jobs()[job]["steps"]
    running = [s for s in steps if "csrc/build/test_" in str(s.get("run", ""))]
    assert running, f"{job} has no step that globs built binaries; this test is now vacuous"
    return "\n".join(str(s["run"]) for s in running)


@pytest.mark.parametrize("job", BUILDING_JOBS)
def test_the_glob_is_not_narrowed_to_a_prefix(job: str) -> None:
    """A prefix leaves everything else built and unrun, and says nothing when it does."""
    globs = set(re.findall(r"csrc/build/(test_[A-Za-z0-9_*]*)", run_step(job)))

    assert globs, f"{job} globs no binaries"
    narrowed = sorted(g for g in globs if g != "test_*")
    assert not narrowed, (
        f"{job} runs only {narrowed}; a binary outside that prefix is built by CI and never "
        f"run, which is what `CPP-LANE-JOB-GLOBS-ONE-PREFIX` was"
    )


@pytest.mark.parametrize("job", BUILDING_JOBS)
def test_a_build_that_produced_nothing_cannot_pass(job: str) -> None:
    r"""The other half: a glob that matches nothing must fail the job rather than loop zero
    times and exit 0.

    `[1-9]` and not `\d`, which matches ZERO: `-ge 0` is true for an empty array, so a
    workflow where a build that produced nothing passes would have satisfied a test with this
    name. That is the shape this PR exists to argue against, found in its own guard by #236's
    review, and the ledger records the scenario: a lane goes red on a runner that built fewer
    binaries than expected and somebody relaxes the threshold to unstick the queue.
    """
    text = run_step(job)

    assert re.search(
        r'test "\$\{#binaries\[@\]\}" -ge [1-9][0-9]*', text
    ), f"{job} does not count what it found, so a build that produced nothing would pass"


@pytest.mark.parametrize("job", BUILDING_JOBS)
def test_object_files_are_not_mistaken_for_binaries(job: str) -> None:
    """`build_csrc.py` leaves `test_<name>.o` beside each binary, and `-x` is true for neither
    reason a reader expects — the build sets no execute bit on objects, but a glob without the
    test would still try to run them on a tree where something else did."""
    assert "!= *.o" in run_step(job)
