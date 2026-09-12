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

#: A glob over built test binaries -- what makes a job subject to the rule in this file.
GLOB = re.compile(r"csrc/build/test_[A-Za-z0-9_]*\*")

#: The jobs deliberately outside it, with why. `cpp-syntax` compiles without linking, so it
#: produces no binary; `cpp-gst-lane` runs ONE named binary with a pixel assertion. Each is
#: held to that reason by `test_an_exempt_job_still_earns_its_exemption` rather than trusted:
#: a list nobody re-checks is the shape `CPP-LANE-JOB-GLOBS-ONE-PREFIX` was.
EXEMPT = {
    "cpp-syntax": "compiles without linking, so there is no binary to run",
    "cpp-gst-lane": "runs one named binary, not a glob",
}


def jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def commands(run: object) -> str:
    """A ``run:`` block with its comment lines dropped.

    Everything here greps that text, and a `run:` block is prose as well as shell: the line
    in `cpp-offline` that explains which prefix was wrong would otherwise be read as the job
    doing it. Writing down why a rule exists must not break the rule.
    """
    return "\n".join(
        line for line in str(run).splitlines() if not line.lstrip().startswith("#")
    )


def run_text(job: str) -> str:
    return "\n".join(commands(step.get("run", "")) for step in jobs()[job]["steps"])


def building_jobs() -> tuple[str, ...]:
    """Every job that globs built binaries, DERIVED from the workflow.

    A hand-written list was the first version, and it repeats on the guard the mistake the
    guard is about: a future `cpp-<something>-lane` would not be in it, would glob one
    prefix, and nothing here would go red. #236's review found that; this is the fix.
    """
    found = tuple(name for name in jobs() if GLOB.search(run_text(name)))
    assert found, "no job in cpp.yml globs built binaries; this module is now vacuous"
    return found


BUILDING_JOBS = building_jobs()


def run_step(job: str) -> str:
    running = [
        s for s in jobs()[job]["steps"] if "csrc/build/test_" in commands(s.get("run", ""))
    ]
    assert running, f"{job} has no step that globs built binaries; this test is now vacuous"
    return "\n".join(commands(s["run"]) for s in running)


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


def test_every_job_is_either_covered_by_the_rule_or_exempt_with_a_reason() -> None:
    """The closure that makes the derivation worth having: a job added to `cpp.yml` is
    covered from the day it lands, or it fails here until somebody says why not. Neither
    half can be satisfied by forgetting."""
    assert set(jobs()) == set(BUILDING_JOBS) | set(EXEMPT), (
        f"cpp.yml jobs {sorted(set(jobs()) - set(BUILDING_JOBS) - set(EXEMPT))} are neither "
        f"covered by this rule nor listed in EXEMPT with a reason"
    )


@pytest.mark.parametrize("job", sorted(EXEMPT))
def test_an_exempt_job_still_earns_its_exemption(job: str) -> None:
    """An exemption is re-earned every run. A job that starts globbing built binaries is
    subject to the rule from that moment, whatever this file remembers about it."""
    assert job in jobs(), f"{job} is exempt from a rule it is no longer in"

    assert not GLOB.search(run_text(job)), (
        f"{job} globs built binaries now, so {EXEMPT[job]!r} is no longer why it is exempt; "
        f"drop it from EXEMPT and let the rule cover it"
    )


def test_a_comment_naming_the_bad_glob_is_not_read_as_doing_it() -> None:
    """Otherwise the rule forbids its own explanation. `cpp.yml` documents the prefix that
    was wrong right above the loop that replaced it, and a guard reading the whole block
    would fail the job for the sentence rather than for the command."""
    block = (
        "# NOT `for candidate in csrc/build/test_tracking_*` -- that skipped 27 binaries\n"
        "for candidate in csrc/build/test_*; do"
    )

    assert commands(block) == "for candidate in csrc/build/test_*; do"
