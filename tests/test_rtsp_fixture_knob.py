"""A knob only a caller already inside the container can turn is not a knob.

`scripts/cpp_bench_over_rtsp.sh` has read `SHIPINFER_RTSP_PERSON_DATA` and
`SHIPINFER_RTSP_SHIP_DATA` since it was written, and `deploy/rootless/cpp.sh` passes the
container the environment it lists and no other -- so the fixture directory could not be
chosen from outside. That matters now that there is a second fixture to choose
(`benchmarks/harness/pan.py`), and it is the kind of gap only a test notices.

Source-read, no container, no device.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "cpp_bench_over_rtsp.sh"
#: Both runners that can launch the wrapper. V168 makes optimisation a loop -- benchmark, then
#: profile -- so the profiler has to be able to launch what the benchmark launches, with the
#: same fixture.
RUNNERS = (
    REPO_ROOT / "deploy" / "rootless" / "cpp.sh",
    REPO_ROOT / "deploy" / "rootless" / "profile.sh",
)
CLI = REPO_ROOT / "csrc" / "shipinfer" / "cli"


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_every_rtsp_variable_the_wrapper_reads_is_forwarded_into_the_container(
    runner: Path,
) -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    text = runner.read_text(encoding="utf-8")
    read = set(re.findall(r"\$\{(SHIPINFER_RTSP_[A-Z_]+):?[-}]", wrapper))

    assert read, "the wrapper reads no SHIPINFER_RTSP_* variable; this test is now vacuous"
    missing = sorted(name for name in read if f"-e {name}" not in text)
    assert not missing, (
        f"{missing} is read inside the container and never forwarded into it by "
        f"{runner.name}, so the default is the only reachable value -- these scripts list "
        f"the environment the run gets"
    )


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_the_cpp_binary_a_runner_names_is_one_the_build_produces(runner: Path) -> None:
    """`profile.sh --cpp` named `shipinfer_pipeline` from the day it was written and
    `scripts/build_csrc.py` has never produced it -- the binaries are `csrc/shipinfer/cli/*.cpp`
    -- so the C++ half of the profiler could not run at all. A default that names nothing is
    the one failure a script like this cannot report."""
    text = runner.read_text(encoding="utf-8")
    built = {path.stem for path in CLI.glob("*.cpp")}
    # BOTH SPELLINGS, because the default lives inline in one script and in its own variable
    # in the other, and a pattern that matches neither passes on anything.
    named = set(re.findall(r"SHIPINFER_CPP_BINARY:-([a-z_]+)\}", text))
    named |= set(re.findall(r"csrc/build/([a-z_]+)\b", text))

    assert built, f"no entry points under {CLI}; this test is now vacuous"
    assert named, (
        f"{runner.name} names no binary under csrc/build/ that this test can see -- the "
        f"spelling moved, and a check that finds nothing passes on everything"
    )
    unknown = sorted(name for name in named if name not in built)
    assert not unknown, (
        f"{runner.name} names {unknown} under csrc/build/, and the build produces "
        f"{sorted(built)} -- a binary nobody builds makes the flag that selects it unusable"
    )
