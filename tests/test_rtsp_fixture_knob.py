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

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "cpp_bench_over_rtsp.sh"
RUNNER = REPO_ROOT / "deploy" / "rootless" / "cpp.sh"


def test_every_rtsp_variable_the_wrapper_reads_is_forwarded_into_the_container() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    read = set(re.findall(r"\$\{(SHIPINFER_RTSP_[A-Z_]+):?[-}]", wrapper))

    assert read, "the wrapper reads no SHIPINFER_RTSP_* variable; this test is now vacuous"
    missing = sorted(name for name in read if f"-e {name}" not in runner)
    assert not missing, (
        f"{missing} is read inside the container and never forwarded into it, so the default "
        f"is the only reachable value -- `cpp.sh` lists the environment the run gets"
    )
