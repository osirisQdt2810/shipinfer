"""The C++ apps the offline build cannot reach are still syntax-checked, when they can be.

`scripts/build_csrc.py --offline` builds only the apps whose include closure never reaches
`core/platform.h`, and CI's `cpp-offline` job builds nothing else. `csrc/shipinfer/cli/bench.cpp`
reaches it through `pipeline/graph/state.h`, so **nothing in this repository ever compiles it** —
and it is the only caller of the perception-event writer.

That is not hypothetical. A `std::mutex` and a `std::map` were used in its sink without being
declared, every test stayed green, and the defect was found by a reviewer reading the diff. A
syntax check is cheap, needs no device, and would have said so in two seconds.

Skipped by name when the CUDA or TensorRT headers are absent, which is CI's case: this closes
the hole on the machines that run the bench, and the ledger carries the CI half.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
#: Where this host keeps the two header sets, matching `build_csrc.py`'s own defaults.
INCLUDES = (Path("/usr/local/cuda/include"), Path("/usr/local/TensorRT/include"))


def _build_module():
    """`build_csrc.py` by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "build_csrc", ROOT / "scripts" / "build_csrc.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cuda_reaching_apps() -> list[Path]:
    """Every app `--offline` refuses, asked of the build script rather than guessed."""
    build = _build_module()
    apps = sorted((CSRC / "shipinfer" / "cli").glob("*.cpp")) + sorted(
        (CSRC / "tests").glob("*.cpp")
    )
    return [app for app in apps if not build.offline_ready(build.include_closure(app), set())]


pytestmark = [
    pytest.mark.skipif(shutil.which("g++") is None, reason="no g++ on PATH"),
    pytest.mark.skipif(
        not all(p.is_dir() for p in INCLUDES),
        reason=f"needs the CUDA and TensorRT headers at {[str(p) for p in INCLUDES]}; "
        f"CI has neither, which is why nothing there compiles these apps",
    ),
]


class TestTheAppsOfflineCannotBuild:
    def test_there_is_at_least_one_of_them(self) -> None:
        """Without this the check below passes by having nothing to compile."""
        found = _cuda_reaching_apps()

        assert found, "no CUDA-reaching app found; this guard would be vacuous"

    def test_each_one_still_compiles(self) -> None:
        """`-fsyntax-only`: no link, no device, no measurement -- just "is this valid C++".

        One subprocess per app, ~2 s each on this host. It is the whole cost of never again
        merging a translation unit that does not compile because no build reaches it.
        """
        failures: list[str] = []
        for app in _cuda_reaching_apps():
            done = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-fsyntax-only",
                    f"-I{CSRC}",
                    *(f"-I{path}" for path in INCLUDES),
                    str(app),
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            if done.returncode != 0:
                errors = [line for line in done.stderr.splitlines() if ": error:" in line][:3]
                failures.append(f"{app.relative_to(ROOT)}:\n  " + "\n  ".join(errors))

        assert not failures, "these do not compile:\n" + "\n".join(failures)
