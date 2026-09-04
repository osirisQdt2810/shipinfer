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

import functools
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
#: Where this host keeps the two header sets, matching `build_csrc.py`'s own defaults.
#: `SHIPINFER_TENSORRT_DIR` moves the second, as it does for the build script.
INCLUDES = (Path("/usr/local/cuda/include"), Path("/usr/local/TensorRT/include"))
#: A translation unit that needs exactly what these apps need. Probed with `g++` rather than
#: tested with `is_dir()`, because a distribution's packages put `NvInfer.h` and
#: `cuda_runtime.h` on the DEFAULT include path (`/usr/include/x86_64-linux-gnu`) where no
#: `-I` names them -- and the `is_dir()` guard then skipped on the one machine that had them.
_PROBE = "#include <NvInfer.h>\n#include <cuda_runtime.h>\nint main() { return 0; }\n"


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


def _include_flags() -> list[str]:
    """The `-I` flags these apps need, or `[]` when the headers are already on the path."""
    tensorrt = Path(os.environ.get("SHIPINFER_TENSORRT_DIR", "/usr/local/TensorRT"))
    candidates = (Path("/usr/local/cuda/include"), tensorrt / "include")
    return [f"-I{path}" for path in candidates if path.is_dir()]


@functools.cache
def _headers_available() -> bool:
    """Whether `NvInfer.h` and `cuda_runtime.h` can be found at all, asked of the compiler."""
    if shutil.which("g++") is None:
        return False
    done = subprocess.run(
        ["g++", "-std=c++17", "-fsyntax-only", *_include_flags(), "-x", "c++", "-"],
        input=_PROBE,
        capture_output=True,
        text=True,
    )
    return done.returncode == 0


pytestmark = pytest.mark.skipif(
    not _headers_available(),
    reason="needs g++ plus the CUDA and TensorRT headers, on the include path or under "
    "SHIPINFER_TENSORRT_DIR; without them nothing in this repository compiles these apps",
)


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
                    *_include_flags(),
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
