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
#: What these apps need, probed with `g++` rather than `is_dir()`: a distribution puts these
#: headers on the DEFAULT include path, where no `-I` names them. `NvInferPlugin.h` is here
#: because `engine.cpp` includes it and NVIDIA ships it SEPARATELY -- a probe certifying a
#: smaller set than the check needs passes, and then the check fails.
_PROBE = (
    "#include <NvInfer.h>\n"
    "#include <NvInferPlugin.h>\n"
    "#include <cuda_runtime.h>\n"
    "int main() { return 0; }\n"
)

#: Set by CI's `cpp-syntax` job: a SKIP is the hole this check exists to close, so where the
#: headers are meant to be installed their absence has to be a failure with a reason.
_REQUIRE = "SHIPINFER_REQUIRE_CSRC_HEADERS"


def _include_flags() -> list[str]:
    """The `-I` flags these apps need, or `[]` when the headers are already on the path.

    `CUDA_HOME` and `SHIPINFER_TENSORRT_DIR` are read exactly as `scripts/build_csrc.py`
    reads them -- the first version honoured only the second, so a box with CUDA elsewhere
    skipped silently while `shipinfer bench` built fine. The `targets/<arch>/include` layout
    is there because that is where `cuda-cudart-dev-12-x` puts the headers on a runner.
    """
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    tensorrt = Path(os.environ.get("SHIPINFER_TENSORRT_DIR", "/usr/local/TensorRT"))
    candidates = (
        cuda / "include",
        cuda / "targets" / "x86_64-linux" / "include",
        tensorrt / "include",
    )
    return [f"-I{path}" for path in candidates if path.is_dir()]


def _build_module():
    """`build_csrc.py` by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "build_csrc", ROOT / "scripts" / "build_csrc.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _apps() -> list[Path]:
    return sorted((CSRC / "shipinfer" / "cli").glob("*.cpp")) + sorted(
        (CSRC / "tests").glob("*.cpp")
    )


def _cuda_reaching_apps() -> list[Path]:
    """Every app `--offline` refuses, asked of the build script rather than guessed."""
    build = _build_module()
    return [
        app for app in _apps() if not build.offline_ready(build.include_closure(app), set())
    ]


def _lanes_available(build, closure: set[Path]) -> bool:
    """Whether every external lane this unit needs is installed, asked of `pkg-config`."""
    for lane in build.lanes_in(closure):
        try:
            build.pkg_config_flags(lane)
        except SystemExit:
            return False  # `pkg-config` answered, and the package is not installed
        except FileNotFoundError:
            return False  # no `pkg-config` binary at all, which is the same answer
    return True


def _uncompiled_units() -> list[Path]:
    """Implementation units nothing in this repository compiles, lanes permitting.

    `-fsyntax-only` on an APP does not parse the `.cpp` files in its closure, so the eight
    units outside the offline build were covered by nothing even with the apps checked --
    including `backends/tensorrt/engine.cpp`, where `initLibNvInferPlugins` lives. Two need an
    external lane (opencv, gstreamer) and are skipped where it is absent, which is the answer
    `build_csrc.py` gives. `.cpp` only: `runtime/ops.cu` stays covered by nothing here,
    because `g++` cannot parse CUDA and `nvcc` is the device tier's job.
    """
    build = _build_module()
    apps = set(_apps())
    units = [p for p in sorted((CSRC / "shipinfer").rglob("*.cpp")) if p not in apps]
    outside = [u for u in units if not build.offline_ready(build.include_closure(u), set())]
    return [u for u in outside if _lanes_available(build, build.include_closure(u))]


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


def _missing_headers_reason() -> str:
    return (
        "needs g++ plus the CUDA and TensorRT headers, on the include path or under "
        "CUDA_HOME / SHIPINFER_TENSORRT_DIR; without them nothing in this repository "
        "compiles these apps"
    )


pytestmark = pytest.mark.skipif(
    not _headers_available() and not os.environ.get(_REQUIRE), reason=_missing_headers_reason()
)


def test_the_headers_are_present_where_they_are_required() -> None:
    """Where CI installs them, a SKIP is the defect -- so it is a FAILURE with a reason.

    The job used to assert this by grepping pytest's `-q` summary for "skipped", which is a
    string match on output whose shape changes between versions, in a `| tee` pipeline whose
    status was `tee`'s. A test that fails is the same statement without either problem.
    """
    if not os.environ.get(_REQUIRE):
        pytest.skip(f"{_REQUIRE} is unset; this asserts what CI's cpp-syntax job installs")

    assert _headers_available(), _missing_headers_reason()


def _compiles(path: Path, extra: list[str]) -> tuple[bool, str]:
    """`-fsyntax-only`: no link, no device, no measurement -- just "is this valid C++"."""
    done = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-fsyntax-only",
            f"-I{CSRC}",
            *_include_flags(),
            *extra,
            str(path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    # ` error:` and not `: error:` -- gcc spells a missing header `fatal error:`, which the
    # narrower filter dropped, so the one failure this job exists to report arrived as a
    # filename and a blank line. The stderr tail is the fallback: a check that fails without
    # saying why is the same defect as a check that skips without saying why.
    errors = [line for line in done.stderr.splitlines() if " error:" in line][:3]
    return done.returncode == 0, "\n  ".join(errors or done.stderr.splitlines()[-3:])


class TestAFailureArrivesWithItsReason:
    """The whole thesis of this job, applied to itself: a red check has to say what broke.

    The first version filtered gcc's stderr for `": error:"`. gcc spells a missing include
    `fatal error:`, which that filter dropped — so the one failure this job exists to report
    (a header the runner does not install) would have arrived as a filename followed by a
    blank line, and a red main would have had to be reproduced by hand to be read.
    """

    def test_a_missing_header_is_reported_by_name(self, tmp_path: Path) -> None:
        unit = tmp_path / "missing_header.cpp"
        unit.write_text("#include <NoSuchHeaderExists.h>\nint main() { return 0; }\n")

        ok, reason = _compiles(unit, [])

        assert not ok
        assert "NoSuchHeaderExists.h" in reason, "gcc says `fatal error:`, not `error:`"

    def test_an_ordinary_error_is_still_reported(self, tmp_path: Path) -> None:
        """The widened filter must not have lost the case the narrow one did catch."""
        unit = tmp_path / "bad_syntax.cpp"
        unit.write_text("int main() { return undeclared_thing; }\n")

        ok, reason = _compiles(unit, [])

        assert not ok
        assert "undeclared_thing" in reason

    def test_a_unit_that_compiles_reports_nothing(self, tmp_path: Path) -> None:
        """Non-vacuity: the fallback tail must not invent a reason for a clean compile."""
        unit = tmp_path / "fine.cpp"
        unit.write_text("int main() { return 0; }\n")

        assert _compiles(unit, []) == (True, "")


class TestTheUnitsNothingCompiles:
    """The eight `.cpp` files outside the offline build, which the app check does not reach."""

    def test_there_are_some(self) -> None:
        assert _uncompiled_units(), "no uncompiled unit found; this guard would be vacuous"

    def test_each_one_compiles(self) -> None:
        build = _build_module()
        failures: list[str] = []
        for unit in _uncompiled_units():
            lanes = build.lanes_in(build.include_closure(unit))
            extra = [
                flag
                for lane in lanes
                for flag in build.pkg_config_flags(lane)
                if flag.startswith("-I")
            ]
            ok, errors = _compiles(unit, extra)
            if not ok:
                failures.append(f"{unit.relative_to(ROOT)}:\n  {errors}")

        assert not failures, "these do not compile:\n" + "\n".join(failures)


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
            ok, errors = _compiles(app, [])
            if not ok:
                failures.append(f"{app.relative_to(ROOT)}:\n  {errors}")

        assert not failures, "these do not compile:\n" + "\n".join(failures)
