"""The C++ apps the offline build cannot reach are still syntax-checked, when they can be.

`scripts/build_csrc.py --offline` builds only the apps whose include closure never reaches
`core/platform.h`, and CI's `cpp-offline` job builds nothing else. `csrc/shipinfer/cli/bench.cpp`
reaches it through `pipeline/graph/state.h`, so **nothing in this repository ever compiles it** —
and it is the only caller of the perception-event writer.

That is not hypothetical. A `std::mutex` and a `std::map` were used in its sink without being
declared, every test stayed green, and the defect was found by a reviewer reading the diff.

The two compile classes are skipped BY NAME where the CUDA and TensorRT headers are absent,
and CI's `cpp-syntax` job installs them -- so there a skip is the defect, and
`SHIPINFER_REQUIRE_CSRC_HEADERS` turns it into a failure naming which headers are short.
`TestAFailureArrivesWithItsReason` needs only `g++`, so it runs everywhere.
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


@functools.cache
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


# doc: long why a syntax check reads `lanes_of` and not `lanes_in`, which cost `bench.cpp`
def _lanes_needed(build, unit: Path) -> set[str]:
    """The lanes COMPILING this unit needs -- `lanes_of`, deliberately not `lanes_in`.

    `lanes_in(closure)` is documented as wider on purpose: "compiling `cli/bench.cpp` needs no
    OpenCV header, but *linking* it needs `replay.cpp`'s object". `-fsyntax-only` never links,
    so gating a compile-only check on a link-time fact skipped the one app this whole file
    exists to compile -- everywhere `libopencv-dev` is absent, which is exactly CI's runner
    (#133 round 3). Measured here: `lanes_in(closure(bench.cpp))` is `{'opencv'}` while
    `lanes_of(bench.cpp)` is empty, and no HEADER in that closure declares a lane at all --
    `EXTERNAL` names only `replay.cpp` and `gstreamer.cpp`, both `.cpp`.
    """
    return build.lanes_of(unit)


def _lanes_available(build, unit: Path) -> bool:
    """Whether every external lane COMPILING this unit needs is installed, per `pkg-config`."""
    for lane in _lanes_needed(build, unit):
        try:
            build.pkg_config_flags(lane)
        except SystemExit:
            return False  # `pkg-config` answered, and the package is not installed
        except FileNotFoundError:
            return False  # no `pkg-config` binary at all, which is the same answer
    return True


def _lane_flags(build, unit: Path) -> list[str]:
    """The `-I` flags an external lane needs, asked of `pkg-config` through `build_csrc.py`."""
    return [
        flag
        for lane in _lanes_needed(build, unit)
        for flag in build.pkg_config_flags(lane)
        if flag.startswith("-I")
    ]


# doc: long the predicate is "in no built closure", and the narrower one it replaced
def _offline_built_units() -> set[Path]:
    """Every `.cpp` the `--offline` build actually compiles: the closure of each app it builds.

    Which is the question, and it took a round to say so. `-fsyntax-only` on an APP does not
    parse the `.cpp` files in its closure, so a unit is compiled only if some app the offline
    build BUILDS links it -- not merely if the unit itself is offline-ready.
    """
    build = _build_module()
    built: set[Path] = set()
    for app in _apps():
        closure = build.include_closure(app)
        if build.offline_ready(closure, set()):
            built |= {path for path in closure if path.suffix == ".cpp"}
    return built


# doc: long the predicate this replaced excluded the one unit the ticket was opened for
def _uncompiled_units() -> list[Path]:
    """Implementation units nothing in this repository compiles, lanes permitting.

    IN NO BUILT CLOSURE, not "not offline-ready" -- which is the ticket's own thesis and what
    the first version got wrong. `obs/sampler.cpp` IS offline-ready, so that predicate filtered
    it out; no app the offline build compiles reaches it, so `cpp-offline` never built it
    either. One unit, compiled by nothing, excluded from the check that exists to find exactly
    that (#133 round 3, note 1).

    Units needing an external lane are skipped where it is absent, which is the answer
    `build_csrc.py` gives. `.cpp` only: `runtime/ops.cu` stays covered by nothing here, because
    `g++` cannot parse CUDA and `nvcc` is the device tier's job.
    """
    build = _build_module()
    apps = set(_apps())
    built = _offline_built_units()
    units = [p for p in sorted((CSRC / "shipinfer").rglob("*.cpp")) if p not in apps]
    return [u for u in units if u not in built and _lanes_available(build, u)]


def _needs_driver_headers(unit: Path) -> bool:
    """Whether compiling this unit reaches a vendor header, so the class gate can be per-unit.

    `core/platform.h` is the one header allowed to name a vendor runtime, and
    `build_csrc.py::needs_accelerator` keys on exactly that. Splitting on it is what lets the
    units that need NO driver headers -- `obs/sampler.cpp` and the policies -- be checked on a
    plain runner instead of only on the one that installs CUDA's.
    """
    return _build_module().needs_accelerator(_build_module().include_closure(unit))


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


# doc: long the header-to-package map, and the round it cost
#: A header a package ships, and the package that ships it -- consulted only after the probe
#: has failed, to turn a CUDA-internal include error into an apt line. `crt/host_defines.h` is
#: the one that cost a round: `cuda-cudart-dev` ships `cuda_runtime.h` but not the `crt/`
#: headers it includes (#133 round 5), and a dev box's full toolkit cannot see that.
_HEADER_PACKAGES: tuple[tuple[str, str], ...] = (
    ("crt/host_defines.h", "cuda-crt-<major>-<minor> (headers only, ~881 KB, no nvcc)"),
    # `cuda_runtime.h` includes this one and `cuda_runtime_api.h` the other; same package, and
    # both are named so a partial install cannot produce a message that looks complete.
    ("crt/host_config.h", "cuda-crt-<major>-<minor>"),
    ("NvInferPlugin.h", "libnvinfer-headers-plugin-dev"),
    ("NvInfer.h", "libnvinfer-headers-dev"),
    ("cuda_runtime.h", "cuda-cudart-dev-<major>-<minor>"),
)


# doc: long why this asks the compiler and not the filesystem
def _absent_headers(table: tuple[tuple[str, str], ...] = _HEADER_PACKAGES) -> list[str]:
    """Which of :data:`_HEADER_PACKAGES` the COMPILER cannot find, one probe each.

    Asked of `g++`, not of `is_file()`, for the reason `_PROBE` states thirty lines up: a
    distribution puts these on the DEFAULT include path, where no `-I` names them. The first
    version walked `_include_flags()` plus `/usr/include` -- and the runner's own apt packages
    put TensorRT's headers under `/usr/include/x86_64-linux-gnu`, so `_headers_available()` was
    True while this reported two of them absent (#133 round 6). Walking a list of roots trades
    one confusing failure for a maintained list of layouts.

    Only reached after the aggregate probe has already failed, so the cost is a handful of
    subprocesses on a path that is about to end the job anyway.
    """
    if shutil.which("g++") is None:
        # Reached at IMPORT time, because `needs_headers`'s `reason=` calls this -- so an
        # unguarded `subprocess.run(["g++", ...])` here is a COLLECTION error rather than a
        # skip, on any box without a toolchain. Which is the offline tier's own image. Found by
        # rehearsing `env -i PATH=/tmp/nobin` after the per-header probe replaced the walk.
        return []
    absent: list[str] = []
    for header, package in table:
        done = subprocess.run(
            ["g++", "-std=c++17", "-fsyntax-only", *_include_flags(), "-x", "c++", "-"],
            input=f"#include <{header}>\nint main() {{ return 0; }}\n",
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            absent.append(f"{header} -> install {package}")
    return absent


def _missing_headers_reason() -> str:
    reason = (
        "needs g++ plus the CUDA and TensorRT headers, on the include path or under "
        "CUDA_HOME / SHIPINFER_TENSORRT_DIR; without them nothing in this repository "
        "compiles these apps"
    )
    absent = _absent_headers()
    return reason if not absent else reason + ". Not found: " + "; ".join(absent)


# doc: long which lanes are another job's, and what adding one costs
#: External lanes a DIFFERENT CI job compiles, so this one may drop their units. `gstreamer`
#: and `nvdec` are both `cpp-gst-lane`'s, which builds `--with-external gstreamer --with-external
#: nvdec`. Everything NOT in here is installed in this job -- `opencv` is, because
#: `ingest/sources/replay.cpp` is compiled by nothing else anywhere (#133 round 4). A lane
#: added here without a job named beside it is a unit going quietly uncovered.
#:
#: `nvdec` arrived RED: #156 landed the lane and nothing in CI could resolve `ffnvcodec`, so
#: this file's own drop guard failed and `cpp-syntax` was red on main for five runs before
#: anyone looked. Adding the lane to `cpp-gst-lane` (which already has the GStreamer packages,
#: and needs only nv-codec-headers on top) is the fix that keeps the guard's meaning: a dropped
#: unit is still the hole this job exists to close.
#: `shipvision` is `cpp-shipvision-lane`'s, and it is the one lane no `-dev` package could
#: satisfy: it is an in-tree submodule this job checks out with `submodules: false`, on
#: purpose (the offline tier must pass without the kernels). So its unit is dropped here and
#: compiled there, by name, with the submodule fetched over anonymous https.
_COVERED_ELSEWHERE = frozenset({"gstreamer", "nvdec", "shipvision"})

# doc: long the guard the module-level pytestmark used to carry, and what needs it
#: `TestAFailureArrivesWithItsReason` is deliberately NOT `needs_headers`-gated -- it needs
#: only a compiler -- but it shells out to `g++`, and the offline tier's image is a `-runtime`
#: one with no toolchain. Without this the tier goes from clean to three raw tracebacks, which
#: is this file's own complaint about checks that fail without saying why (#133 round 6).
no_gpp = pytest.mark.skipif(shutil.which("g++") is None, reason="no g++ on PATH")

#: Applied to the two COMPILE classes and not to the module: the guard tests below need `g++`
#: and nothing else, so a module-level mark would have run them only on the `cpp-syntax`
#: runner -- a harness whose own guards execute in one place is a harness nobody checks.


# doc: long why this is a fixture and not a `skipif`, which probed at import time
#: Applied with `@pytest.mark.usefixtures`, because a `skipif` evaluates its condition -- and
#: its `reason` -- WHEN THE MODULE IS IMPORTED. Both call `_headers_available()`, which shells
#: out to `g++`: so a plain offline `pytest` collecting this file paid a compiler spawn for
#: classes it was about to skip, on every run, including on a box with no `g++` where the
#: answer is known without asking. As a fixture the probe happens only if one of these tests
#: actually runs (#133 round 3, note 2).
@pytest.fixture(scope="session")
def csrc_headers() -> None:
    if not _headers_available() and not os.environ.get(_REQUIRE):
        pytest.skip(_missing_headers_reason())


needs_headers = pytest.mark.usefixtures("csrc_headers")


@needs_headers
def test_the_headers_are_present_where_they_are_required() -> None:
    """Where CI installs them, a SKIP is the defect -- so it is a FAILURE with a reason.

    The job used to assert this by grepping pytest's `-q` summary for "skipped", which is a
    string match on output whose shape changes between versions, in a `| tee` pipeline whose
    status was `tee`'s. A test that fails is the same statement without either problem.
    """
    if not os.environ.get(_REQUIRE):
        pytest.skip(f"{_REQUIRE} is unset; this asserts what CI's cpp-syntax job installs")

    assert _headers_available(), _missing_headers_reason()


# doc: long the flags this passes are the BUILD's, and the two it used to drop
def _build_flags() -> list[str]:
    """`-Wall -Wextra` and `-DSHIPINFER_OMITTED_LANES`, which `scripts/build_csrc.py` passes.

    A syntax check that compiles a unit under different flags from the build is checking a
    different translation unit. The define is the one that matters: `ingest/omitted_lanes.h`
    has an `#ifdef` branch on it, so without it this leg parsed the OTHER side of that branch
    from the one every real binary compiles. `lane_defines(frozenset())` is the shape the
    build passes when no lane is enabled, which is this check's situation for every unit whose
    lanes `pkg-config` could not resolve -- and the value is derived from `EXTERNAL` rather
    than written here, so adding a lane cannot leave this behind.
    """
    return ["-Wall", "-Wextra", *_build_module().lane_defines(_resolvable_lanes())]


@functools.cache
def _resolvable_lanes() -> frozenset[str]:
    """The external lanes `pkg-config` can answer for on THIS host.

    Which is what `--with-external` would be given here, so `lane_defines` names the lanes
    this check genuinely left out. Spelling `frozenset()` instead would tell a unit compiled
    WITH opencv that opencv was omitted -- a define that contradicts the compile it labels.
    """
    build = _build_module()
    resolvable = set()
    for lane in build.EXTERNAL:
        try:
            build.pkg_config_flags(lane)
        except (SystemExit, FileNotFoundError):
            continue
        resolvable.add(lane)
    return frozenset(resolvable)


def _compiles(path: Path, extra: list[str]) -> tuple[bool, str]:
    """`-fsyntax-only`: no link, no device, no measurement -- just "is this valid C++"."""
    done = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-fsyntax-only",
            f"-I{CSRC}",
            *_build_flags(),
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


@no_gpp
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

    def test_the_builds_own_defines_reach_the_compile(self, tmp_path: Path) -> None:
        """`-DSHIPINFER_OMITTED_LANES` is not decoration: `ingest/omitted_lanes.h` has an
        `#ifdef` branch on it, so a check that omits it parses the OTHER side of that branch
        from the one every real binary compiles. Asserted by compiling a unit that REQUIRES
        the define rather than by reading the flag list back, which would only restate
        `_build_flags`.
        """
        unit = tmp_path / "needs_the_define.cpp"
        unit.write_text(
            "#ifndef SHIPINFER_OMITTED_LANES\n"
            '#error "the syntax check must pass the build\'s own defines"\n'
            "#endif\n"
            "int main() { return 0; }\n"
        )

        ok, errors = _compiles(unit, [])

        assert ok, errors


@no_gpp
class TestTheDriverlessUnitsNothingCompiles:
    """The subset needing no vendor header, checked on ANY box with a compiler.

    Split out of the class below, which is `needs_headers`-gated: widening the predicate to "in
    no built closure" brought in units that reach no vendor header at all, and gating those on
    CUDA's headers would have run them only on the one runner that installs them. A harness
    whose checks execute in one place is a harness nobody checks -- the same argument the marks
    above make about themselves.
    """

    def test_each_one_compiles(self) -> None:
        build = _build_module()
        units = [u for u in _uncompiled_units() if not _needs_driver_headers(u)]
        assert units, "no driverless uncompiled unit; this guard would be vacuous"
        failures: list[str] = []
        for unit in units:
            ok, errors = _compiles(unit, _lane_flags(build, unit))
            if not ok:
                failures.append(f"{unit.relative_to(ROOT)}:\n  {errors}")

        assert not failures, "these do not compile:\n" + "\n".join(failures)

    def test_the_sampler_is_one_of_them(self) -> None:
        """Non-vacuity, named: `obs/sampler.cpp` is the unit this widening was opened for."""
        sampler = CSRC / "shipinfer" / "obs" / "sampler.cpp"
        assert sampler in _uncompiled_units(), (
            "obs/sampler.cpp is compiled by nothing -- it is offline-ready, so the old "
            "`not offline_ready` predicate excluded it, and no app the offline build compiles "
            "reaches it either"
        )


@needs_headers
class TestTheUnitsNothingCompiles:
    """The `.cpp` files outside the offline build that DO reach a vendor header."""

    def test_there_are_some(self) -> None:
        assert _uncompiled_units(), "no uncompiled unit found; this guard would be vacuous"

    def test_each_one_compiles(self) -> None:
        build = _build_module()
        failures: list[str] = []
        for unit in _uncompiled_units():
            if not _needs_driver_headers(unit):
                continue  # the class above, on a runner that needs no CUDA packages
            ok, errors = _compiles(unit, _lane_flags(build, unit))
            if not ok:
                failures.append(f"{unit.relative_to(ROOT)}:\n  {errors}")

        assert not failures, "these do not compile:\n" + "\n".join(failures)

    def test_replay_cpp_is_this_jobs_to_compile_and_nobody_elses(self) -> None:
        """Checkable on any host, unlike the guard below, which needs the package absent.

        `_COVERED_ELSEWHERE` is the whole judgement in this file: a lane in it is somebody
        else's job, and a lane out of it must be installed HERE. `opencv` is out of it because
        `ingest/sources/replay.cpp` is compiled by nothing else anywhere -- `cpp-gst-lane`
        builds the other two. So this pins the reasoning rather than the environment: on a host
        that HAS libopencv-dev, the guard below cannot tell a correct exclusion from a missing
        one.
        """
        build = _build_module()
        replay = CSRC / "shipinfer" / "ingest" / "sources" / "replay.cpp"

        assert replay.is_file(), "the unit this reasoning is about"
        assert _lanes_needed(build, replay) == {"opencv"}, "it declares the opencv lane"
        assert "opencv" not in _COVERED_ELSEWHERE, (
            "opencv is not covered by another job, so the cpp-syntax job installs it; putting "
            "it here would drop replay.cpp from every CI job at once"
        )
        # DERIVED, not pinned: the old `== {"gstreamer", "nvdec"}` said its own reason was
        # "a third entry without a job building it is a unit going quietly uncovered", and
        # that PROPERTY is what is worth asserting. A literal set needs editing for every
        # legitimate lane and says nothing about the job, which is the half that can be gone.
        workflow = (ROOT / ".github" / "workflows" / "cpp.yml").read_text(encoding="utf-8")
        unbuilt = [
            lane
            for lane in sorted(_COVERED_ELSEWHERE)
            if f"--with-external {lane}" not in workflow
        ]
        assert not unbuilt, (
            f"{unbuilt} are excused from this job as 'covered elsewhere', but no job in "
            f"cpp.yml builds them (`--with-external <lane>`), so their units are compiled by "
            f"nothing anywhere -- which is the hole this whole file exists to close"
        )

    def test_no_unit_is_dropped_for_a_missing_lane_where_this_is_required(self) -> None:
        """The loud skip belongs on THIS leg, and BOTH legs can now fire.

        It used to say the apps leg was "unreachable by construction" -- true until the
        `shipvision` lane declared a test app, and a comment asserting a dead invariant is
        what cost #169 a round. Here it is reachable and it matters: without `libopencv-dev`,
        `ingest/sources/replay.cpp` falls out and NOTHING in CI compiles it -- `cpp-gst-lane`
        covers `gstreamer.cpp`, not that one -- which is this file's own thesis.
        """
        if not os.environ.get(_REQUIRE):
            pytest.skip(f"{_REQUIRE} is unset; this asserts what CI's cpp-syntax job installs")

        build = _build_module()
        apps = set(_apps())
        units = [p for p in sorted((CSRC / "shipinfer").rglob("*.cpp")) if p not in apps]
        outside = [u for u in units if not build.offline_ready(build.include_closure(u), set())]
        dropped = [
            u.relative_to(ROOT).as_posix()
            for u in outside
            if not _lanes_available(build, u)
            and not (_lanes_needed(build, u) <= _COVERED_ELSEWHERE)
        ]

        assert outside, "no unit is outside the offline build; this guard would be vacuous"
        assert not dropped, (
            f"these units were dropped for a missing external lane: {dropped}. Where this job "
            f"runs, a drop is the hole it exists to close -- install the lane's `-dev` package, "
            f"or add the lane to `_COVERED_ELSEWHERE` and name the job that compiles it"
        )


class TestThisFileStandsAlone:
    """The `--noconftest -c /dev/null` in `ci.yml` is load-bearing, so it is asserted here.

    Every `pytest` in this repository loads `tests/conftest.py`, which imports numpy, pydantic
    and -- on the offline path -- torch, and `pyproject.toml`'s `--strict-config` additionally
    demands pytest-asyncio and pytest-timeout. The `cpp-syntax` job's interpreter is
    `setup-python`'s and has none of them, so it died at COLLECTION with rc=4 before one
    `g++ -fsyntax-only` ran (#133 round 4). Installing torch into a headers-only job is ~200 MB
    for nothing, so the job bypasses the suite's config -- and a bare flag in a workflow is a
    knob a later edit removes without knowing why it was there. This is the why.
    """

    #: Everything this module may import: the standard library, plus pytest. `__future__` is
    #: the compiler's, not a package -- listed because it is an import statement all the same.
    _ALLOWED = frozenset(
        {
            "__future__",
            "ast",
            "functools",
            "importlib",
            "os",
            "pathlib",
            "pytest",
            "re",
            "shutil",
            "subprocess",
            "sys",
            "types",
        }
    )

    def test_a_missing_header_names_the_package_that_ships_it(self) -> None:
        """Because the raw compiler error names a file nobody here has heard of.

        `cuda-cudart-dev` ships `cuda_runtime.h` WITHOUT the `crt/` headers it includes, so the
        first run on main failed with `crt/host_defines.h: No such file or directory` on all
        three compile legs -- which reads as an NVIDIA packaging problem rather than as a short
        apt line (#133 round 5). A dev box has a full toolkit and cannot see it, so the message
        has to carry the answer.
        """
        headers = dict(_HEADER_PACKAGES)

        assert "crt/host_defines.h" in headers, "the one that cost a round"
        assert "cuda-crt" in headers["crt/host_defines.h"], "and it names cuda-crt"
        assert set(headers) >= {"NvInfer.h", "NvInferPlugin.h", "cuda_runtime.h"}, (
            "every header the probe includes needs a package beside it, or its absence "
            "produces an error message with no action in it"
        )

    def test_the_reason_stays_plain_when_the_headers_are_there(self) -> None:
        """On a box that has them, the reason must not grow a misleading `Not found` tail.

        This is what caught round 6: the first version WALKED directories, and the runner's own
        apt packages put TensorRT's headers under `/usr/include/x86_64-linux-gnu` -- on `g++`'s
        default search list and in none of the roots it walked. So `_headers_available()` was
        True while this reported two absent, and the job's first act on `main` was to fail for
        a reason unrelated to any C++ in the tree.
        """
        if not _headers_available():
            pytest.skip("this box has no headers; the tail is correct here")

        assert "Not found" not in _missing_headers_reason()

    def test_a_header_the_compiler_cannot_find_is_reported_with_its_package(self) -> None:
        """The other direction, and checkable anywhere: the probe has to actually report."""
        if shutil.which("g++") is None:
            pytest.skip("no g++ on PATH; the probe returns nothing by design")

        absent = _absent_headers((("definitely/not/here.h", "some-package"),))

        assert absent == ["definitely/not/here.h -> install some-package"]

    def test_this_file_needs_no_conftest(self) -> None:
        """Its imports are stdlib plus pytest, so it runs on an interpreter with only pytest."""
        import ast

        source = Path(__file__).read_text(encoding="utf-8")
        found: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])

        allowed = {name.split(".")[0] for name in self._ALLOWED}
        assert found <= allowed, (
            f"{sorted(found - allowed)} would need installing in the cpp-syntax job, which "
            f"carries pytest and nothing else. Either keep this file self-contained or make "
            f"ci.yml install the dev environment -- but not silently, because the job then "
            f"fails at collection and the redness reads as a packaging problem"
        )

    def test_nothing_probes_the_compiler_at_import(self) -> None:
        """The header probe shells out to `g++`, so it must not run when the module is merely
        collected: a plain offline `pytest` paid a compiler spawn for classes it then skipped.
        A `skipif` did exactly that -- its condition AND its reason are evaluated at import --
        which is why the gate is a fixture (#133 round 3, note 2).
        """
        import ast

        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        inside: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for child in ast.walk(node):
                    inside.add(id(child))
        eager = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"_headers_available", "_missing_headers_reason"}
            and id(node) not in inside
        ]
        assert eager == [], (
            f"{eager} runs at import, so collecting this file spawns a compiler even where "
            f"every class using it is about to skip"
        )

    def test_it_uses_no_fixture_of_ours(self) -> None:
        """`--noconftest` also removes our fixtures, so this file may only use pytest's own."""
        import ast

        ours = {"repository", "server", "settings", "metrics", "device", "ops"}
        source = Path(__file__).read_text(encoding="utf-8")
        used: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                used.update(arg.arg for arg in node.args.args if arg.arg != "self")

        assert not (used & ours), f"these come from conftest.py: {sorted(used & ours)}"
        assert used <= {"tmp_path", "monkeypatch", "capsys", "caplog"}, (
            f"unexpected fixtures {sorted(used)}; only pytest's built-ins survive "
            f"--noconftest"
        )


@needs_headers
class TestTheAppsOfflineCannotBuild:
    def test_there_is_at_least_one_of_them(self) -> None:
        """Without this the check below passes by having nothing to compile."""
        found = _cuda_reaching_apps()

        assert found, "no CUDA-reaching app found; this guard would be vacuous"

    def test_each_one_still_compiles(self) -> None:
        """`-fsyntax-only`: no link, no device, no measurement -- just "is this valid C++".

        Through the same lane reading as the unit check, so the two legs answer one question
        one way -- and that reading is `lanes_of`, not `lanes_in`. Round 3 wrote "it changes
        nothing today", reasoning from a dev box that has `libopencv-dev`; CI's runner does
        not, so `cli/bench.cpp` hit the `continue` and was never compiled there. The job stayed
        green with the app it exists for uncovered.
        """
        build = _build_module()
        failures: list[str] = []
        skipped: list[str] = []
        for app in _cuda_reaching_apps():
            if not _lanes_available(build, app):
                # Covered elsewhere is not a hole: `_COVERED_ELSEWHERE` names the lanes
                # another cpp.yml job builds, and the same reading applies to an APP as to a
                # unit. `test_tracking_shard.cpp` is the first app to declare a lane at all.
                if not _lanes_needed(build, app) <= _COVERED_ELSEWHERE:
                    skipped.append(app.relative_to(ROOT).as_posix())
                continue
            ok, errors = _compiles(app, _lane_flags(build, app))
            if not ok:
                failures.append(f"{app.relative_to(ROOT)}:\n  {errors}")

        assert not failures, "these do not compile:\n" + "\n".join(failures)
        # ARMOUR THAT HAS NOW FIRED, on #169: it used to say `skipped` was always empty
        # because no app declared a lane, and the `shipvision` lane declares a test app. The
        # exclusion above is what keeps this meaning something -- a skip is a hole UNLESS a
        # named job compiles it, the same judgement the unit leg makes.
        if os.environ.get(_REQUIRE):
            assert not skipped, (
                f"these apps were skipped for a missing external lane: {skipped}. Where this "
                f"job runs, a skip is the hole it exists to close -- install the lane's `-dev` "
                f"package or narrow `EXTERNAL` so the unit does not claim it"
            )
