#!/usr/bin/env python3
"""Build the C++ data plane on the host, the way the baseline binary is already built.

WHY ON THE HOST
---------------
`benchmarks/harness/baseline.py` compiles `sim_pipeline_v2.cpp` on the host and stages its
shared-library closure into a directory the container mounts, and the reason applies unchanged
here: the container image has a toolchain but no OpenCV, the host has both, and TensorRT lives
on the host at `/usr/local/TensorRT`. The *run* is still in a container — that is the rule, and
it is about measurements, which a compile is not.

Host `nvcc` is 11.5 against a 12.6 driver. Fine for this box: the A5000s are `sm_86`, which
11.5 supports, and a cudart-linked binary is forward-compatible with a newer driver. It is
also why `sm_89` cannot be built here, and why a production build belongs in a container with
a matching toolkit.

No CMake. Two translation units and a link is a shell command, and adding a build system would
mean the container needs one too.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
BUILD = CSRC / "build"
TENSORRT = Path(os.environ.get("SHIPINFER_TENSORRT_DIR", "/usr/local/TensorRT"))
CUDA = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
#: sm_86 is the RTX A5000 on this box. sm_80 covers the A100s a deployment might use, and both
#: are supported by CUDA 11.5. Listing them explicitly rather than using `-arch=native` keeps
#: the artefact portable across the two devices this project actually targets.
ARCHES = ("86", "80")


def run(command: list[str], what: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{what} failed (exit {result.returncode}):\n{result.stderr[-6000:]}")


def opencv_flags() -> list[str]:
    probe = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "opencv4"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "pkg-config could not find opencv4. The replay source decodes JPEGs with it, the "
            "same way the baseline letterboxes with it, so the build needs it."
        )
    return probe.stdout.split()


_INCLUDE = re.compile(r'^\s*#include\s+"(shipinfer/[^"]+)"', re.MULTILINE)


def include_closure(app: Path) -> set[Path]:
    """Every translation unit ``app`` reaches through ``#include "shipinfer/..."`` lines.

    A header's definitions live in the ``.cpp`` (or ``.cu``) beside it with the same stem —
    the layout rule this tree follows — so reaching a header means linking its unit. Returns
    the units and the headers, so :func:`needs_accelerator` can look for ``core/platform.h``.
    """
    seen: set[Path] = set()
    todo = [app]
    while todo:
        path = todo.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for name in _INCLUDE.findall(path.read_text(errors="replace")):
            header = CSRC / name
            todo.append(header)
            todo.extend(header.with_suffix(suffix) for suffix in (".cpp", ".cu"))
    return seen


def needs_accelerator(closure: set[Path]) -> bool:
    """Whether any unit in the closure includes the driver's headers (through ``platform.h``).

    Keyed on ``core/platform.h`` alone because that is the one header allowed to name a
    vendor runtime (the architecture test enforces it). A header that included ``<NvInfer.h>``
    without ``platform.h`` would be misclassified — and fail loudly at link, not silently.
    """
    return any(p.name == "platform.h" and p.parent.name == "core" for p in closure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="build only the apps whose includes never reach core/platform.h, with g++ alone: "
        "no nvcc, no CUDA, no TensorRT — the C++ offline tier, runnable on a machine with no "
        "driver",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--debug", action="store_true", help="-O0 -g instead of -O2")
    args = parser.parse_args()

    # The tree mirrors `src/shipinfer/` — a thing's header and its translation unit live next
    # to each other, so there is no `include/` to search separately and the include root is
    # `csrc/` itself. Entry points are `cli/` (the binaries) and `tests/`.
    pkg = CSRC / "shipinfer"
    apps = sorted((pkg / "cli").glob("*.cpp")) + sorted((CSRC / "tests").glob("*.cpp"))
    closures = {app: include_closure(app) for app in apps}
    cuda_free = {app for app, closure in closures.items() if not needs_accelerator(closure)}
    if args.offline:
        # Only the apps whose include closure never reaches `core/platform.h`: buildable and
        # runnable on a machine with no driver, no nvcc and no TensorRT — the C++ offline tier.
        apps = [app for app in apps if app in cuda_free]
        if not apps:
            raise SystemExit("no CUDA-free apps found under csrc/")
    sources = [q for q in sorted(pkg.rglob("*.cpp")) if q not in set(apps)]
    cuda_sources = sorted((CSRC / "shipinfer").rglob("*.cu"))
    if not sources:
        raise SystemExit(f"no sources under {CSRC}")
    if not args.offline and not (TENSORRT / "include" / "NvInfer.h").is_file():
        raise SystemExit(
            f"no TensorRT headers under {TENSORRT}. Set SHIPINFER_TENSORRT_DIR to the "
            f"install root."
        )
    needed_tools = ["g++"] if args.offline else ["g++", "nvcc"]
    for tool in needed_tools:
        if shutil.which(tool) is None and not (CUDA / "bin" / tool).is_file():
            raise SystemExit(f"{tool} is not on PATH and not under {CUDA / 'bin'}")

    newest = max(
        p.stat().st_mtime for p in [*sources, *apps, *cuda_sources, *CSRC.rglob("*.h")]
    )
    targets = [BUILD / p.stem for p in apps]
    if not args.force and all(t.is_file() and t.stat().st_mtime >= newest for t in targets):
        print("up to date: " + ", ".join(str(t) for t in targets))
        return 0

    BUILD.mkdir(parents=True, exist_ok=True)
    optimise = ["-O0", "-g"] if args.debug else ["-O2"]
    includes = [f"-I{CSRC}"]
    if not args.offline:
        includes += [f"-I{TENSORRT / 'include'}", f"-I{CUDA / 'include'}"]

    objects: list[str] = []
    object_of: dict[Path, str] = {}
    nvcc = str(CUDA / "bin" / "nvcc") if (CUDA / "bin" / "nvcc").is_file() else "nvcc"
    # The CUDA-free *units*: every translation unit whose own include closure never reaches
    # `core/platform.h`. A CUDA-free binary links all of them, not just the ones its own
    # closure names — policies register through file-scope `PolicyRegistrar`s, so a unit
    # left off the link line is a policy missing from that binary's registry, and two
    # binaries would then answer `build_policy("power_of_two")` differently.
    free_units = {q for q in sources if not needs_accelerator(include_closure(q))}
    if args.offline:
        # Only CUDA-free units are compiled: the offline build must not even *compile* a
        # unit that includes the driver's headers.
        sources = [q for q in sources if q in free_units]
        cuda_sources = []
    for source in cuda_sources:
        # Named by the path under csrc/, not the stem: two files with one stem in different
        # directories used to overwrite each other's object and one was never linked.
        obj = BUILD / (str(source.relative_to(CSRC)).replace("/", "__") + ".o")
        gencode: list[str] = []
        for arch in ARCHES:
            gencode += [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
        print(f"nvcc  {source.name}")
        run(
            [
                nvcc,
                "-std=c++17",
                *optimise,
                "-c",
                str(source),
                "-o",
                str(obj),
                "--compiler-options",
                "-fPIC",
                *gencode,
                *includes,
            ],
            f"compiling {source.name}",
        )
        objects.append(str(obj))
        object_of[source] = str(obj)

    # OpenCV is reached only through `ingest/sources/replay.*`, which no CUDA-free closure
    # contains — so the offline build must not even ask for it: `pkg-config` refusing on a
    # runner with no OpenCV was the third undeclared prerequisite of a flag that promised
    # "g++ alone".
    cv_flags = [] if args.offline else opencv_flags()
    for source in sources:
        obj = BUILD / (str(source.relative_to(CSRC)).replace("/", "__") + ".o")
        print(f"g++   {source.name}")
        run(
            [
                "g++",
                "-std=c++17",
                *optimise,
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "-c",
                str(source),
                "-o",
                str(obj),
                *includes,
                *[f for f in cv_flags if f.startswith("-I")],
            ],
            f"compiling {source.name}",
        )
        objects.append(str(obj))
        object_of[source] = str(obj)

    # One library of shared objects, then one binary per entry point. A CUDA binary links
    # every object; a CUDA-free binary links every CUDA-free object — so a registrar's unit is
    # always on the line of any binary that can link it, and the registry is the same in
    # every binary. A test cannot pass against different code than the binary runs.
    for app in apps:
        obj = BUILD / f"{app.stem}.o"
        print(f"g++   {app.name}")
        run(
            [
                "g++",
                "-std=c++17",
                *optimise,
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "-c",
                str(app),
                "-o",
                str(obj),
                *includes,
                *[f for f in cv_flags if f.startswith("-I")],
            ],
            f"compiling {app.name}",
        )
        binary = BUILD / app.stem
        print(f"link  {binary.name}" + ("  (CUDA-free)" if app in cuda_free else ""))
        if app in cuda_free:
            # Every CUDA-free object, and no accelerator library: `ldd` on the result must
            # show neither libcuda nor libnvinfer, which is what makes it runnable — and
            # meaningful — on a machine with no driver; every CUDA-free registrar is linked,
            # which is what makes its registry the production one.
            link_objects = [object_of[q] for q in sorted(free_units) if q in object_of]
            link_libs = ["-pthread"]
        else:
            link_objects = objects
            link_libs = [
                f"-L{TENSORRT / 'lib'}",
                f"-L{CUDA / 'lib64'}",
                f"-Wl,-rpath,{TENSORRT / 'lib'}",
                f"-Wl,-rpath,{CUDA / 'lib64'}",
                "-lnvinfer",
                "-lnvinfer_plugin",
                "-lcudart",
                "-pthread",
                *[f for f in cv_flags if not f.startswith("-I")],
            ]
        run(
            [
                "g++",
                "-std=c++17",
                *optimise,
                *link_objects,
                str(obj),
                "-o",
                str(binary),
                *link_libs,
            ],
            f"linking {binary.name}",
        )
        print(f"built {binary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
