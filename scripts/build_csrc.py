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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--debug", action="store_true", help="-O0 -g instead of -O2")
    args = parser.parse_args()

    # The tree mirrors `src/shipinfer/` — a thing's header and its translation unit live next
    # to each other, so there is no `include/` to search separately and the include root is
    # `csrc/` itself. Entry points are `cli/` (the binaries) and `tests/`.
    pkg = CSRC / "shipinfer"
    apps = sorted((pkg / "cli").glob("*.cpp")) + sorted((CSRC / "tests").glob("*.cpp"))
    sources = [q for q in sorted(pkg.rglob("*.cpp")) if q not in set(apps)]
    cuda_sources = sorted((CSRC / "shipinfer").rglob("*.cu"))
    if not sources:
        raise SystemExit(f"no sources under {CSRC}")
    if not (TENSORRT / "include" / "NvInfer.h").is_file():
        raise SystemExit(
            f"no TensorRT headers under {TENSORRT}. Set SHIPINFER_TENSORRT_DIR to the "
            f"install root."
        )
    for tool in ("g++", "nvcc"):
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
    includes = [
        f"-I{CSRC}",
        f"-I{TENSORRT / 'include'}",
        f"-I{CUDA / 'include'}",
    ]

    objects: list[str] = []
    nvcc = str(CUDA / "bin" / "nvcc") if (CUDA / "bin" / "nvcc").is_file() else "nvcc"
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

    cv_flags = opencv_flags()
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

    # One library of shared objects, then one binary per entry point. The test binary links
    # the same objects the pipeline does, so a test cannot pass against different code.
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
        print(f"link  {binary.name}")
        run(
            [
                "g++",
                "-std=c++17",
                *optimise,
                *objects,
                str(obj),
                "-o",
                str(binary),
                f"-L{TENSORRT / 'lib'}",
                f"-L{CUDA / 'lib64'}",
                f"-Wl,-rpath,{TENSORRT / 'lib'}",
                f"-Wl,-rpath,{CUDA / 'lib64'}",
                "-lnvinfer",
                "-lnvinfer_plugin",
                "-lcudart",
                "-pthread",
                *[f for f in cv_flags if not f.startswith("-I")],
            ],
            f"linking {binary.name}",
        )
        print(f"built {binary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
