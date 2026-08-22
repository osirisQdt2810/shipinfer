#!/usr/bin/env python3
"""Configure and build the native extension, then verify it imports.

A thin wrapper over CMake, and it exists for one reason: the extension must be built
against the *same* interpreter that will import it, and getting that wrong produces an
``ImportError`` about undefined symbols that tells you nothing about the cause.
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
NATIVE = ROOT / "native"


def discover_cuda_home() -> Path | None:
    """Pick a CUDA toolkit, newest first.

    Hosts accumulate toolkits, and ``/usr/local/cuda`` is often a symlink to whichever one
    was installed first — frequently an old nvcc that cannot parse the system libstdc++ at
    all. Choosing the newest installed toolkit is both more likely to work and more likely
    to match the driver.
    """
    if "CUDA_HOME" in os.environ:
        return Path(os.environ["CUDA_HOME"])
    candidates = []
    for entry in Path("/usr/local").glob("cuda-*"):
        match = re.fullmatch(r"cuda-(\d+)\.(\d+)", entry.name)
        if match and (entry / "bin" / "nvcc").is_file():
            candidates.append(((int(match[1]), int(match[2])), entry))
    if candidates:
        return max(candidates)[1]
    default = Path("/usr/local/cuda")
    return default if (default / "bin" / "nvcc").is_file() else None


def host_compiler_for(nvcc: Path) -> str | None:
    """The newest GCC this nvcc actually supports.

    nvcc rejects a host compiler newer than it knows about, and the failure is a wall of
    template errors from inside libstdc++ that looks like a bug in your own code. The
    mapping below is nvcc's documented maximum supported GCC per release.
    """
    try:
        version = subprocess.run(
            [str(nvcc), "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", version)
    if not match:
        return None
    major, minor = int(match[1]), int(match[2])
    if major >= 12:
        maximum = 13 if minor >= 4 else 12
    elif (major, minor) >= (11, 6):
        maximum = 11
    else:
        maximum = 10
    for candidate in range(maximum, 8, -1):
        path = shutil.which(f"g++-{candidate}")
        if path:
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hip", action="store_true", help="build for ROCm instead of CUDA")
    parser.add_argument("--debug", action="store_true", help="build with -O0 and symbols")
    parser.add_argument("--clean", action="store_true", help="delete the build directory first")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 4)
    parser.add_argument(
        "--arch",
        default=None,
        help="CUDA architectures, e.g. '86' for a single-node build (default: 61;70;75;80;86;89)",
    )
    parser.add_argument(
        "--cuda-home", default=None, help="CUDA toolkit to build with (default: newest found)"
    )
    args = parser.parse_args()

    if shutil.which("cmake") is None:
        print("cmake is not on PATH (need >= 3.22)", file=sys.stderr)
        return 2

    build_dir = NATIVE / "build"
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    configure = [
        "cmake",
        "-S",
        str(NATIVE),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={'Debug' if args.debug else 'Release'}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if args.hip:
        configure += ["-DSHIPINFER_WITH_HIP=ON", "-DSHIPINFER_WITH_CUDA=OFF"]
    else:
        cuda_home = Path(args.cuda_home) if args.cuda_home else discover_cuda_home()
        if cuda_home is None:
            print(
                "no CUDA toolkit found; install one or pass --cuda-home "
                "(or build for ROCm with --hip)",
                file=sys.stderr,
            )
            return 2
        nvcc = cuda_home / "bin" / "nvcc"
        configure += [
            f"-DCUDAToolkit_ROOT={cuda_home}",
            f"-DCMAKE_CUDA_COMPILER={nvcc}",
        ]
        host = host_compiler_for(nvcc)
        if host:
            # Both the CUDA host compiler *and* the C++ compiler, deliberately. pybind11
            # links with LTO, and LTO bytecode from two different GCC major versions does
            # not combine — the failure is an "LTO version" error at the final link that
            # says nothing about the mismatched compilers that caused it.
            configure += [
                f"-DCMAKE_CUDA_HOST_COMPILER={host}",
                f"-DCMAKE_CXX_COMPILER={host}",
            ]
        print(f"# cuda toolkit: {cuda_home}  host compiler: {host or 'default'}")
    if args.arch:
        configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}")

    for command in (configure, ["cmake", "--build", str(build_dir), "-j", str(args.jobs)]):
        print("$", " ".join(command))
        result = subprocess.run(command)
        if result.returncode != 0:
            return result.returncode

    # Import it here, in a subprocess, so a build that produces an unloadable module fails
    # now rather than at server start-up.
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import shipinfer._C as c; "
            "print(f'shipinfer._C {c.__version__} platform={c.platform} "
            "devices={c.device_count()}')",
        ],
        cwd=ROOT,
    )
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
