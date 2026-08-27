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

EXTERNAL LIBRARIES, AND THE ONE FLAG THAT OPTS A LANE BACK IN
-------------------------------------------------------------
A unit that reaches outside this tree declares what it needs in :data:`EXTERNAL`, and there are
two axes now rather than one:

*accelerator* — ``core/platform.h`` in the closure means the driver's headers, so the unit is
built by ``nvcc`` and the binary links ``libcudart``/``libnvinfer``.

*external* — a ``pkg-config`` package somebody has to have installed. ``--offline`` leaves those
units out for the same reason it leaves the accelerator ones out: that flag promises ``g++``
alone on a machine with nothing installed, and ``pkg-config`` refusing on a bare runner is
exactly the undeclared prerequisite it exists to prevent.

``--with-external <lane>`` opts one named lane back into the offline build. That is not a
loophole, it is the only way the GStreamer source gets compiled and tested anywhere: the one
image with both the GStreamer headers and the GStreamer plugins (``shipinfer-gst:jammy``) has no
nvcc and no TensorRT, so the full build cannot run there and the offline build is what has to
stretch. A lane whose units reach ``core/platform.h`` is refused rather than stretched to — see
:func:`main`.

A full build (no ``--offline``) compiles every unit and therefore needs every lane's packages.
When one is missing it fails naming the package and where to get it, rather than quietly
dropping the unit: a binary whose registry silently lacks a source reports "unknown video
source" for what is really "that library is not installed", and ``ingest/registry.h`` says at
length why those must not be confused.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

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


class ExternalLane(NamedTuple):
    """One library outside this tree, and the translation units that reach it.

    ``units`` are paths under ``csrc/``, listed rather than sniffed: an ``#include <gst/gst.h>``
    is invisible to the closure walker (which follows only ``"shipinfer/..."`` lines), so the
    declaration has to be here for the build to know. ``hint`` is what an operator does about a
    ``pkg-config`` that cannot resolve ``packages`` — the message is the whole value of failing
    loudly instead of dropping the unit.
    """

    units: tuple[str, ...]
    packages: tuple[str, ...]
    hint: str


#: The external lanes, keyed by the name ``--with-external`` takes. OpenCV is in here with
#: GStreamer rather than staying a global: it was reached by exactly one unit all along, and two
#: mechanisms for one idea is how the second one ends up subtly different. Behaviour for
#: ``replay`` is unchanged either way — it is the only unit that names ``cv::``, it is the only
#: one that gets the ``-I`` flags, and it is in every link that used to receive the libraries,
#: because a link that reaches it reaches ``core/platform.h`` too.
EXTERNAL: dict[str, ExternalLane] = {
    "opencv": ExternalLane(
        units=("shipinfer/ingest/sources/replay.cpp",),
        packages=("opencv4",),
        hint=(
            "install libopencv-dev. The replay source decodes JPEGs with it, the same way the "
            "baseline letterboxes with it, so the build needs it."
        ),
    ),
    "gstreamer": ExternalLane(
        units=("shipinfer/ingest/sources/gstreamer.cpp",),
        packages=("gstreamer-1.0", "gstreamer-app-1.0"),
        hint=(
            "install libgstreamer1.0-dev and libgstreamer-plugins-base1.0-dev, or build inside "
            "shipinfer-gst:jammy, which has both (deploy/rootless/gst-image.sh). `--offline` "
            "without `--with-external gstreamer` leaves this unit out entirely."
        ),
    ),
}

_PKG_CONFIG_CACHE: dict[str, list[str]] = {}


def pkg_config_flags(lane: str) -> list[str]:
    """``pkg-config --cflags --libs`` for one lane, or a ``SystemExit`` naming what is missing.

    Cached, because a lane's flags are asked for once per translation unit and once per link
    line, and shelling out forty times to read the same answer is forty processes.
    """
    if lane not in _PKG_CONFIG_CACHE:
        spec = EXTERNAL[lane]
        probe = subprocess.run(
            ["pkg-config", "--cflags", "--libs", *spec.packages],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise SystemExit(
                f"pkg-config could not resolve {', '.join(spec.packages)}, needed by "
                f"{', '.join(spec.units)} (the '{lane}' external lane): {spec.hint}"
            )
        _PKG_CONFIG_CACHE[lane] = probe.stdout.split()
    return _PKG_CONFIG_CACHE[lane]


def compile_flags(lanes: set[str]) -> list[str]:
    """The ``-I`` half of every lane's flags."""
    return [f for lane in sorted(lanes) for f in pkg_config_flags(lane) if f.startswith("-I")]


def link_flags(lanes: set[str]) -> list[str]:
    """The library half — everything that is not an include path."""
    return [
        f for lane in sorted(lanes) for f in pkg_config_flags(lane) if not f.startswith("-I")
    ]


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


def lanes_of(unit: Path) -> set[str]:
    """The external lanes this one translation unit is declared to need."""
    name = unit.relative_to(CSRC).as_posix()
    return {lane for lane, spec in EXTERNAL.items() if name in spec.units}


def lanes_in(closure: set[Path]) -> set[str]:
    """Every external lane reachable from a closure, because a *unit* in it needs the library.

    Wider than :func:`lanes_of` on purpose: compiling ``cli/bench.cpp`` needs no OpenCV header,
    but *linking* it needs ``replay.cpp``'s object and therefore ``libopencv_core``.
    """
    lanes: set[str] = set()
    for path in closure:
        lanes |= lanes_of(path)
    return lanes


def offline_ready(closure: set[Path], enabled: frozenset[str]) -> bool:
    """Whether the offline lane can build this closure: no driver, and no unasked-for library.

    Two reasons to stay out, and they are the same reason twice — ``--offline`` promises ``g++``
    alone on a machine with nothing installed. ``core/platform.h`` in the closure means the
    driver's headers; an external lane means somebody's ``-dev`` package. ``--with-external``
    names the lanes this particular run does have, which is still ``g++`` alone plus one
    ``pkg-config``.
    """
    return not needs_accelerator(closure) and lanes_in(closure) <= enabled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="build only the apps whose includes never reach core/platform.h, with g++ alone: "
        "no nvcc, no CUDA, no TensorRT — the C++ offline tier, runnable on a machine with no "
        "driver",
    )
    parser.add_argument(
        "--with-external",
        action="append",
        metavar="LANE",
        choices=sorted(EXTERNAL),
        default=[],
        help="add one external-library lane to the offline build: its units are compiled with "
        "`pkg-config --cflags` and linked with `--libs`. Repeatable. Lanes: "
        + ", ".join(
            f"{lane} ({' '.join(EXTERNAL[lane].packages)})" for lane in sorted(EXTERNAL)
        )
        + ". No effect without --offline, because a full build compiles every unit and "
        "therefore needs every lane",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--debug", action="store_true", help="-O0 -g instead of -O2")
    args = parser.parse_args()

    # A full build has every lane on — every lane it CAN have. A lane whose -dev package is
    # missing is left out with a loud warning rather than failing the whole build (#45
    # follow-up): the host that builds `bench` has nvcc and OpenCV but no GStreamer headers,
    # and failing the bench lane to protect a registry answer trades a real workflow for a
    # hypothetical confusion — which the warning below names instead. Asking for a lane BY
    # NAME is different: `--with-external <lane>` is a promise the package is there, and a
    # broken promise stays a hard failure (the probe loop further down).
    if args.offline:
        enabled = frozenset(args.with_external)
    else:
        available: set[str] = set()
        for lane in sorted(EXTERNAL):
            try:
                pkg_config_flags(lane)
            except SystemExit as refusal:
                print(
                    f"WARNING: external lane '{lane}' left out of this build — {refusal}\n"
                    f'         binaries from this build answer "unknown video source" for '
                    f"that lane's sources; that is the missing -dev package talking, not the "
                    f"registry.",
                    file=sys.stderr,
                )
            else:
                available.add(lane)
        enabled = frozenset(available)
    for unit in (CSRC / u for spec in EXTERNAL.values() for u in spec.units):
        if not unit.is_file():
            raise SystemExit(
                f"EXTERNAL names {unit.relative_to(CSRC)}, which does not exist. A stale path "
                f"there means that unit compiles with no external flags at all, and the error "
                f"an hour later is a missing system header in a file that looks innocent."
            )
    for lane in sorted(enabled if args.offline else ()):
        for unit in (CSRC / u for u in EXTERNAL[lane].units):
            if needs_accelerator(include_closure(unit)):
                # The whole point of the flag is one more `pkg-config` on an otherwise offline
                # line. A unit that also reached the driver would need nvcc, CUDA and TensorRT,
                # at which point this is the full build wearing a smaller name.
                raise SystemExit(
                    f"--with-external {lane}: {unit.relative_to(CSRC)} reaches core/platform.h, "
                    f"so it cannot join an offline build — that lane is g++ alone, with no CUDA "
                    f"and no TensorRT. Either the include is a mistake, or this unit belongs to "
                    f"the full build only."
                )
    # Probed here rather than at the first unit that needs it, so a missing `-dev` package is a
    # start-up failure with a sentence in it instead of a surprise forty compiles in
    # (CONVENTIONS 2.6: validate at start-up, not at first use).
    for lane in sorted(enabled):
        pkg_config_flags(lane)

    # The tree mirrors `src/shipinfer/` — a thing's header and its translation unit live next
    # to each other, so there is no `include/` to search separately and the include root is
    # `csrc/` itself. Entry points are `cli/` (the binaries) and `tests/`.
    pkg = CSRC / "shipinfer"
    apps = sorted((pkg / "cli").glob("*.cpp")) + sorted((CSRC / "tests").glob("*.cpp"))
    closures = {app: include_closure(app) for app in apps}
    # `cuda_free` decides how a binary is *linked* and is the accelerator axis alone; the offline
    # set is narrower, because it also has to have every external library the closure names.
    cuda_free = {app for app, closure in closures.items() if not needs_accelerator(closure)}
    if args.offline:
        # Only the apps this machine can actually build: no `core/platform.h` in the closure, and
        # no external lane that was not asked for — buildable and runnable with no driver, no
        # nvcc and no TensorRT, which is the C++ offline tier.
        apps = [app for app in apps if offline_ready(closures[app], enabled)]
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
    # `--with-external` changes which units are on the link line without touching a single
    # mtime, so the freshness check has to know which lanes the last build used. Without this,
    # adding a lane prints "up to date" and leaves the new unit out of a binary whose tests then
    # report a skip — a green run that proves nothing, which is the failure mode this whole file
    # is careful about.
    stamp = BUILD / ".lanes"
    lane_key = ("offline" if args.offline else "full") + ":" + ",".join(sorted(enabled))
    same_lanes = stamp.is_file() and stamp.read_text() == lane_key
    if (
        not args.force
        and same_lanes
        and all(t.is_file() and t.stat().st_mtime >= newest for t in targets)
    ):
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
    # The units a CUDA-free binary links: every translation unit whose own include closure never
    # reaches `core/platform.h` and whose external lanes this run has. All of them, not just the
    # ones the binary's own closure names — sources and policies register through file-scope
    # registrars, so a unit left off the link line is a source missing from that binary's
    # registry, and two binaries would then answer `create_source("gstreamer")` differently.
    free_units = {q for q in sources if offline_ready(include_closure(q), enabled)}
    if args.offline:
        # Only those units are compiled: the offline build must not even *compile* a unit that
        # includes the driver's headers, or one that needs a `-dev` package nobody asked for.
        sources = [q for q in sources if q in free_units]
    else:
        # The full build mirrors the rule for the lanes it had to leave out (warned above):
        # compiling a unit without its headers is the same failure, later and uglier.
        sources = [q for q in sources if lanes_in(include_closure(q)) <= enabled]
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

    # Per unit, not globally: only `replay.cpp` names `cv::` and only `gstreamer.cpp` names
    # `gst_`, so only they get those `-I` paths. `pkg-config` is asked nothing at all for a lane
    # no compiled unit needs — it refusing on a runner with no OpenCV was the third undeclared
    # prerequisite of a flag that promised "g++ alone".
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
                *compile_flags(lanes_of(source)),
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
                *compile_flags(lanes_of(app)),
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
            linked = [q for q in sorted(free_units) if q in object_of]
            link_objects = [object_of[q] for q in linked]
            # The lanes of what is actually on the line, so an offline binary that took
            # `--with-external gstreamer` links `libgstreamer-1.0` and one that did not asks for
            # nothing.
            link_libs = ["-pthread", *link_flags(lanes_in(set(linked)))]
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
                *link_flags(lanes_in(set(object_of))),
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
    # Last, so a failed build never leaves the tree looking fresh for a lane set it does not
    # have on disk.
    stamp.write_text(lane_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
