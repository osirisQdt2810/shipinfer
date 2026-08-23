"""Drive ``benchmarks/baseline``'s own binary — the real one, not a re-implementation.

``benchmarks/baseline`` is the ``counting-simulation`` repository, pinned as a submodule and
**read-only**. Nothing here edits it. What this module does is compile
``sim_pipeline_v2.cpp`` unchanged, assemble the shared libraries it needs so it can run
inside the container, launch it with the harness's configuration, and hand the JSONL it
writes to :mod:`benchmarks.harness.analysis`.

Why the binary is compiled on the host and run in the container
--------------------------------------------------------------
The project rule is that measurements run in a container. This box cannot build container
images — rootless Docker here cannot mount ``/proc`` from an unprivileged user namespace, so
buildkit fails (``deploy/rootless/setup.sh``) — and containers have no outbound network, so
``apt install g++ libopencv-dev`` is not available either. The only image that can be used is
``pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime``, which has neither a C++ compiler nor
OpenCV.

So the split is: **compile on the host** (a compiler is not a measurement) and **run in the
container**. To make the host-built binary runnable there, :func:`stage_runtime_libs` copies
the closure of its shared libraries into ``benchmarks/build/baseline-libs`` and the container
gets that on ``LD_LIBRARY_PATH``. The C runtime, libstdc++ and the driver are deliberately
**excluded** from the closure: the host and the image are both Ubuntu 22.04 so the ABI
matches, and shadowing the image's own libc or libstdc++ would break python and torch in the
same container.

What the baseline does with a partial batch
-------------------------------------------
The engines here are static-batch plans, and ``TrtRunner::infer`` calls ``setInputShape``
with the batch it actually assembled. On a static plan that call fails for any batch smaller
than the plan's, the C++ code throws inside a worker thread, and the process aborts. It
therefore only survives while its queue stays non-empty — which is exactly the saturated
regime the methodology measures, and is why a short or lightly-loaded run of the baseline
can die where a saturated one runs for minutes. :attr:`BaselineResult.aborted` reports it
rather than letting a truncated log look like a completed experiment.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.harness.config import BenchConfig

__all__ = [
    "BaselineResult",
    "build_binary",
    "run_baseline",
    "stage_runtime_libs",
]

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "benchmarks" / "baseline" / "sim_pipeline_v2.cpp"
BUILD_DIR = REPO / "benchmarks" / "build"
BINARY = BUILD_DIR / "sim_pipeline_v2"
LIB_DIR = BUILD_DIR / "baseline-libs"

#: Where TensorRT lives on this host. Overridable because it is the one path that is not the
#: repository's to decide.
TENSORRT_ROOT = Path(os.environ.get("TRT_ROOT", "/usr/local/TensorRT"))
CUDA_ROOT = Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda-12.6"))

#: Libraries the container supplies and must keep supplying. Copying the host's versions of
#: these into the closure would put a second libc on ``LD_LIBRARY_PATH`` ahead of the image's
#: own, which breaks the interpreter that is sharing the container.
_HOST_ONLY = (
    "libc.so",
    "libm.so",
    "libdl.so",
    "libpthread.so",
    "librt.so",
    "libstdc++.so",
    "libgcc_s.so",
    "ld-linux",
    "libcuda.so",
    "libnvidia-ml.so",
    "libcudart.so",
    "libnvinfer.so",
    "libnvinfer_plugin.so",
)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """What one baseline run produced."""

    log: Path
    console: Path
    exit_code: int
    aborted: bool
    elapsed_s: float

    @property
    def ok(self) -> bool:
        """A run is usable when it wrote samples, whether or not it aborted at the end.

        ``aborted`` is not automatically fatal: the binary dies on a partial batch, which on
        a saturated run happens only as the sources are shutting down and after the samples
        that matter are already on disk. What is fatal is an empty log, and the analysis
        raises on that.
        """
        return self.log.exists() and self.log.stat().st_size > 0


# ------------------------------------------------------------------------------- build


def build_binary(*, force: bool = False) -> Path:
    """Compile ``sim_pipeline_v2.cpp`` with the same flags its own runner script uses.

    Idempotent: a binary newer than the source is reused, so a benchmark sweep does not
    rebuild fifty times. ``force`` rebuilds regardless.

    Raises:
        FileNotFoundError: the submodule is not checked out, or TensorRT is not where
            ``TRT_ROOT`` says.
        RuntimeError: the compile failed, with the compiler's own diagnostics attached —
            they are the actionable part and swallowing them would leave "build failed".
    """
    # Reuse first, and check the toolchain only if a build is actually needed. Reusing a
    # binary does not require a compiler, and the container this runs in deliberately has
    # neither g++ nor the TensorRT headers -- the binary is built on the host precisely
    # because the container cannot build it. Demanding headers before noticing the binary
    # already exists turns a working configuration into a hard failure, and reports the
    # missing headers as the problem when nothing needed them.
    if (
        not force
        and BINARY.is_file()
        and (not SOURCE.is_file() or BINARY.stat().st_mtime >= SOURCE.stat().st_mtime)
    ):
        return BINARY

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"{SOURCE} is missing. Check the submodule out with "
            f"`git submodule update --init benchmarks/baseline`"
        )
    if not (TENSORRT_ROOT / "include" / "NvInfer.h").is_file():
        raise FileNotFoundError(
            f"no TensorRT headers under {TENSORRT_ROOT}. Set TRT_ROOT to the install root. "
            f"If you are inside the benchmark container, the binary should already have "
            f"been built on the host -- {BINARY} is missing or older than the source."
        )

    opencv = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "opencv4"],
        capture_output=True,
        text=True,
        check=False,
    )
    if opencv.returncode != 0:
        raise RuntimeError(
            "pkg-config could not find opencv4. The baseline letterboxes with OpenCV on the "
            "CPU, so it cannot be built without it"
        )

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        str(SOURCE),
        "-o",
        str(BINARY),
        *opencv.stdout.split(),
        f"-I{TENSORRT_ROOT / 'include'}",
        f"-I{CUDA_ROOT / 'include'}",
        f"-L{TENSORRT_ROOT / 'lib'}",
        f"-L{CUDA_ROOT / 'lib64'}",
        f"-Wl,-rpath,{TENSORRT_ROOT / 'lib'}",
        f"-Wl,-rpath,{CUDA_ROOT / 'lib64'}",
        "-lnvinfer",
        "-lnvinfer_plugin",
        "-lcudart",
        "-pthread",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not BINARY.is_file():
        raise RuntimeError(
            f"compiling {SOURCE.name} failed (exit {result.returncode}):\n{result.stderr[-4000:]}"
        )
    return BINARY


def stage_runtime_libs(*, force: bool = False) -> Path:
    """Copy the binary's shared-library closure into a directory a container can mount.

    Idempotent: an existing non-empty directory is reused unless ``force``. See the module
    docstring for why the C runtime and the driver are excluded.

    Raises:
        RuntimeError: ``ldd`` reported an unresolved dependency on the host, which means the
            staged set would be incomplete in a way that only shows up as a container
            failure minutes later.
    """
    if not force and LIB_DIR.is_dir() and any(LIB_DIR.iterdir()):
        return LIB_DIR
    binary = build_binary()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(TENSORRT_ROOT / "lib"), str(CUDA_ROOT / "lib64"), env.get("LD_LIBRARY_PATH", "")]
    )
    listing = subprocess.run(
        ["ldd", str(binary)], capture_output=True, text=True, check=True, env=env
    ).stdout
    if "not found" in listing:
        missing = [line.strip() for line in listing.splitlines() if "not found" in line]
        raise RuntimeError(
            "the freshly built baseline has unresolved libraries on the host, so staging "
            f"them for the container cannot succeed either: {missing}"
        )

    if LIB_DIR.is_dir():
        shutil.rmtree(LIB_DIR)
    LIB_DIR.mkdir(parents=True)
    for line in listing.splitlines():
        parts = line.split("=>")
        if len(parts) != 2:
            continue
        target = parts[1].strip().split(" ")[0]
        if not target or not Path(target).is_file():
            continue
        if any(marker in Path(target).name for marker in _HOST_ONLY):
            continue
        shutil.copy(Path(target).resolve(), LIB_DIR / Path(target).name)
    return LIB_DIR


def library_path() -> str:
    """``LD_LIBRARY_PATH`` for running the staged binary, host or container.

    The NVIDIA runtime libraries shipped inside the pytorch image's site-packages are
    included when present: the image has no ``/usr/local/cuda``, and TensorRT needs a
    ``libcudart`` from somewhere.
    """
    parts = [str(TENSORRT_ROOT / "lib"), str(LIB_DIR)]
    conda = Path("/opt/conda/lib")
    if conda.is_dir():
        for major in sorted(conda.glob("python3.*/site-packages/nvidia/*/lib")):
            parts.append(str(major))
    if (CUDA_ROOT / "lib64").is_dir():
        parts.append(str(CUDA_ROOT / "lib64"))
    parts.append("/usr/lib/x86_64-linux-gnu")
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


# --------------------------------------------------------------------------------- run


def command_line(config: BenchConfig, log: Path) -> list[str]:
    """The exact argv, so a report can quote it and a reader can re-run it by hand."""
    resolved = config.resolved()
    assert resolved.person_frames and resolved.ship_frames
    assert resolved.det_engine and resolved.seg_engine
    return [
        str(BINARY),
        "--det-folder",
        str(resolved.person_frames),
        "--seg-folder",
        str(resolved.ship_frames),
        "--det-engine",
        str(resolved.det_engine),
        "--seg-engine",
        str(resolved.seg_engine),
        "--num-det-source-workers",
        str(config.sources_per_module),
        "--num-seg-source-workers",
        str(config.sources_per_module),
        "--num-det-workers",
        str(config.workers_per_module),
        "--num-seg-workers",
        str(config.workers_per_module),
        # Physical ordinals, but CUDA_VISIBLE_DEVICES has already narrowed the world to
        # `config.gpus`, so the binary must be told 0..n-1. Passing physical ids *and*
        # restricting visibility would put every worker on the wrong device or none.
        "--gpu-ids",
        ",".join(str(i) for i in range(len(config.gpus))),
        "--det-batch-size",
        str(config.batch),
        "--seg-batch-size",
        str(config.batch),
        "--det-source-fps",
        str(int(config.fps)),
        "--seg-source-fps",
        str(int(config.fps)),
        "--det-buffer-capacity",
        str(config.buffer_capacity),
        "--seg-buffer-capacity",
        str(config.buffer_capacity),
        "--log-buffer-capacity",
        str(config.buffer_capacity),
        "--log-jsonl",
        str(log),
    ]


def run_baseline(config: BenchConfig, out_dir: Path | None = None) -> BaselineResult:
    """Run the baseline for ``config.seconds`` and return where its log landed.

    The binary runs until interrupted — it has no ``--seconds`` — so the harness sends
    ``SIGINT`` at the deadline, which is what its own signal handler expects and what
    ``auto_experiments_v2_cpp.py`` does. It is started in its own process group so the
    signal reaches the whole thing rather than only the shell in front of it.

    Raises:
        FileNotFoundError: an input is missing; the message names it.
    """
    config = config.resolved()
    config.require_inputs()
    out = out_dir or config.out_dir / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "buffers.jsonl"
    console = out / "console.log"
    if log.exists():
        log.unlink()

    build_binary()
    stage_runtime_libs()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = library_path()
    env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices()
    # The baseline letterboxes on the CPU inside every inference thread. Left unpinned,
    # OpenCV spawns a pool per thread and the threads fight over the whole box, which
    # changes the answer run to run. One thread each is also what each worker gets in
    # practice once the machine is busy.
    env["OMP_NUM_THREADS"] = "1"

    argv = command_line(config, log)
    started = time.monotonic()
    with console.open("w", encoding="utf-8") as sink:
        sink.write(" ".join(argv) + "\n\n")
        sink.flush()
        process = subprocess.Popen(
            argv,
            cwd=str(REPO),
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = started + config.seconds
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.2)
        if process.poll() is None:
            _terminate(process)
        exit_code = process.wait()
    elapsed = time.monotonic() - started

    text = console.read_text(encoding="utf-8", errors="ignore")
    aborted = "terminate called" in text or "Error:" in text
    return BaselineResult(
        log=log,
        console=console,
        exit_code=exit_code,
        aborted=aborted,
        elapsed_s=elapsed,
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """SIGINT, then SIGTERM, then SIGKILL — the escalation the binary's handler expects."""
    for sig, grace in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
