"""Build a TensorRT plan from an ONNX in the model repository, on demand.

The reference detector libraries do this — `BaseDetector::loadEngine` deserialises the plan
and, on any failure, calls `buildEngine` and retries up to three times
(`references/generic-object-detection-yolo26-trt/src/process/BaseDetector.cpp:225-231`). The
reason is operational: an engine is only valid for the GPU architecture and the TensorRT
version that produced it, so a plan cannot be shipped. What ships is the ONNX, and the plan is
a **build artefact of this machine**.

Three things the reference gets wrong, and this does not:

**It shells out.** `Yolo26Detector::buildEngine` runs `system("python3 pytools/onnx2trt.py ...")`
and recovers its own configuration by regex-parsing `mbs(\\d+)` and `imgsz(\\d+)` out of the
ONNX *filename*. A library that shells out to a script at a compile-time-baked path cannot be
packaged. Here the build is in-process, through `shipvision.detection.engine_build`.

**It rebuilds per instance.** A build takes ~90 s for yolo26n. With two instances on each of
eight GPUs that is sixteen builds of the same plan, and they race to write the same file. A
lock makes exactly one build and the rest wait for it.

**It trusts a plan it finds.** A stale plan built by a different TensorRT, or for a different
GPU, deserialises to `None` — or worse, loads and misbehaves. The cache key therefore carries
the TensorRT version and the device's compute capability, so a plan from another environment
is a miss rather than a landmine.

And one thing neither could have known: **TensorRT's ONNX parser segmentation-faults on
malformed input.** Measured on 10.14.1 — a text file named `*.onnx` dumps core with no
exception to catch. Since this code path is reached at server start-up, a truncated download or
an unresolved Git-LFS pointer would kill the worker and leave an operator with a core dump.
`engine_build` validates the ONNX before the parser sees it; this module never bypasses it.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

from shipinfer.core.errors import BackendLoadError
from shipinfer.core.logging import get_logger

__all__ = ["cache_name", "resolve_engine"]

_LOG = get_logger("backends.tensorrt.autobuild")

#: How long to wait for another process's build before giving up. A yolo26n plan takes ~90 s
#: and a larger model several minutes, so the timeout is generous — waiting is always better
#: than a second concurrent build, which would double the work and race on the output file.
_BUILD_TIMEOUT_S = 900.0
_POLL_S = 0.5


def cache_name(onnx: Path, *, trt_version: str, capability: str, fp16: bool) -> str:
    """A plan filename that changes when anything invalidating it changes.

    Everything that makes a plan valid or not is in the name: the ONNX's content, the
    TensorRT version, the GPU's compute capability, and the precision. A plan built for sm_86
    on TensorRT 10.14 is simply a different file from one built for sm_89 on 10.13 — naming
    them the same is how a stale artefact gets loaded and then misbehaves.

    The ONNX is hashed by content rather than by mtime: a re-export with the same timestamp is
    a real thing (a build system that preserves times), and an mtime cache would serve the old
    plan for the new model.
    """
    digest = hashlib.sha256(onnx.read_bytes()).hexdigest()[:16]
    precision = "fp16" if fp16 else "fp32"
    return f"{onnx.stem}.{digest}.trt{trt_version}.sm{capability}.{precision}.plan"


def resolve_engine(
    directory: Path,
    *,
    engine_file: str,
    trt: Any,
    device_index: int,
    fp16: bool = False,
    onnx_file: str | None = None,
    builder: Any = None,
) -> Path:
    """The path to a loadable plan, building it from an ONNX if one is needed.

    Args:
        directory: the model version directory, e.g. ``model_repository/ship_detector/1``.
        engine_file: the configured plan name. Used as-is when the file is present, which
            keeps a hand-built or vendor-supplied plan working unchanged.
        onnx_file: the ONNX to build from. Defaults to the only ``*.onnx`` in the directory,
            because a directory with one ONNX needs no configuration to say which.
        builder: injected for testing; defaults to
            :func:`shipvision.detection.engine_build.build_engine`.

    Resolution order, and each step is there for a reason:

    1. **The configured plan, if it exists.** An operator who supplied a plan gets that plan.
    2. **A cached plan for this exact (ONNX, TensorRT, GPU, precision).** Restarts are free.
    3. **Build it**, under a lock so concurrent instances produce one plan between them.

    Raises:
        BackendLoadError: if there is neither a plan nor an ONNX, if several ONNX files make
            the choice ambiguous, or if a concurrent build does not finish in time.
    """
    configured = directory / engine_file
    if configured.is_file():
        return configured

    onnx = _find_onnx(directory, onnx_file)
    version = str(getattr(trt, "__version__", "unknown"))
    capability = _capability(device_index)
    target = directory / cache_name(onnx, trt_version=version, capability=capability, fp16=fp16)
    if target.is_file():
        _LOG.info("reusing cached plan %s", target.name)
        return target

    return _build_once(onnx, target, fp16=fp16, builder=builder)


def _find_onnx(directory: Path, onnx_file: str | None) -> Path:
    if onnx_file is not None:
        candidate = directory / onnx_file
        if not candidate.is_file():
            raise BackendLoadError(f"configured onnx_file not found: {candidate}")
        return candidate

    candidates = sorted(directory.glob("*.onnx"))
    if not candidates:
        raise BackendLoadError(
            f"{directory} has neither a TensorRT plan nor an ONNX to build one from. "
            f"Drop a `.onnx` beside the config, or set `parameters.engine_file` to a plan "
            f"that exists."
        )
    if len(candidates) > 1:
        raise BackendLoadError(
            f"{directory} holds {len(candidates)} ONNX files "
            f"({', '.join(p.name for p in candidates)}); set `parameters.onnx_file` to say "
            f"which. Picking one would be a guess, and the wrong guess builds a plan for the "
            f"wrong network."
        )
    return candidates[0]


def _capability(device_index: int) -> str:
    """``"86"`` for an RTX A5000. Part of the cache key — a plan is architecture-specific.

    Raises rather than falling back. It used to return ``"unknown"`` so that a missing
    capability "must not stop a build" — but the cache key exists precisely to stop a plan
    built for one architecture being loaded on another, and a key of ``smunknown`` matches
    every machine that also failed to introspect. An sm_86 plan would then be handed to an
    sm_89 device, which is the exact landmine this filename was designed to defuse. If we
    cannot name the architecture we cannot name the file, so we do not build.
    """
    try:
        import torch

        major, minor = torch.cuda.get_device_capability(device_index)
    except Exception as exc:
        raise BackendLoadError(
            f"cannot determine the compute capability of device {device_index} ({exc}), so "
            f"a built plan could not be keyed to this architecture. A plan is only valid "
            f"for the architecture it was built on, so caching one under an unknown key "
            f"would let it be reused on a different GPU. Supply a prebuilt plan instead."
        ) from exc
    return f"{major}{minor}"


def _build_once(onnx: Path, target: Path, *, fp16: bool, builder: Any) -> Path:
    """Build `target`, or wait for whoever is already building it.

    The lock is an exclusive-create of a sidecar file rather than `flock`: it works across
    processes and containers on a shared mount, and a crashed builder leaves a lock whose age
    is visible, which is a better failure than a silently held kernel lock.
    """
    if builder is None:
        try:
            from shipvision.detection.engine_build import build_engine as builder
        except ImportError as exc:
            raise BackendLoadError(
                f"cannot build a plan from {onnx.name}: shipvision is not importable "
                f"({exc}). Install it, or supply a prebuilt plan."
            ) from exc

    lock = target.with_suffix(".building")
    try:
        handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if _lock_is_abandoned(lock):
            # A SIGKILLed builder never reaches the `finally` below, so its lock outlives
            # it. Every later start-up then waited the full 900 s and failed, needing manual
            # cleanup — the docstring claimed the lock's age was "visible" and nothing ever
            # looked at it. Now it is read, and an abandoned lock is taken over.
            _LOG.warning("taking over an abandoned build lock %s", lock.name)
            lock.unlink(missing_ok=True)
            return _build_once(onnx, target, fp16=fp16, builder=builder)
        return _wait_for(target, lock)

    try:
        os.write(handle, f"{os.getpid()}\n".encode())
        os.close(handle)
        started = time.monotonic()
        _LOG.info("building %s from %s (fp16=%s)", target.name, onnx.name, fp16)
        # Written to a temporary name and renamed, so a reader never sees a partial plan:
        # rename is atomic within a filesystem and a half-written plan deserialises to None.
        staging = target.with_suffix(".partial")
        builder(onnx, staging, fp16=fp16, timing_cache=onnx.parent / "timing.cache")
        staging.replace(target)
        _LOG.info("built %s in %.1fs", target.name, time.monotonic() - started)
        return target
    except Exception:
        target.with_suffix(".partial").unlink(missing_ok=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def _lock_is_abandoned(lock: Path) -> bool:
    """Whether the process that wrote this lock is gone.

    Two independent signals, because either alone is wrong. The pid in the file is checked
    first and is decisive when it is readable: a build that died leaves no live process, and
    a *live* pid means someone really is building. Age is the fallback for a lock whose pid
    cannot be read or was recycled — a build that has run longer than the timeout has
    already failed by this function's own definition.
    """
    try:
        recorded = int(lock.read_text().strip().splitlines()[0])
    except (OSError, ValueError, IndexError):
        recorded = 0
    if recorded > 0:
        try:
            os.kill(recorded, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Alive and owned by someone else; that still counts as a live builder.
            return False
        else:
            return False
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return True
    return age > _BUILD_TIMEOUT_S


def _wait_for(target: Path, lock: Path) -> Path:
    """Wait for another builder, then return its plan."""
    _LOG.info("another process is building %s; waiting", target.name)
    deadline = time.monotonic() + _BUILD_TIMEOUT_S
    while time.monotonic() < deadline:
        if target.is_file():
            return target
        if not lock.exists():
            # The builder went away without producing a plan. One retry, by falling through
            # to a fresh attempt, is better than failing a start-up on someone else's crash.
            if target.is_file():
                return target
            raise BackendLoadError(
                f"the process building {target.name} exited without producing it; "
                f"check its log, then restart"
            )
        time.sleep(_POLL_S)
    raise BackendLoadError(
        f"waited {_BUILD_TIMEOUT_S:.0f}s for another process to build {target.name} and it "
        f"did not appear. If no build is running, remove {lock.name} and restart."
    )
