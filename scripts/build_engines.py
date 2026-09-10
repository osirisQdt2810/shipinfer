#!/usr/bin/env python3
"""Build every TensorRT engine this host needs, from the ONNX in ``models/``.

Three places already told readers to run this — `models/README.md`, the bench's
`require_inputs`, and three model configs — and it did not exist, so anyone trying to
reproduce the benchmark got `No such file or directory`. The mechanism was always there
(`shipinfer.backends.tensorrt.autobuild`); what was missing was one command that uses it.

WHY THIS EXISTS AS A SCRIPT AT ALL

The server builds engines on demand, so a deployment never needs this. A *benchmark* does:
both systems must load plans built from the same ONNX at the same precision, or the
comparison measures the engines rather than the serving layers. Building them in one place,
with the precision printed, is what makes that checkable.

    python scripts/build_engines.py                 # everything, fp32
    python scripts/build_engines.py --fp16          # half precision
    python scripts/build_engines.py --int8          # INT8, calibrated on the bench's frames
    python scripts/build_engines.py --only ship_detector
    python scripts/build_engines.py --check         # report, build nothing

An engine is valid only for the GPU architecture and TensorRT version it was built on, so
this refuses to run without a device rather than producing a plan that will not load.

MEASURED HERE (A5000): the detector quantises to int8, the segmenter does not -- TensorRT
finds no implementation for its mask-prototype head with INT8 and FP16 both set. Reported
rather than worked around; `_apply_precision`'s comment says why dropping FP16 would be a
slower engine under a name that says int8.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MODELS = REPO / "models"
REPOSITORY = REPO / "model_repository"


@dataclass(frozen=True, slots=True)
class Target:
    """One engine: where the ONNX is, and where the plan has to land.

    The bench and the model repository want the same engine in two places — the baseline
    binary takes a path on its command line, the server finds its plan in the model's
    version directory — so each target names both rather than leaving one of them to a
    copy step somebody forgets.
    """

    name: str
    onnx: Path
    #: The flat path the benchmark's `BenchConfig` resolves to.
    engine: Path
    # doc: long why this is a tuple and not one directory
    #: Every repository version directory whose backend loads this plan. A TUPLE because one
    #: target can feed several models: `reid` feeds `person_embedder` AND `ship_embedder`, and
    #: while it was `None` neither got a plan -- so `--force` printed success, installed
    #: nothing, and the bench's byte-identity guard refused every run while naming that command
    #: as the remedy. The operator ran it, the guard failed identically, and the loop had no
    #: exit but a manual `cp`. Empty still means "flat engine only".
    version_dirs: tuple[Path, ...] = ()
    #: ``(height, width)`` this engine is fed, and whether it is fed WHOLE frames. INT8
    #: calibration needs both: the extent to preprocess to, and which of the two transforms
    #: the pipeline applies -- a letterboxed frame for a detector, a crop for an embedder.
    #: Absent from an fp32/fp16 build, which needs no calibration set at all.
    fed: tuple[int, int] = (640, 640)
    whole_frame: bool = True


#: The build targets. Nothing outside this file reads them: `shipinfer plan` used to, to
#: prescribe a build command, and that prescription is gone -- a command is only ever right
#: for one repository and that note runs against any of them. Each version directory's
#: `README.md` carries the command for ITS model, which is where the knowledge is true.
TARGETS = (
    Target(
        "ship_detector",
        MODELS / "yolo26n.onnx",
        MODELS / "yolo26n_fp32.engine",
        (REPOSITORY / "ship_detector" / "1",),
    ),
    Target(
        "ship_segmenter",
        MODELS / "yolo26n-seg.onnx",
        MODELS / "yolo26n-seg_fp32.engine",
        (REPOSITORY / "ship_segmenter" / "1",),
    ),
    Target(
        "reid",
        MODELS / "reid_r50.onnx",
        MODELS / "reid_r50_fp32.engine",
        # BOTH embedders, which is the fanout that made `--force` true for them.
        (REPOSITORY / "person_embedder" / "1", REPOSITORY / "ship_embedder" / "1"),
        fed=(256, 128),
        whole_frame=False,
    ),
)


#: The precisions this script builds, in the order they cost less accuracy.
PRECISIONS = ("fp32", "fp16", "int8")


def _engine_path(target: Target, precision: str) -> Path:
    """The flat engine path for this precision.

    The default names carry `_fp32` because that is what they are; asking for another
    precision and silently overwriting an fp32 plan of the same name is how two runs end up
    comparing different engines while claiming to compare architectures.
    """
    if precision == "fp32":
        return target.engine
    return target.engine.with_name(target.engine.name.replace("_fp32", f"_{precision}"))


def report(precision: str) -> int:
    print(f"{'model':<16} {'onnx':<10} {'engine':<10} path")
    print("-" * 72)
    missing = 0
    for target in TARGETS:
        engine = _engine_path(target, precision)
        onnx_state = "present" if target.onnx.is_file() else "MISSING"
        engine_state = "present" if engine.is_file() else "absent"
        missing += not target.onnx.is_file()
        print(f"{target.name:<16} {onnx_state:<10} {engine_state:<10} {engine}")
    if missing:
        print(
            f"\n{missing} ONNX file(s) missing. They are exported from the checkpoints in "
            f"models/ — see models/README.md.",
            file=sys.stderr,
        )
    return 1 if missing else 0


#: The benchmark's own frame folders, so the activations calibrated for are the ones the
#: measurement produces. 2K and not `BenchConfig.resolution`'s 4k because every batch is
#: letterboxed to the engine's extent first: the source resolution changes the resampling,
#: not the activation ranges the scales come from.
CALIBRATION_DIRS = (
    REPO / "benchmarks" / "baseline" / "data" / "person_2K",
    REPO / "benchmarks" / "baseline" / "data" / "ship_2K",
)
#: A CAP, not a promise: calibration runs one forward pass per batch INSIDE the build, so the
#: set size is build time, and the entropy criterion's histograms stop moving after a few
#: hundred images. 256 is TensorRT's own guidance; this box holds 14 usable frames, and that
#: build took 426.6 s -- so the number below is the ceiling and the corpus is the limit.
CALIBRATION_IMAGES = 256
CALIBRATION_BATCH = 8


def _calibration_files() -> list[Path]:
    """Every frame in the replay folders, interleaved so a short set holds both subjects.

    Not concatenated: the folders are `person_2K` and `ship_2K`, so taking the first 256 of
    a concatenation would calibrate on people only -- and a scale chosen without a single
    ship in it is the mismatch this module's docstring warns about, arriving by a different
    door.
    """
    per_dir = [sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")) for d in CALIBRATION_DIRS]
    interleaved: list[Path] = []
    for index in range(max((len(files) for files in per_dir), default=0)):
        interleaved.extend(files[index] for files in per_dir if index < len(files))
    return interleaved


def _calibration_batches(target: Target, files: list[Path]) -> Iterator[Any]:
    """Batches preprocessed by THE PIPELINE'S OWN OPS, which is the whole point.

    `calibration.py`'s trap: data preprocessed differently from inference produces scales
    that are wrong by that difference, the engine builds without complaint, and the output is
    plausible rather than broken. So this calls `ImageOps.letterbox_batch` with
    `NormalizeParams()` -- mean 0, std 255, swap_rb -- and pad 114, which is exactly what
    `plan_stages.h`'s `pad_value = 114.f / 255.f` and `runtime/ops.cu`'s `/ 255.f` do on the
    serving path, and what `tests/runtime/test_ops_parity.py` holds the two implementations to.
    """
    import cv2
    import numpy as np

    from shipinfer.runtime.ops import IMAGE_OPS, NormalizeParams

    # doc: long why the registry by NAME here rather than `get_image_ops`
    # NUMPY BY NAME, and not `get_image_ops`, which would hand back the fused kernels or torch
    # -- both of which want the device that the TensorRT builder owns while it calibrates. The
    # readable implementation is also the right one to calibrate through:
    # `tests/runtime/test_ops_parity.py` is what says it agrees with the kernels on this exact
    # transform, so a scale calibrated here is a scale the served path will see.
    image_ops = IMAGE_OPS.create("numpy")
    params = NormalizeParams()
    height, width = target.fed
    pending: list[np.ndarray] = []
    for path in files:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        pending.append(frame if target.whole_frame else _centre_crop(frame, target.fed))
        if len(pending) < CALIBRATION_BATCH:
            continue
        yield image_ops.letterbox_batch(pending, (height, width), params, pad_value=114).tensor
        pending = []
    # A SHORT FINAL BATCH IS FINE: `CalibrationBatchFeeder.next_batch` pads it "by REPEATING
    # ITS OWN ROWS rather than with zeros", because a batch of zeros "drags the activation
    # histograms towards zero exactly where the entropy criterion is choosing a clipping
    # threshold". 14 frames at batch 8 is [8, 6], and the 6 becomes 8 real images.
    if pending:
        yield image_ops.letterbox_batch(pending, (height, width), params, pad_value=114).tensor


def _centre_crop(frame: Any, fed: tuple[int, int]) -> Any:
    """The middle of the frame at the embedder's aspect ratio, as a stand-in for a detection.

    An embedder is fed CROPS, not frames, so calibrating it on letterboxed 1080p would put
    two grey bars in every calibration image and none in any served one. A centre crop at the
    engine's own aspect ratio is not a person box -- it is stated here rather than implied,
    and the honest fix is to calibrate from the detector's own boxes once the two builds can
    be chained. What it does get right is the scale of the content and the absence of bars.
    """
    height, width = frame.shape[:2]
    fed_h, fed_w = fed
    side_h = min(height, round(width * fed_h / fed_w))
    side_w = min(width, round(side_h * fed_w / fed_h))
    top = (height - side_h) // 2
    left = (width - side_w) // 2
    return frame[top : top + side_h, left : left + side_w]


def _feeder(target: Target) -> Any:
    """The calibration batch source for one target, or a refusal naming what is missing.

    A feeder rather than a calibrator, because `build_engine` builds the calibrator itself
    (`_apply_precision`): handing it one would mean a second door onto `config.int8_calibrator`
    with the cache wired by a different hand.
    """
    from shipvision.detection.backends.tensorrt.calibration import CalibrationBatchFeeder

    files = _calibration_files()[:CALIBRATION_IMAGES]
    if not files:
        raise SystemExit(
            f"int8 needs calibration frames and none were found in "
            f"{', '.join(str(d) for d in CALIBRATION_DIRS)}. They come with the baseline "
            f"submodule: `git submodule update --init benchmarks/baseline`"
        )
    height, width = target.fed
    feeder = CalibrationBatchFeeder(
        _calibration_batches(target, files),
        batch_shape=(CALIBRATION_BATCH, 3, height, width),
        # THE LOUD VERSION OF THE TRAP: `NormalizeParams()` is mean 0 / std 255, so every
        # value is in [0, 1]. A batch that arrives in 0-255 raises here instead of building a
        # plausible engine whose activations all live in the bottom half-percent of the range.
        value_range=(0.0, 1.0),
        # A CEILING FROM THE FILE COUNT, which is more batches than an undecodable frame
        # will produce -- `_calibration_batches` skips whatever `cv2.imread` refuses. That is
        # the right direction: `limit` stops the feeder EARLY, so over-stating it costs
        # nothing and under-stating it would silently calibrate on less than the corpus.
        limit=(len(files) + CALIBRATION_BATCH - 1) // CALIBRATION_BATCH,
    )
    print(
        f"{target.name}: calibrating on {len(files)} frame(s) at {height}x{width}, "
        f"{'letterboxed' if target.whole_frame else 'centre-cropped'}",
        flush=True,
    )
    return feeder


def build(targets: tuple[Target, ...], *, precision: str, force: bool) -> int:
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        print(
            "TensorRT is not importable here. An engine is specific to the GPU "
            "architecture and TensorRT version it is built on, so building one without "
            "them would produce a plan that cannot load. Run this inside the benchmark "
            "container (deploy/rootless/bench.sh drops you in the right image).",
            file=sys.stderr,
        )
        return 2

    from shipvision.detection.engine_build import build_engine

    timing_cache = MODELS / "timing.cache"
    failures = 0
    for target in targets:
        engine = _engine_path(target, precision)
        if not target.onnx.is_file():
            print(f"{target.name}: no ONNX at {target.onnx}", file=sys.stderr)
            failures += 1
            continue
        if engine.is_file() and not force:
            print(f"{target.name}: {engine.name} already built (use --force to rebuild)")
            _install(target, engine)
            continue

        started = time.monotonic()
        print(f"{target.name}: building {engine.name} ({precision}) ...", flush=True)
        try:
            # doc: long why an int8 build sets fp16 too
            # FP16 STAYS ON FOR AN INT8 BUILD, because INT8 is per-layer: TensorRT keeps a
            # layer in float when quantising it would cost more than it saves, and with
            # `fp16=False` that fallback is fp32 -- a slower engine than the fp16 one it is
            # meant to beat. Both flags is what `trtexec --int8 --fp16` does.
            build_engine(
                target.onnx,
                engine,
                fp16=precision in ("fp16", "int8"),
                int8=precision == "int8",
                int8_calibration=_feeder(target) if precision == "int8" else None,
                # BESIDE THE ENGINE, and named for it: the scales belong to one network, so a
                # cache shared between the detector and the segmenter would hand one model the
                # other's ranges. `CalibrationCache.read` is what makes a rebuild cheap.
                calibration_cache=(
                    engine.with_suffix(".calib") if precision == "int8" else None
                ),
                timing_cache=timing_cache,
            )
        except Exception as exc:  # the builder's own diagnostics are the actionable part
            print(f"{target.name}: FAILED — {exc}", file=sys.stderr)
            failures += 1
            continue
        elapsed = time.monotonic() - started
        size_mb = engine.stat().st_size / 1e6
        print(f"{target.name}: {engine.name}  {size_mb:.1f} MB in {elapsed:.1f}s")

        _install(target, engine)
    return 1 if failures else 0


# doc: long why the model name comes from the path and not from `Target.name`
def _artefact_name(version_dir: Path, engine: Path) -> str:
    """The file name THIS model's config asks for, not the conventional one.

    `parameters.engine_file` is configurable and defaults to `model.plan`; the install wrote
    the default unconditionally. A model naming anything else got its plan under a name nothing
    loads: the builder reports success, `shipinfer plan` names the configured file, and the
    failure arrives at the next start-up, minutes of TensorRT later.

    THE MODEL NAME COMES FROM THE PATH, `<repository>/<name>/<version>`, and not from
    `Target.name` -- those are not the same thing. `reid` is one target that feeds TWO
    repository models and has no `version_dir` at all, so a name-keyed lookup would be right
    for two of the three targets by luck.

    Read through `ModelRepository` because that is the reader that already knows; a second
    parser here is the second door ADR-020 argues against, one artefact along.
    """
    from shipinfer.core.errors import ShipInferError
    from shipinfer.repository import ModelRepository

    name = version_dir.parent.name
    try:
        return ModelRepository.load(version_dir.parents[1]).entry(name).config.engine_file
    except (ShipInferError, OSError) as error:
        # Refused, not defaulted. Guessing `model.plan` here is exactly the defect above, and
        # the flat engine is already built and named in the output, so nothing is lost.
        raise SystemExit(
            f"{name}: built the engine, but cannot read the repository to learn where to "
            f"install it ({error}). The flat engine is at {engine}; fix "
            f"{version_dir.parent / 'config.yaml'} and re-run to install it"
        ) from None


def _install(target: Target, engine: Path) -> None:
    """Put the plan where the server looks for it.

    Copying rather than symlinking: a plan is an artefact, and a dangling link inside a
    container whose mount layout differs is a confusing way to fail at start-up.

    Called on the skip path too. Skipping the copy when the flat engine already existed
    left the artefact absent, `autobuild` then built the server a *different* plan from
    ONNX, and the benchmark's identity check had nothing to compare — so the two sides ran
    different engines and the guard passed.
    """
    blob = engine.read_bytes()
    for version_dir in target.version_dirs:
        destination = version_dir / _artefact_name(version_dir, engine)
        if destination.is_file() and destination.read_bytes() == blob:
            continue
        version_dir.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        # `relative_to` RAISES for a path outside the repository rather than returning the
        # absolute one, so printing where the file went could be the thing that fails.
        shown = (
            destination.relative_to(REPO) if destination.is_relative_to(REPO) else destination
        )
        print(f"{'':<16}  -> {shown}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fp16", action="store_true", help="build half-precision plans")
    parser.add_argument(
        "--int8",
        action="store_true",
        help="build INT8 plans, calibrated on the benchmark's own replay frames through the "
        "pipeline's own letterbox. Implies fp16 for the layers TensorRT declines to quantise.",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument(
        "--check", action="store_true", help="report what exists, build nothing"
    )
    parser.add_argument("--only", action="append", default=[], help="build just this model")
    args = parser.parse_args(argv)

    # REFUSED rather than resolved by precedence. `--fp16 --int8` reads as "both", and both is
    # what an int8 build already does per layer -- so a reader who typed it meant something
    # this script cannot tell apart from a mistake, and guessing which is how the wrong plan
    # gets built under the right name.
    if args.fp16 and args.int8:
        print(
            "--fp16 and --int8 together is ambiguous: an int8 build already allows fp16 for "
            "the layers TensorRT declines to quantise, so pass just --int8.",
            file=sys.stderr,
        )
        return 2
    precision = "int8" if args.int8 else ("fp16" if args.fp16 else "fp32")

    if args.check:
        # Reports what exists and builds nothing, so it is inspection rather than a build --
        # allowed on the host on purpose, the way `shipinfer repo ls` and a `--version` query
        # are. The gate belongs below it, not above.
        return report(precision)

    selected = TARGETS
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {t.name for t in TARGETS}
        if unknown:
            print(f"unknown model(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        selected = tuple(t for t in TARGETS if t.name in wanted)

    from shipinfer.runtime import containment

    # CLAUDE.md's list of what must run in a container names "any engine build", and this was
    # the one entry in that list with no gate. A plan is valid only for the architecture and
    # TensorRT version it was built on -- host `nvcc` here is 11.5 against a 12.6 driver -- so
    # a host-built engine is the WRONG artefact rather than a slower one.
    containment.require_container("an engine build")

    return build(selected, precision=precision, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
