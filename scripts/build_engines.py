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
    python scripts/build_engines.py --only ship_detector
    python scripts/build_engines.py --check         # report, build nothing

An engine is valid only for the GPU architecture and TensorRT version it was built on, so
this refuses to run without a device rather than producing a plan that will not load.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
    #: The model repository directory whose backend loads the same plan, if any.
    version_dir: Path | None


TARGETS = (
    Target(
        "ship_detector",
        MODELS / "yolo26n.onnx",
        MODELS / "yolo26n_fp32.engine",
        REPOSITORY / "ship_detector" / "1",
    ),
    Target(
        "ship_segmenter",
        MODELS / "yolo26n-seg.onnx",
        MODELS / "yolo26n-seg_fp32.engine",
        REPOSITORY / "ship_segmenter" / "1",
    ),
    Target("reid", MODELS / "reid_r50.onnx", MODELS / "reid_r50_fp32.engine", None),
)


def _precision_suffix(fp16: bool) -> str:
    return "fp16" if fp16 else "fp32"


def _engine_path(target: Target, fp16: bool) -> Path:
    """The flat engine path for this precision.

    The default names carry `_fp32` because that is what they are; asking for fp16 and
    silently overwriting an fp32 plan of the same name is how two runs end up comparing
    different engines while claiming to compare architectures.
    """
    if not fp16:
        return target.engine
    return target.engine.with_name(target.engine.name.replace("_fp32", "_fp16"))


def report(fp16: bool) -> int:
    print(f"{'model':<16} {'onnx':<10} {'engine':<10} path")
    print("-" * 72)
    missing = 0
    for target in TARGETS:
        engine = _engine_path(target, fp16)
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


def build(targets: tuple[Target, ...], *, fp16: bool, force: bool) -> int:
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
        engine = _engine_path(target, fp16)
        if not target.onnx.is_file():
            print(f"{target.name}: no ONNX at {target.onnx}", file=sys.stderr)
            failures += 1
            continue
        if engine.is_file() and not force:
            print(f"{target.name}: {engine.name} already built (use --force to rebuild)")
            continue

        started = time.monotonic()
        print(
            f"{target.name}: building {engine.name} ({_precision_suffix(fp16)}) ...", flush=True
        )
        try:
            build_engine(target.onnx, engine, fp16=fp16, timing_cache=timing_cache)
        except Exception as exc:  # the builder's own diagnostics are the actionable part
            print(f"{target.name}: FAILED — {exc}", file=sys.stderr)
            failures += 1
            continue
        elapsed = time.monotonic() - started
        size_mb = engine.stat().st_size / 1e6
        print(f"{target.name}: {engine.name}  {size_mb:.1f} MB in {elapsed:.1f}s")

        # The server loads its plan from the model's own version directory. Copying rather
        # than symlinking: a plan is an artefact, and a dangling link inside a container
        # whose mount layout differs is a confusing way to fail at start-up.
        if target.version_dir is not None:
            target.version_dir.mkdir(parents=True, exist_ok=True)
            destination = target.version_dir / "model.plan"
            destination.write_bytes(engine.read_bytes())
            print(f"{'':<16}  -> {destination.relative_to(REPO)}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fp16", action="store_true", help="build half-precision plans")
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument(
        "--check", action="store_true", help="report what exists, build nothing"
    )
    parser.add_argument("--only", action="append", default=[], help="build just this model")
    args = parser.parse_args(argv)

    if args.check:
        return report(args.fp16)

    selected = TARGETS
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {t.name for t in TARGETS}
        if unknown:
            print(f"unknown model(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        selected = tuple(t for t in TARGETS if t.name in wanted)

    return build(selected, fp16=args.fp16, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
