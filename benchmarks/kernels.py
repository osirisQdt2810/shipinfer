#!/usr/bin/env python3
"""The **kernel tier**: how fast is each `ImageOps` implementation, at the sizes we run.

WHY THIS EXISTS
---------------
`tests/runtime/test_ops_parity.py` proves the implementations *agree*. Nothing measured how
much faster the fused ones are, and the repository has been quoting an inherited "50x on
preprocessing" figure from the reference project that nobody here reproduced. R44 asks for
three tiers — system, algo, kernel — and only the system tier existed.

This is the bottom one. It answers a narrow question precisely: for one op, at one shape,
what does each registered implementation cost? That is the number an optimisation is judged
against, and it is the number that says whether a fused kernel is worth its build.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not report a speed-up for the *pipeline*. A kernel that is 50x faster on an op that
is 2% of the frame budget buys 2%. `benchmarks/stages.py` is the tier that puts these costs
into per-frame context, and `run_bench.py` is the one that measures the system. Reading a
kernel number as a system number is the error this file's own docstring exists to prevent.

METHOD
------
Warm up (allocators settle, CUDA contexts initialise, the first launch pays for JIT), then
time `--repeat` batches of `--iterations` calls and report the **median batch**, not the mean:
a single scheduler hiccup on a shared box moves a mean and does not move a median. Device
implementations synchronise inside the timed region, because a kernel launch that returns
immediately has measured nothing.

    deploy/rootless/test.sh --version    # (any container entry point)
    python benchmarks/kernels.py --op all
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

#: The sizes this project actually runs, so a number here transfers to the deployment.
#: 1080p in, 640x640 to the detector, 256x128 person crops, 15 objects a frame.
FRAME_HW = (1080, 1920)
DETECT_HW = (640, 640)
CROP_HW = (256, 128)
OBJECTS = 15
#: 25 000 candidate boxes is what a YOLO head emits before suppression — the case the op
#: exists for, rather than the twenty that survive it.
NMS_CANDIDATES = 25_000


@dataclass(frozen=True, slots=True)
class Measurement:
    """One implementation's cost for one op."""

    op: str
    implementation: str
    #: Median seconds for one call. Median rather than mean: see the module docstring.
    seconds: float
    #: Spread across the timed batches, as a fraction of the median. A wide spread on a
    #: shared box means the number is not reproducible and should be said so, not smoothed.
    spread: float
    calls: int
    note: str = ""

    @property
    def per_call_us(self) -> float:
        return self.seconds * 1e6

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "implementation": self.implementation,
            "per_call_us": round(self.per_call_us, 2),
            "spread": round(self.spread, 3),
            "calls": self.calls,
            "note": self.note,
        }


@dataclass
class OpResult:
    """Every implementation's cost for one op, plus the ratio between them."""

    op: str
    measurements: list[Measurement] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def baseline(self) -> Measurement | None:
        """numpy, when it ran. The readable implementation is the thing to be faster *than*."""
        for measurement in self.measurements:
            if measurement.implementation == "numpy":
                return measurement
        return None

    def speedup(self, measurement: Measurement) -> float | None:
        base = self.baseline
        if base is None or measurement.seconds <= 0:
            return None
        return base.seconds / measurement.seconds


def _time(
    call: Callable[[], Any], *, iterations: int, repeat: int, warmup: int
) -> tuple[float, float]:
    """Median per-call seconds and the spread across batches.

    `perf_counter` around a batch rather than around each call: at a few microseconds an op,
    the clock read is a measurable share of what is being measured.
    """
    for _ in range(warmup):
        call()
    batches: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(iterations):
            call()
        batches.append((time.perf_counter() - start) / iterations)
    median = statistics.median(batches)
    spread = 0.0 if median <= 0 else (max(batches) - min(batches)) / median
    return median, spread


def _synchronised(
    ops: Any, call: Callable[[], Any], device: int | None = None
) -> Callable[[], Any]:
    """Wrap a device call so the timed region includes the work, not just the launch.

    A CUDA launch is asynchronous. Timing one without a synchronise measures the cost of
    *asking*, which for a fused kernel is close to zero and would produce a spectacular,
    entirely fictional speed-up. This is the single easiest way to publish a fake number in
    a kernel benchmark, so it is handled once, here.

    The synchronise is on ``device`` — the one the ops are bound to — not on the current
    device. ``torch.cuda.synchronize()`` with no argument waits on cuda:0, so a run on
    ``--device 2`` would have timed the launch and waited for the wrong GPU.
    """
    # `on_device` is the ABC's own answer (`runtime/ops/base.py`: "callers should branch on
    # `on_device` rather than catching"). The first version sniffed `describe()` for the words
    # "cuda" or "torch", which worked for the two device implementations that happened to
    # mention them and would have left any other one unsynchronised — timing its launches.
    if not getattr(ops, "on_device", False):
        return call
    try:
        import torch
    except ImportError:
        return call
    if not torch.cuda.is_available():
        return call

    def synchronised() -> Any:
        result = call()
        torch.cuda.synchronize(device)
        return result

    return synchronised


def _destination(device: int | None) -> Any:
    """The device the `letterbox_to_device` output lives on: the one the ops are bound to.

    No `ImageOps` implementation exposes a public `device` attribute (`TorchImageOps` keeps
    `_device`, `NativeImageOps` `_device_index`), so the first version's
    ``getattr(ops, "device", "cuda")`` always fell through to the *current* device. On
    ``--device 2`` the native path then refused the cross-device write (a skip, so the
    device-fair column vanished) and the torch path silently ran the whole op on cuda:0
    while the table said cuda:2. The index is threaded through instead of guessed.
    """
    import torch

    return torch.device("cuda") if device is None else torch.device("cuda", device)


def _implementations(
    only: str | None, device: int | None
) -> tuple[list[tuple[str, Any]], dict[str, str]]:
    """Every registered `ImageOps` that could be constructed, and why the rest could not.

    **Construction is where an implementation usually fails, not the call.** `native` raises
    `ConfigurationError` from its constructor when the submodule is not built — with a
    remedy in the message — so catching only at call time reports nothing and crashes the
    run. A missing native build is a fact about this machine and belongs in the table; a
    shorter table with no explanation is how "we never measured it" becomes "it is not
    faster".
    """
    # Imported for the side effect: each module registers itself on import, so the registry
    # is empty until the package has been touched.
    from shipinfer.runtime.ops import IMAGE_OPS

    built: list[tuple[str, Any]] = []
    unavailable: dict[str, str] = {}
    for name in sorted(IMAGE_OPS.names()):
        if only and name != only:
            continue
        try:
            built.append((name, _bind(IMAGE_OPS, name, device)))
        except Exception as exc:
            # First line only: the remedy is several lines long and the table has one column.
            unavailable[name] = str(exc).splitlines()[0]
    return built, unavailable


def _bind(registry: Any, name: str, device: int | None) -> Any:
    """Construct one implementation **the way production constructs it**.

    `TorchImageOps.__init__` falls back to `torch.device("cpu")` unless it is given a
    `device_index`, and `PipelineRunner._build_ops` always gives it one. The first version of
    this file called `create(name)` with no arguments and therefore timed torch on the *CPU*:
    it came out 7-13x slower than numpy, which is a true fact about a configuration nobody
    runs and a false one about this project.

    That is the recurring error in every measurement in this repository — benchmarking
    something adjacent to what production does — so the binding is explicit here and the
    device each implementation ended up on is printed in the table.
    """
    if device is None:
        return registry.create(name)
    try:
        return registry.create(name, device_index=device)
    except TypeError:
        # A host-only implementation takes no device. numpy is the case; it is not an error.
        return registry.create(name)


def _to_device_case(
    ops: Any, frame: np.ndarray, params: Any, device: int | None
) -> Callable[[], Any]:
    """A closure for `letterbox_to_device`, with the destination allocated once, on ``device``.

    The contract is strict about `out` — right device, float32, contiguous, rank 4 NCHW —
    and every violation is silent at the pixel level, so it is checked rather than assumed.
    Allocating it here rather than inside the timed call is the point: the production caller
    owns a buffer for the life of the instance, and timing an allocation per call would
    measure something nobody does.

    Host-only implementations raise `NotImplementedError` from the call, which `measure`
    already reports as a skip with the reason.
    """
    try:
        import torch
    except ImportError:
        return lambda: ops.letterbox_to_device([frame], None, params)

    if not getattr(ops, "on_device", False) or not torch.cuda.is_available():
        # Let the call raise its own `NotImplementedError`, so the table says *why*.
        return lambda: ops.letterbox_to_device([frame], None, params)

    out = torch.empty((1, 3, *DETECT_HW), dtype=torch.float32, device=_destination(device))
    return lambda: ops.letterbox_to_device([frame], out, params)


@dataclass(frozen=True)
class Inputs:
    """One op's inputs, built once per op and shared by every implementation timed on it.

    Seeded, and built **outside** the per-implementation loop: NMS cost depends on the overlap
    structure of the boxes, so timing numpy and native on different random box sets put a data
    difference on top of the implementation difference in the `vs numpy` ratio.
    """

    frame: np.ndarray
    boxes: np.ndarray
    nms_boxes: np.ndarray
    nms_scores: np.ndarray


def _inputs(seed: int) -> Inputs:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (*FRAME_HW, 3), dtype=np.uint8)
    boxes = np.stack(
        [
            rng.uniform(0, FRAME_HW[1] - 64, OBJECTS),
            rng.uniform(0, FRAME_HW[0] - 64, OBJECTS),
            rng.uniform(64, FRAME_HW[1], OBJECTS),
            rng.uniform(64, FRAME_HW[0], OBJECTS),
        ],
        axis=1,
    ).astype(np.float32)
    nms_boxes = np.stack(
        [
            rng.uniform(0, 1000, NMS_CANDIDATES),
            rng.uniform(0, 1000, NMS_CANDIDATES),
            rng.uniform(1000, 2000, NMS_CANDIDATES),
            rng.uniform(1000, 2000, NMS_CANDIDATES),
        ],
        axis=1,
    ).astype(np.float32)
    nms_scores = rng.uniform(0, 1, NMS_CANDIDATES).astype(np.float32)
    return Inputs(frame=frame, boxes=boxes, nms_boxes=nms_boxes, nms_scores=nms_scores)


def _cases(ops: Any, inputs: Inputs, device: int | None) -> dict[str, Callable[[], Any]]:
    """One closure per op over ``inputs``, allocated *outside* the timed region — allocation
    is not what is being measured. ``device`` is where a device-side output lives."""
    from shipinfer.runtime.ops.base import NormalizeParams

    params = NormalizeParams()
    frame, boxes = inputs.frame, inputs.boxes
    nms_boxes, nms_scores = inputs.nms_boxes, inputs.nms_scores

    return {
        "letterbox": lambda: ops.letterbox_batch([frame], DETECT_HW, params),
        # The same work without the trip home. `letterbox_batch` returns numpy by contract, so
        # a device implementation pays a device->host copy of the result that numpy never
        # makes — comparing only that column rigs the table against exactly the implementations
        # this project exists to justify. `letterbox_to_device` is the honest device number,
        # and it is also the call the production path actually makes: the letterboxed tensor's
        # next stop is a TensorRT binding, not the host.
        "letterbox_to_device": _to_device_case(ops, frame, params, device),
        "crop_batch": lambda: ops.crop_batch(frame, boxes, CROP_HW, params),
        "nms": lambda: ops.nms(nms_boxes, nms_scores, 0.45, 0.25, 300),
    }


def measure(
    op: str,
    *,
    only: str | None,
    device: int | None,
    iterations: int,
    repeat: int,
    warmup: int,
    seed: int = 0,
) -> OpResult:
    result = OpResult(op=op)
    available, unavailable = _implementations(only, device)
    result.skipped.update(unavailable)
    inputs = _inputs(seed)  # once per op: every implementation is timed on the same data
    for name, ops in available:
        cases = _cases(ops, inputs, device)
        if op not in cases:
            continue
        call = _synchronised(ops, cases[op], device)
        try:
            call()  # once, outside the timing, so an unsupported op is reported not timed
        except NotImplementedError as exc:
            result.skipped[name] = f"not implemented: {exc}"
            continue
        except Exception as exc:  # the reason belongs in the report, not in a traceback
            result.skipped[name] = f"{type(exc).__name__}: {exc}"
            continue
        seconds, spread = _time(call, iterations=iterations, repeat=repeat, warmup=warmup)
        result.measurements.append(
            Measurement(
                op=op,
                implementation=name,
                seconds=seconds,
                spread=spread,
                calls=iterations * repeat,
                note=ops.describe(),
            )
        )
    return result


#: Above this fraction of the CPU count a one-minute load average means the box has a busy
#: neighbour, and a microbenchmark is the first thing that stops being reproducible.
LOAD_WARN_FRACTION = 0.5


def load_note() -> str:
    """One line naming the host's load, and saying plainly when it is too high to trust.

    A kernel benchmark is far more sensitive to a noisy neighbour than a system benchmark:
    the system one measures a saturated pipeline where the contention is part of the picture,
    while this one is timing microseconds of one thread. Recorded rather than assumed, because
    the first run of this file was taken at load 41 of 48 and its spreads reached 76%.
    """
    one, five, fifteen = os.getloadavg()
    cpus = os.cpu_count() or 1
    line = f"host: load {one:.1f}/{five:.1f}/{fifteen:.1f} over {cpus} cpus"
    if one > cpus * LOAD_WARN_FRACTION:
        line += "  <- BUSY. These numbers are not reproducible; re-run on a quiet box."
    return line


def render(results: list[OpResult]) -> str:
    lines = [
        "",
        load_note(),
        f"{'op':<20} {'impl':<10} {'per call':>12} {'spread':>8} {'vs numpy':>10}   where",
        "-" * 88,
    ]
    for result in results:
        for measurement in sorted(result.measurements, key=lambda m: m.seconds):
            ratio = result.speedup(measurement)
            # A spread over 20% on a shared box means the median is not reproducible, and
            # saying so is more useful than printing three significant figures of noise.
            flag = " (noisy)" if measurement.spread > 0.20 else ""
            lines.append(
                f"{result.op:<20} {measurement.implementation:<10} "
                f"{measurement.per_call_us:>9.1f} us {measurement.spread:>7.1%} "
                f"{('-' if ratio is None else f'{ratio:>9.2f}x'):>10}{flag}"
                f"   {measurement.note}"
            )
        for name, why in sorted(result.skipped.items()):
            lines.append(f"{result.op:<20} {name:<10} {'skipped':>12}   {why}")
    lines += [
        "",
        "`letterbox` returns numpy by contract, so a device implementation pays a copy home",
        "that numpy never makes. `letterbox_to_device` is the device-fair column, and the one",
        "the production path actually calls.",
        "",
        "A kernel speed-up is not a system speed-up: an op that is 2% of the frame budget",
        "caps out at 2% however fast it gets. `benchmarks/stages.py` puts these in per-frame",
        "context; `benchmarks/run_bench.py` measures the system.",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--op",
        default="all",
        choices=("all", "letterbox", "letterbox_to_device", "crop_batch", "nms"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the inputs; the same seed times every implementation on the same data",
    )
    parser.add_argument("--implementation", default=None, help="only this one, e.g. native")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help=(
            "the device index each implementation is bound to, as PipelineRunner binds it. "
            "Pass -1 to construct with no device, which measures torch on the CPU — a "
            "configuration nothing in this project runs."
        ),
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=7, help="timed batches; the median wins")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from shipinfer.runtime import containment

    # A measurement, so the container rule applies (CLAUDE.md). The gate lives in the process
    # that would do the work, because a deny-list over command text cannot be made sound.
    containment.require_container("the kernel benchmark")

    args.device = None if args.device < 0 else args.device
    ops = (
        ("letterbox", "letterbox_to_device", "crop_batch", "nms")
        if args.op == "all"
        else (args.op,)
    )
    results = [
        measure(
            op,
            only=args.implementation,
            device=args.device,
            iterations=args.iterations,
            repeat=args.repeat,
            warmup=args.warmup,
            seed=args.seed,
        )
        for op in ops
    ]
    print(render(results))
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "host": {
                        "load_average": [round(v, 2) for v in os.getloadavg()],
                        "cpu_count": os.cpu_count(),
                    },
                    "shapes": {
                        "frame": list(FRAME_HW),
                        "detect": list(DETECT_HW),
                        "crop": list(CROP_HW),
                        "objects": OBJECTS,
                        "nms_candidates": NMS_CANDIDATES,
                    },
                    "results": [
                        {
                            "op": r.op,
                            "measurements": [m.as_dict() for m in r.measurements],
                            "skipped": r.skipped,
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
