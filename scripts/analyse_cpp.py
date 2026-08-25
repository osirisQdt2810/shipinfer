#!/usr/bin/env python3
"""Score a C++ data-plane run with the **same** judge the other two systems use.

There is deliberately no new measurement code here. `benchmarks/harness/analysis.py` fits the
line, decides SATURATED / SUSTAINED / DRAINING / UNMEASURED and applies every guard it applies
to the Python driver and to the baseline binary. Given that this port exists to look good, the
one thing it must not have is a friendlier judge than the thing it is being compared against.

    python scripts/analyse_cpp.py .artifacts/cpp/balanced.jsonl --read 70808 --elapsed 72
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness import analysis


def capacities(meta: dict, samples) -> dict[str, int]:
    """The bound each module's occupancy can actually reach."""
    workers = int(meta["config"]["workers"])
    out = {}
    for module in samples.modules:
        out[module] = (
            int(meta["config"]["buffer_capacity"]) if module == "pipeline" else workers
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--read",
        type=int,
        required=True,
        help="frames_read from the binary's own report — the *measured* offered load, not the "
        "configured target. A run that never delivered its load is a different experiment.",
    )
    parser.add_argument("--elapsed", type=float, required=True, help="wall seconds of the run")
    parser.add_argument("--warmup", type=float, default=10.0)
    args = parser.parse_args()

    meta = json.loads(args.log.read_text().splitlines()[0])["meta"]
    achieved = args.read / args.elapsed
    samples = analysis.read_log(args.log, sample_interval_s=1.0)
    run = analysis.analyse(
        samples,
        system="cpp",
        warmup_s=args.warmup,
        # Only the entry module gets an offered rate from the frame count. The detector is fed by
        # the worker drain loop, not by the cameras — under saturation only a fraction of what
        # was read ever reaches it — so giving it the cameras' rate printed a detector row that
        # "sustained" more than the whole run retired. The object-fed models are data-driven for
        # the same reason the Python harness measures theirs instead of asserting them.
        offered={"pipeline": achieved},
        # Per module, never one scalar. The pipeline queue's bound is the configured capacity;
        # a model module's series is `ModelPool::waiting()`, which is bounded by the *worker*
        # count — so a pool with every worker blocked in `lease()` plateaus at `workers`, and a
        # plateau guard set at 0.95 x 65536 could never trip for it. The first version passed the
        # scalar, and the four model rows it printed as SUSTAINED were structurally unreachable
        # by the guard: the friendlier judge this file says it must never be.
        capacity=capacities(meta, samples),
        entry_modules=("pipeline",),
    )
    print(analysis.render([run]))
    target = meta["config"]["cameras"] * meta["config"]["fps"]
    print(f"\noffered: {achieved:.1f} img/s of {target:g} target ({achieved / target:.0%})")
    print(f"verdict: {run.verdict}")
    binding = run.binding_module
    if binding is not None:
        print(f"binding module: {binding.module} (growth {binding.fit.slope:+.1f}/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
