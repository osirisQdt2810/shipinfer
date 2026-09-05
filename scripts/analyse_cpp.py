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


# doc: long two bounds apply to a model module and the judge must take the smaller
def capacities(meta: dict, samples) -> dict[str, int]:
    """The bound each module's occupancy can actually reach.

    `pipeline` is the frame queue: `pipeline_queue`, which was `buffer_capacity` until P5-C
    renamed it -- a renamed key travels to this judge as a KeyError, after the run.

    A MODEL module samples `Model::total_depth()`, the sum of its instances' queue depths, and
    TWO bounds apply. Structurally it cannot pass `instance_queue x instances`; in practice the
    workers are the concurrency feeding it. Whichever is smaller is the one the series can
    reach, and this file's whole rule is that it must not be the friendlier judge -- so a
    ceiling that is too loose (a plateau guard that can never trip, every model row SUSTAINED
    and unreachable) is the failure to avoid, and the minimum is what avoids it.

    The worker count alone was right only while every instance queue was 65536 and could
    therefore never bind. P5-C made 64 real, and 64 x 2 x 2 is below a 48-worker run.
    """
    config = meta["config"]
    per_device = {model["name"]: int(model["instances_per_device"]) for model in meta["models"]}
    devices = len(config["gpus"])
    instance_queue = int(config["instance_queue"])
    workers = int(config["workers"])
    out = {}
    for module in samples.modules:
        if module == "pipeline":
            out[module] = int(config["pipeline_queue"])
        elif module in per_device:
            out[module] = min(workers, instance_queue * per_device[module] * devices)
        else:
            # A module the run record does not name: the worker count is the only bound left,
            # and saying so beats refusing to score a run over a module nobody added here.
            out[module] = workers
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
        # Per module, never one scalar. The first version passed one, and the four model rows
        # it printed as SUSTAINED were structurally unreachable by the plateau guard: the
        # friendlier judge this file says it must never be. `capacities` states which bound
        # each module actually has, and why a model module takes the smaller of two.
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
