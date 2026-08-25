#!/usr/bin/env python3
"""The **algo tier**: where does a frame's time actually go, stage by stage.

WHY THIS EXISTS
---------------
R44 asks for three tiers — system, algo, kernel — and for a long time only the system tier
existed. The system tier says *which queue grows*; it does not say *where the time goes*, and
every optimisation in this repository so far has been chosen by reasoning from a symptom.
The operator was right to call that out: the answer should be a profile, not an inference.

This is the middle tier. It runs the real pipeline at a load it can comfortably keep up with
and reports what each stage cost, per call and per frame. Deliberately **not** at saturation:
under saturation a stage's measured latency includes the time it spent waiting behind other
frames, so a queueing artefact reads as an expensive stage. A profile wants the service time.

WHAT IT REUSES, AND WHY THAT MATTERS
------------------------------------
Nothing here re-implements a measurement. `PipelineStage.run` is already a template method
that stamps `elapsed_us` on every `StageOutcome`, and `_CollectorObserver` already feeds that
into the `shipinfer_pipeline_stage_latency_us` histogram, labelled by stage. So the algo tier
is a *reader*: it drives `benchmarks.harness.shipinfer.run_shipinfer` and then renders
histograms the production server already keeps. A separate timing path would be a second
implementation that could disagree with the one operators actually watch.

READING THE OUTPUT
------------------
`per frame` is the number to look at: a stage that costs 8 ms per call but runs on one frame
in three costs 2.7 ms per frame. The `serial` total at the bottom is what one frame would cost
with no concurrency at all; the gap between it and the achieved frame time is what the
instance pools and the worker pool are buying.

    deploy/rootless/bench.sh ...        # the system tier
    python benchmarks/stages.py         # this one, inside a container
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness.config import BenchConfig
from benchmarks.harness.histograms import HistogramCell, read_cell

#: **Fallback only.** The stage names come from the run — `ShipInferResult.stages` is read off
#: `runner.graph.stage_names` — because a stage added to the graph and not listed here would
#: be invisible, which is the failure mode a profile can least afford. These two are what
#: every graph has, and are used only for a result that carries no names at all.
FRAME_STAGES = ("detect", "crop")


@dataclass(frozen=True, slots=True)
class StageCost:
    """One stage's cost, per call and per frame."""

    stage: str
    calls: int
    #: Exact per-call mean: the histogram's `sum / count`. The cost model is built on this.
    mean_us: float
    #: Bucket-edge quantiles — the histogram's resolution, kept as distribution colour. Two
    #: stages in one bucket share a p50 while their means differ; the first version of this
    #: tier charged stages by p50 and rendered a 2.3x difference as a tie.
    p50_us: float
    p95_us: float
    #: Calls per frame. Below 1.0 for a conditional branch — `ship_segmenter` runs only on
    #: frames that contain a ship, and charging it to every frame would overstate it.
    calls_per_frame: float
    per_frame_us: float

    @property
    def per_frame_ms(self) -> float:
        return self.per_frame_us / 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "calls": self.calls,
            "mean_us": round(self.mean_us, 1),
            "p50_us": round(self.p50_us, 1),
            "p95_us": round(self.p95_us, 1),
            "calls_per_frame": round(self.calls_per_frame, 3),
            "per_frame_us": round(self.per_frame_us, 1),
        }


@dataclass(frozen=True, slots=True)
class Profile:
    """Every stage's cost, plus the arithmetic that puts them in context."""

    stages: tuple[StageCost, ...]
    frames: int
    elapsed_s: float
    #: Which window every number above describes — "steady window" normally; the whole run,
    #: named as such, when the run was shorter than its warm-up or the result predates the
    #: steady-window cells. Printed, because a profile that mixes windows lies quietly.
    window: str
    #: Whether the run stayed unsaturated. A saturated run's latencies include queueing, so
    #: the profile would be measuring the backlog rather than the work.
    offered: float
    achieved: float

    @property
    def serial_per_frame_us(self) -> float:
        """What one frame costs with no concurrency — the sum of every stage's per-frame cost."""
        return sum(s.per_frame_us for s in self.stages)

    @property
    def wall_per_frame_us(self) -> float:
        return 0.0 if self.frames <= 0 else self.elapsed_s * 1e6 / self.frames

    @property
    def kept_up(self) -> bool:
        """98% of the offered load, the same bar `check_offer` holds the system tier to."""
        return self.offered > 0 and self.achieved >= 0.98 * self.offered

    def share(self, cost: StageCost) -> float:
        total = self.serial_per_frame_us
        return 0.0 if total <= 0 else cost.per_frame_us / total


STEADY_WINDOW = "steady window"


def profile_from(
    result: Any, metrics: Any, config: BenchConfig, stages: tuple[str, ...]
) -> Profile:
    """Read the histograms the run already filled. No new instrumentation.

    **One window.** Every number comes from the steady window: `steady_stage_latency` is each
    stage's histogram at the end minus the histogram at the warm-up boundary, and the frame
    count it is divided by is `steady_frames_accepted`, read at the same two instants. The
    first version divided a steady duration by a whole-run frame count and charged warm-up
    latencies — engines deserialising, first calls paying JIT — to the service-time profile.

    **Exact totals.** `per_frame_us` is `total_us / frames`, both exact. `p50`/`p95` are the
    histogram's bucket edges and are reported as colour, not used as a cost.

    `calls_per_frame` comes from the stage's own run count rather than being assumed to be
    one: `crop` runs once a frame, the embedders run once per *object batch*, and a
    conditional branch runs on a subset of frames. Assuming one would overstate the cheap
    stages and understate the expensive ones by the same factor.
    """
    steady_cells = getattr(result, "steady_stage_latency", None)
    cells: dict[str, HistogramCell] = dict(steady_cells or {})
    if steady_cells is not None:
        # The result carries the steady window — even one in which no stage recorded, which
        # is a fact about the run and not a reason to fall back to a different window. The
        # duration comes from the same branch as the frame count: `steady_s` here, never
        # `elapsed_s`, so the two cannot describe different windows.
        frames = int(getattr(result, "steady_frames_accepted", 0))
        elapsed = float(getattr(result, "steady_s", 0.0))
        read = int(getattr(result, "steady_frames_read", 0))
        window = (
            "whole run — the run ended before its own warm-up, so there is no steady window"
            if getattr(result, "steady_is_whole_run", False)
            else STEADY_WINDOW
        )
    else:
        # A result with no per-window cells (an older run file, or a run that recorded no
        # stage). Read the cumulative histogram and say which window that is.
        frames = int(getattr(result, "frames_accepted", 0))
        elapsed = float(getattr(result, "elapsed_s", 0.0))
        read = int(getattr(result, "frames_read", 0))
        window = "whole run — this result carries no steady-window histograms"
        if metrics is not None:
            cells = {s: read_cell(metrics.stage_latency_us, stage=s) for s in stages}
    costs: list[StageCost] = []
    for stage in stages:
        cell = cells.get(stage)
        if cell is None or cell.count == 0:
            continue
        per_frame_calls = 0.0 if frames <= 0 else cell.count / frames
        costs.append(
            StageCost(
                stage=stage,
                calls=cell.count,
                mean_us=cell.mean,
                p50_us=cell.quantile(0.5),
                p95_us=cell.quantile(0.95),
                calls_per_frame=per_frame_calls,
                per_frame_us=0.0 if frames <= 0 else cell.total / frames,
            )
        )
    return Profile(
        stages=tuple(sorted(costs, key=lambda c: -c.per_frame_us)),
        frames=frames,
        elapsed_s=elapsed,
        window=window,
        offered=config.offered_total,
        achieved=0.0 if elapsed <= 0 else read / elapsed,
    )


def render(profile: Profile) -> str:
    lines = [
        "",
        f"host: load {'/'.join(f'{v:.1f}' for v in os.getloadavg())} over {os.cpu_count()} cpus",
        f"offered {profile.offered:g} img/s, delivered {profile.achieved:.1f}, "
        f"{profile.frames} frames in {profile.elapsed_s:.1f}s ({profile.window})",
        "",
    ]
    if not profile.kept_up:
        lines += [
            "WARNING: the pipeline did not keep up with the offered load, so these latencies",
            "include time spent queueing behind other frames. A profile wants service time —",
            "lower --cameras or --fps until it does, then read the table.",
            "",
        ]
    lines += [
        f"{'stage':<18} {'mean':>10} {'p50':>9} {'p95':>9} {'calls/frame':>12} "
        f"{'per frame':>11} {'share':>7}",
        "-" * 82,
    ]
    for cost in profile.stages:
        lines.append(
            f"{cost.stage:<18} {cost.mean_us:>8.0f}us {cost.p50_us:>7.0f}us "
            f"{cost.p95_us:>7.0f}us {cost.calls_per_frame:>12.2f} "
            f"{cost.per_frame_ms:>9.2f}ms {profile.share(cost):>6.1%}"
        )
    lines += [
        "-" * 82,
        f"{'serial per frame':<18} {'':>10} {'':>9} {'':>9} {'':>12} "
        f"{profile.serial_per_frame_us / 1000:>9.2f}ms",
        f"{'wall per frame':<18} {'':>10} {'':>9} {'':>9} {'':>12} "
        f"{profile.wall_per_frame_us / 1000:>9.2f}ms",
        "",
        "`mean` and `per frame` are exact (the histogram's sum over its count, over frames",
        "accepted in the same window). `p50`/`p95` are bucket upper edges — the histogram's",
        "resolution — and two stages in one bucket share them while their means differ.",
        "`serial per frame` is one frame with no concurrency at all. The gap between it and",
        "`wall per frame` is what the instance pools and the worker pool are buying — if they",
        "are close, adding workers will not help and the answer is a cheaper stage.",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Defaults chosen to stay *below* saturation on this box: the Python plane retires ~77
    # img/s, so 60 leaves headroom and the latencies are service time rather than queueing.
    parser.add_argument("--cameras", type=int, default=12)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--warmup", type=float, default=10.0, dest="warmup_s")
    parser.add_argument("--gpus", default="2,3,4,5")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from shipinfer.runtime import containment

    # A measurement, so the container rule applies (CLAUDE.md). The gate lives in the process
    # that would do the work, because a deny-list over command text cannot be made sound.
    containment.require_container("the algo benchmark")

    config = BenchConfig(
        cameras=args.cameras,
        fps=args.fps,
        gpus=tuple(int(g) for g in args.gpus.split(",") if g.strip()),
        seconds=args.seconds,
        warmup_s=args.warmup_s,
        out_dir=args.out_dir or (BenchConfig().out_dir / "stages"),
    ).resolved()

    # torch reads CUDA_VISIBLE_DEVICES once, at its first CUDA call, so it has to be set
    # before anything imports it — the same reason `run_bench` sets it in `main`.
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices()

    from benchmarks.harness import shipinfer as harness

    result = harness.run_shipinfer(config, config.out_dir)
    metrics = harness.last_pipeline_metrics()
    if metrics is None:
        print(
            "the run did not expose its pipeline metrics, so there is nothing to profile",
            file=sys.stderr,
        )
        return 1

    stages = tuple(getattr(result, "stages", ()) or FRAME_STAGES)
    profile = profile_from(result, metrics, config, stages)
    print(render(profile))
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "config": config.as_dict(),
                    "frames": profile.frames,
                    "elapsed_s": profile.elapsed_s,
                    "window": profile.window,
                    "offered": profile.offered,
                    "achieved": round(profile.achieved, 1),
                    "kept_up": profile.kept_up,
                    "serial_per_frame_us": round(profile.serial_per_frame_us, 1),
                    "wall_per_frame_us": round(profile.wall_per_frame_us, 1),
                    "stages": [s.as_dict() for s in profile.stages],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
