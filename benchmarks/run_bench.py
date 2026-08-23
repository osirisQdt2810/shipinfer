#!/usr/bin/env python3
"""Run both systems under one load and report sustained image throughput.

This is the entry point for the comparison the project exists to justify: ShipInfer against
`counting-simulation`, at 50 cameras x 20 fps on four GPUs, measured by the baseline's own
saturation methodology.

WHAT MAKES THE TWO NUMBERS COMPARABLE
-------------------------------------
Both systems are offered the same load: ``cameras x fps`` images per second, 1000 by
default. Both write a once-a-second buffer-occupancy log in the same shape. Both are then
analysed by the same fit: a buffer whose occupancy grows over the steady window is a module
that cannot keep up, and its sustained throughput is ``offered - growth``.

The one thing that is *not* symmetric is where an image enters, and getting this wrong is
the easiest way to publish a fake speed-up:

- The **baseline** runs two independent single-model pipelines. Half its source workers feed
  the detector queue, half feed the segmenter queue, and an image belongs to exactly one of
  them. Its system throughput is therefore ``det_sustained + seg_sustained`` — a sum over
  disjoint image streams, which is what its own report sums.

- **ShipInfer** runs one DAG. Every camera frame enters the pipeline queue exactly once and
  then fans out into crops: the detector sees the frame, the segmenter and the embedders see
  crops *derived from* it. Those crops are not new images. Summing every module's sustained
  rate the way the baseline's report does would count each frame once at the pipeline queue
  and again at the detector, and report roughly twice the real throughput.

So the comparison metric is defined once, here, as **images accepted per second at the point
every image enters exactly once** — a sum over the baseline's two disjoint queues, and the
pipeline queue alone for ShipInfer. :func:`system_throughput` is the only place that
decision lives, and :attr:`RunAnalysis.total_sustained` is deliberately not used for
ShipInfer because it is a sum over modules that are not disjoint.

This is conservative in ShipInfer's favour being *understated*, not overstated: one baseline
image passes through one model, while one ShipInfer frame passes through detect, then
conditional segmentation, then one or two embedders. Equal frames-per-second therefore
represents strictly more work on our side. The report says so rather than quietly banking it.

WHAT THE VERDICT MEANS
----------------------
A speed-up is only reported when both runs produced a usable measurement. If either side is
SATURATED its sustained rate is a *bound*, not a rate, and the ratio of two bounds is not a
speed-up — the report prints the bound and says the comparison is not available. A partial
number presented as a total is the failure mode this whole harness was built to avoid.

USAGE
-----
Runs in a container, on real GPUs, against real engines::

    deploy/rootless/test.sh --bench                 # the wrapper, defaults below
    python benchmarks/run_bench.py --seconds 70     # inside a container

Everything lands in ``--out-dir`` (default ``benchmarks/build/run-<label>``): each system's
occupancy JSONL, its console capture, and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness import analysis, baseline, shipinfer
from benchmarks.harness.analysis import SATURATED, RunAnalysis
from benchmarks.harness.config import BenchConfig

#: The baseline's two queues, each fed by its own disjoint half of the source workers.
BASELINE_ENTRY_MODULES = ("det", "seg")


@dataclass(frozen=True, slots=True)
class SystemThroughput:
    """Images per second a system sustained, and whether that number is a rate or a bound."""

    system: str
    images_per_s: float | None
    saturated: bool
    binding_module: str | None
    detail: str

    @property
    def is_rate(self) -> bool:
        """True when the number is a throughput; False when it is only an upper bound."""
        return self.images_per_s is not None and not self.saturated

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "images_per_s": None if self.images_per_s is None else round(self.images_per_s, 1),
            "saturated": self.saturated,
            "binding_module": self.binding_module,
            "is_rate": self.is_rate,
            "detail": self.detail,
        }


def system_throughput(run: RunAnalysis) -> SystemThroughput:
    """Images per second, counted once per image. The only definition in the harness.

    See this module's docstring for why the two systems sum different module sets. The rule
    in one line: add up the modules at which an image enters, and never add a module that
    only ever sees work derived from an image already counted.
    """
    by_name = {m.module: m for m in run.modules}
    binding = run.binding_module
    saturated = run.verdict == SATURATED

    if run.system == "baseline":
        entries = [by_name[n] for n in BASELINE_ENTRY_MODULES if n in by_name]
        if not entries:
            return SystemThroughput(
                run.system,
                None,
                saturated,
                binding.module if binding else None,
                f"no entry module in the log (saw {sorted(by_name)})",
            )
        if any(m.sustained is None for m in entries):
            return SystemThroughput(
                run.system,
                None,
                saturated,
                binding.module if binding else None,
                "an entry queue's offered rate was not known, so its sustained rate is not "
                "defined and the sum would be partial",
            )
        total = sum(m.sustained or 0.0 for m in entries)
        names = " + ".join(m.module for m in entries)
        return SystemThroughput(
            run.system,
            total,
            saturated,
            binding.module if binding else None,
            f"{names}: two disjoint image streams, so the sum is the system's image rate",
        )

    entry = by_name.get(shipinfer.PIPELINE_MODULE)
    if entry is None or entry.sustained is None:
        return SystemThroughput(
            run.system,
            None,
            saturated,
            binding.module if binding else None,
            f"no {shipinfer.PIPELINE_MODULE!r} module with a known offered rate in the log",
        )
    return SystemThroughput(
        run.system,
        entry.sustained,
        saturated,
        binding.module if binding else None,
        f"{shipinfer.PIPELINE_MODULE}: every camera frame enters here exactly once; the "
        f"downstream models see crops derived from those frames, not new images",
    )


def compare(base: SystemThroughput, ours: SystemThroughput, *, target: float) -> str:
    """The verdict paragraph. Refuses to divide two numbers that are not both rates."""
    lines = ["", "COMPARISON", "-" * 78]
    for t in (base, ours):
        value = "unmeasured" if t.images_per_s is None else f"{t.images_per_s:>8.1f} img/s"
        kind = "" if t.is_rate else "   (a BOUND, not a rate — the run saturated)"
        lines.append(f"  {t.system:<10} {value}{kind}")
        lines.append(f"             {t.detail}")
    lines.append("")

    if not (base.is_rate and ours.is_rate):
        which = [t.system for t in (base, ours) if not t.is_rate]
        lines += [
            f"  Speed-up: NOT AVAILABLE — {', '.join(which)} did not produce a sustained "
            f"rate.",
            "  A saturated run's number is an upper bound, and the ratio of two bounds is",
            "  not a speed-up. Re-run with a lower offered rate to find each system's",
            "  sustainable point, or with more seconds if the window was too short.",
        ]
        return "\n".join(lines)

    ratio = ours.images_per_s / base.images_per_s if base.images_per_s else float("inf")
    verdict = "MET" if ratio >= target else "NOT MET"
    lines += [
        f"  Speed-up: {ratio:.2f}x   (target >= {target:g}x — {verdict})",
        "",
        "  Read this conservatively: one baseline image passes through one model, while one",
        "  ShipInfer frame passes through detect, conditional segmentation and one or two",
        "  embedders. Equal images-per-second is strictly more work on our side.",
    ]
    return "\n".join(lines)


def _analyse(system: str, log: Path, cfg: BenchConfig, offered, entries) -> RunAnalysis:
    return analysis.analyse(
        analysis.read_log(log, sample_interval_s=cfg.sample_interval_s),
        system=system,
        warmup_s=cfg.warmup_s,
        offered=offered,
        capacity=cfg.buffer_capacity,
        entry_modules=entries,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cameras", type=int, default=50)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--gpus", default="2,3,4,5", help="physical CUDA ordinals, comma separated")
    p.add_argument("--seconds", type=float, default=70.0)
    p.add_argument("--warmup", type=float, default=10.0, dest="warmup_s")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--resolution", choices=("2k", "4k"), default="2k")
    p.add_argument(
        "--systems",
        default="baseline,shipinfer",
        help="which to run; run one at a time to keep the GPUs uncontended",
    )
    p.add_argument("--target", type=float, default=5.0, help="required speed-up")
    p.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="OMP_NUM_THREADS, applied to both systems; unset leaves both unpinned",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--label", default=None, help="names the output directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - {"baseline", "shipinfer"}
    if unknown:
        print(f"unknown system(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    # The default lives on BenchConfig, so a run directory is derived from it rather than
    # from a second copy of the same path that could drift out of step with it.
    base_out = args.out_dir or (BenchConfig().out_dir / f"run-{label}")
    base_out.mkdir(parents=True, exist_ok=True)

    cfg = BenchConfig(
        cameras=args.cameras,
        fps=args.fps,
        gpus=tuple(int(g) for g in args.gpus.split(",") if g.strip()),
        batch=args.batch,
        seconds=args.seconds,
        warmup_s=args.warmup_s,
        resolution=args.resolution,
        omp_threads=args.omp_threads,
        out_dir=base_out,
    ).resolved()
    out_dir = cfg.out_dir

    # torch reads CUDA_VISIBLE_DEVICES once, when it first initialises, so it has to be set
    # before either system starts rather than inside the driver that needs it. Here is the
    # only place that both knows the config and runs before anything imports torch -- the
    # harness modules import shipinfer lazily for exactly this reason.
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices()
    # The same OpenMP policy for both systems, or neither. Pinning only the baseline gave
    # our torch pre-processing the whole box while the baseline letterboxed on the CPU
    # inside its own single-threaded workers.
    if cfg.omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(cfg.omp_threads)

    print(
        f"load: {cfg.cameras} cameras x {cfg.fps:g} fps = {cfg.offered_total:g} img/s offered"
    )
    print(f"gpus: {list(cfg.gpus)}   seconds: {cfg.seconds:g} (warmup {cfg.warmup_s:g})")
    print(cfg.concurrency_note)
    print(f"out:  {out_dir}")

    runs: list[RunAnalysis] = []
    throughputs: dict[str, SystemThroughput] = {}

    if "baseline" in systems:
        print("\n=== baseline (counting-simulation, its own binary) ===", flush=True)
        result = baseline.run_baseline(cfg, out_dir / "baseline")
        if not result.ok:
            print("baseline produced no samples; aborting", file=sys.stderr)
            return 1
        # Each of the baseline's two queues is fed by its own half of the source
        # workers, so both offered rates are known exactly from the configuration --
        # unlike ShipInfer's crop-fed models, which have to be measured.
        run = _analyse(
            "baseline",
            result.log,
            cfg,
            offered=dict.fromkeys(BASELINE_ENTRY_MODULES, cfg.offered_per_module),
            entries=BASELINE_ENTRY_MODULES,
        )
        runs.append(run)
        throughputs["baseline"] = system_throughput(run)

    if "shipinfer" in systems:
        print("\n=== shipinfer (ingest -> scheduler -> engines -> reassembly) ===", flush=True)
        result = shipinfer.run_shipinfer(cfg, out_dir / "shipinfer")
        # Refuse before analysing. A run whose generator never delivered the load is not a
        # slower measurement, it is a different experiment, and reporting it against the
        # configured target publishes a rate that was never sustained.
        shipinfer.check_offer(cfg, result)
        achieved = shipinfer.achieved_offer(cfg, result)
        print(
            f"offered: {achieved:.1f} img/s achieved of {cfg.offered_total:g} target "
            f"({achieved / cfg.offered_total:.0%})"
        )
        run = _analyse(
            "shipinfer",
            result.log,
            cfg,
            offered=shipinfer.offered_rates(cfg, result),
            entries=(shipinfer.PIPELINE_MODULE,),
        )
        runs.append(run)
        throughputs["shipinfer"] = system_throughput(run)
        if result.per_device:
            print("\nper-device execution (the balancing evidence):")
            for model, devices in sorted(result.per_device.items()):
                spread = "  ".join(f"gpu{d}={n}" for d, n in sorted(devices.items()))
                print(f"  {model:<18} {spread}")

    print()
    print(analysis.render(runs))

    summary: dict[str, Any] = {
        "config": cfg.as_dict(),
        "runs": json.loads(analysis.as_json(runs)),
        "throughput": {k: v.as_dict() for k, v in throughputs.items()},
        "target": args.target,
    }

    if "baseline" in throughputs and "shipinfer" in throughputs:
        text = compare(throughputs["baseline"], throughputs["shipinfer"], target=args.target)
        print(text)
        base, ours = throughputs["baseline"], throughputs["shipinfer"]
        summary["speedup"] = (
            round(ours.images_per_s / base.images_per_s, 3)
            if base.is_rate and ours.is_rate and base.images_per_s
            else None
        )
        summary["target_met"] = (
            summary["speedup"] is not None and summary["speedup"] >= args.target
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
