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

WHAT THE VERDICT MEANS, AND WHICH DIRECTION EACH ONE ERRS IN
------------------------------------------------------------
Every run yields one of three things, and conflating them is how a harness publishes a
number it did not measure. This used to refuse SATURATED as "a bound, not a rate", which had
it exactly backwards and made the headline deliverable unreachable: both systems are offered
the same 1000 img/s by construction, so either neither saturated and each reported its own
offered rate back (ratio 1.00x, "NOT MET"), or one did and the comparison was declared
unavailable. There was no run that could print a speed-up.

- **SATURATED** (and not capped) is a **capacity**. The buffer grew linearly, so
  ``offered - growth`` is the rate the module actually retired. This is the whole of the
  buffer-growth methodology, and the one regime in which the number means something exact.
- **SUSTAINED** or **DRAINING** is a **floor**. Nothing grew, so the system kept up with
  what it was given and its real capacity is *at least* the offered rate — how much more,
  this run cannot say.
- **UNMEASURED** is nothing. A capped buffer sheds instead of growing, so its slope stops
  meaning anything; a module whose fit will not bound is the same.

The ratio inherits its meaning from the pair, and :func:`ratio_of` says which it has:

===================  ===================  ==============================================
baseline             shipinfer            the ratio is
===================  ===================  ==============================================
capacity             capacity             an exact speed-up
capacity             floor                a **floor** on the speed-up — ">= Nx", which is
                                          enough to *meet* a target
floor                capacity             a **ceiling** — enough to miss a target, never
                                          enough to meet one
floor                floor                nothing. Both kept up; raise the offered rate.
either               UNMEASURED           nothing
===================  ===================  ==============================================

Which is why ``--sweep`` exists: one offered rate can leave both sides on the floor, and the
answer is to climb until somebody saturates rather than to argue about the point you have.
A partial number presented as a total is the failure mode this whole harness was built to
avoid; a measurable regime refused on principle is the one it acquired in trying.

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness import analysis, baseline, shipinfer
from benchmarks.harness.analysis import SATURATED, RunAnalysis
from benchmarks.harness.config import BenchConfig

#: The baseline's two queues, each fed by its own disjoint half of the source workers.
BASELINE_ENTRY_MODULES = ("det", "seg")

#: What a system's number is. See the module docstring's table for how a pair combines.
#: A measured rate: the buffer grew, so `offered - growth` is what the module retired.
CAPACITY = "capacity"
#: A lower bound: nothing grew, so the system can do at least this and possibly much more.
FLOOR = "floor"
#: No measurement at all.
NOTHING = "none"

#: What the *ratio* of two of those is.
EXACT = "exact"
#: The true speed-up is this or better — enough to meet a target, never to miss one.
AT_LEAST = "at_least"
#: The true speed-up is this or worse — enough to miss a target, never to meet one.
AT_MOST = "at_most"

MET = "MET"
NOT_MET = "NOT MET"
INCONCLUSIVE = "INCONCLUSIVE"
UNAVAILABLE = "NOT AVAILABLE"


@dataclass(frozen=True, slots=True)
class SystemThroughput:
    """Images per second a system sustained, and whether that number is a rate or a bound."""

    system: str
    images_per_s: float | None
    saturated: bool
    binding_module: str | None
    detail: str
    #: The run's own verdict, carried whole rather than flattened. Reducing it to
    #: ``saturated`` lost UNMEASURED, and UNMEASURED then read as "not saturated" and
    #: therefore as a rate — so a pipeline queue pegged at 65000/65536, which is shedding
    #: and certainly saturated, printed `Speed-up: 5.26x (MET)` with no bound label.
    verdict: str = analysis.SUSTAINED

    @property
    def kind(self) -> str:
        """What the number is: a measured CAPACITY, a FLOOR under it, or NOTHING.

        Allow-lists, not deny-lists. `not saturated` admitted every verdict anyone adds
        later, and the one that already existed — UNMEASURED — is precisely the case the
        analysis raises to say "this run cannot support a number".
        """
        if self.images_per_s is None:
            return NOTHING
        if self.verdict == analysis.SATURATED:
            # Not capped: `RunAnalysis.verdict` downgrades a capped run to UNMEASURED before
            # it ever reaches here, because a bound buffer sheds rather than grows and its
            # slope stops meaning anything.
            return CAPACITY
        if self.verdict in (analysis.SUSTAINED, analysis.DRAINING):
            return FLOOR
        return NOTHING

    @property
    def is_rate(self) -> bool:
        """Whether the number can be used at all — for the summary and the exit code."""
        return self.kind != NOTHING

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "images_per_s": None if self.images_per_s is None else round(self.images_per_s, 1),
            "saturated": self.saturated,
            "verdict": self.verdict,
            "kind": self.kind,
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
                run.verdict,
            )
        if any(m.sustained is None for m in entries):
            return SystemThroughput(
                run.system,
                None,
                saturated,
                binding.module if binding else None,
                "an entry queue's offered rate was not known, so its sustained rate is not "
                "defined and the sum would be partial",
                run.verdict,
            )
        total = sum(m.sustained or 0.0 for m in entries)
        names = " + ".join(m.module for m in entries)
        return SystemThroughput(
            run.system,
            total,
            saturated,
            binding.module if binding else None,
            f"{names}: two disjoint image streams, so the sum is the system's image rate",
            run.verdict,
        )

    entry = by_name.get(shipinfer.PIPELINE_MODULE)
    if entry is None or entry.sustained is None:
        return SystemThroughput(
            run.system,
            None,
            saturated,
            binding.module if binding else None,
            f"no {shipinfer.PIPELINE_MODULE!r} module with a known offered rate in the log",
            run.verdict,
        )
    return SystemThroughput(
        run.system,
        entry.sustained,
        saturated,
        binding.module if binding else None,
        f"{shipinfer.PIPELINE_MODULE}: every camera frame enters here exactly once; the "
        f"downstream models see crops derived from those frames, not new images",
        run.verdict,
    )


@dataclass(frozen=True, slots=True)
class Ratio:
    """The speed-up, what kind of claim it supports, and why.

    One object rather than a rendered paragraph and a separately computed `summary.json`
    field. Those were two implementations of the same decision, and the JSON one divided
    whenever both sides were "a rate" — which under the table above includes floor over
    floor, the case that is purely an artefact of both systems being handed the same load.
    """

    value: float | None
    kind: str | None
    verdict: str
    reason: str

    @property
    def met(self) -> bool:
        return self.verdict == MET

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else round(self.value, 3),
            "kind": self.kind,
            "verdict": self.verdict,
            "reason": self.reason,
        }

    def headline(self, target: float) -> str:
        if self.value is None:
            return f"  Speed-up: {self.verdict} — {self.reason}"
        prefix = {EXACT: "", AT_LEAST: ">= ", AT_MOST: "<= "}[self.kind or EXACT]
        return (
            f"  Speed-up: {prefix}{self.value:.2f}x   "
            f"(target >= {target:g}x — {self.verdict})"
        )


def ratio_of(base: SystemThroughput, ours: SystemThroughput, *, target: float) -> Ratio:
    """Divide two numbers only when the pair supports a claim, and say which claim.

    The four cases are the module docstring's table. The one that used to be missing is the
    useful one: a baseline at its measured wall against a ShipInfer run that never grew a
    buffer gives a *floor* on the speed-up, and a floor above the target meets it.
    """
    if NOTHING in (base.kind, ours.kind):
        which = ", ".join(t.system for t in (base, ours) if t.kind == NOTHING)
        return Ratio(None, None, UNAVAILABLE, f"{which} produced no measurement")

    if base.kind == FLOOR and ours.kind == FLOOR:
        return Ratio(
            None,
            None,
            UNAVAILABLE,
            "neither system saturated, so both numbers are floors and their ratio is an "
            "artefact of the offered rate being equal by construction",
        )

    if not base.images_per_s:
        return Ratio(None, None, UNAVAILABLE, "the baseline sustained nothing to divide by")

    value = ours.images_per_s / base.images_per_s
    if base.kind == CAPACITY and ours.kind == CAPACITY:
        return Ratio(
            value,
            EXACT,
            MET if value >= target else NOT_MET,
            "both sides saturated, so both numbers are measured capacities",
        )
    if ours.kind == FLOOR:
        return Ratio(
            value,
            AT_LEAST,
            MET if value >= target else INCONCLUSIVE,
            f"{base.system} saturated at its capacity while {ours.system} never grew a "
            f"buffer, so the real ratio is this or better",
        )
    return Ratio(
        value,
        AT_MOST,
        NOT_MET if value < target else INCONCLUSIVE,
        f"{ours.system} saturated while {base.system} did not, so the baseline's number is "
        f"a floor and this ratio is a ceiling",
    )


def compare(base: SystemThroughput, ours: SystemThroughput, *, target: float) -> str:
    """The verdict paragraph. Renders :func:`ratio_of`; decides nothing itself."""
    labels = {
        CAPACITY: "measured capacity",
        FLOOR: "a FLOOR — the system kept up, so its capacity is at least this",
        NOTHING: "no measurement",
    }
    lines = ["", "COMPARISON", "-" * 78]
    for t in (base, ours):
        value = "unmeasured" if t.images_per_s is None else f"{t.images_per_s:>8.1f} img/s"
        lines.append(f"  {t.system:<10} {value}   ({labels[t.kind]}; run was {t.verdict})")
        lines.append(f"             {t.detail}")

    ratio = ratio_of(base, ours, target=target)
    lines += ["", ratio.headline(target), "", f"  {ratio.reason}."]

    if ratio.verdict == INCONCLUSIVE and ratio.kind == AT_LEAST:
        lines.append("  A floor can meet a target but cannot miss one — raise the load.")
    if ratio.verdict == UNAVAILABLE and base.kind == FLOOR and ours.kind == FLOOR:
        lines += [
            "",
            "  Raise the load until one of them grows a buffer:",
            "      python benchmarks/run_bench.py --sweep",
        ]
    if ratio.value is not None:
        lines += [
            "",
            "  Read this conservatively in one further respect: one baseline image passes",
            "  through one model, while one ShipInfer frame passes through detect,",
            "  conditional segmentation and one or two embedders. Equal images-per-second is",
            "  strictly more work on our side.",
        ]
    return "\n".join(lines)


def _analyse(
    system: str, log: Path, cfg: BenchConfig, offered, entries, capacity=None
) -> RunAnalysis:
    return analysis.analyse(
        analysis.read_log(log, sample_interval_s=cfg.sample_interval_s),
        system=system,
        warmup_s=cfg.warmup_s,
        offered=offered,
        capacity=cfg.buffer_capacity if capacity is None else capacity,
        entry_modules=entries,
    )


def measure_baseline(cfg: BenchConfig, out_dir: Path) -> tuple[RunAnalysis, SystemThroughput]:
    """One baseline run at one offered rate. Raises if it produced no samples."""
    print("\n=== baseline (counting-simulation, its own binary) ===", flush=True)
    result = baseline.run_baseline(cfg, out_dir / "baseline")
    if not result.ok:
        raise RuntimeError("baseline produced no samples")
    # Each of the baseline's two queues is fed by its own half of the source workers, so
    # both offered rates are known exactly from the configuration -- unlike ShipInfer's
    # crop-fed models, which have to be measured.
    run = _analyse(
        "baseline",
        result.log,
        cfg,
        offered=dict.fromkeys(BASELINE_ENTRY_MODULES, cfg.offered_per_module),
        entries=BASELINE_ENTRY_MODULES,
    )
    return run, system_throughput(run)


def measure_shipinfer(cfg: BenchConfig, out_dir: Path) -> tuple[RunAnalysis, SystemThroughput]:
    """One ShipInfer run at one offered rate, with every guard the harness has."""
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
        capacity=shipinfer.per_module_capacity(cfg),
    )
    ours = system_throughput(run)
    if ours.images_per_s is not None:
        # Cross-checked against what came out of the far end. The buffer-growth method
        # cannot tell a flat queue from a refused one, and an emitted-event count can.
        shipinfer.reconcile(result, ours.images_per_s)
    if result.per_device:
        print("\nper-device execution (the balancing evidence):")
        for model, devices in sorted(result.per_device.items()):
            spread = "  ".join(f"gpu{d}={n}" for d, n in sorted(devices.items()))
            print(f"  {model:<18} {spread}")
    return run, ours


#: One entry per system, so the sweep and the single-point path cannot drift apart.
MEASURE = {"baseline": measure_baseline, "shipinfer": measure_shipinfer}


def sweep_system(
    system: str, cfg: BenchConfig, out_dir: Path, multipliers: Sequence[float]
) -> tuple[list[RunAnalysis], SystemThroughput]:
    """Climb the offered rate until the system saturates, and return that rung.

    A single offered rate cannot settle the comparison, because both systems are handed the
    same load by construction: if neither saturates, both report that load back and the
    ratio is 1.00 regardless of how much headroom either has.

    **Stop at the first saturated rung.** That rung is the measured capacity; the higher
    ones only saturate harder, and this box is shared — every extra rung is minutes of four
    GPUs spent to learn nothing. If no rung saturates, the highest floor stands, and the
    caller reports it as a floor rather than promoting it to a capacity.
    """
    runs: list[RunAnalysis] = []
    best: SystemThroughput | None = None
    for multiplier in sorted(multipliers):
        rung = cfg.at_offer(multiplier)
        print(
            f"\n--- {system}: rung x{multiplier:g} = {rung.offered_total:g} img/s offered ---",
            flush=True,
        )
        run, throughput = MEASURE[system](rung, out_dir / f"x{multiplier:g}")
        runs.append(run)
        if throughput.kind == CAPACITY:
            print(f"    {system} saturated at x{multiplier:g}; the ladder stops here.")
            return runs, throughput
        if throughput.kind == FLOOR and (
            best is None or (throughput.images_per_s or 0) > (best.images_per_s or 0)
        ):
            best = throughput
    if best is None:
        return runs, SystemThroughput(
            system, None, False, None, "no rung produced a measurement", analysis.UNMEASURED
        )
    print(f"    {system} never saturated; its capacity is above the top rung.")
    return runs, best


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
    p.add_argument(
        "--pipeline-workers",
        type=int,
        default=None,
        help=(
            "frames in flight through the DAG (default 96). Exposed because it is the knob "
            "most likely to be mistaken for a result: a wall at the pipeline queue while "
            "every model queue stays flat is either the worker pool or the interpreter, "
            "and only sweeping this tells you which."
        ),
    )
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
    p.add_argument(
        "--sweep",
        nargs="?",
        const="0.25,0.5,1,2,4",
        default=None,
        metavar="MULTIPLIERS",
        help=(
            "climb the offered rate until a system saturates, instead of measuring one "
            "point. Comma-separated multipliers of --cameras x --fps, low to high "
            "(default 0.25,0.5,1,2,4). A system stops at its first saturated rung: that "
            "rung is its measured capacity, and higher rungs only burn GPU time to "
            "saturate harder."
        ),
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
        **(
            {"pipeline_workers": args.pipeline_workers}
            if args.pipeline_workers is not None
            else {}
        ),
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

    multipliers = None
    if args.sweep:
        multipliers = [float(m) for m in args.sweep.split(",") if m.strip()]
        print("sweep: rungs x" + ", x".join(f"{m:g}" for m in multipliers))

    runs: list[RunAnalysis] = []
    throughputs: dict[str, SystemThroughput] = {}

    for system in ("baseline", "shipinfer"):
        if system not in systems:
            continue
        try:
            if multipliers:
                rung_runs, throughput = sweep_system(system, cfg, out_dir / system, multipliers)
                runs.extend(rung_runs)
            else:
                run, throughput = MEASURE[system](cfg, out_dir)
                runs.append(run)
        except RuntimeError as exc:
            print(f"{system}: {exc}; aborting", file=sys.stderr)
            return 1
        throughputs[system] = throughput

    print()
    print(analysis.render(runs))

    summary: dict[str, Any] = {
        "config": cfg.as_dict(),
        "sweep": multipliers,
        "runs": json.loads(analysis.as_json(runs)),
        "throughput": {k: v.as_dict() for k, v in throughputs.items()},
        "target": args.target,
    }

    if "baseline" in throughputs and "shipinfer" in throughputs:
        base, ours = throughputs["baseline"], throughputs["shipinfer"]
        print(compare(base, ours, target=args.target))
        # `ratio_of`, not a second division here. The paragraph and this field were two
        # implementations of the same decision, and this one divided whenever both sides
        # were "a rate" — which includes floor over floor, the case that is an artefact of
        # both systems being handed the same load by construction.
        ratio = ratio_of(base, ours, target=args.target)
        summary["speedup"] = ratio.value if ratio.kind == EXACT else None
        summary["ratio"] = ratio.as_dict()
        summary["target_met"] = ratio.met

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
