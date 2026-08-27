"""The harness driving the shards: one ShipInfer process per shard, through the real launcher.

`run_shipinfer` measures one process over every GPU. That is topology A, and it is also the
generator's ceiling: one interpreter reading fifty replay cameras delivers ~2% of 50 x 20 fps.
The multi-process shapes (`fleet`, `service` — ledger T2/T3) are measured the way they are
deployed: the parent plans the shards, starts one child per shard through
:class:`shipinfer.launch.Fleet`, and each child runs today's `run_shipinfer` on **its**
cameras and **its** GPU, writing its own occupancy log and a `summary.json`. The parent waits
for every child to finish, re-analyses each shard's log, and sums.

**This harness is not the product's control plane, and says so on its own argv.** A shard of
a *deployment* is told what to read over gRPC (arch.md section 2); a shard of a *benchmark*
is this module, run with `--config` and `--out`, and its cameras go on the same line — one
more flag on a command the harness both writes and parses. The `service` tier's per-child
keys still travel in the environment, because a child joins the tier while it starts up,
before it could answer anything.

Per-device by construction: a shard is a GPU, so "per-device execution" is the per-shard table,
and under `service` a request that left its shard shows up in the peer's counts.

The split is either the launcher's plan (LPT by offered fps — balanced by construction, which
is B doing its job) or an explicit `shard_cameras` — a crowded shard next to quiet ones, which
is the shape B cannot fix and C exists for. Both go through the same code; the split is data.

Every `shipinfer` import is inside a function, like the rest of the harness: the config,
analysis and sampler modules stay importable with no torch, and so does this one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmarks.harness.config import BenchConfig

__all__ = [
    "aggregate",
    "camera_names",
    "child_command",
    "child_config",
    "plan_for",
    "run_sharded",
    "tier_env",
]

#: Set in every child's environment, so `run_shipinfer` can tell a shard child from a run of
#: its own (a child never serves RTSP: the parent does, once).
CHILD_ENV = "SHIPINFER_BENCH_SHARD_CHILD"


def camera_names(config: BenchConfig) -> list[str]:
    """The harness's camera ids, in order — what `_cameras` names them and the plan splits."""
    return [f"cam{index:02d}" for index in range(config.cameras)]


def plan_for(config: BenchConfig) -> Any:
    """The plan the harness will launch.

    With ``config.shard_cameras`` it is the explicit split — contiguous slices of the camera
    list, one GPU per shard — and otherwise the launcher's own LPT balance, which is `fleet`
    doing its job. Either way the split is data and both go through the same code.
    """
    from shipinfer.scheduling.sharding import Shard, ShardPlan, plan_shards

    if config.topology == "single":
        raise ValueError("plan_for: `single` has no shards; call run_shipinfer directly")
    names = camera_names(config)
    gpus = list(config.gpus)
    count = config.shards or len(gpus)
    if not config.shard_cameras:
        return plan_shards(dict.fromkeys(names, config.fps), shards=count, gpus=gpus)
    shards = []
    start = 0
    for index, size in enumerate(config.shard_cameras):
        cameras = tuple(names[start : start + size])
        start += size
        shards.append(
            Shard(
                index=index, cameras=cameras, gpus=(gpus[index],), offered_fps=size * config.fps
            )
        )
    return ShardPlan(shards=tuple(shards), shards_per_gpu=dict.fromkeys(gpus[: len(shards)], 1))


def tier_env(config: BenchConfig, plan: Any) -> tuple[dict[str, str], Any]:
    """What a `service` child needs before it starts: the ring names and its own index.

    Returns:
        ``(fleet-wide env, per-shard env callable)``. Empty and ``None`` for `fleet`, which
        tells its children nothing — under the benchmark as under a deployment, the shape
        differs in what the children are told, not in how they are supervised.

    This is the one thing the control plane cannot carry (see the module docstring): a shard
    opens its peers' rings while the engine starts, which is before its first RPC. One run id
    per launch names the rings, so two benchmark runs on one box cannot open each other's
    memory and a restarted run cannot attach to a dead one's leftovers.
    """
    if config.topology != "service":
        return {}, None
    import uuid

    from shipinfer.core.settings.runner import (
        SERVICE_PEERS_ENV,
        SERVICE_RUN_ENV,
        SERVICE_SHARD_ENV,
    )

    run_id = uuid.uuid4().hex[:12]
    env = {
        SERVICE_RUN_ENV: run_id,
        SERVICE_PEERS_ENV: json.dumps(list(range(len(plan.shards)))),
    }
    return env, lambda shard: {SERVICE_SHARD_ENV: str(shard.index)}


def child_config(
    parent: BenchConfig, *, cameras: Sequence[str], gpus: Sequence[int], out_dir: Path
) -> BenchConfig:
    """The parent's configuration narrowed to one shard: its cameras, its GPU, its directory.

    ``topology`` becomes ``single`` — a child is one process over its own devices and must not
    plan shards of its own — and the offered rate follows the slice (`BenchConfig.offered_total`).
    """
    return replace(
        parent,
        topology="single",
        shards=0,
        shard_cameras=(),
        gpus=tuple(gpus),
        camera_ids=tuple(cameras),
        out_dir=out_dir,
    )


def child_command(shard: Any, config_path: Path, out_dir: Path) -> list[str]:
    """The argv of one shard child: this interpreter, this module, its slice of the run.

    The cameras are on the line. They used to ride in ``SHIPINFER_SHARD_CAMERAS``, because the
    command being launched was ``shipinfer serve``, which has no camera flag and could not
    grow one for a benchmark. This child is the harness's own module, so the harness gives it
    a flag — and the variable that existed for the other case is gone with the argv mechanism
    it belonged to (arch.md section 2).
    """
    return [
        sys.executable,
        "-m",
        "benchmarks.harness.shards",
        "--config",
        str(config_path),
        "--out",
        str(out_dir / f"shard-{shard.index}"),
        "--cameras",
        ",".join(shard.cameras),
    ]


def run_sharded(
    config: BenchConfig,
    out_dir: Path,
    *,
    startup_timeout_s: float = 1800.0,
) -> list[dict[str, Any]]:
    """Start one child per shard, wait for all of them, return their summaries in shard order.

    Raises:
        RuntimeError: a child exited non-zero (its own traceback is on stderr), or did not
            finish within start-up plus the run plus a margin.
    """
    from shipinfer.launch import Fleet

    plan = plan_for(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config.as_dict(), indent=1))
    tier, per_shard = tier_env(config, plan)
    fleet = Fleet(
        plan=plan,
        command=lambda shard: child_command(shard, config_path, out_dir),
        env={**tier, CHILD_ENV: "1"},
        shard_env=per_shard,
        drain_s=30.0,
    )
    print(f"topology: {config.topology}", flush=True)
    print(plan.describe(), flush=True)
    fleet.start()
    children = list(fleet.running)
    deadline = time.monotonic() + startup_timeout_s + config.seconds + 120.0
    try:
        while any(child.process.poll() is None for child in children):
            if time.monotonic() > deadline:
                raise RuntimeError("shard children did not finish within the deadline")
            time.sleep(1.0)
    finally:
        fleet.stop(drain_s=30.0)
    failed = [c for c in children if c.process.returncode != 0]
    if failed:
        codes = ", ".join(f"shard {c.shard.index}: exit {c.process.returncode}" for c in failed)
        raise RuntimeError(f"shard child(ren) failed — {codes}; their tracebacks are above")
    summaries = []
    for shard in plan.shards:
        path = out_dir / f"shard-{shard.index}" / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"shard {shard.index} exited 0 but wrote no {path.name}")
        summaries.append(json.loads(path.read_text()))
    return summaries


def aggregate(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum what sums, keep what does not.

    ``images_per_s`` is the sum over shards, or ``None`` if any shard has none: a fleet's
    throughput is the sum of its shards' only when every shard produced a number. The verdict
    is the *worst* shard's — one saturated shard makes the fleet's number a capacity, exactly
    as one saturated module makes a single process's.
    """
    rates = [s["throughput"]["images_per_s"] for s in summaries]
    total = None if any(r is None for r in rates) else float(sum(rates))
    verdicts = [s["throughput"]["verdict"] for s in summaries]
    saturated = any(s["throughput"]["saturated"] for s in summaries)
    order = ("SATURATED", "UNMEASURED", "DRAINING", "SUSTAINED")
    worst = next((v for v in order if v in verdicts), verdicts[0] if verdicts else "UNMEASURED")
    binding = next(
        (s["throughput"]["binding_module"] for s in summaries if s["throughput"]["saturated"]),
        None,
    )
    per_device: dict[str, dict[str, int]] = {}
    for summary in summaries:
        for model, devices in summary.get("per_device", {}).items():
            bucket = per_device.setdefault(model, {})
            for device, count in devices.items():
                bucket[device] = bucket.get(device, 0) + int(count)
    return {
        "images_per_s": total,
        "verdict": worst,
        "saturated": saturated,
        "binding_module": binding,
        "per_device": per_device,
        "shards": [
            {
                "shard": s["shard"],
                "gpus": s["gpus"],
                "cameras": len(s["cameras"]),
                "offered_total": s["offered_total"],
                "achieved": s["achieved"],
                "images_per_s": s["throughput"]["images_per_s"],
                "verdict": s["throughput"]["verdict"],
            }
            for s in summaries
        ],
    }


# -- the child ---------------------------------------------------------------------------


def _relabel(
    per_device: Mapping[str, Mapping[str, int]], gpus: Sequence[int]
) -> dict[str, dict[str, int]]:
    """A child sees its GPUs as ``cuda:0..n-1``; the table reads in physical ordinals."""
    out: dict[str, dict[str, int]] = {}
    for model, devices in per_device.items():
        row: dict[str, int] = {}
        for device, count in devices.items():
            label = device
            if device.startswith("cuda:"):
                index = int(device.split(":", 1)[1])
                if index < len(gpus):
                    label = f"cuda:{gpus[index]}"
            row[label] = row.get(label, 0) + int(count)
        out[model] = row
    return out


def _child_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="one shard of a sharded ShipInfer benchmark run"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", default="", help="comma-separated ids; set by the parent")
    args = parser.parse_args(argv)

    cameras = tuple(c for c in args.cameras.split(",") if c)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = tuple(int(g) for g in visible.split(",") if g.strip())
    if not cameras or not gpus:
        print(
            f"shard child: --cameras={args.cameras!r} CUDA_VISIBLE_DEVICES={visible!r} — "
            "this module is started by run_sharded, not by hand",
            file=sys.stderr,
        )
        return 2
    parent = BenchConfig.from_dict(json.loads(args.config.read_text()))
    config = child_config(parent, cameras=cameras, gpus=gpus, out_dir=args.out)
    shard_index = int(args.out.name.rsplit("-", 1)[-1]) if "-" in args.out.name else -1

    from benchmarks import run_bench
    from benchmarks.harness import shipinfer as system

    try:
        run, ours, result, offered, capacity = run_bench.measure_shipinfer_in_full(
            config, args.out, serve_rtsp=False, label=f"shard {shard_index} (gpu {list(gpus)})"
        )
    except Exception:
        traceback.print_exc()
        return 1
    summary = {
        "shard": shard_index,
        "gpus": list(gpus),
        "cameras": list(cameras),
        "offered_total": config.offered_total,
        "achieved": system.achieved_offer(config, result),
        "offered": offered,
        "capacity": dict(capacity),
        "log": str(result.log),
        "throughput": ours.as_dict(),
        "verdict": run.verdict,
        "per_device": _relabel(result.per_device, gpus),
        "requests_total": result.requests_total,
        "requests_rejected": result.requests_rejected,
        "frames_read": result.frames_read,
        "frames_dropped": result.frames_dropped,
        "steady_s": result.steady_s,
        "startup_s": result.startup_s,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover - the child entry point
    sys.exit(_child_main())
