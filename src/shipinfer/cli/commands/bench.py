"""``shipinfer bench`` — the evidence that the scheduler does what it claims."""

from __future__ import annotations

import time
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
from typing import Any

import numpy as np

from shipinfer.cli.common import build_settings, console, print_table
from shipinfer.core.errors import QueueFullError
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.types import Tensor
from shipinfer.server import InferenceServer

__all__ = ["bench"]


def bench(
    repository: Path,
    model: str,
    *,
    cameras: int = 50,
    fps: int = 20,
    seconds: float = 5.0,
    policy: str | None = None,
    gpus: str | None = None,
    skew: float = 1.0,
    in_flight: int = 256,
    log_level: str = "WARNING",
) -> int:
    """Drive a synthetic multi-camera load and report balance, fairness and latency.

    This exists because "the offline suite is green" is not evidence that a scheduler
    balances. The numbers that matter here are the *per-device* request counts and the
    p99 — a policy can have a perfect mean and still starve one camera.

    ``--in-flight`` bounds outstanding requests, which every real producer must do against
    a bounded pool: firing everything at once measures the load generator's lack of
    backpressure rather than the server.

    ``--skew`` reproduces the failure this system was rebuilt around: at ``skew=8`` camera 0
    submits eight times the traffic of the others, which is what a crowded platform looks
    like next to an empty corridor. With fair queueing on, the quiet cameras keep their
    share; with it off, they do not.
    """
    out = console()
    settings = build_settings(repository, gpus=gpus, policy=policy, log_level=log_level)

    with InferenceServer(settings) as server:
        handle = server.model(model)
        spec = handle.artifact.config.input_specs[0]
        shape = tuple(max(d, 1) for d in spec.shape)
        payload = Tensor.from_numpy(np.zeros((1, *shape), dtype=spec.dtype.numpy_dtype))

        weights = np.array([skew] + [1.0] * (cameras - 1), dtype=np.float64)
        weights /= weights.sum()
        target = int(cameras * fps * seconds)
        interval = 1.0 / (cameras * fps) if cameras * fps else 0.0

        rng = np.random.default_rng(0)
        pending: set = set()
        finished: list = []
        submitted = 0
        rejected = 0
        started = time.monotonic()
        next_send = started

        while submitted < target:
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.001))
                continue
            next_send += interval

            camera = int(rng.choice(cameras, p=weights))
            request = InferenceRequest(
                model_name=model,
                inputs={spec.name: payload},
                context=RequestContext(camera_id=f"cam{camera:02d}", frame_id=submitted),
            )
            try:
                pending.add(handle.infer(request))
            except QueueFullError:
                rejected += 1
            submitted += 1

            if len(pending) >= in_flight:
                done, pending = wait(pending, return_when=FIRST_COMPLETED, timeout=60.0)
                # Accumulate: a completed future dropped here is a response missing from
                # the report, which silently understates both throughput and fairness.
                finished.extend(done)

        done, not_done = wait(pending, timeout=120.0)
        finished.extend(done)
        elapsed = time.monotonic() - started
        if not_done:
            out.print(f"[yellow]{len(not_done)} request(s) never completed[/yellow]")

    _report(out, server, finished, submitted, rejected, elapsed, skew)
    return 0


def _report(
    out: Any,
    server: InferenceServer,
    done: Iterable[Any],
    submitted: int,
    rejected: int,
    elapsed: float,
    skew: float,
) -> None:
    per_device: dict[str, int] = {}
    per_camera: dict[str, int] = {}
    latencies: list[float] = []
    failures = 0

    for future in done:
        try:
            response = future.result()
        except Exception:
            failures += 1
            continue
        device = str(response.executed_on)
        per_device[device] = per_device.get(device, 0) + 1
        camera = response.context.camera_id
        per_camera[camera] = per_camera.get(camera, 0) + 1
        latencies.append(response.timings.total_us / 1000.0)

    completed = len(latencies)
    out.print(
        f"\n[bold]submitted[/bold] {submitted}  "
        f"[bold]completed[/bold] {completed}  "
        f"[bold]rejected[/bold] {rejected}  "
        f"[bold]failed[/bold] {failures}  "
        f"in {elapsed:.2f}s -> {completed / elapsed:.0f} req/s"
    )

    if latencies:
        values = np.sort(np.asarray(latencies))
        out.print(
            "[bold]latency ms[/bold]  "
            f"p50={values[len(values) // 2]:.2f}  "
            f"p95={values[int(len(values) * 0.95)]:.2f}  "
            f"p99={values[int(len(values) * 0.99)]:.2f}  "
            f"max={values[-1]:.2f}"
        )

    if per_device:
        total = sum(per_device.values())
        print_table(
            "Load balance (requests per device)",
            ["device", "requests", "share"],
            [
                [device, str(count), f"{count / total:.1%}"]
                for device, count in sorted(per_device.items())
            ],
        )

    spills = server.metrics.spills_total.total()
    if spills:
        # Spills are the locality policy admitting it had to move work off the GPU that
        # held the data. A few are healthy; a lot means the threshold is set too low.
        out.print(
            f"[bold]spillovers[/bold] {spills:.0f} ({spills / max(completed, 1):.1%} of served)"
        )

    if per_camera:
        counts = np.array(sorted(per_camera.values()))
        # The number that matters when skew > 1: how much of the service the *quietest*
        # camera got relative to the busiest. Fair queueing keeps this close to the
        # submission ratio; an evict-oldest shared buffer drives it toward zero.
        out.print(
            f"[bold]per-camera served[/bold]  min={counts[0]}  median={int(np.median(counts))}  "
            f"max={counts[-1]}  (submission skew was {skew:g}x)"
        )
