#!/usr/bin/env python3
"""ShipInfer's scheduler against the `counting-simulation` architecture, head to head.

WHY THIS SHAPE OF EXPERIMENT

`references/counting-simulation` cannot be run on this host: it needs TensorRT, OpenCV and
a pair of built `.engine` files, and the repository ships none of them. What it does ship
is its architecture, and that is the thing worth comparing — in production both systems
call the same TensorRT engine, so the engine is a constant and the serving layer is the
variable. Running both against an identical synthetic backend isolates exactly that, and is
a *cleaner* measurement than comparing two separately-built engines would have been.

The baseline below is a faithful re-implementation of `sim_pipeline_v2.py`, from reading it:

  * ONE shared bounded `queue.Queue` per model, holding bare frames with no identity
  * a global `pop_lock` held while a worker dequeues its whole batch, each `get` with a
    0.2 s timeout
  * workers statically bound to GPUs by `gpu_ids[i % len(gpu_ids)]`
  * a fixed batch size
  * a producer that blocks and retries forever on a full queue rather than shedding

Nothing here is a strawman: those are the five decisions in that file, and each is a
reasonable first thing to write. The comparison is about what they cost at 50 cameras.

FAIRNESS RULES, so the numbers mean something

  * identical per-batch cost model for both (the same formula ShipInfer's mock backend uses)
  * the same number of execution threads and the same devices
  * the same total queue capacity, offered load, skew and duration
  * latency measured the same way on both: submit -> completion, from the caller's side

Run:
    python benchmarks/compare_baseline.py --seconds 5 --cameras 50 --fps 20 --skew 8
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


# =============================================================================================
# The shared cost model
# =============================================================================================


@dataclass(frozen=True)
class CostModel:
    """What one batch costs, for both systems.

    Matches `MockBackend`: a fixed per-launch cost plus a per-row cost. That shape is what
    makes batching worth anything — if execution were purely per-row, a batch of 32 would
    cost exactly 32 singles and no scheduler could help.
    """

    fixed_ms: float = 1.2
    per_item_ms: float = 0.03

    def seconds_for(self, batch_size: int) -> float:
        return (self.fixed_ms + self.per_item_ms * batch_size) / 1000.0


# =============================================================================================
# Baseline: the counting-simulation architecture
# =============================================================================================


@dataclass
class Sample:
    camera: str
    frame: int
    submitted: float
    completed: float = 0.0
    device: str = ""

    @property
    def latency_ms(self) -> float:
        return (self.completed - self.submitted) * 1000.0


class BaselineServer:
    """One shared queue, a global pop lock, workers pinned to GPUs at start-up."""

    def __init__(
        self,
        *,
        devices: list[int],
        batch_size: int,
        capacity: int,
        cost: CostModel,
    ) -> None:
        self._queue: queue.Queue[Sample] = queue.Queue(maxsize=capacity)
        self._pop_lock = threading.Lock()
        self._stop = threading.Event()
        self._cost = cost
        self._batch_size = batch_size
        self._done: list[Sample] = []
        self._done_lock = threading.Lock()
        self._batches = 0
        # `assert det_workers == len(gpu_ids)` in the original: one worker per device.
        self._threads = [
            threading.Thread(target=self._worker, args=(f"cuda:{gpu}",), daemon=True)
            for gpu in devices
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def submit(self, sample: Sample) -> bool:
        """Blocking retry, exactly as the original's source loop does.

        It never shows a caller that it is behind: the frame is not dropped, the producer
        is stalled. At 50 cameras that means one saturated model back-pressures the thread
        feeding *every* camera, and the only visible symptom is that the run takes longer
        than it should.
        """
        while not self._stop.is_set():
            try:
                self._queue.put(sample, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self, device: str) -> None:
        while not self._stop.is_set():
            batch: list[Sample] = []
            # The global pop lock. Dequeue is serialised across every worker, and a worker
            # that finds the queue empty holds it for the full 0.2 s timeout while the
            # others wait behind it.
            with self._pop_lock:
                for _ in range(self._batch_size):
                    try:
                        batch.append(self._queue.get(timeout=0.2))
                    except queue.Empty:
                        break
            if not batch:
                continue

            time.sleep(self._cost.seconds_for(len(batch)))
            finished = time.perf_counter()
            for sample in batch:
                sample.completed = finished
                sample.device = device
            with self._done_lock:
                self._done.extend(batch)
                self._batches += 1

    def drain(self, timeout: float) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline and not self._queue.empty():
            time.sleep(0.01)
        time.sleep(0.2)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)

    @property
    def completed(self) -> list[Sample]:
        with self._done_lock:
            return list(self._done)

    @property
    def batches(self) -> int:
        return self._batches


# =============================================================================================
# The load generator, shared by both systems
# =============================================================================================


def build_schedule(cameras: int, fps: int, seconds: float, skew: float) -> list[str]:
    """The camera each request comes from, in submission order.

    `skew` is how much more traffic camera 0 sends than each of the others — the crowded
    platform next to the empty corridor, and the case the whole fairness design exists for.
    """
    weights = np.array([skew] + [1.0] * (cameras - 1), dtype=np.float64)
    weights /= weights.sum()
    total = int(cameras * fps * seconds)
    rng = np.random.default_rng(0)
    return [f"cam{int(c):02d}" for c in rng.choice(cameras, size=total, p=weights)]


def paced(total: int, seconds: float):
    """Yield indices at a steady rate, so both systems face the same arrival process."""
    interval = seconds / total if total else 0.0
    start = time.perf_counter()
    for index in range(total):
        target = start + index * interval
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        yield index


# =============================================================================================
# Reporting
# =============================================================================================


@dataclass
class Result:
    name: str
    offered: int
    completed: int
    rejected: int
    elapsed: float
    latencies_ms: list[float]
    per_device: Counter = field(default_factory=Counter)
    per_camera: Counter = field(default_factory=Counter)
    camera_latencies: dict = field(default_factory=dict)
    batches: int = 0

    @property
    def throughput(self) -> float:
        return self.completed / self.elapsed if self.elapsed else 0.0

    def percentile(self, q: float) -> float:
        if not self.latencies_ms:
            return 0.0
        values = sorted(self.latencies_ms)
        return values[min(len(values) - 1, int(len(values) * q))]

    def device_spread(self) -> tuple[float, float]:
        if not self.per_device:
            return (0.0, 0.0)
        total = sum(self.per_device.values())
        shares = [c / total for c in self.per_device.values()]
        return (min(shares), max(shares))

    def quiet_vs_loud_p99(self) -> tuple[float, float]:
        """p99 latency for the quietest camera and for the loudest one.

        This is where fair queueing shows up, and per-camera *counts* do not: when nothing
        is dropped both systems serve every request, but in a single FIFO queue a quiet
        camera's frame waits behind whatever the loud camera has already queued. Round-robin
        draining puts one frame from each active camera into the batch instead, so a quiet
        camera's latency stops depending on how busy its neighbour is.
        """
        if not self.camera_latencies:
            return (0.0, 0.0)
        by_volume = sorted(self.camera_latencies.items(), key=lambda kv: len(kv[1]))
        quiet, loud = by_volume[0][1], by_volume[-1][1]

        def p99(values: list[float]) -> float:
            ordered = sorted(values)
            return ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]

        return (p99(quiet), p99(loud))

    def camera_service(self) -> tuple[int, int, int]:
        if not self.per_camera:
            return (0, 0, 0)
        counts = sorted(self.per_camera.values())
        return (counts[0], int(statistics.median(counts)), counts[-1])


def _group_latencies(pairs: list[tuple[str, float]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for camera, latency in pairs:
        grouped.setdefault(camera, []).append(latency)
    return grouped


def render(results: list[Result], offered_rate: float) -> str:
    lines: list[str] = []
    header = f"{'':30}" + "".join(f"{r.name:>22}" for r in results)
    lines.append(header)
    lines.append("-" * len(header))

    def row(label: str, fn) -> None:
        lines.append(f"{label:30}" + "".join(f"{fn(r):>22}" for r in results))

    row("offered (req/s)", lambda _r: f"{offered_rate:.0f}")
    row("completed", lambda r: f"{r.completed}/{r.offered}")
    row("achieved (req/s)", lambda r: f"{r.throughput:.0f}")
    row("wall clock (s)", lambda r: f"{r.elapsed:.2f}")
    row("rejected (visible)", lambda r: f"{r.rejected}")
    row("batches executed", lambda r: f"{r.batches}")
    row("mean batch size", lambda r: f"{(r.completed / r.batches) if r.batches else 0:.1f}")
    lines.append("")
    row("latency p50 (ms)", lambda r: f"{r.percentile(0.50):.1f}")
    row("latency p95 (ms)", lambda r: f"{r.percentile(0.95):.1f}")
    row("latency p99 (ms)", lambda r: f"{r.percentile(0.99):.1f}")
    row("latency max (ms)", lambda r: f"{max(r.latencies_ms) if r.latencies_ms else 0:.1f}")
    lines.append("")
    row("device share min", lambda r: f"{r.device_spread()[0]:.1%}")
    row("device share max", lambda r: f"{r.device_spread()[1]:.1%}")
    lines.append("")
    row("camera served min", lambda r: f"{r.camera_service()[0]}")
    row("camera served median", lambda r: f"{r.camera_service()[1]}")
    row("camera served max", lambda r: f"{r.camera_service()[2]}")
    lines.append("")
    row("quietest cam p99 (ms)", lambda r: f"{r.quiet_vs_loud_p99()[0]:.1f}")
    row("loudest cam p99 (ms)", lambda r: f"{r.quiet_vs_loud_p99()[1]:.1f}")
    return "\n".join(lines)


# =============================================================================================
# Runners
# =============================================================================================


def run_baseline(
    schedule: list[str],
    seconds: float,
    devices: list[int],
    batch: int,
    capacity: int,
    cost: CostModel,
) -> Result:
    server = BaselineServer(devices=devices, batch_size=batch, capacity=capacity, cost=cost)
    server.start()

    started = time.perf_counter()
    for index in paced(len(schedule), seconds):
        server.submit(
            Sample(camera=schedule[index], frame=index, submitted=time.perf_counter())
        )
    server.drain(timeout=60.0)
    elapsed = time.perf_counter() - started
    server.stop()

    done = server.completed
    return Result(
        name="counting-simulation",
        offered=len(schedule),
        completed=len(done),
        rejected=0,  # it blocks rather than refusing; that is the point
        elapsed=elapsed,
        latencies_ms=[s.latency_ms for s in done],
        per_device=Counter(s.device for s in done),
        per_camera=Counter(s.camera for s in done),
        camera_latencies=_group_latencies([(s.camera, s.latency_ms) for s in done]),
        batches=server.batches,
    )


def run_shipinfer(
    schedule: list[str],
    seconds: float,
    devices: list[int],
    batch: int,
    capacity: int,
    cost: CostModel,
    *,
    dynamic_batching: bool,
    label: str,
    tmp: Path,
) -> Result:
    from concurrent.futures import FIRST_COMPLETED, wait

    from shipinfer.core.errors import QueueFullError
    from shipinfer.core.request import InferenceRequest, RequestContext
    from shipinfer.core.settings import ServerSettings
    from shipinfer.core.types import Tensor
    from shipinfer.server import InferenceServer

    root = tmp / f"repo_{label}"
    (root / "m" / "1").mkdir(parents=True, exist_ok=True)
    (root / "m" / "config.yaml").write_text(
        "platform: mock\n"
        f"max_batch_size: {batch}\n"
        "inputs: [{name: x, data_type: FP32, dims: [8]}]\n"
        "outputs: [{name: y, data_type: FP32, dims: [8]}]\n"
        # One instance per device, matching the baseline's one worker per GPU.
        f"instance_groups: [{{kind: KIND_GPU, count: 1, gpus: {devices}}}]\n"
        "dynamic_batching:\n"
        f"  enabled: {'true' if dynamic_batching else 'false'}\n"
        "  max_queue_delay_us: 3000\n"
        "parameters:\n"
        f"  latency_ms: {cost.fixed_ms}\n"
        f"  per_item_latency_ms: {cost.per_item_ms}\n"
    )

    settings = ServerSettings(
        model_repository=root,
        devices={"visible_gpus": devices},
        execution={"warmup_iterations": 0},
        # Same TOTAL capacity as the baseline's single shared queue.
        scheduler={"max_queue_size": max(1, capacity // len(devices))},
        observability={"log_level": "ERROR"},
    )

    payload = Tensor.from_numpy(np.zeros((1, 8), dtype=np.float32))
    latencies: list[float] = []
    per_device: Counter = Counter()
    per_camera: Counter = Counter()
    camera_latencies: dict[str, list[float]] = {}
    rejected = 0
    pending: set = set()

    def harvest(futures) -> None:
        for future in futures:
            if future.exception() is not None:
                return
            response = future.result()
            latencies.append(response.timings.total_us / 1000.0)
            per_device[str(response.executed_on)] += 1
            camera = response.context.camera_id
            per_camera[camera] += 1
            camera_latencies.setdefault(camera, []).append(response.timings.total_us / 1000.0)

    with InferenceServer(settings) as server:
        model = server.model("m")
        started = time.perf_counter()
        for index in paced(len(schedule), seconds):
            if len(pending) >= 512:
                done, pending = wait(pending, return_when=FIRST_COMPLETED, timeout=60)
                harvest(done)
            request = InferenceRequest(
                model_name="m",
                inputs={"x": payload},
                context=RequestContext(camera_id=schedule[index], frame_id=index),
            )
            try:
                pending.add(model.infer(request))
            except QueueFullError:
                rejected += 1
        done, _ = wait(pending, timeout=120)
        harvest(done)
        elapsed = time.perf_counter() - started
        batches = int(server.metrics.batches_total.total())

    return Result(
        name=label,
        offered=len(schedule),
        completed=len(latencies),
        rejected=rejected,
        elapsed=elapsed,
        latencies_ms=latencies,
        per_device=per_device,
        per_camera=per_camera,
        camera_latencies=camera_latencies,
        batches=batches,
    )


# =============================================================================================
# Preprocessing: the one comparison that needs no synthetic backend
# =============================================================================================


def compare_preprocessing(count: int = 8, repeats: int = 5) -> str:
    """cv2 letterbox in a Python loop against the fused device kernel.

    Not simulated. `counting-simulation` letterboxes every frame with `cv2.resize` +
    `cv2.cvtColor` on the CPU, one image at a time; ShipInfer does the resize, colour
    convert, normalise and NHWC->NCHW as a single CUDA kernel over the whole batch.

    OpenCV's thread count is pinned rather than left at its default, because leaving it
    free made this measurement swing by 4x between runs on a shared box — and an unstable
    number is not evidence. Both settings are reported, and the honest one to quote depends
    on the deployment:

      * ALL CORES flatters the CPU path relative to how it actually runs. In their
        pipeline every worker thread calls cv2 concurrently, so under load each gets
        roughly one core, not all of them.
      * ONE CORE is what a worker actually gets when the others are busy, which at
        50 cameras is all of the time.

    The median of `repeats` runs is reported, with the spread, so a single lucky or unlucky
    run cannot be mistaken for a result.
    """
    try:
        import cv2
    except ImportError:
        return "  (opencv not installed; skipped)"

    from shipinfer.runtime.ops import NormalizeParams, get_image_ops
    from shipinfer.runtime.platform import is_available

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(count)]
    dst = (640, 640)
    params = NormalizeParams(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0), swap_rb=True)

    def cv2_letterbox() -> np.ndarray:
        """Their path, transcribed from sim_pipeline_v2.py."""
        out = np.empty((len(frames), 3, *dst), dtype=np.float32)
        for i, image in enumerate(frames):
            h, w = image.shape[:2]
            r = min(dst[0] / h, dst[1] / w)
            # Transcribed verbatim, redundant int() included: a "tidied" baseline is not
            # the baseline.
            nh, nw = int(round(h * r)), int(round(w * r))  # noqa: RUF046
            resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            top, left = (dst[0] - nh) // 2, (dst[1] - nw) // 2
            canvas = cv2.copyMakeBorder(
                resized,
                top,
                dst[0] - nh - top,
                left,
                dst[1] - nw - left,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            out[i] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return out

    def timed(fn, reps: int = 5) -> tuple[float, float, float]:
        """(median, min, max) milliseconds over `repeats` samples of `reps` calls each."""
        fn()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(reps):
                fn()
            samples.append((time.perf_counter() - start) / reps * 1000.0)
        samples.sort()
        return (samples[len(samples) // 2], samples[0], samples[-1])

    lines = [f"  {count} x 1080p -> 640x640, median of {repeats} samples (min-max shown)"]

    available = cv2.getNumberOfCPUs()
    for threads, note in ((0, f"all {available} cores"), (1, "one core")):
        cv2.setNumThreads(threads if threads else available)
        median, lo, hi = timed(cv2_letterbox)
        lines.append(
            f"    cv2 loop, {note:<16}: {median:7.1f} ms  ({lo:.0f}-{hi:.0f})"
            f"  {count / median * 1000:6.0f} img/s"
        )
    cv2.setNumThreads(available)

    if not is_available():
        lines.append("    fused device kernel             :  (no accelerator on this host)")
        return "\n".join(lines)

    import torch

    ops = get_image_ops(device_index=0)
    out = torch.empty((count, 3, *dst), dtype=torch.float32, device="cuda:0")

    def fused() -> None:
        ops.letterbox_to_device(frames, out, params)
        torch.cuda.synchronize()

    median, lo, hi = timed(fused)
    lines.append(
        f"    fused kernel -> device tensor   : {median:7.1f} ms  ({lo:.0f}-{hi:.0f})"
        f"  {count / median * 1000:6.0f} img/s"
    )
    lines.append(f"    (implementation: {ops.describe()})")
    return "\n".join(lines)


# =============================================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=int, default=50)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--skew", type=float, default=8.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=512, help="total queued requests")
    parser.add_argument("--gpus", type=str, default="", help="comma-separated; default all")
    parser.add_argument(
        "--fixed-ms", type=float, default=1.2, help="per-batch cost, whatever the batch size"
    )
    parser.add_argument(
        "--per-item-ms", type=float, default=0.03, help="additional cost per row in the batch"
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from shipinfer.runtime.platform import device_count

    if args.gpus:
        devices = [int(g) for g in args.gpus.split(",")]
    else:
        devices = list(range(max(1, device_count())))

    cost = CostModel(fixed_ms=args.fixed_ms, per_item_ms=args.per_item_ms)
    schedule = build_schedule(args.cameras, args.fps, args.seconds, args.skew)
    offered_rate = len(schedule) / args.seconds

    print(f"\n{'=' * 96}")
    print("ShipInfer vs counting-simulation — same load, same cost model, same devices")
    print(f"{'=' * 96}")
    print(
        f"  {args.cameras} cameras x {args.fps} fps for {args.seconds:g}s "
        f"= {len(schedule)} requests at {offered_rate:.0f} req/s, camera 0 at {args.skew:g}x"
    )
    print(f"  devices: {devices}   batch: {args.batch}   total queue capacity: {args.capacity}")
    print(f"  cost model: {cost.fixed_ms} ms per batch + {cost.per_item_ms} ms per row\n")

    tmp = Path("/tmp/shipinfer-bench")
    tmp.mkdir(exist_ok=True)

    results = [
        run_baseline(schedule, args.seconds, devices, args.batch, args.capacity, cost),
        run_shipinfer(
            schedule,
            args.seconds,
            devices,
            args.batch,
            args.capacity,
            cost,
            dynamic_batching=False,
            label="shipinfer (no batch)",
            tmp=tmp,
        ),
        run_shipinfer(
            schedule,
            args.seconds,
            devices,
            args.batch,
            args.capacity,
            cost,
            dynamic_batching=True,
            label="shipinfer",
            tmp=tmp,
        ),
    ]

    print(render(results, offered_rate))
    print("\nPreprocessing (measured, not simulated):")
    print(compare_preprocessing())

    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "name": r.name,
                        "offered": r.offered,
                        "completed": r.completed,
                        "rejected": r.rejected,
                        "elapsed_s": round(r.elapsed, 3),
                        "throughput": round(r.throughput, 1),
                        "p50_ms": round(r.percentile(0.50), 2),
                        "p95_ms": round(r.percentile(0.95), 2),
                        "p99_ms": round(r.percentile(0.99), 2),
                        "device_share": dict(r.per_device),
                        "camera_min_median_max": r.camera_service(),
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
