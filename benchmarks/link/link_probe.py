"""Per-pair GPU link probe — the measurement behind `docs/arch.md` §3.2 and ADR-016.

`cudaDeviceCanAccessPeer` answers *capable*, not *fast*. This script times, for every GPU
pair it is given, the two payloads a DataPool ticket would reference — a 12 MB frame (4K
NV12) and a 128 KB crop batch — over three paths:

    direct   torch peer copy, device→device (P2P where the driver enables it)
    staged   device → pinned host → device, two explicit copies
    same     device→device on the source GPU alone (bandwidth ceiling, for scale)

and reports `can_device_access_peer` alongside, so a "capable but 1000x slower" pair is
visible as a row rather than a surprise. Median of N after warm-up; CUDA events for the
on-device paths, wall clock for the staged one (it has a host hop by definition).

Run INSIDE the container (CLAUDE.md rule): see `benchmarks/link/run.sh`. Output: a JSON
document on stdout plus `nvidia-smi topo -m` verbatim, so the pair classes are traceable.

Pairs default to one of each link class on the dev box (nvidia-smi topo -m, 27 Aug 2026):
NV4 0-1 and 3-4; PXB 0-3, 1-3, 2-4; SYS 3-5 and 4-6. Override with --pairs "0-1,2-4".
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import torch

FRAME = 12 * 1024 * 1024
CROP = 128 * 1024
SIZES = {"frame_12MB": FRAME, "crop_128KB": CROP}


def _fmt(ms: float, nbytes: int) -> dict:
    return {"us": round(ms * 1000, 1), "GBps": round((nbytes / (ms / 1e3)) / 1e9, 2)}


def direct(src: int, dst: int, nbytes: int, iters: int, warmup: int) -> float:
    a = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{src}")
    b = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dst}")
    times = []
    try:
        for i in range(warmup + iters):
            torch.cuda.synchronize(src)
            torch.cuda.synchronize(dst)
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(dst):
                t0.record()
                b.copy_(a, non_blocking=True)
                t1.record()
            torch.cuda.synchronize(dst)
            if i >= warmup:
                times.append(t0.elapsed_time(t1))
    finally:
        del a, b
    return statistics.median(times)


def staged(src: int, dst: int, nbytes: int, iters: int, warmup: int) -> float:
    a = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{src}")
    h = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    b = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dst}")
    times = []
    try:
        for i in range(warmup + iters):
            torch.cuda.synchronize(src)
            torch.cuda.synchronize(dst)
            t0 = time.perf_counter()
            h.copy_(a, non_blocking=True)
            torch.cuda.synchronize(src)
            b.copy_(h, non_blocking=True)
            torch.cuda.synchronize(dst)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    finally:
        del a, h, b
    return statistics.median(times)


def topo() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "topo", "-m"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - env
        return f"(nvidia-smi topo -m unavailable: {exc})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", default="0-1,3-4,0-3,1-3,2-4,3-5,4-6")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    pairs = [tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",") if p]
    report: dict = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "pairs": {},
        "same_device_baseline": {},
    }
    try:
        for label, nbytes in SIZES.items():
            ms = direct(pairs[0][0], pairs[0][0], nbytes, args.iters, args.warmup)
            report["same_device_baseline"][label] = _fmt(ms, nbytes)
        for src, dst in pairs:
            row = {"can_access_peer": torch.cuda.can_device_access_peer(src, dst)}
            for label, nbytes in SIZES.items():
                row[f"direct_{label}"] = _fmt(
                    direct(src, dst, nbytes, args.iters, args.warmup), nbytes
                )
                row[f"staged_{label}"] = _fmt(
                    staged(src, dst, nbytes, args.iters, args.warmup), nbytes
                )
            report["pairs"][f"{src}-{dst}"] = row
            print(f"# {src}-{dst}: {json.dumps(row)}", flush=True)
    finally:
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=1))
    print("\n# nvidia-smi topo -m\n" + topo())


if __name__ == "__main__":
    main()
