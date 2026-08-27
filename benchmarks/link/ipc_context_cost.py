"""How much VRAM does one foreign CUDA context cost? — the `C_ctx` input of ADR-016.

ADR-015 refused CUDA IPC because opening a peer's device memory creates a CUDA context on
the peer's device inside the opening process, one per peer. ADR-016 bounds that to K
neighbours and budgets it — which needs the number, not the fear. This script measures it:

    parent  (device --owner)  allocates a slab and hands its CUDA-IPC handle over a
                              torch.multiprocessing queue (the same machinery the DataPool
                              stands on, ADR-003);
    child   (device --opener) first touches its OWN device (so its own context exists and
                              is excluded from the delta), then opens the handle and reads
                              one byte through it.

`nvidia-smi --query-gpu=memory.used` is sampled on both devices before and after the open,
from the child; the slab's own bytes are resident before the baseline and so cancel. The
delta on the OWNER's device is the cost of the child having *any* CUDA context on the
owner's device — the mapping plus the context the driver creates to hold it (the child also
synchronises the owner device inside the window, deliberately, so the context is fully
materialised). That whole-context number is the quantity ADR-016's budget needs; it is NOT
"the IPC mapping alone". The delta on the opener's device is what the mapping adds locally.

This measures K = 1 on one NVLink pair. ADR-016's budget assumes `K x C_ctx` is linear in
K; probe K = 2 and K = 3 (two/three openers on one owner) before the start-up refusal
becomes an enforced gate.

Run INSIDE the container (CLAUDE.md rule): see `benchmarks/link/run.sh`. Uses only the
dev trio (defaults owner=3, opener=4).
"""

from __future__ import annotations

import argparse
import json
import subprocess

import torch
import torch.multiprocessing as mp

SLAB_BYTES = 64 * 1024 * 1024


def used_mib(index: int) -> int:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(out)


def child(q_in, q_out, owner: int, opener: int) -> None:
    torch.cuda.set_device(opener)
    torch.ones(1, device=f"cuda:{opener}")  # own context exists before the baseline
    torch.cuda.synchronize(opener)
    before = {"owner": used_mib(owner), "opener": used_mib(opener)}
    slab = q_in.get()  # CUDA IPC handle travels here; opening happens on first use
    first = int(slab[0].item())  # one read through the mapping
    torch.cuda.synchronize(owner)
    after = {"owner": used_mib(owner), "opener": used_mib(opener)}
    q_out.put(
        {
            "before": before,
            "after": after,
            "first_byte": first,
            "delta_owner_MiB": after["owner"] - before["owner"],
            "delta_opener_MiB": after["opener"] - before["opener"],
        }
    )
    del slab
    q_in.get()  # hold the mapping until the parent has read its own numbers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--owner", type=int, default=3)
    ap.add_argument("--opener", type=int, default=4)
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    q_in, q_out = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=child, args=(q_in, q_out, args.owner, args.opener))
    p.start()
    try:
        torch.cuda.set_device(args.owner)
        slab = torch.full((SLAB_BYTES,), 7, dtype=torch.uint8, device=f"cuda:{args.owner}")
        torch.cuda.synchronize(args.owner)
        parent_before = {"owner": used_mib(args.owner), "opener": used_mib(args.opener)}
        q_in.put(slab)
        result = q_out.get(timeout=120)
        parent_after = {"owner": used_mib(args.owner), "opener": used_mib(args.opener)}
        result.update(
            {
                "slab_MiB": SLAB_BYTES // (1024 * 1024),
                "parent_view_before": parent_before,
                "parent_view_after": parent_after,
                "owner": args.owner,
                "opener": args.opener,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "note": "delta_owner_MiB is the foreign-context cost the opener pays on the owner's "
                "device (slab bytes already resident before the baseline); delta_opener_MiB "
                "is what the mapping adds on the opener's own device.",
            }
        )
        print(json.dumps(result, indent=1))
    finally:
        q_in.put(None)
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
