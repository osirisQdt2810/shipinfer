#!/usr/bin/env python3
"""The device indices CUDA can actually open, as `nvidia-smi` numbers them.

`_gpus.sh` exists because a card the driver enumerates but the CUDA runtime cannot open takes
the whole GPU tier down -- `torch.cuda.__init__` queues `_check_capability`, which walks EVERY
visible device, so a run that asked for GPUs 0-3 dies on a card it never wanted:

    RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED ... device=7, num_gpus=7

That file has documented the incident since 1 Sep and still defaults to `all`, so the same card
takes the same tier down for anyone who does not know the incantation. This is the detection it
was missing: print the healthy indices, or nothing at all when it cannot tell.

Matched by PCI BUS ID rather than by ordinal, because the two enumerations are exactly what
disagree -- CUDA's list is dense over the cards it can open, `nvidia-smi`'s is over all of them.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys


def _key(bus: str) -> str:
    """A bus id the two enumerations agree on.

    CUDA writes a four-digit domain (`0000:D2:00.0`) and the driver an eight-digit one
    (`00000000:D2:00.0`), so the raw strings never match. The bus and device fields do, and a
    box with two PCI domains would still differ in them.
    """
    return ":".join(bus.upper().strip().split(":")[-2:])


def _cuda_bus_ids() -> list[str] | None:
    """Bus ids CUDA can open, RAW, or None when there is no runtime to ask.

    Raw on purpose: normalising is `main`'s job, so both enumerations go through the same
    `_key` and neither side can be keyed while the other is not.
    """
    for soname in ("libcudart.so.12", "libcudart.so.11.0", "libcudart.so"):
        try:
            lib = ctypes.CDLL(soname)
        except OSError:
            continue
        lib.cudaGetDeviceCount.restype = ctypes.c_int
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaDeviceGetPCIBusId.restype = ctypes.c_int
        lib.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        count = ctypes.c_int()
        if lib.cudaGetDeviceCount(ctypes.byref(count)) != 0:
            return None
        out, buf = [], ctypes.create_string_buffer(64)
        for device in range(count.value):
            if lib.cudaDeviceGetPCIBusId(buf, 64, device) != 0:
                return None
            out.append(buf.value.decode())
        return out
    return None


def _smi_bus_ids() -> list[tuple[str, str]] | None:
    """(index, bus id) as the driver numbers them, or None when `nvidia-smi` is absent."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    rows = []
    for line in done.stdout.splitlines():
        index, _, bus = line.partition(",")
        if index.strip():
            # The raw bus id is kept for the message; only the key is normalised.
            rows.append((index.strip(), bus.strip()))
    return rows or None


def main() -> int:
    cuda, smi = _cuda_bus_ids(), _smi_bus_ids()
    if cuda is None or smi is None:
        # SILENT, not a guess: with nothing to compare, the caller keeps its own default.
        return 1
    healthy = {_key(bus) for bus in cuda}
    usable = [index for index, bus in smi if _key(bus) in healthy]
    if not usable or len(usable) == len(smi):
        # Nothing to route around -- or nothing left, which is a driver problem this script
        # must not paper over by handing the caller an empty list.
        return 1
    missing = [f"{index} ({bus})" for index, bus in smi if _key(bus) not in healthy]
    print(",".join(usable))
    print("  ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
