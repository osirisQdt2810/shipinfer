"""The inside half of the attestation. Runs in the container; writes to a bind mount.

Records the facts only a process inside a container can observe, then draws the announced
VRAM staircase so an outside recorder can catch it. Nothing here reads the operator's log —
the point is that two independent observations have to agree.
"""

from __future__ import annotations

import time as _time
import os
import pathlib
import socket
import subprocess
import sys
import time

#: MiB per step. Uneven and announced in advance: a flat allocation could be anything, whereas
#: 1536/3072/4608/6144 rising in a stated order is a shape a shared box does not produce by
#: accident. Large enough to dwarf the ~18 MiB idle floor, small enough to fit any GPU here.
STEPS_MIB = (1536, 3072, 4608, 6144)


def _local_stamp() -> str:
    """`HH:MM:SS.mmm` in local time, matching the operator's recorder byte for byte.

    Deliberately naive rather than timezone-aware: `vram_log.sh` writes
    `date '+%H:%M:%S.%3N'`, so an offset here would make the two timestamps
    un-greppable against each other, which is the entire point of printing them.
    """
    now = _time.time()
    return _time.strftime("%H:%M:%S", _time.localtime(now)) + f".{int(now % 1 * 1000):03d}"


def _read(path: str) -> str:
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError as exc:
        return f"<unreadable: {exc}>"


def main() -> int:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    say("--- observed from INSIDE ---")
    say(f"/.dockerenv present   : {pathlib.Path('/.dockerenv').exists()}")
    say(f"hostname              : {socket.gethostname()}")
    say(f"PID 1 comm            : {_read('/proc/1/comm')}")
    say(f"our PID              : {os.getpid()}")
    cgroup = _read("/proc/self/cgroup")
    say(f"cgroup                : {cgroup.splitlines()[0] if cgroup else '<empty>'}")
    say(f"container id (mountinfo): {'docker' in _read('/proc/self/mountinfo')}")
    say(f"python                : {sys.version.split()[0]} at {sys.executable}")
    say(f"CUDA_VISIBLE_DEVICES  : {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    import torch

    say()
    say("--- what CUDA reports in here ---")
    say(f"torch                 : {torch.__version__} (cuda {torch.version.cuda})")
    say(f"is_available          : {torch.cuda.is_available()}")
    count = torch.cuda.device_count()
    say(f"visible devices       : {count}")
    for i in range(count):
        free, total = torch.cuda.mem_get_info(i)
        say(
            f"  cuda:{i} {torch.cuda.get_device_name(i)}  "
            f"{total / 2**20:.0f} MiB total, {(total - free) / 2**20:.0f} MiB used"
        )

    if count < len(STEPS_MIB):
        say(f"\nREFUSING: need {len(STEPS_MIB)} devices to draw the signature, have {count}")
        return 1

    say()
    say("--- drawing the announced VRAM staircase ---")
    held = []
    for index, mib in enumerate(STEPS_MIB):
        elements = mib * 2**20 // 4  # float32
        held.append(torch.empty(elements, dtype=torch.float32, device=f"cuda:{index}"))
        torch.cuda.synchronize(index)
        stamp = _local_stamp()
        say(f"  {stamp}  cuda:{index} <- {mib} MiB allocated")

    # Real work on top of the allocation, so the GPUs are busy and not merely occupied: a
    # matmul per device, which also proves the devices are usable rather than just visible.
    for index in range(len(STEPS_MIB)):
        a = torch.randn(4096, 4096, device=f"cuda:{index}")
        for _ in range(20):
            a = a @ a.T / 4096.0
        torch.cuda.synchronize(index)
        say(f"  cuda:{index} ran 20 chained 4096x4096 matmuls, checksum {float(a.sum()):.3f}")

    hold = float(os.environ.get("HOLD_S", "12"))
    start = _local_stamp()
    say(f"  holding for {hold:.0f}s from {start} — this is the window to grep")
    time.sleep(hold)
    end = _local_stamp()
    say(f"  released at {end}")

    say()
    say("--- nvidia-smi, as seen from inside the container ---")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    for line in smi.stdout.strip().splitlines():
        say(f"  {line}")

    say()
    say(
        f"VERDICT: ran inside a container ({pathlib.Path('/.dockerenv').exists()}), "
        f"on {count} GPUs, holding {sum(STEPS_MIB)} MiB between {start} and {end}"
    )

    if out is not None:
        out.write_text(
            out.read_text() + "\n" + "\n".join(lines) + "\n"
            if out.exists()
            else "\n".join(lines) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
