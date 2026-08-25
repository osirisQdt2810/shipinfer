"""Start the local RTSP server for a benchmark run, and be sure it is gone afterwards.

WHY A SUBPROCESS AND NOT AN IMPORT
----------------------------------
`scripts/rtsp_serve.py` runs a GLib main loop, which owns the thread it is started on for the
life of the process. Importing and calling it would hand the benchmark's own thread away.
More importantly, the point of an RTSP run is that frames cross a **real socket** with a real
H.264 payload — an in-process shortcut would quietly remove the thing being measured.

WHY THE READINESS CHECK IS A SOCKET AND NOT A SLEEP
---------------------------------------------------
A fixed sleep is either too short — fifty cameras all fail their first connect, back off, and
the run measures the backoff — or too long, which is dead time in every run forever. Polling
the port answers the actual question.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from benchmarks.harness.config import BenchConfig

__all__ = ["serving"]

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rtsp_serve.py"


def _accepting(port: int, host: str = "127.0.0.1") -> bool:
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=0.25):
        return True
    return False


@contextlib.contextmanager
def serving(config: BenchConfig, *, timeout_s: float = 60.0) -> Iterator[None]:
    """Run `rtsp_serve` for the duration of the block. A no-op unless `source == "rtsp"`.

    Raises:
        RuntimeError: the server did not accept a connection within ``timeout_s``, or exited
            on its own. Both are refused rather than tolerated: a run whose cameras cannot
            connect produces a clean-looking zero, and this project has already published one
            of those.
    """
    if config.source != "rtsp":
        yield
        return
    if not SCRIPT.is_file():
        raise RuntimeError(f"no RTSP server at {SCRIPT}")

    resolved = config.resolved()
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--streams",
            str(config.cameras),
            "--port",
            str(config.rtsp_port),
            "--fps",
            str(int(config.fps)),
            "--data",
            str(resolved.person_frames),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = (process.stdout.read() if process.stdout else "") or ""
                raise RuntimeError(
                    f"the RTSP server exited with {process.returncode} before it was "
                    f"ready:\n{output[-2000:]}"
                )
            if _accepting(config.rtsp_port):
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(
                f"the RTSP server did not accept a connection on port {config.rtsp_port} "
                f"within {timeout_s:g}s. A run whose cameras cannot connect reads as a clean "
                f"zero, so it is refused here instead."
            )
        yield
    finally:
        # Terminate, then kill. A GLib loop that ignores SIGTERM would otherwise hold the
        # port and make the *next* run fail with an address already in use, minutes later and
        # nowhere near the cause.
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
