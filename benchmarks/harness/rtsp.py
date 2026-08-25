"""Start the local RTSP server for a benchmark run, and be sure it is gone afterwards.

WHY A SUBPROCESS AND NOT AN IMPORT
----------------------------------
`scripts/rtsp_serve.py` runs a GLib main loop, which owns the thread it is started on for the
life of the process. Importing and calling it would hand the benchmark's own thread away.
More importantly, the point of an RTSP run is that frames cross a **real socket** with a real
H.264 payload — an in-process shortcut would quietly remove the thing being measured.

WHY TWO SERVERS
---------------
`rtsp_serve.py --data` serves one directory. The benchmark's cameras are half person frames
and half ship frames — the split that decides how many crops the detector produces, and
therefore the downstream load — so a person server listens on ``rtsp_port`` and a ship server
on the next port. One server fed with person frames measured a different experiment: the ship
branch of the graph saw no ship, its queues stayed empty, and the analysis blamed the detector.

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

__all__ = ["serving", "ship_port"]

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rtsp_serve.py"


def _accepting(port: int, host: str = "127.0.0.1") -> bool:
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=0.25):
        return True
    return False


def ship_port(config: BenchConfig) -> int:
    """The port the ship-content server listens on: the one after the person server's."""
    return config.rtsp_port + 1


def _servers(config: BenchConfig) -> list[tuple[str, int, int, Path]]:
    """``(content, port, streams, directory)`` for each server this run needs.

    Mirrors ``_cameras``: cameras ``[0, half)`` are person content, ``[half, cameras)`` are
    ship content. A half with no cameras starts no server.
    """
    resolved = config.resolved()
    half = config.cameras // 2
    plan: list[tuple[str, int, int, Path]] = []
    if half > 0:
        plan.append(("person", config.rtsp_port, half, Path(resolved.person_frames)))
    if config.cameras - half > 0:
        plan.append(
            ("ship", ship_port(config), config.cameras - half, Path(resolved.ship_frames))
        )
    return plan


def _stop(process: subprocess.Popen[str]) -> None:
    # Terminate, then kill. A GLib loop that ignores SIGTERM would otherwise hold the port
    # and make the *next* run fail with an address already in use, minutes later and nowhere
    # near the cause.
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


@contextlib.contextmanager
def serving(config: BenchConfig, *, timeout_s: float = 60.0) -> Iterator[None]:
    """Run the RTSP servers for the duration of the block. A no-op unless ``source == "rtsp"``.

    Raises:
        RuntimeError: a server did not accept a connection within ``timeout_s``, or exited on
            its own. Both are refused rather than tolerated: a run whose cameras cannot connect
            produces a clean-looking zero, and this project has already published one of those.
    """
    if config.source != "rtsp":
        yield
        return
    if not SCRIPT.is_file():
        raise RuntimeError(f"no RTSP server at {SCRIPT}")

    started: list[tuple[str, int, subprocess.Popen[str]]] = []
    try:
        for content, port, streams, directory in _servers(config):
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--streams",
                    str(streams),
                    "--port",
                    str(port),
                    "--fps",
                    str(int(config.fps)),
                    "--data",
                    str(directory),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            started.append((content, port, process))
        deadline = time.monotonic() + timeout_s
        pending = list(started)
        while pending:
            for content, port, process in list(pending):
                if process.poll() is not None:
                    output = (process.stdout.read() if process.stdout else "") or ""
                    raise RuntimeError(
                        f"the {content} RTSP server (port {port}) exited with "
                        f"{process.returncode} before it was ready:\n{output[-2000:]}"
                    )
                if _accepting(port):
                    pending.remove((content, port, process))
            if pending and time.monotonic() >= deadline:
                waiting = ", ".join(f"{c} on port {p}" for c, p, _ in pending)
                raise RuntimeError(
                    f"the RTSP server(s) did not accept a connection within {timeout_s:g}s: "
                    f"{waiting}. A run whose cameras cannot connect reads as a clean zero, so "
                    f"it is refused here instead."
                )
            if pending:
                time.sleep(0.25)
        yield
    finally:
        for _content, _port, process in started:
            _stop(process)
