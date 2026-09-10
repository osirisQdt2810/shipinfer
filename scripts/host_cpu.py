#!/usr/bin/env python3
"""Run a command and report the host CPU it used, apart from its load generators.

The RTSP arm starts its two `rtsp_serve.py` servers *inside* the bench container, because the
rootless daemon has no NAT and a second container cannot be reached. So the run pays to
generate its own load, and its events figure is depressed by work no deployment does.
`NOT-GPU-BOUND-AT-FIVE-GPUS` measured that penalty at up to ~17% and recorded the split as
"unknown, between the servers (ours to discount) and our own decode threads (ours to
optimise)". This is what makes it a number.

    python scripts/host_cpu.py --pid 41 --pid 42 -- ./bench --cameras 50

Exact rather than sampled, on both sides: `os.wait4` hands back the child's own rusage, and
`utime + stime` in `/proc/<pid>/stat` is the kernel's total across every thread of a process,
so one read before and one after bound the window with nothing to miss.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK")


def ticks_from_stat(line: str) -> float:
    """The `utime + stime` ticks in one `/proc/<pid>/stat` line.

    Parsed from the LAST `)` rather than by splitting on spaces: field 2 is the executable
    name in parentheses, and it can contain both -- `python (old)` is a legal `comm`, and so
    is anything a process writes to `/proc/self/comm`. Splitting naively shifts every field
    after it and reports another process's numbers as this one's.
    """
    fields = line[line.rindex(")") + 2 :].split()
    # `stat` field 14 is utime and 15 is stime, 1-indexed; field 3 is the first after `comm`.
    return float(fields[11]) + float(fields[12])


def cpu_seconds(pid: int) -> float | None:
    """CPU-seconds this process has used, or ``None`` if it is gone.

    ``None`` and not ``0.0``: a server that exited early used its CPU and then vanished, and
    reporting that as zero would silently discount the very cost this measures.
    """
    try:
        line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return ticks_from_stat(line) / TICKS_PER_SECOND


def thread_cpu(pid: int) -> dict[int, tuple[str, float]]:
    """``{tid: (name, cpu_seconds)}`` for one process, from `/proc/<pid>/task/`.

    The same two fields :func:`cpu_seconds` reads, one directory deeper. `comm` separately
    rather than out of `stat`: a thread name can contain a `)` and the parse below already
    keys on the LAST one, so reading the name from its own file is both simpler and exact.
    """
    out: dict[int, tuple[str, float]] = {}
    try:
        tasks = sorted(Path(f"/proc/{pid}/task").iterdir())
    except OSError:
        return out
    for task in tasks:
        try:
            name = (task / "comm").read_text(encoding="utf-8").strip()
            line = (task / "stat").read_text(encoding="utf-8")
        except OSError:
            continue  # the thread exited between the listing and the read
        out[int(task.name)] = (name, ticks_from_stat(line) / TICKS_PER_SECOND)
    return out


def thread_class(name: str) -> str:
    """The class a thread name belongs to: everything before the discriminator.

    `core/thread_name.h` and `core/thread_name.py` build `<class>-<discriminator>`, except the
    model instances, whose class carries the device (`m3.0-ship_detec`) because that is the
    thing worth grouping BY. So the split is on the first `-`, and a `m<device>.<index>` head
    keeps its device: `cam-`, `pipe-`, `m3.0-` and the two singletons.
    """
    head, _, _rest = name.partition("-")
    return head if _rest else name


class ThreadSampler:
    """Samples one process's per-thread CPU until asked to stop.

    Sampled, not exact, and the report says so: a thread that starts and dies between two
    ticks is missed entirely, so the per-class sum is a LOWER bound on the `wait4` total the
    caller already has. Printing both makes the breakdown's own trustworthiness a number
    rather than a hope -- and the interval is 200 ms because the threads this exists to
    measure live for the whole run.
    """

    def __init__(self, pid: int, interval_s: float = 0.2) -> None:
        self._pid = pid
        self._interval_s = interval_s
        self._stop = threading.Event()
        #: The HIGHEST reading per tid, because a thread's CPU only ever grows and a tid that
        #: has exited must keep the CPU it used rather than falling out of the total.
        self._seen: dict[int, tuple[str, float]] = {}
        self._thread = threading.Thread(target=self._run, name="host-cpu-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Join, THEN take one last reading -- which still works while the child is alive.

        In that order because `_sample` is a read-modify-write per tid: sampling from this
        thread while the sampler is inside its own `_sample` can let a lower reading win and
        break the "highest reading per tid" invariant this class documents. Bounded by one
        tick, so it would not have shown up in a total -- and the invariant is the contract.
        """
        self._stop.set()
        self._thread.join(timeout=self._interval_s * 5)
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def _sample(self) -> None:
        for tid, (name, cpu) in thread_cpu(self._pid).items():
            previous = self._seen.get(tid)
            if previous is None or cpu >= previous[1]:
                self._seen[tid] = (name, cpu)

    def by_class(self) -> dict[str, dict[str, float]]:
        """Per class: how much CPU, and how many threads carried it."""
        totals: dict[str, dict[str, float]] = {}
        for name, cpu in self._seen.values():
            entry = totals.setdefault(thread_class(name), {"cpu_s": 0.0, "threads": 0})
            entry["cpu_s"] += cpu
            entry["threads"] += 1
        return {
            name: {"cpu_s": round(value["cpu_s"], 2), "threads": int(value["threads"])}
            for name, value in sorted(totals.items(), key=lambda pair: -pair[1]["cpu_s"])
        }

    def top(self, count: int = 12) -> list[dict[str, object]]:
        """The heaviest individual threads, because a class can hide the answer.

        A class is `m<device>.<ordinal>`, so it holds one thread per MODEL and only the name
        says which model; the measurement is in `WHICH-THREADS-SPEND-THE-HOST-CPU`, because
        numbers in a docstring drift.

        ROWS AND NOT A MAPPING, keyed by tid: a name is NOT unique -- fifty cameras' GStreamer
        jitterbuffer threads share one `comm` -- so a dict dropped every duplicate but the
        LAST, the smallest of a collided set, and read as "few and cheap".
        """
        ranked = sorted(self._seen.items(), key=lambda item: -item[1][1])[:count]
        return [
            {"tid": tid, "name": name, "cpu_s": round(cpu, 2)} for tid, (name, cpu) in ranked
        ]

    def total(self) -> float:
        return sum(cpu for _name, cpu in self._seen.values())


def window(before: dict[int, float | None], after: dict[int, float | None]) -> dict[str, float]:
    """Per-pid CPU-seconds spent *during* the window, keyed by pid as a string.

    A delta, because a server is started before the command and has already burned CPU on its
    fixture cache and its first clients by the time the run begins. Charging its lifetime
    total to the run overstates the penalty; charging zero when it died understates it, so a
    pid that vanished is reported as the deficit it is rather than dropped.
    """
    out: dict[str, float] = {}
    for pid, start in before.items():
        end = after.get(pid)
        if start is None or end is None:
            out[str(pid)] = float("nan")
        else:
            out[str(pid)] = max(0.0, end - start)
    return out


def _relay_signals_to(child: list[int]) -> list[tuple[int, object]]:
    """Pass SIGTERM and SIGINT on, so `docker stop` still reaches the bench.

    Installed BEFORE the spawn and reading a one-element list, because a handler installed
    after it has a window in which a signal kills this wrapper and leaves the bench running
    with its GPU contexts held -- the leak GPU hygiene exists to prevent. The previous
    handlers come back, so calling `main` from a test leaves the caller's own intact.
    """

    def relay(number: int, _frame: object) -> None:
        if child:
            os.kill(child[0], number)
        else:
            raise SystemExit(128 + number)

    return [
        (number, signal.signal(number, relay)) for number in (signal.SIGTERM, signal.SIGINT)
    ]


def report(
    command_cpu_s: float,
    generators: dict[str, float],
    wall_s: float,
    threads: ThreadSampler | None = None,
) -> dict:
    """The accounting, as one flat mapping — every field a reader needs to divide.

    `cores` is here because CPU-seconds alone cannot say whether a host was saturated: 300
    CPU-seconds over 70 s is four busy cores on this 48-core box and an impossibility on a
    two-core one.
    """
    generator_cpu_s = sum(value for value in generators.values() if value == value)
    accounting = {
        "command_cpu_s": round(command_cpu_s, 2),
        "generator_cpu_s": round(generator_cpu_s, 2),
        "generators": {pid: round(value, 2) for pid, value in generators.items()},
        "wall_s": round(wall_s, 2),
        "cores": os.cpu_count() or 0,
        "command_cores_busy": round(command_cpu_s / wall_s, 2) if wall_s > 0 else 0.0,
        "generator_cores_busy": (round(generator_cpu_s / wall_s, 2) if wall_s > 0 else 0.0),
    }
    if threads is not None:
        sampled = threads.total()
        accounting["threads"] = threads.by_class()
        accounting["threads_top"] = threads.top()
        # BOTH numbers, because the sampled one is a lower bound: a thread that starts and
        # dies between two ticks is missed. `accounted_pct` is the breakdown's own
        # trustworthiness -- at 100% nothing was missed, and a low figure means the process
        # spends its CPU in threads too short-lived for this to see.
        accounting["threads_cpu_s"] = round(sampled, 2)
        accounting["accounted_pct"] = (
            round(100.0 * sampled / command_cpu_s, 1) if command_cpu_s > 0 else 0.0
        )
    return accounting


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pid",
        type=int,
        action="append",
        default=[],
        help="a load generator to account for separately (repeatable)",
    )
    parser.add_argument(
        "--threads",
        action="store_true",
        help="also break the command's CPU down by thread class (`/proc/<pid>/task/`)",
    )
    parser.add_argument(
        "--threads-interval",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help="how often to sample the threads. The instrument costs ~4 small /proc reads per "
        "thread per second, and the host it measures is the one this is arguing is tight, so "
        "a long run should ask for less: 0.5 at 70 s still catches every thread that lives "
        "for the window.",
    )
    parser.add_argument("--out", type=Path, help="also write the accounting here as JSON")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("no command given; usage: host_cpu.py [--pid N] -- <command>", file=sys.stderr)
        return 2

    before = {pid: cpu_seconds(pid) for pid in args.pid}
    started = time.monotonic()
    child: list[int] = []
    sampler: ThreadSampler | None = None
    restore = _relay_signals_to(child)
    try:
        # `posix_spawnp` rather than `subprocess`: `os.wait4` is the only stdlib call that
        # hands back a child's rusage, and `Popen` would then reap the same pid twice.
        child.append(os.posix_spawnp(command[0], command, os.environ))
        # After the spawn and before the wait: the sampler needs a pid, and every thread this
        # exists to measure outlives its first tick.
        if args.threads:
            sampler = ThreadSampler(child[0], interval_s=args.threads_interval)
            sampler.start()
        _, status, usage = os.wait4(child[0], 0)
    finally:
        if sampler is not None:
            sampler.stop()
        for number, previous in restore:
            signal.signal(number, previous)
    wall_s = time.monotonic() - started
    after = {pid: cpu_seconds(pid) for pid in args.pid}

    accounting = report(usage.ru_utime + usage.ru_stime, window(before, after), wall_s, sampler)
    print("host cpu: " + json.dumps(accounting), file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(accounting, indent=2) + "\n", encoding="utf-8")

    # The command's own outcome, not this wrapper's: a signalled bench must not look like a
    # clean exit to whatever reads the run's status.
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return os.WEXITSTATUS(status)


if __name__ == "__main__":
    raise SystemExit(main())
