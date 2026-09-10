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
import tempfile
import threading
import time
from collections.abc import Container
from pathlib import Path

TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK")

#: Where a load generator inside the bench's own process tree is declared, one pid per line.
#: The wrapper creates the path and exports it; `benchmarks/harness/rtsp.py` writes each RTSP
#: server's `Popen.pid` as it starts one.
GENERATOR_PIDFILE_ENV = "SHIPINFER_HOST_CPU_PIDFILE"


def declare_generator(pid: int | None = None) -> bool:
    """Say that a process generates load rather than serving it. False if nobody is asking.

    Called by the SPAWNER -- `benchmarks/harness/rtsp.py` passes each `Popen.pid` -- because
    those servers run inside the bench container (the rootless daemon has no NAT) and are
    therefore children of the bench, already inside `wait4`'s rusage. The no-argument form
    declares the caller, for a generator that would rather say so itself.
    """
    path = os.environ.get(GENERATOR_PIDFILE_ENV)
    if not path:
        return False
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid() if pid is None else pid}\n")
    return True


def ticks_from_stat(line: str, *, with_children: bool = False) -> float:
    """The `utime + stime` ticks in one `/proc/<pid>/stat` line, optionally with its children.

    Parsed from the LAST `)` rather than by splitting on spaces: field 2 is the executable
    name in parentheses, and it can contain both -- `python (old)` is a legal `comm`, and so
    is anything a process writes to `/proc/self/comm`. Splitting naively shifts every field
    after it and reports another process's numbers as this one's.
    """
    fields = line[line.rindex(")") + 2 :].split()
    # `stat` field 14 is utime and 15 is stime, 1-indexed; field 3 is the first after `comm`.
    total = float(fields[11]) + float(fields[12])
    if with_children:
        # 16 and 17, cutime/cstime: what this process's REAPED children used. `wait4` already
        # charges the command that way, so a generator measured without it is measured on a
        # different rule from the total it is subtracted from.
        total += float(fields[13]) + float(fields[14])
    return total


def cpu_seconds(pid: int) -> float | None:
    """CPU-seconds this process and its reaped children have used, or ``None`` if it is gone.

    ``None`` and not ``0.0``: a server that exited early used its CPU and then vanished, and
    reporting that as zero would silently discount the very cost this measures.

    CHILDREN INCLUDED, because every caller is measuring a load *generator* and a generator
    that forks is still generator cost: `scripts/rtsp_serve.py` shells out to `ffmpeg` for the
    whole JPEG set whenever the `.h264` fixture is cold, which on a fresh tree is the largest
    single piece of it -- and it was landing in the bench's column.
    """
    try:
        line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return ticks_from_stat(line, with_children=True) / TICKS_PER_SECOND


def thread_cpu(pid: int) -> dict[int, tuple[str, float]]:
    """``{tid: (name, cpu_seconds)}`` for one process, from `/proc/<pid>/task/`.

    The process's OWN two fields, without `cutime`/`cstime`: a per-thread `stat` reports the
    THREAD GROUP's figures for those, so the leader's row would carry every reaped child (and
    non-leaders zero) -- the CPU `cpu_seconds` subtracts, added back on one row.

    One directory deeper than :func:`cpu_seconds`, and `comm` separately
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


def process_tree(pid: int, skip: Container[int] = frozenset()) -> list[int]:
    """``pid`` and every descendant, from `/proc/<pid>/task/<tid>/children`.

    The default topology spawns one shard PROCESS per GPU (`--topology fleet`), so a sampler
    reading only the parent's task directory attributes a few percent of a run whose
    `accounted_pct` then reads as a broken instrument rather than as a missing tree walk.
    Depth-first over `children`, which lists a task's *immediate* children only.

    ``skip`` is neither returned nor descended into: a load generator the bench spawned is
    inside this tree and is not the bench.
    """
    seen: list[int] = []
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in seen or current in skip:
            continue
        seen.append(current)
        try:
            tasks = sorted(Path(f"/proc/{current}/task").iterdir())
        except OSError:
            continue  # the process exited between the walk and the read
        for task in tasks:
            try:
                children = (task / "children").read_text(encoding="utf-8")
            except OSError:
                continue
            pending.extend(int(child) for child in children.split())
    return seen


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
    measure live for the whole run. Walks the process TREE, so a sharded run's children
    count; that multiplies the /proc reads per tick, which is what the interval is for.
    """

    def __init__(
        self, pid: int, interval_s: float = 0.2, generators: Path | None = None
    ) -> None:
        self._pid = pid
        self._interval_s = interval_s
        self._generators_path = generators
        self._stop = threading.Event()
        #: tid -> (owning pid, name, HIGHEST cpu seen). Highest, because a thread's CPU only
        #: ever grows and a tid that has exited must keep the CPU it used. The pid is kept so
        #: a process declared a generator LATER can have its threads forgotten.
        self._seen: dict[int, tuple[int, str, float]] = {}
        #: Threads whose tid was reused, as (pid, tid, name, cpu). A LOWER reading than the
        #: one stored is a new thread on a recycled id, not a counter going backwards.
        self._retired: list[tuple[int, int, str, float]] = []
        #: Lifetime CPU of each generator the bench spawned, sampled while it is still alive
        #: -- it dies with the bench, so there is no reading it afterwards.
        self._generator_cpu: dict[int, float] = {}
        #: Every pid ever seen UNDER a declared generator. Accumulated because an exited child
        #: must not come back and `cutime` has charged it already; never pruned, which is safe
        #: only while pid reuse cannot happen inside one run (`pid_max` is 4194304 here).
        self._generator_tree: set[int] = set()
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

    def _generator_pids(self) -> frozenset[int]:
        """The pids the bench has declared to be load generators, so far.

        Re-read every tick rather than once: the servers are started after the spawn, and a
        run that adds one later is the same case as a run that starts with two.
        """
        if self._generators_path is None:
            return frozenset()
        try:
            text = self._generators_path.read_text(encoding="utf-8")
        except OSError:
            return frozenset()
        # A partial last line is a write in flight, not a corrupt file.
        return frozenset(int(word) for word in text.split() if word.isdigit())

    def _sample(self) -> None:
        generators = self._generator_pids()
        # The generator's OWN DESCENDANTS as well as the generator: `cutime` charges a reaped
        # `ffmpeg` to the server that waited for it, so a subtree left in the breakdown is
        # counted twice over -- once in `threads_cpu_s` and once out of `bench_cpu_s`, which
        # is how `accounted_pct` passes 100%.
        for generator in generators:
            self._generator_tree.update(process_tree(generator))
        excluded = frozenset(generators | self._generator_tree)
        # EVERY tick, not only when the set grows. A generator is declared once its pid
        # exists, so a tick can precede the declaration -- and `_generator_pids` answers with
        # an empty set on `OSError`, so one unreadable tick can re-add threads a growth-only
        # guard would then never purge again. It is a comprehension over a few hundred tids.
        self._forget(excluded)
        for pid in generators:
            cpu = cpu_seconds(pid)
            if cpu is not None:
                self._generator_cpu[pid] = max(self._generator_cpu.get(pid, 0.0), cpu)
        # Every process in the tree, because a tid is unique host-wide and a shard child's
        # model threads are the ones this instrument exists to name -- but not a generator's,
        # which is inside the tree and is not the bench.
        for pid in process_tree(self._pid, skip=excluded):
            for tid, (name, cpu) in thread_cpu(pid).items():
                previous = self._seen.get(tid)
                if previous is None or cpu >= previous[2]:
                    self._seen[tid] = (pid, name, cpu)
                else:
                    self._retired.append((previous[0], tid, previous[1], previous[2]))
                    self._seen[tid] = (pid, name, cpu)

    def _forget(self, pids: frozenset[int]) -> None:
        """Drop every thread owned by one of ``pids``, live or retired.

        Retroactive, because a generator is declared only once its pid exists and a tick can
        precede that. Reaches only what was seen ALIVE: a grandchild caught by a
        pre-declaration tick and gone by the next stays -- the boundary, and not expected,
        since `Popen` and the declaration are two statements apart in `harness/rtsp.py`.
        A reused pid would now also stop the walk descending, taking a whole subtree.
        """
        self._seen = {tid: row for tid, row in self._seen.items() if row[0] not in pids}
        self._retired = [row for row in self._retired if row[0] not in pids]

    def generator_cpu(self) -> dict[int, float]:
        """Per-pid lifetime CPU of the generators the bench spawned."""
        return dict(self._generator_cpu)

    def _rows(self) -> list[tuple[int, str, float]]:
        """Every thread seen, live and retired, as ``(tid, name, cpu_s)``."""
        live = [(tid, name, cpu) for tid, (_pid, name, cpu) in self._seen.items()]
        return live + [(tid, name, cpu) for _pid, tid, name, cpu in self._retired]

    def by_class(self) -> dict[str, dict[str, float]]:
        """Per class: how much CPU, and how many threads carried it."""
        totals: dict[str, dict[str, float]] = {}
        for _tid, name, cpu in self._rows():
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
        ranked = sorted(self._rows(), key=lambda row: -row[2])[:count]
        return [{"tid": tid, "name": name, "cpu_s": round(cpu, 2)} for tid, name, cpu in ranked]

    def total(self) -> float:
        return sum(cpu for _tid, _name, cpu in self._rows())


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


def _generator_pidfile(out: Path | None) -> Path:
    """An empty drop-box for generator pids, beside the accounting or in the temp dir."""
    directory = out.parent if out is not None else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"host-cpu-generators-{os.getpid()}.pids"
    path.write_text("", encoding="utf-8")
    return path


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

    ABSENT AND EMPTY MEAN DIFFERENT THINGS. With no sampler there is no `spawned_generators`
    key at all -- "not measured" -- while an empty one is "measured, and there were none". A
    zero would be a claim an unsampled run cannot make, and `bench_cpu_s` with it.
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
        spawned = threads.generator_cpu()
        # SUBTRACTED, unlike `--pid`. An external generator is a separate process tree and is
        # absent from `wait4`'s rusage; one the bench SPAWNED is a reaped child, so it is
        # already inside `command_cpu_s`. Naming it without discounting it would leave the
        # RTSP arm's ~17% generator penalty inside a figure read as the bench's own.
        spawned_cpu_s = sum(spawned.values())
        accounting["spawned_generators"] = {
            str(pid): round(value, 2) for pid, value in sorted(spawned.items())
        }
        accounting["spawned_generator_cpu_s"] = round(spawned_cpu_s, 2)
        bench_cpu_s = max(0.0, command_cpu_s - spawned_cpu_s)
        accounting["bench_cpu_s"] = round(bench_cpu_s, 2)
        sampled = threads.total()
        accounting["threads"] = threads.by_class()
        accounting["threads_top"] = threads.top()
        # BOTH numbers, because the sampled one is a lower bound: a thread that starts and
        # dies between two ticks is missed. `accounted_pct` is the breakdown's own
        # trustworthiness -- at 100% nothing was missed, and a low figure means the process
        # spends its CPU in threads too short-lived for this to see.
        accounting["threads_cpu_s"] = round(sampled, 2)
        accounting["accounted_pct"] = (
            round(100.0 * sampled / bench_cpu_s, 1) if bench_cpu_s > 0 else 0.0
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
        "for the window. ZERO turns the sampling off, which also gives up the spawned-"
        "generator discount -- both need a reading taken while the run is alive.",
    )
    parser.add_argument("--out", type=Path, help="also write the accounting here as JSON")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("no command given; usage: host_cpu.py [--pid N] -- <command>", file=sys.stderr)
        return 2

    before = {pid: cpu_seconds(pid) for pid in args.pid}
    sampling = args.threads and args.threads_interval > 0
    # Created before the spawn and exported to the child, so a bench that starts a load
    # generator has somewhere to say so. Truncated rather than appended: a stale file from a
    # previous run would discount pids this run never spawned.
    pidfile = _generator_pidfile(args.out) if sampling else None
    environ = dict(os.environ)
    if pidfile is not None:
        environ[GENERATOR_PIDFILE_ENV] = str(pidfile)
    started = time.monotonic()
    child: list[int] = []
    sampler: ThreadSampler | None = None
    restore = _relay_signals_to(child)
    try:
        # `posix_spawnp` rather than `subprocess`: `os.wait4` is the only stdlib call that
        # hands back a child's rusage, and `Popen` would then reap the same pid twice.
        child.append(os.posix_spawnp(command[0], command, environ))
        # After the spawn and before the wait: the sampler needs a pid, and every thread this
        # exists to measure outlives its first tick.
        if sampling:
            sampler = ThreadSampler(
                child[0], interval_s=args.threads_interval, generators=pidfile
            )
            sampler.start()
        _, status, usage = os.wait4(child[0], 0)
    finally:
        if sampler is not None:
            sampler.stop()
        for number, previous in restore:
            signal.signal(number, previous)
        # After `stop`, which took the last reading: the totals live in the sampler, not here.
        if pidfile is not None:
            pidfile.unlink(missing_ok=True)
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
