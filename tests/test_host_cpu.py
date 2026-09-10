"""``scripts/host_cpu.py`` separates a run's own CPU from its load generators'.

Why this file exists: `NOT-GPU-BOUND-AT-FIVE-GPUS` ruled the GPUs out as the limit and left
the host cost recorded as "unknown, split between the RTSP servers (ours to discount) and our
own decode threads (ours to optimise)". The split decides how much of the RTSP arm's ~17%
penalty is a benchmark artefact, so the arithmetic that produces it is the thing worth
pinning -- the same argument `benchmarks/tests/test_occupancy_window.py` makes about the
occupancy percentage.

Offline by design (ADR-001): `/proc` and `wait4` are the kernel's, not a driver's. The
subprocesses here spin on the CPU for fractions of a second and touch no device.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from tests.support.subprocess_env import checkout_env

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "host_cpu.py"

#: Long enough that a tick-resolution counter (usually 100 Hz) reads several ticks, short
#: enough that the whole file stays under a second of CPU.
SPIN_S = 0.35

_SPIN = (
    "import time\n"
    "end = time.monotonic() + {seconds}\n"
    "while time.monotonic() < end:\n"
    "    pass\n"
)


@pytest.fixture(scope="module")
def host_cpu() -> ModuleType:
    """Import by path, and registered first -- `scripts/` is not a package on `sys.path`."""
    spec = importlib.util.spec_from_file_location("scripts.host_cpu", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.host_cpu"] = module
    spec.loader.exec_module(module)
    return module


def _spinner(seconds: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _SPIN.format(seconds=seconds)], env=checkout_env()
    )


class TestReadingOneProcessesCpu:
    def test_a_comm_with_spaces_and_parentheses_does_not_shift_the_fields(
        self, host_cpu: ModuleType
    ) -> None:
        """`comm` is whatever the process wrote to `/proc/self/comm`, and a `split()` on the
        whole line reports another field's value as utime. 13 and 7 are fields 14 and 15."""
        fields = [str(n) for n in range(3, 53)]
        fields[14 - 3] = "13"
        fields[15 - 3] = "7"
        line = "4242 (py (weird) name) S " + " ".join(fields[1:])

        assert host_cpu.ticks_from_stat(line) == 20.0

    def test_this_process_has_used_some_cpu(self, host_cpu: ModuleType) -> None:
        assert (host_cpu.cpu_seconds(os.getpid()) or 0.0) > 0.0

    def test_a_process_that_is_gone_reads_none_rather_than_zero(
        self, host_cpu: ModuleType
    ) -> None:
        """Zero would say "this generator was free", which is the opposite of the truth."""
        done = _spinner(0.0)
        done.wait()

        assert host_cpu.cpu_seconds(done.pid) is None


class TestTheGeneratorsCpuIsAWindowAndNotALifetime:
    """The property the whole measurement rests on.

    A server is started before the run -- it builds its fixture cache and serves the first
    connects -- so charging its lifetime total to the run overstates the penalty this exists
    to quantify, in the direction that flatters us.
    """

    def test_only_the_cpu_inside_the_window_is_charged(self, host_cpu: ModuleType) -> None:
        spinner = _spinner(SPIN_S * 3)
        try:
            time.sleep(SPIN_S)
            before = {spinner.pid: host_cpu.cpu_seconds(spinner.pid)}
            time.sleep(SPIN_S)
            after = {spinner.pid: host_cpu.cpu_seconds(spinner.pid)}
        finally:
            spinner.kill()
            spinner.wait()

        charged = host_cpu.window(before, after)[str(spinner.pid)]
        lifetime = after[spinner.pid] or 0.0

        assert 0.0 < charged < lifetime, (
            f"charged {charged:.3f}s of a {lifetime:.3f}s lifetime; a delta must be strictly "
            "less than the total when the process was already burning CPU beforehand"
        )
        assert charged == pytest.approx(SPIN_S, abs=SPIN_S)

    def test_a_generator_that_died_mid_run_is_not_counted_as_free(
        self, host_cpu: ModuleType
    ) -> None:
        """`nan` and not `0.0`, so `generator_cpu_s` under-reports visibly rather than
        silently: a server killed by the OOM killer took its CPU with it."""
        charged = host_cpu.window({7: 1.0}, {7: None})

        assert math.isnan(charged["7"])

    def test_a_counter_that_appears_to_go_backwards_reads_zero(
        self, host_cpu: ModuleType
    ) -> None:
        """Not expected -- these only grow -- but a negative would send a reader looking for
        a kernel problem instead of at the pid that was reused."""
        assert host_cpu.window({7: 9.0}, {7: 1.0}) == {"7": 0.0}


class TestTheAccountingIsDivisible:
    def test_cores_busy_is_cpu_seconds_over_the_wall_window(self, host_cpu: ModuleType) -> None:
        """CPU-seconds alone cannot say whether a host was saturated: 280 over 70 s is four
        busy cores here and impossible on a two-core box, which is why `cores` is reported."""
        out = host_cpu.report(280.0, {"1": 70.0, "2": 70.0}, 70.0)

        assert out["command_cores_busy"] == 4.0
        assert out["generator_cpu_s"] == 140.0
        assert out["generator_cores_busy"] == 2.0
        assert out["cores"] == os.cpu_count()

    def test_a_dead_generator_does_not_poison_the_sum(self, host_cpu: ModuleType) -> None:
        """`nan + 1.0` is `nan`, and one dead server would erase the whole accounting."""
        out = host_cpu.report(1.0, {"1": 2.0, "2": float("nan")}, 1.0)

        assert out["generator_cpu_s"] == 2.0
        assert math.isnan(out["generators"]["2"])

    def test_a_zero_window_reports_no_rate_rather_than_dividing(
        self, host_cpu: ModuleType
    ) -> None:
        out = host_cpu.report(1.0, {}, 0.0)

        assert out["command_cores_busy"] == 0.0
        assert out["generator_cores_busy"] == 0.0


#: Named threads that burn a per-thread CPU budget, with no project imports: a test for a
#: `/proc` reader must not depend on the package it happens to be shipped with.
#: `time.thread_time()` and not `process_time()` -- the latter counts every thread, so five
#: threads sharing one budget all exit at once and the sampler sees nothing.
_THREADS = """
import ctypes, ctypes.util, threading, time

libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
libc.pthread_self.restype = ctypes.c_void_p
libc.pthread_self.argtypes = []
libc.pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


def burn(name, seconds):
    libc.pthread_setname_np(libc.pthread_self(), name.encode())
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        pass


plan = {plan}
threads = [threading.Thread(target=burn, args=pair) for pair in plan]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
"""


class TestWhichThreadsSpentIt:
    """`NOT-GPU-BOUND-AT-FIVE-GPUS` ruled the GPUs out and asked which threads spend the host
    CPU. Both planes name their threads now, so `/proc/<pid>/task/` answers it -- and this is
    the arithmetic that turns those names into a breakdown.
    """

    #: One long, one medium, two short of a class, and a singleton. Long enough that a 20 ms
    #: sampler sees several ticks of each.
    _PLAN = (("pipe-0", 0.6), ("cam-a", 0.3), ("cam-b", 0.3), ("m0.0-detect", 0.15))

    def _run(self, host_cpu: ModuleType, tmp_path: Path) -> dict:
        out = tmp_path / "cpu.json"
        code = host_cpu.main(
            [
                "--threads",
                # Explicit, so the class docstring's "a 20 ms sampler" is true rather than
                # aspirational: the default is 200 ms and these threads live under a second.
                "--threads-interval",
                "0.02",
                "--out",
                str(out),
                "--",
                sys.executable,
                "-c",
                _THREADS.format(plan=repr(self._PLAN)),
            ]
        )
        assert code == 0
        return json.loads(out.read_text(encoding="utf-8"))

    def test_the_breakdown_names_the_classes_and_ranks_them(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        """The whole point: `pipe` burned twice what each `cam` did, and the two `cam` threads
        are one row of two. Ranked by CPU, because the question is where the time goes."""
        accounting = self._run(host_cpu, tmp_path)
        threads = accounting["threads"]

        assert {"pipe", "cam", "m0.0"} <= set(threads), threads
        assert threads["cam"]["threads"] == 2, threads
        assert threads["pipe"]["cpu_s"] > threads["m0.0"]["cpu_s"], threads
        assert list(threads) == sorted(threads, key=lambda k: -threads[k]["cpu_s"]), threads

    def test_the_sampled_sum_is_a_lower_bound_and_says_how_much_it_saw(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        """Sampling can only miss CPU, never invent it, so the breakdown's own
        trustworthiness is a number in every run rather than a caveat in a docstring."""
        accounting = self._run(host_cpu, tmp_path)

        assert accounting["threads_cpu_s"] <= accounting["command_cpu_s"] + 0.05
        assert 0.0 < accounting["accounted_pct"] <= 100.5, accounting
        assert accounting["threads_cpu_s"] == pytest.approx(
            sum(entry["cpu_s"] for entry in accounting["threads"].values()), abs=0.05
        )

    def test_it_is_off_unless_asked_for(self, host_cpu: ModuleType, tmp_path: Path) -> None:
        """Every existing reader of this JSON predates the breakdown, so the default output
        must not change shape."""
        out = tmp_path / "plain.json"
        host_cpu.main(
            ["--out", str(out), "--", sys.executable, "-c", _SPIN.format(seconds=SPIN_S)]
        )

        accounting = json.loads(out.read_text(encoding="utf-8"))
        assert "threads" not in accounting
        assert "accounted_pct" not in accounting

    def test_a_class_is_the_name_up_to_its_discriminator(self, host_cpu: ModuleType) -> None:
        """The grouping both planes' schemes were designed for -- and `m3.0` keeps its DEVICE,
        because which device's instances are hot is the question behind the question."""
        assert host_cpu.thread_class("cam-camera-0049") == "cam"
        assert host_cpu.thread_class("pipe-127") == "pipe"
        assert host_cpu.thread_class("m3.0-ship_detec") == "m3.0"
        assert host_cpu.thread_class("sweeper") == "sweeper"
        assert host_cpu.thread_class("bench") == "bench"

    def test_the_heaviest_individual_threads_are_named(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        """A class can hide the answer: at the design load each `m<device>.<ordinal>` class
        holds one thread per MODEL, so only the individual names say which model costs most."""
        accounting = self._run(host_cpu, tmp_path)

        top = accounting["threads_top"]
        names = [row["name"] for row in top]
        assert "pipe-0" in names, top
        assert [row["cpu_s"] for row in top] == sorted(
            (row["cpu_s"] for row in top), reverse=True
        ), top
        assert max(row["cpu_s"] for row in top) <= accounting["command_cpu_s"] + 0.05
        assert len({row["tid"] for row in top}) == len(top), top

    def test_two_threads_with_one_name_are_two_rows(self, host_cpu: ModuleType) -> None:
        """A name is not unique -- GStreamer gives all fifty cameras' jitterbuffer threads the
        same `comm` -- and a mapping keyed by name dropped every duplicate but the last, i.e.
        the SMALLEST of a collided set. The class row was right all along; only this one lied,
        and it lied in the direction of "those threads are few and cheap".
        """
        sampler = host_cpu.ThreadSampler(os.getpid(), interval_s=0.01)
        # `(owning pid, name, cpu)`: the pid is kept so a process declared a load generator
        # after the first tick can have its threads forgotten.
        sampler._seen = {
            11: (os.getpid(), "rtpjitterbuffer", 0.9),
            12: (os.getpid(), "rtpjitterbuffer", 0.5),
            13: (os.getpid(), "rtpjitterbuffer", 0.1),
        }

        top = sampler.top()

        assert [row["cpu_s"] for row in top] == [0.9, 0.5, 0.1], top
        assert {row["tid"] for row in top} == {11, 12, 13}, top
        assert sampler.by_class()["rtpjitterbuffer"] == {"cpu_s": 1.5, "threads": 3}

    def test_a_thread_that_exits_keeps_the_cpu_it_used(self, host_cpu: ModuleType) -> None:
        """A tid that vanishes between ticks must not fall out of the total: the highest
        reading per tid is kept, because a thread's CPU only ever grows."""
        sampler = host_cpu.ThreadSampler(os.getpid(), interval_s=0.01)
        sampler._seen = {7: (os.getpid(), "cam-a", 1.5), 8: (os.getpid(), "cam-b", 0.5)}

        assert sampler.total() == 2.0
        assert sampler.by_class()["cam"] == {"cpu_s": 2.0, "threads": 2}


class TestTheWrapperIsTransparentToTheCommand:
    """It sits between the harness and the bench, so its own behaviour must be invisible."""

    def test_it_measures_the_command_and_passes_its_exit_code_through(
        self, host_cpu: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        out = tmp_path / "nested" / "cpu.json"

        code = host_cpu.main(
            ["--out", str(out), "--", sys.executable, "-c", _SPIN.format(seconds=SPIN_S)]
        )

        assert code == 0
        accounting = json.loads(out.read_text(encoding="utf-8"))
        assert accounting["command_cpu_s"] == pytest.approx(SPIN_S, abs=SPIN_S)
        assert accounting["command_cores_busy"] > 0.5, accounting
        assert "host cpu:" in capsys.readouterr().err

    def test_a_failing_command_fails_the_wrapper(self, host_cpu: ModuleType) -> None:
        assert host_cpu.main(["--", sys.executable, "-c", "raise SystemExit(3)"]) == 3

    def test_a_signalled_command_is_not_reported_as_a_clean_exit(
        self, host_cpu: ModuleType
    ) -> None:
        """`docker stop` on a bench must not leave a run's status saying it finished."""
        code = host_cpu.main(
            [
                "--",
                sys.executable,
                "-c",
                "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
            ]
        )

        assert code == 128 + 15

    def test_no_command_is_a_usage_error(self, host_cpu: ModuleType) -> None:
        """The exit code the harness's own tests use for a malformed command line."""
        assert host_cpu.main([]) == 2
        assert host_cpu.main(["--pid", "1"]) == 2


class TestAShardedRunsChildrenCountToo:
    """`--topology fleet` -- the default runner -- is one PROCESS per GPU, so the threads this
    instrument exists to name live in a grandchild of the wrapper. Reading only the direct
    child's task directory reported a few percent of a run and called it `accounted_pct`.
    """

    #: The child spawns the grandchild and waits: the wrapper's own child owns no named
    #: thread, so anything found under these names came from walking the tree.
    _NESTED = (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', {inner!r}], check=True)\n"
    )
    _PLAN = (("shard-0", 0.4), ("shard-1", 0.4))

    def test_a_grandchilds_threads_are_named_in_the_breakdown(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        out = tmp_path / "cpu.json"
        inner = _THREADS.format(plan=repr(self._PLAN))
        code = host_cpu.main(
            [
                "--threads",
                "--threads-interval",
                "0.02",
                "--out",
                str(out),
                "--",
                sys.executable,
                "-c",
                self._NESTED.format(inner=inner),
            ]
        )
        assert code == 0

        accounting = json.loads(out.read_text(encoding="utf-8"))
        assert accounting["threads"].get("shard", {}).get("threads") == 2, accounting["threads"]
        assert accounting["accounted_pct"] > 50, accounting

    def test_the_walk_names_the_whole_tree_and_survives_a_dead_pid(
        self, host_cpu: ModuleType
    ) -> None:
        """Two claims in one place: a grandchild is reachable, and a pid that has exited is a
        skipped branch rather than an exception mid-sample."""
        child = _spinner(SPIN_S)
        try:
            tree = host_cpu.process_tree(os.getpid())
            assert os.getpid() in tree and child.pid in tree, tree
        finally:
            child.wait()
        assert host_cpu.process_tree(2**22) == [2**22]


#: A bench that starts a load generator: spawn it, declare its pid in the drop-box the
#: wrapper exported, then burn some CPU of its own so the two are told apart by more than zero.
_WITH_GENERATOR = """
import os, subprocess, sys, time

generator = subprocess.Popen([sys.executable, '-c', {inner!r}])
with open(os.environ['SHIPINFER_HOST_CPU_PIDFILE'], 'a') as handle:
    handle.write(str(generator.pid) + chr(10))
end = time.thread_time() + {own}
while time.thread_time() < end:
    pass
generator.wait()
"""


def _fake_tree(children: dict[int, list[int]]):
    """A `process_tree` stand-in that respects parentage AND `skip`.

    Both matter now: `_sample` asks for a *generator's* subtree as well as the bench's, so a
    fake that answered with the whole map would sweep the bench's own pid into the exclusion.
    """

    def walk(pid: int, skip: object = frozenset()) -> list[int]:
        found: list[int] = []
        pending = [pid]
        while pending:
            current = pending.pop()
            if current in found or current in skip:  # type: ignore[operator]
                continue
            found.append(current)
            pending.extend(children.get(current, []))
        return found

    return walk


class TestAGeneratorTheBenchSpawnedIsNotTheBench:
    """The RTSP arm starts its two `rtsp_serve.py` servers INSIDE the bench container, because
    the rootless daemon has no NAT. So they are children of the bench and their CPU is already
    in `wait4`'s rusage -- unlike a `--pid` generator, which is a separate tree. #204's tree
    walk started sampling them; this is the discount that keeps the figure honest.
    """

    _PLAN = (("gen-0", 0.4), ("gen-1", 0.4))

    def _run(self, host_cpu: ModuleType, tmp_path: Path) -> dict:
        out = tmp_path / "cpu.json"
        code = host_cpu.main(
            [
                "--threads",
                "--threads-interval",
                "0.02",
                "--out",
                str(out),
                "--",
                sys.executable,
                "-c",
                _WITH_GENERATOR.format(
                    inner=_THREADS.format(plan=repr(self._PLAN)), own=SPIN_S
                ),
            ]
        )
        assert code == 0
        return json.loads(out.read_text(encoding="utf-8"))

    def test_its_threads_are_not_in_the_breakdown(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        """Declared AFTER its first tick, which is the real sequence -- the pid does not exist
        until the spawn returns -- so this also pins that a late declaration is retroactive."""
        accounting = self._run(host_cpu, tmp_path)

        assert "gen" not in accounting["threads"], accounting["threads"]
        assert all(not row["name"].startswith("gen-") for row in accounting["threads_top"])

    def test_its_cpu_is_named_and_taken_out_of_the_denominator(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        accounting = self._run(host_cpu, tmp_path)

        assert accounting["spawned_generators"], accounting
        assert accounting["spawned_generator_cpu_s"] > 0.3, accounting
        assert accounting["bench_cpu_s"] == pytest.approx(
            accounting["command_cpu_s"] - accounting["spawned_generator_cpu_s"], abs=0.02
        ), accounting

    def test_the_accounted_share_is_measured_against_the_bench(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        """The whole point of subtracting: divide by the tree total and a correct breakdown
        reads as a broken instrument, which is how #204's 3.3% was mistaken for one."""
        accounting = self._run(host_cpu, tmp_path)

        by_bench = 100.0 * accounting["threads_cpu_s"] / accounting["bench_cpu_s"]
        by_tree = 100.0 * accounting["threads_cpu_s"] / accounting["command_cpu_s"]
        # `rel`, not `abs`: the JSON fields are rounded to two places and these totals are
        # fractions of a second, so recomputing from them cannot land on the exact figure.
        assert accounting["accounted_pct"] == pytest.approx(by_bench, rel=0.02), accounting
        assert abs(accounting["accounted_pct"] - by_bench) < abs(
            accounting["accounted_pct"] - by_tree
        ), f"divided by the tree total, not the bench's: {accounting}"
        assert accounting["accounted_pct"] > 50, accounting

    def test_no_generator_leaves_the_fields_empty_rather_than_absent(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        out = tmp_path / "cpu.json"
        code = host_cpu.main(
            [
                "--threads",
                "--threads-interval",
                "0.02",
                "--out",
                str(out),
                "--",
                sys.executable,
                "-c",
                _SPIN.format(seconds=SPIN_S),
            ]
        )
        assert code == 0

        accounting = json.loads(out.read_text(encoding="utf-8"))
        assert accounting["spawned_generators"] == {}
        assert accounting["spawned_generator_cpu_s"] == 0.0
        assert accounting["bench_cpu_s"] == accounting["command_cpu_s"]

    def test_the_drop_box_does_not_outlive_the_run(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        """A stale file would discount pids a later run never spawned."""
        self._run(host_cpu, tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["cpu.json"]

    def test_a_generator_declared_after_the_first_tick_is_purged(
        self, host_cpu: ModuleType, tmp_path: Path, monkeypatch
    ) -> None:
        """The real sequence, made deterministic. The RTSP servers take seconds to accept a
        connection and the wrapper ticks every 500 ms, so several samples land before the pid
        is ever written -- and whatever they recorded has to leave the table when it is.
        """
        box = tmp_path / "generators.pids"
        box.write_text("", encoding="utf-8")
        tree = {os.getpid(): {1: ("pipe-0", 1.0)}, 100: {2: ("gen-0", 5.0)}}
        monkeypatch.setattr(host_cpu, "process_tree", _fake_tree({os.getpid(): [100]}))
        monkeypatch.setattr(host_cpu, "thread_cpu", lambda pid: tree[pid])
        monkeypatch.setattr(host_cpu, "cpu_seconds", lambda pid: 5.0)
        sampler = host_cpu.ThreadSampler(os.getpid(), interval_s=10.0, generators=box)

        sampler._sample()
        assert "gen" in sampler.by_class(), "the pre-condition: undeclared, so it is sampled"

        box.write_text("100\n", encoding="utf-8")
        sampler._sample()

        assert "gen" not in sampler.by_class(), sampler.by_class()
        assert sampler.generator_cpu() == {100: 5.0}
        assert sampler.total() == pytest.approx(1.0)

    def test_an_unreadable_tick_cannot_undo_a_declaration(
        self, host_cpu: ModuleType, tmp_path: Path, monkeypatch
    ) -> None:
        """`_generator_pids` answers with an empty set on `OSError`, so an unreadable tick
        says "no generators" — and the pids that were declared are remembered rather than
        re-admitted. Three ticks: undeclared, declared, unreadable.
        """
        box = tmp_path / "generators.pids"
        tree = {os.getpid(): {1: ("pipe-0", 1.0)}, 100: {2: ("gen-0", 5.0)}}
        monkeypatch.setattr(host_cpu, "process_tree", _fake_tree({os.getpid(): [100]}))
        monkeypatch.setattr(host_cpu, "thread_cpu", lambda pid: tree[pid])
        monkeypatch.setattr(host_cpu, "cpu_seconds", lambda pid: 5.0)
        sampler = host_cpu.ThreadSampler(os.getpid(), interval_s=10.0, generators=box)

        sampler._sample()
        assert "gen" in sampler.by_class(), "the pre-condition: undeclared, so it is sampled"
        box.write_text("100\n", encoding="utf-8")
        sampler._sample()
        assert "gen" not in sampler.by_class(), sampler.by_class()
        box.unlink()
        sampler._sample()

        assert "gen" not in sampler.by_class(), sampler.by_class()
        assert sampler.total() == pytest.approx(1.0)

    def test_a_generators_own_child_is_purged_with_it(
        self, host_cpu: ModuleType, tmp_path: Path, monkeypatch
    ) -> None:
        """`cutime` charges a reaped `ffmpeg` to the server that waited for it, so leaving the
        child's THREADS in the breakdown counts that CPU twice -- once in `threads_cpu_s` and
        once out of `bench_cpu_s`. The first tick here is the unreadable one, which is the only
        way the subtree gets in: the walk does not descend into a declared generator.
        """
        box = tmp_path / "generators.pids"
        tree = {
            os.getpid(): {1: ("pipe-0", 1.0)},
            100: {2: ("gen-0", 5.0)},
            200: {3: ("ffmpeg", 9.0)},
        }
        monkeypatch.setattr(
            host_cpu, "process_tree", _fake_tree({os.getpid(): [100], 100: [200], 200: []})
        )
        monkeypatch.setattr(host_cpu, "thread_cpu", lambda pid: tree[pid])
        monkeypatch.setattr(host_cpu, "cpu_seconds", lambda pid: 14.0)
        sampler = host_cpu.ThreadSampler(os.getpid(), interval_s=10.0, generators=box)

        sampler._sample()
        assert {"gen", "ffmpeg"} <= set(sampler.by_class()), sampler.by_class()

        box.write_text("100\n", encoding="utf-8")
        sampler._sample()

        assert set(sampler.by_class()) == {"pipe"}, sampler.by_class()
        assert sampler.total() == pytest.approx(1.0)

    def test_a_subtree_missed_on_one_tick_is_purged_on_the_next(
        self, host_cpu: ModuleType, tmp_path: Path, monkeypatch
    ) -> None:
        """Why the purge must not go back behind a grew-since-last-tick guard.

        `process_tree` swallows `OSError` per pid, so a walk can miss a live grandchild and
        answer with the generator alone. Reinstating the guard passes every other test in this
        file: the residual difference is exactly this sequence, where the declaration set stops
        changing and the subtree is discovered a tick later.
        """
        box = tmp_path / "generators.pids"
        tree = {
            os.getpid(): {1: ("pipe-0", 1.0)},
            100: {2: ("gen-0", 5.0)},
            200: {3: ("ffmpeg", 9.0)},
        }
        full = _fake_tree({os.getpid(): [100], 100: [200], 200: []})
        blind = _fake_tree({os.getpid(): [100], 100: [], 200: []})
        seen_100 = [0]

        def walk(pid: int, skip: object = frozenset()) -> list[int]:
            if pid == 100:
                seen_100[0] += 1
                # The first read of 100's `children` fails; the second sees 200.
                return (blind if seen_100[0] == 1 else full)(pid, skip)
            return full(pid, skip)

        monkeypatch.setattr(host_cpu, "process_tree", walk)
        monkeypatch.setattr(host_cpu, "thread_cpu", lambda pid: tree[pid])
        monkeypatch.setattr(host_cpu, "cpu_seconds", lambda pid: 14.0)
        sampler = host_cpu.ThreadSampler(os.getpid(), interval_s=10.0, generators=box)

        sampler._sample()
        assert {"gen", "ffmpeg"} <= set(sampler.by_class()), sampler.by_class()
        box.write_text("100\n", encoding="utf-8")
        sampler._sample()
        assert set(sampler.by_class()) == {
            "pipe",
            "ffmpeg",
        }, "the pre-condition: the walk missed 200, so only the generator itself is purged"
        sampler._sample()

        assert set(sampler.by_class()) == {"pipe"}, sampler.by_class()

    def test_declaring_is_a_no_op_when_nobody_is_measuring(
        self, host_cpu: ModuleType, monkeypatch
    ) -> None:
        """A bench run without the wrapper must not need to know it is not being measured."""
        monkeypatch.delenv(host_cpu.GENERATOR_PIDFILE_ENV, raising=False)

        assert host_cpu.declare_generator(123) is False

    def test_declaring_appends_rather_than_replacing(
        self, host_cpu: ModuleType, tmp_path, monkeypatch
    ) -> None:
        """Two servers, one drop-box: the second must not erase the first."""
        box = tmp_path / "generators.pids"
        box.write_text("", encoding="utf-8")
        monkeypatch.setenv(host_cpu.GENERATOR_PIDFILE_ENV, str(box))

        assert host_cpu.declare_generator(11) is True
        assert host_cpu.declare_generator(22) is True

        assert box.read_text(encoding="utf-8").split() == ["11", "22"]

    def test_the_harness_declares_each_server_it_starts(self) -> None:
        """The link a unit test cannot reach: the declaration has to sit inside the loop that
        starts the servers, or only the last of the two is discounted."""
        text = (ROOT / "benchmarks" / "harness" / "rtsp.py").read_text(encoding="utf-8")
        loop = text.split("for content, port, streams, directory in _servers(config):")[1]

        assert "declare_generator(process.pid)" in loop.split("deadline =")[0]


#: A generator whose real cost is a child: `scripts/rtsp_serve.py` shells out to `ffmpeg` over
#: the whole JPEG set whenever the `.h264` fixture is cold. The burner is reaped, so the
#: kernel folds it into the generator's `cutime`.
_GENERATOR_WITH_A_CHILD = """
import subprocess, sys, time

subprocess.run([sys.executable, '-c', {inner!r}], check=True)
end = time.thread_time() + {own}
while time.thread_time() < end:
    pass
"""


class TestAGeneratorsOwnChildrenAreDiscountedToo:
    """`cpu_seconds` read `utime + stime` and never `cutime + cstime`, so the largest single
    piece of a cold-fixture run -- the `ffmpeg` encode -- stayed in the bench's column. `wait4`
    charges the command WITH its reaped children, so a generator measured without them is
    measured on a different rule from the total it is subtracted from.
    """

    def test_the_stat_parse_can_include_the_children(self, host_cpu: ModuleType) -> None:
        """Fields 16 and 17, past a `comm` chosen to break a naive split. The list below is
        the fields AFTER `comm`, so index 11 is field 14 -- which is what the parse slices."""
        after_comm = ["S", *(str(n) for n in range(4, 55))]
        after_comm[11], after_comm[12] = "10", "10"  # utime, stime
        after_comm[13], after_comm[14] = "30", "20"  # cutime, cstime
        line = "7 (py (thread) 1) " + " ".join(after_comm)

        assert host_cpu.ticks_from_stat(line) == 20.0
        assert host_cpu.ticks_from_stat(line, with_children=True) == 70.0

    def test_a_live_generator_reads_higher_with_its_children(
        self, host_cpu: ModuleType
    ) -> None:
        """The kernel folds a REAPED child into `cutime`, so this reads the generator while it
        is still alive -- a dead pid answers `None`, which is the documented behaviour and not
        what is being measured here."""
        script = (
            "import subprocess, sys, time\n"
            f"subprocess.run([sys.executable, '-c', {_SPIN.format(seconds=SPIN_S)!r}])\n"
            "time.sleep(12)\n"
        )
        child = subprocess.Popen([sys.executable, "-c", script], env=checkout_env())
        own = with_children = 0.0
        try:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                try:
                    line = Path(f"/proc/{child.pid}/stat").read_text(encoding="utf-8")
                except OSError:  # pragma: no cover - the child outlives this loop
                    break
                own = host_cpu.ticks_from_stat(line) / host_cpu.TICKS_PER_SECOND
                with_children = host_cpu.cpu_seconds(child.pid) or 0.0
                if with_children > SPIN_S * 0.7:
                    break
                time.sleep(0.05)
        finally:
            child.terminate()
            child.wait()

        assert with_children > SPIN_S * 0.7, (own, with_children)
        assert with_children > own * 2, (own, with_children)

    def test_the_whole_child_tree_leaves_the_benchs_column(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        """End to end: the generator burns a quarter of what its child does, and both are out
        of `bench_cpu_s`."""
        out = tmp_path / "cpu.json"
        inner = _GENERATOR_WITH_A_CHILD.format(
            inner=_SPIN.format(seconds=SPIN_S), own=SPIN_S / 4
        )
        code = host_cpu.main(
            [
                "--threads",
                "--threads-interval",
                "0.02",
                "--out",
                str(out),
                "--",
                sys.executable,
                "-c",
                _WITH_GENERATOR.format(inner=inner, own=SPIN_S),
            ]
        )
        assert code == 0

        accounting = json.loads(out.read_text(encoding="utf-8"))
        # The child alone burns SPIN_S; charging only the generator's own time would report
        # about a quarter of that.
        assert accounting["spawned_generator_cpu_s"] > SPIN_S, accounting


class TestTheSamplerCanBeTurnedOff:
    def test_a_zero_interval_reports_no_thread_breakdown(
        self, host_cpu: ModuleType, tmp_path: Path
    ) -> None:
        """The instrument costs /proc reads on the host it exists to argue is tight, so the
        design-load run needs a way out -- and the way out gives up the discount with it."""
        out = tmp_path / "cpu.json"
        code = host_cpu.main(
            [
                "--threads",
                "--threads-interval",
                "0",
                "--out",
                str(out),
                "--",
                sys.executable,
                "-c",
                _SPIN.format(seconds=SPIN_S),
            ]
        )
        assert code == 0

        accounting = json.loads(out.read_text(encoding="utf-8"))
        assert "threads" not in accounting, accounting
        assert accounting["command_cpu_s"] > 0

    def test_the_wrapper_takes_the_interval_from_the_environment(self) -> None:
        text = (ROOT / "deploy" / "rootless" / "bench.sh").read_text(encoding="utf-8")

        assert '"${SHIPINFER_BENCH_HOST_CPU_INTERVAL:-0.5}"' in text
        assert (
            "-e SHIPINFER_BENCH_HOST_CPU_INTERVAL" in text
        ), "and it has to reach the container"


class TestARecycledTidLosesNeitherThread:
    """`_seen` is keyed by tid and keeps the highest reading, so a reused id used to drop
    whichever thread read lower -- the new one. The tree walk widened the window: a tid is
    unique host-wide, but across a whole shard fleet there are far more of them.
    """

    def test_both_threads_keep_their_cpu(self, host_cpu: ModuleType, monkeypatch) -> None:
        readings = [
            {7: ("pipe-0", 1.0)},
            {7: ("pipe-0", 2.0)},
            {7: ("cam-a", 0.5)},  # same tid, new thread: the counter cannot go backwards
            {7: ("cam-a", 0.9)},
        ]
        monkeypatch.setattr(host_cpu, "process_tree", lambda pid, skip=frozenset(): [pid])
        sampler = host_cpu.ThreadSampler(os.getpid(), interval_s=10.0)
        for reading in readings:
            monkeypatch.setattr(host_cpu, "thread_cpu", lambda _pid, r=reading: r)
            sampler._sample()

        assert sampler.total() == pytest.approx(2.9)
        assert sampler.by_class()["pipe"]["cpu_s"] == pytest.approx(2.0)
        assert sampler.by_class()["cam"]["cpu_s"] == pytest.approx(0.9)
        assert [row["name"] for row in sampler.top()] == ["pipe-0", "cam-a"]


class TestBothArmsAreActuallyWiredToIt:
    """The one link the tests above cannot reach, and it is the link that goes missing.

    A measurement whose wiring is deleted leaves every unit test green and the number gone --
    `benchmarks/tests/test_occupancy_window.py` exists because that happened twice to the
    occupancy counter. Weaker than the tests above and says so: what it catches is a wrapper
    dropped from one arm, or a line printed into a log and into no summary.
    """

    RTSP = ROOT / "scripts" / "cpp_bench_over_rtsp.sh"
    RUNNER = ROOT / "scripts" / "run_cpp_bench.sh"

    def test_the_rtsp_arm_accounts_for_both_of_its_servers(self) -> None:
        text = self.RTSP.read_text(encoding="utf-8")

        assert "scripts/host_cpu.py" in text
        assert '--pid "$person_pid" --pid "$ship_pid"' in text, (
            "the RTSP arm must charge BOTH servers; one of them is half the load and half "
            "the CPU this exists to discount"
        )

    def test_the_replay_arm_is_wrapped_too(self) -> None:
        """Without this the two arms are not comparable, and their difference -- our own
        decode cost -- is the whole reason the accounting was added."""
        text = self.RUNNER.read_text(encoding="utf-8")

        assert "host_cpu.py" in text.split('if [ "$SOURCE" = "replay" ]')[1].split("else")[0]

    def test_the_python_bench_is_wrapped_as_well(self) -> None:
        """The instrument that produced the C++ finding was on one plane only, so four
        design-load arms of the Python A/B recorded no `host cpu:` line at all."""
        text = (ROOT / "deploy" / "rootless" / "bench.sh").read_text(encoding="utf-8")

        statement = text.split("exec python")[1]
        assert statement.startswith(" /work/scripts/host_cpu.py --threads"), statement[:70]
        assert (
            "run_bench.py" in statement.split("--threads-interval")[1].split("'")[0]
        ), "the wrapper has to be what runs the bench, not a line beside it"

    def test_the_number_reaches_the_summary_and_not_only_the_log(self) -> None:
        """`run_cpp_bench.sh` says this in its own words about the `chain` line: it was in
        every log and in none of these summaries, so a reader of the documented output could
        quote a figure without ever seeing the caveat. Same alternation, same reason."""
        assert "host cpu:" in self.RUNNER.read_text(encoding="utf-8").split("grep -E")[1]
