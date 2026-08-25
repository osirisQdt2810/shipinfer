"""Supervising one process per shard, tested against processes that are not servers.

Everything worth getting wrong here is about *supervision*, not about inference: does the child
get the right ``CUDA_VISIBLE_DEVICES``, does a dead shard take the fleet down rather than
leaving a quarter of the cameras dark behind green dashboards, does a shard that will not drain
get killed rather than leaked, and is anything left as a zombie afterwards.

None of that needs a GPU, and testing it through a real server would test CUDA instead — which
is why :class:`~shipinfer.server.launcher.Fleet` takes its command as a callable. The children
below are ``python -c`` one-liners that do exactly the one thing each test is about.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.scheduling.sharding import plan_shards
from shipinfer.server.launcher import (
    Fleet,
    ShardExitedError,
    ShardProcess,
    serve_command,
)

FLEET = {f"cam{i}": 20.0 for i in range(8)}


def plan(shards: int = 2, gpus: tuple[int, ...] = (2, 3, 4, 5)):
    return plan_shards(FLEET, shards=shards, gpus=list(gpus))


def sleeps(seconds: float = 30.0):
    """A child that stays up until it is told to go."""
    return lambda shard: [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def exits(code: int = 1, after: float = 0.0):
    """A child that goes away on its own — a shard whose engine failed to load."""
    return lambda shard: [
        sys.executable,
        "-c",
        f"import sys, time; time.sleep({after}); sys.exit({code})",
    ]


def ignores_sigterm():
    """A child that will not drain. A shard blocked in a CUDA call is not interruptible."""
    return lambda shard: [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]


def reports_env(path_for):
    """A child that writes its own ``CUDA_VISIBLE_DEVICES`` where the test can read it."""
    return lambda shard: [
        sys.executable,
        "-c",
        "import os, sys; open(sys.argv[1], 'w').write("
        "os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))",
        str(path_for(shard)),
    ]


class TestEachShardGetsItsOwnDevices:
    """The one thing that cannot be fixed later: it must be set before the child starts."""

    def test_the_child_sees_only_its_own_gpus(self, tmp_path) -> None:
        written = {}

        def path_for(shard):
            written[shard.index] = tmp_path / f"shard{shard.index}.txt"
            return written[shard.index]

        fleet = Fleet(plan=plan(shards=2), command=reports_env(path_for))
        fleet.start()
        for running in fleet.running:
            running.process.wait(timeout=30)
        fleet.stop()

        assert written[0].read_text() == "2,3"
        assert written[1].read_text() == "4,5"

    def test_it_is_in_the_environment_before_the_interpreter_starts(self, tmp_path) -> None:
        """Not set by the child. A shard that sets it itself is racing whatever imported
        torch first, and in this codebase that is a module-scope import two packages deep —
        by then the runtime has already chosen its devices."""
        target = tmp_path / "seen.txt"
        fleet = Fleet(
            plan=plan(shards=1, gpus=(4,)),
            command=lambda shard: [
                sys.executable,
                "-c",
                # Reads the variable in the first statement the interpreter runs.
                f"import os; open({str(target)!r}, 'w').write("
                "os.environ['CUDA_VISIBLE_DEVICES'])",
            ],
        )
        fleet.start()
        for running in fleet.running:
            running.process.wait(timeout=30)
        fleet.stop()

        assert target.read_text() == "4"

    def test_the_parents_environment_survives(self, tmp_path, monkeypatch) -> None:
        """A child needs PATH, HOME and whatever the container set. Replacing the environment
        rather than extending it is how a shard ends up unable to find its own model files."""
        monkeypatch.setenv("SHIPINFER_TEST_MARKER", "kept")
        target = tmp_path / "marker.txt"
        fleet = Fleet(
            plan=plan(shards=1, gpus=(2,)),
            command=lambda shard: [
                sys.executable,
                "-c",
                f"import os; open({str(target)!r}, 'w').write("
                "os.environ.get('SHIPINFER_TEST_MARKER', '<lost>'))",
            ],
        )
        fleet.start()
        for running in fleet.running:
            running.process.wait(timeout=30)
        fleet.stop()

        assert target.read_text() == "kept"

    def test_an_operators_cuda_visible_devices_does_not_override_the_plan(
        self, tmp_path
    ) -> None:
        """A shard's device set is decided by the plan, not by an inherited variable — one
        that leaked in would give every shard the same devices and rebuild the failure this
        whole design exists to fix, silently."""
        target = tmp_path / "devices.txt"
        fleet = Fleet(
            plan=plan(shards=1, gpus=(3,)),
            command=lambda shard: [
                sys.executable,
                "-c",
                f"import os; open({str(target)!r}, 'w').write("
                "os.environ['CUDA_VISIBLE_DEVICES'])",
            ],
            env={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
        )
        fleet.start()
        for running in fleet.running:
            running.process.wait(timeout=30)
        fleet.stop()

        assert target.read_text() == "3"


class TestADeadShardTakesTheFleetDown:
    """A shard is a slice of the fleet, so one dying is not degradation — it is dark cameras."""

    def test_supervise_raises_when_a_shard_exits(self) -> None:
        fleet = Fleet(plan=plan(shards=2), command=exits(code=3, after=0.05))
        fleet.start()

        with pytest.raises(ShardExitedError, match="no longer being read"):
            fleet.supervise(poll_s=0.02)

    def test_the_message_names_the_shard_and_its_exit_code(self) -> None:
        fleet = Fleet(plan=plan(shards=1, gpus=(2,)), command=exits(code=42, after=0.05))
        fleet.start()

        with pytest.raises(ShardExitedError, match=r"shard 0 .*exited with 42"):
            fleet.supervise(poll_s=0.02)

    def test_the_rest_of_the_fleet_is_stopped_before_the_error_is_raised(self) -> None:
        """Otherwise the caller has to remember to, and the surviving shards keep holding
        their GPUs while the operator reads a traceback."""

        def command(shard):
            return exits(code=1, after=0.05)(shard) if shard.index == 0 else sleeps()(shard)

        fleet = Fleet(plan=plan(shards=2), command=command)
        fleet.start()
        survivors = [r for r in fleet.running if r.shard.index == 1]

        with pytest.raises(ShardExitedError):
            fleet.supervise(poll_s=0.02)

        assert survivors[0].process.poll() is not None, "the healthy shard was left running"
        assert fleet.running == ()

    def test_a_shard_that_exits_zero_still_counts_as_dead(self) -> None:
        """A server that returns cleanly has still stopped serving. Treating 0 as fine is how
        a fleet ends up with three shards and a green dashboard."""
        fleet = Fleet(plan=plan(shards=1, gpus=(2,)), command=exits(code=0, after=0.05))
        fleet.start()

        with pytest.raises(ShardExitedError):
            fleet.supervise(poll_s=0.02)

    def test_supervise_returns_when_asked_to_stop(self) -> None:
        fleet = Fleet(plan=plan(shards=2), command=sleeps())
        fleet.start()
        try:
            calls = {"n": 0}

            def until() -> bool:
                calls["n"] += 1
                return calls["n"] >= 2

            fleet.supervise(poll_s=0.01, until=until)
        finally:
            fleet.stop()

        assert calls["n"] == 2

    def test_supervising_a_fleet_that_never_started_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="not running"):
            Fleet(plan=plan(), command=sleeps()).supervise()


class TestStartUpIsAllOrNothing:
    """Three of four shards up means three-quarters of the cameras are watched, silently."""

    def test_a_failure_mid_start_stops_what_already_started(self) -> None:
        started: list[ShardProcess] = []

        def command(shard):
            if shard.index == 1:
                raise RuntimeError("no engine for this shard")
            return sleeps()(shard)

        fleet = Fleet(plan=plan(shards=2), command=command)
        original = fleet._spawn

        def spy(shard):
            running = original(shard)
            started.append(running)
            return running

        fleet._spawn = spy  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="no engine"):
            fleet.start()

        assert started, "shard 0 should have started before shard 1 failed"
        assert all(r.process.poll() is not None for r in started), "shard 0 was left running"
        assert fleet.running == ()

    def test_an_empty_command_is_refused(self) -> None:
        fleet = Fleet(plan=plan(shards=1, gpus=(2,)), command=lambda shard: [])

        with pytest.raises(ConfigurationError, match="empty command"):
            fleet.start()

    def test_starting_twice_is_refused(self) -> None:
        fleet = Fleet(plan=plan(shards=1, gpus=(2,)), command=sleeps())
        fleet.start()
        try:
            with pytest.raises(ConfigurationError, match="already running"):
                fleet.start()
        finally:
            fleet.stop()


class TestStoppingLeavesNothingBehind:
    """This box is shared: a finished-but-alive shard keeps a CUDA context per device."""

    def test_a_shard_that_will_not_drain_is_killed(self) -> None:
        fleet = Fleet(plan=plan(shards=1, gpus=(2,)), command=ignores_sigterm(), drain_s=0.3)
        fleet.start()
        pid = fleet.running[0].process.pid

        started = time.monotonic()
        fleet.stop()
        elapsed = time.monotonic() - started

        assert elapsed < 5.0, "a shard blocked in a CUDA call must not hold the launcher open"
        with pytest.raises(OSError):
            os.kill(pid, 0)

    def test_the_drain_deadline_is_shared_not_per_shard(self) -> None:
        """Otherwise stopping N shards takes N times as long, and a rolling restart of a
        sixteen-GPU deployment turns a 20-second drain into five minutes."""
        fleet = Fleet(plan=plan(shards=4), command=ignores_sigterm(), drain_s=0.4)
        fleet.start()

        started = time.monotonic()
        fleet.stop()
        elapsed = time.monotonic() - started

        assert elapsed < 4 * 0.4, f"stopping four shards took {elapsed:.2f}s"

    def test_no_child_is_left_as_a_zombie(self) -> None:
        fleet = Fleet(plan=plan(shards=2), command=ignores_sigterm(), drain_s=0.2)
        fleet.start()
        processes = [r.process for r in fleet.running]

        fleet.stop()

        for process in processes:
            assert process.returncode is not None, "a killed child was never reaped"

    def test_stop_is_idempotent(self) -> None:
        fleet = Fleet(plan=plan(shards=2), command=sleeps())
        fleet.start()
        fleet.stop()

        fleet.stop()

        assert fleet.running == ()

    def test_stopping_a_fleet_that_never_started_does_nothing(self) -> None:
        Fleet(plan=plan(), command=sleeps()).stop()

    def test_the_context_manager_stops_on_the_way_out(self) -> None:
        with Fleet(plan=plan(shards=2), command=sleeps()) as fleet:
            processes = [r.process for r in fleet.running]
            assert all(p.poll() is None for p in processes)

        assert all(p.poll() is not None for p in processes)

    def test_the_context_manager_stops_on_an_exception_too(self) -> None:
        processes: list[subprocess.Popen] = []
        with (
            pytest.raises(ValueError, match="boom"),
            Fleet(plan=plan(shards=2), command=sleeps()) as fleet,
        ):
            processes = [r.process for r in fleet.running]
            raise ValueError("boom")

        assert all(p.poll() is not None for p in processes)

    def test_a_non_positive_drain_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="killed mid-batch"):
            Fleet(plan=plan(), command=sleeps(), drain_s=0.0)


class TestTheDefaultCommand:
    def test_it_runs_this_interpreter_not_whatever_is_on_path(self) -> None:
        """A shard that starts under a different interpreter is a debugging session nobody
        wants, and the console script may not be on PATH inside a container."""
        argv = serve_command(plan(shards=1, gpus=(2,)).shards[0], repository="/models")

        assert argv[0] == sys.executable
        assert argv[1:4] == ["-m", "shipinfer", "serve"]

    def test_every_flag_it_emits_is_one_serve_defines(self) -> None:
        """The first version of this passed `--cameras` and a positional repository. `serve`
        has neither — it takes `-r`. The argv read plausibly, the CLI would have rejected it on
        both counts, and the test that "covered" it asserted the shape I had invented rather
        than the shape the parser accepts.

        So this reads the real command's real signature. A flag renamed in `serve.py` breaks
        this test, which is the point: the launcher and the command it launches are two halves
        of one contract, and nothing else holds them together.
        """
        import inspect

        from shipinfer.cli.commands.serve import serve

        argv = serve_command(plan(shards=1, gpus=(2,)).shards[0], repository="/models")
        # After the subcommand name: `-m` before it belongs to the interpreter, not to `serve`.
        after_serve = argv[argv.index("serve") + 1 :]
        emitted = {arg for arg in after_serve if arg.startswith("-")}
        accepted = {"-r", "--repository"} | {
            f"--{name.replace('_', '-')}" for name in inspect.signature(serve).parameters
        }

        assert emitted, "the command emits no flags at all; it cannot be right by accident"
        assert emitted <= accepted, f"{emitted - accepted} are not flags `serve` defines"

    def test_it_names_the_repository_with_the_flag_serve_defines(self) -> None:
        argv = serve_command(plan(shards=1, gpus=(2,)).shards[0], repository="/models")

        assert argv[argv.index("-r") + 1] == "/models"
        assert (
            "--cameras" not in argv
        ), "serve has no camera flag; a shard's cameras travel in SHARD_CAMERAS_ENV"

    def test_the_cameras_travel_in_the_environment_instead(self, tmp_path) -> None:
        """The half that replaced the invented flag, checked end to end: a child really does
        come up knowing which cameras are its own."""
        written = {}

        def path_for(shard):
            written[shard.index] = tmp_path / f"cams{shard.index}.txt"
            return written[shard.index]

        fleet = Fleet(
            plan=plan(shards=2),
            command=lambda shard: [
                sys.executable,
                "-c",
                f"import os; open({'{}'!r}, 'w')".replace("{}", str(path_for(shard)))
                + ".write(os.environ['SHIPINFER_SHARD_CAMERAS'])",
            ],
        )
        fleet.start()
        for running in fleet.running:
            running.process.wait(timeout=30)
        fleet.stop()

        by_index = {s.index: s for s in fleet.plan.shards}
        for index, path in written.items():
            assert path.read_text() == ",".join(by_index[index].cameras)
        assert written[0].read_text() != written[1].read_text()

    def test_it_does_not_pass_gpus(self) -> None:
        """The child sees only its own devices, so to it they are 0..n-1. Passing the physical
        ordinals as well would ask it to select devices it cannot see."""
        argv = serve_command(plan(shards=2).shards[1], repository="/models")

        assert "--gpus" not in argv
        assert "4" not in argv and "5" not in argv

    def test_extra_arguments_are_forwarded(self) -> None:
        argv = serve_command(
            plan(shards=1, gpus=(2,)).shards[0], repository="/models", extra=("--http",)
        )

        assert argv[-1] == "--http"
