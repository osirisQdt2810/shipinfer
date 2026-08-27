"""The argv a shard is started with — the last thing left in ``server/launcher.py``.

Supervision moved to :mod:`shipinfer.launch` and its tests moved with it
(``tests/launch/test_supervisor.py``). What stays here is the half arch.md §2 deletes: the two
functions that render a child's command line. They are worth testing until they go, because the
launcher and the CLI command it launches are two halves of one contract and nothing else holds
them together — the first version of ``serve_command`` emitted ``--cameras`` and a positional
repository, neither of which ``serve`` accepts.

``Fleet`` still appears below, once. ``serve_command``'s promise is that the cameras travel in
the environment *instead of* on the argv, and only a real spawn shows the other half of that
promise being kept.
"""

from __future__ import annotations

import sys

from shipinfer.launch import Fleet
from shipinfer.scheduling.sharding import plan_shards
from shipinfer.server.launcher import serve_command

FLEET = {f"cam{i}": 20.0 for i in range(8)}


def plan(shards: int = 2, gpus: tuple[int, ...] = (2, 3, 4, 5)):
    return plan_shards(FLEET, shards=shards, gpus=list(gpus))


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
        ), "serve has no camera flag; a shard's cameras travel in SHIPINFER_SHARD_CAMERAS"

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


class TestAShardCanBeAddressedOverHttp:
    """One port for every shard is not an option — the second bind fails — so HTTP is per shard,
    and only when asked for: a warm shard has no ingress."""

    def test_no_http_unless_asked(self) -> None:
        argv = serve_command(plan(shards=2).shards[1], repository="/models")
        assert "--http" not in argv and "--port" not in argv

    def test_each_shard_gets_its_own_port(self) -> None:
        shards = plan(shards=2).shards
        first = serve_command(shards[0], repository="/models", http_port_base=8100)
        second = serve_command(shards[1], repository="/models", http_port_base=8100)
        assert first[-3:] == ["--http", "--port", "8100"]
        assert second[-3:] == ["--http", "--port", "8101"]
        assert first[:-3] == second[:-3], "everything else is identical across shards"
