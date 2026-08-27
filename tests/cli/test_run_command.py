"""``shipinfer run``: what it resolves, what it refuses, and what it prints before spawning.

Offline throughout — every test here stops at ``--dry-run``, which returns before the
container gate and before a process exists. What is asserted is the wiring: the chain is read
*once* and both halves of it reach the runner, the two placement flags land in the settings
tree rather than in a runner-specific keyword, and a flag that does nothing yet says so
instead of being ignored.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from shipinfer.cli.commands.run import run
from shipinfer.core.errors import ConfigurationError

CHAIN = textwrap.dedent("""
    name: mock_chain
    elements:
      decode: {impl: mock}
      detect: {impl: mock, model: ship_detector}
      output: {impl: mock}
    """)


@pytest.fixture()
def chain_file(tmp_path: Path) -> Path:
    path = tmp_path / "mock_chain.yaml"
    path.write_text(CHAIN)
    return path


class TestWhatItResolves:
    def test_a_dry_run_prints_the_plan_and_spawns_nothing(
        self, chain_file: Path, capsys
    ) -> None:
        """The plan is the decision: which camera lands on which GPU is stable across
        restarts, so it is worth reading before fifty of them start reconnecting."""
        assert run(chain_file, runner="fleet", gpus="2,3", dry_run=True) == 0

        out = capsys.readouterr().out
        assert "runner fleet" in out
        assert "2 shard(s)" in out and "gpu(s) [2]" in out

    def test_the_shards_flag_reaches_the_plan_through_the_settings(
        self, chain_file: Path, capsys
    ) -> None:
        """`--shards` is `runner.shards`, not a keyword this command hands one named runner —
        which is what lets it name none (CONVENTIONS 2.3)."""
        run(chain_file, runner="fleet", gpus="4,5", shards=4, dry_run=True)

        assert "4 shard(s)" in capsys.readouterr().out

    def test_the_in_process_runner_has_no_plan_and_says_so(
        self, chain_file: Path, capsys
    ) -> None:
        assert run(chain_file, runner="inprocess", dry_run=True) == 0
        assert "no plan: one process" in capsys.readouterr().out

    def test_the_chain_name_and_size_are_printed_from_the_parsed_chain(
        self, chain_file: Path, capsys
    ) -> None:
        """Read once, here: the *text* is what a shard is sent and the parsed chain is what
        this process validates, so a mistyped chain fails on this line and not on sixteen
        children at once."""
        run(chain_file, runner="inprocess", dry_run=True)

        assert "topology: mock_chain (3 element(s))" in capsys.readouterr().out


class TestWhatItRefuses:
    # `--inputs` has its own file now that it does something: `tests/cli/test_run_inputs.py`.

    def test_an_unknown_runner_lists_the_ones_there_are(self, chain_file: Path) -> None:
        with pytest.raises(ConfigurationError, match="inprocess"):
            run(chain_file, runner="deepstream-ish", dry_run=True)

    def test_an_unreadable_chain_names_the_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="cannot read topology"):
            run(tmp_path / "nope.yaml", runner="inprocess", dry_run=True)

    def test_a_chain_that_does_not_load_fails_before_anything_is_spawned(
        self, tmp_path: Path
    ) -> None:
        """`ChainSpecError` is a `TopologyError` is a `ConfigurationError`: the operator gets
        the loader's own message, on the command they typed."""
        path = tmp_path / "broken.yaml"
        path.write_text("elements: {detect: {impl: mock, model: m}}")

        with pytest.raises(ConfigurationError):
            run(path, runner="fleet", gpus="0", dry_run=True)

    def test_a_fleet_with_no_visible_gpus_is_refused(
        self, chain_file: Path, monkeypatch
    ) -> None:
        """The driver is asked once, and when it reports none there is nothing to plan."""
        monkeypatch.setattr("shipinfer.runtime.platform.device_count", lambda: 0)

        with pytest.raises(ConfigurationError, match="which GPUs"):
            run(chain_file, runner="fleet", dry_run=True)


class TestWhoAnswersWhichDevices:
    """`--gpus` is not the only way an operator answers, and the driver is the last word.

    `build_settings` puts a flag in as an *init keyword argument*, which is
    pydantic-settings' highest-priority source. So a driver answer resolved before the
    settings and passed in as if it were a flag outranks `SHIPINFER_DEVICES__VISIBLE_GPUS` -
    which is how a two-GPU deployment silently became eight shards on eight devices, each
    holding a CUDA context on a shared box.
    """

    def test_the_env_setting_is_the_operators_answer_and_the_driver_is_not_asked(
        self, chain_file: Path, capsys, monkeypatch
    ) -> None:
        """The documented way to restrict a deployment (`launch/supervisor.py` calls it
        that). The stub raises rather than returning 8: a plan that came out right while the
        driver was still consulted would be right for today's box only."""

        def refuse() -> int:
            raise AssertionError("the driver was asked although the operator had answered")

        monkeypatch.setenv("SHIPINFER_DEVICES__VISIBLE_GPUS", "[2,3]")
        monkeypatch.setattr("shipinfer.runtime.platform.device_count", refuse)

        assert run(chain_file, runner="fleet", dry_run=True) == 0

        out = capsys.readouterr().out
        assert "2 shard(s)" in out, "the plan was not the two devices the operator owns"
        assert "gpu(s) [2]" in out and "gpu(s) [3]" in out

    def test_the_flag_still_wins_over_the_env_setting(
        self, chain_file: Path, capsys, monkeypatch
    ) -> None:
        """`build_settings`' own rule: flags win over `SHIPINFER_*`. A one-off `--gpus 5`
        must not require unsetting anything."""
        monkeypatch.setenv("SHIPINFER_DEVICES__VISIBLE_GPUS", "[2,3]")

        run(chain_file, runner="fleet", gpus="5", dry_run=True)

        out = capsys.readouterr().out
        assert "1 shard(s)" in out and "gpu(s) [5]" in out

    def test_with_nothing_configured_the_driver_fills_it_in_once(
        self, chain_file: Path, capsys, monkeypatch
    ) -> None:
        """One shard per visible GPU (ADR-006) is still the default a bare `run` gets."""
        calls: list[int] = []

        def counted() -> int:
            calls.append(1)
            return 3

        monkeypatch.delenv("SHIPINFER_DEVICES__VISIBLE_GPUS", raising=False)
        monkeypatch.setattr("shipinfer.runtime.platform.device_count", counted)

        run(chain_file, runner="fleet", dry_run=True)

        assert "3 shard(s)" in capsys.readouterr().out
        assert len(calls) == 1, "the driver was asked more than once"
