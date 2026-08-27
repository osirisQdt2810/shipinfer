"""The topology seam: a registry, a contract, and `fleet` as its first instance.

The operator's target is topology C (a cross-process inference tier); `fleet` is B, and the
point of putting B behind the contract first is that C and the DeepStream competitor become a
file and a decorator each. What is worth pinning here is the contract itself — what a topology
decides and what it does not — and that the command reaches the launcher through it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings import ServerSettings
from shipinfer.scheduling.sharding import plan_shards
from shipinfer.server.launcher import serve_command
from shipinfer.server.topology import TOPOLOGIES, TOPOLOGY_ENV, Topology, build_topology
from shipinfer.server.topology.fleet import FleetTopology

CAMERAS = {"cam00": 20.0, "cam01": 20.0, "cam02": 5.0, "cam03": 5.0}
GPUS = (2, 3)


class TestTheRegistryIsTheSwitch:
    def test_fleet_is_registered_under_its_name(self) -> None:
        assert "fleet" in TOPOLOGIES
        assert isinstance(build_topology("fleet"), FleetTopology)
        assert build_topology("fleet").name == "fleet"

    def test_an_unknown_topology_is_refused_with_the_known_names(self) -> None:
        with pytest.raises(ConfigurationError, match=r"unknown topology 'mesh'.*fleet"):
            build_topology("mesh")

    def test_the_settings_default_is_fleet(self) -> None:
        settings = ServerSettings(model_repository=Path("model_repository"))

        assert settings.runner.runner == "fleet"
        assert settings.runner.shards is None  # one per visible GPU, decided at launch
        assert settings.runner.drain_s == 20.0

    def test_the_runner_is_env_overridable_like_every_other_section(self, monkeypatch) -> None:
        monkeypatch.setenv(TOPOLOGY_ENV, "fleet")
        monkeypatch.setenv("SHIPINFER_RUNNER__SHARDS", "3")

        settings = ServerSettings(model_repository=Path("model_repository"))

        assert settings.runner.runner == "fleet"
        assert settings.runner.shards == 3

    def test_a_topology_must_subclass_the_contract(self) -> None:
        with pytest.raises(TypeError, match="does not subclass Topology"):

            @TOPOLOGIES.register("not-a-topology")
            class _Wrong:
                pass


class TestFleetIsTheLauncherBehindTheContract:
    """B "chỉ cần chỉnh 1 chút": registration and conformance, behaviour unchanged."""

    def test_the_plan_is_plan_shards(self) -> None:
        settings = ServerSettings(model_repository=Path("model_repository"))

        plan = FleetTopology().plan(settings, cameras=CAMERAS, gpus=GPUS, shards=2)

        assert plan == plan_shards(CAMERAS, shards=2, gpus=GPUS)
        assert plan.cameras == tuple(sorted(CAMERAS))

    def test_the_command_is_serve_on_the_same_repository(self) -> None:
        plan = plan_shards(CAMERAS, shards=2, gpus=GPUS)
        shard = plan.shards[0]

        assert FleetTopology().command(shard, repository="repo") == serve_command(
            shard, repository="repo"
        )
        assert "serve" in FleetTopology().command(shard, repository="repo")

    def test_every_child_is_told_which_topology_it_is_part_of(self) -> None:
        settings = ServerSettings(model_repository=Path("model_repository"))

        env = FleetTopology().environment(settings)

        assert env[TOPOLOGY_ENV] == "fleet"
        # Read back through the settings tree: one switch, not two.
        assert ServerSettings.model_fields["runner"] is not None
        assert TOPOLOGY_ENV == "SHIPINFER_RUNNER__RUNNER"

    def test_describe_is_one_line(self) -> None:
        assert "\n" not in FleetTopology().describe()
        assert FleetTopology().describe()


class TestTheFleetCommandGoesThroughTheTopology:
    def _config(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "SHIPINFER_INGEST__CAMERAS",
            json.dumps(
                [
                    {"camera_id": n, "uri": f"rtsp://host/{n}", "fps": f}
                    for n, f in CAMERAS.items()
                ]
            ),
        )
        monkeypatch.delenv("SHIPINFER_SHARD_CAMERAS", raising=False)

    def test_the_dry_run_names_the_topology(self, monkeypatch, capsys) -> None:
        from shipinfer.cli.commands.fleet import fleet

        self._config(monkeypatch)

        assert fleet(Path("model_repository"), shards=2, gpus="2,3", dry_run=True) == 0
        assert "topology: fleet" in capsys.readouterr().out

    def test_shards_default_to_one_per_visible_gpu(self, monkeypatch, capsys) -> None:
        from shipinfer.cli.commands.fleet import fleet

        self._config(monkeypatch)

        assert fleet(Path("model_repository"), gpus="2,3,4,5", dry_run=True) == 0
        out = capsys.readouterr().out
        assert out.count("shard ") >= 4 or "4 shard" in out

    def test_an_unknown_topology_stops_before_anything_is_planned(self, monkeypatch) -> None:
        from shipinfer.cli.commands.fleet import fleet

        self._config(monkeypatch)

        with pytest.raises(ConfigurationError, match="unknown topology"):
            fleet(Path("model_repository"), shards=2, gpus="2,3", topology="mesh", dry_run=True)

    def test_the_settings_section_selects_the_topology(self, monkeypatch, capsys) -> None:
        from shipinfer.cli.commands.fleet import fleet

        self._config(monkeypatch)
        monkeypatch.setenv(TOPOLOGY_ENV, "fleet")

        assert fleet(Path("model_repository"), shards=2, gpus="2,3", dry_run=True) == 0
        assert "topology: fleet" in capsys.readouterr().out

    def test_the_contract_is_what_the_launcher_receives(self, monkeypatch) -> None:
        """Not a dry run: `Fleet` is replaced, so the test sees what it would have been
        given — the topology's command and environment, not the module's own."""
        import importlib

        module = importlib.import_module("shipinfer.cli.commands.fleet")
        self._config(monkeypatch)
        monkeypatch.setattr(module, "require_container", lambda *_a, **_k: None)
        received: dict[str, object] = {}

        class _Fleet:
            def __init__(self, *, plan, command, env, shard_env, drain_s) -> None:
                received.update(
                    plan=plan, command=command, env=env, shard_env=shard_env, drain_s=drain_s
                )

            def start(self) -> None: ...

            def supervise(self, *_a, **_k) -> None: ...

            def stop(self, *_a, **_k) -> None: ...

        monkeypatch.setattr(module, "Fleet", _Fleet)
        monkeypatch.setattr(module, "forward_signals", lambda *_a, **_k: None)

        assert module.fleet(Path("model_repository"), shards=2, gpus="2,3", drain_s=7.0) == 0
        assert received["env"][TOPOLOGY_ENV] == "fleet"
        assert (
            received["shard_env"](received["plan"].shards[0]) == {}
        ), "fleet says nothing per shard"
        assert received["drain_s"] == 7.0
        shard = received["plan"].shards[0]
        assert received["command"](shard) == serve_command(shard, repository="model_repository")


class TestTheContractIsSmallOnPurpose:
    def test_a_topology_decides_plan_command_and_environment_only(self) -> None:
        abstract = set(Topology.__abstractmethods__)

        assert abstract == {"plan", "command"}
        assert callable(Topology.environment) and callable(Topology.describe)
