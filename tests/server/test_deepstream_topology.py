"""`deepstream` registers against the topology seam, and its children are not servers.

Topology D is the competitor: the same plan, the same environment discipline, a completely
different child. What is worth pinning here is exactly the three things a topology decides —
the plan, the command and the environment — plus the two refusals that keep this one honest: a
shard with more than one GPU, and an HTTP port for a process that serves no API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.topology import (
    DEEPSTREAM_CONFIG_DIR_ENV,
    DEEPSTREAM_RUN_ENV,
    DEEPSTREAM_SHARD_ENV,
)
from shipinfer.scheduling.sharding import Shard, ShardPlan, plan_shards
from shipinfer.server.launcher import deepstream_command
from shipinfer.server.topology import TOPOLOGIES, DeepStreamTopology, build_topology
from shipinfer.server.topology.base import TOPOLOGY_ENV

CAMERAS = {f"cam{i}": 20.0 for i in range(8)}
GPUS = (0, 1, 2, 3)


class TestTheRegistryIsTheSwitch:
    def test_deepstream_is_a_registered_topology(self) -> None:
        assert "deepstream" in TOPOLOGIES
        assert isinstance(build_topology("deepstream"), DeepStreamTopology)
        assert build_topology("deepstream").name == "deepstream"

    def test_it_is_listed_beside_the_others(self) -> None:
        assert {"fleet", "service", "deepstream"} <= set(TOPOLOGIES.names())

    def test_the_settings_read_what_an_operator_and_the_launcher_set(self, monkeypatch) -> None:
        monkeypatch.setenv(TOPOLOGY_ENV, "deepstream")
        monkeypatch.setenv("SHIPINFER_TOPOLOGY__DEEPSTREAM__MUX_WIDTH", "1280")
        monkeypatch.setenv("SHIPINFER_TOPOLOGY__DEEPSTREAM__NETWORK_MODE", "int8")
        monkeypatch.setenv(DEEPSTREAM_RUN_ENV, "abc123def456")
        monkeypatch.setenv(DEEPSTREAM_SHARD_ENV, "2")

        settings = ServerSettings(model_repository=Path("model_repository"))

        assert settings.topology.kind == "deepstream"
        assert settings.topology.deepstream.mux_width == 1280
        assert settings.topology.deepstream.network_mode == "int8"
        assert settings.topology.deepstream.run_id == "abc123def456"
        assert settings.topology.deepstream.shard == 2

    def test_the_defaults_are_the_shipped_dag(self) -> None:
        deepstream = ServerSettings().topology.deepstream

        assert deepstream.detector == "ship_detector"
        assert deepstream.secondaries == ["person_embedder", "ship_embedder"]
        assert deepstream.operate_on == {
            "person_embedder": ["person"],
            "ship_embedder": ["ship"],
        }

    def test_a_class_filter_for_a_gie_that_does_not_run_is_refused(self) -> None:
        with pytest.raises(ValueError, match="operate_on names"):
            ServerSettings(
                topology={
                    "kind": "deepstream",
                    "deepstream": {
                        "secondaries": ["person_embedder"],
                        "operate_on": {"ship_embedder": ["ship"]},
                    },
                }
            )

    def test_a_tracker_extent_the_library_refuses_is_refused_here(self) -> None:
        with pytest.raises(ValueError, match="multiples of 32"):
            ServerSettings(topology={"deepstream": {"tracker_width": 961}})


class TestThePlanIsTheFleetsPlusOneRefusal:
    def test_the_plan_is_plan_shards(self) -> None:
        topology = DeepStreamTopology()

        plan = topology.plan(ServerSettings(), cameras=CAMERAS, gpus=GPUS, shards=4)

        assert plan == plan_shards(CAMERAS, shards=4, gpus=GPUS)
        assert plan.cameras == tuple(sorted(CAMERAS))

    def test_a_shard_with_two_gpus_is_refused_naming_gpu_id_and_the_fix(self) -> None:
        """Fewer shards than GPUs gives a shard two devices, and every element in a DeepStream
        graph takes one `gpu-id`."""
        topology = DeepStreamTopology()

        with pytest.raises(ConfigurationError, match=r"gpu-id.*--shards") as excinfo:
            topology.plan(ServerSettings(), cameras=CAMERAS, gpus=GPUS, shards=2)

        # The fix names the number of GPUs, not the number of shards that just failed.
        assert "N >= 4" in str(excinfo.value)

    def test_an_explicit_plan_is_adopted_and_validated_the_same_way(self) -> None:
        """The harness splits cameras unevenly and hands the topology the plan; the one-GPU
        rule has to hold there too, or a shard fails inside a GStreamer element instead."""
        good = ShardPlan(
            shards=(
                Shard(index=0, cameras=("a", "b"), gpus=(2,), offered_fps=40.0),
                Shard(index=1, cameras=("c",), gpus=(3,), offered_fps=20.0),
            ),
            shards_per_gpu={2: 1, 3: 1},
        )
        topology = DeepStreamTopology()
        topology.adopt(good)  # keeps it, says nothing

        bad = ShardPlan(
            shards=(Shard(index=0, cameras=("a",), gpus=(2, 3), offered_fps=20.0),),
            shards_per_gpu={2: 1, 3: 1},
        )
        with pytest.raises(ConfigurationError, match="exactly one GPU per shard"):
            DeepStreamTopology().adopt(bad)


class TestTheChildIsNotAServer:
    def test_the_command_is_the_deepstream_subcommand_on_the_same_repository(self) -> None:
        plan = plan_shards(CAMERAS, shards=4, gpus=GPUS)
        shard = plan.shards[0]

        argv = DeepStreamTopology().command(shard, repository="repo")

        assert list(argv) == [sys.executable, "-m", "shipinfer", "deepstream", "-r", "repo"]
        assert list(argv) == deepstream_command(shard, repository="repo")

    def test_the_command_carries_no_cameras_no_gpus_and_no_port(self) -> None:
        """All three travel in the environment, or do not exist at all — see `serve_command`."""
        shard = plan_shards(CAMERAS, shards=4, gpus=GPUS).shards[1]

        argv = list(DeepStreamTopology().command(shard, repository="repo"))

        assert not any(a.startswith("--port") or a == "--http" for a in argv)
        assert not any(a in {"--gpus", "--cameras"} for a in argv)

    def test_asking_a_deepstream_fleet_for_http_is_refused_not_ignored(self) -> None:
        shard = plan_shards(CAMERAS, shards=4, gpus=GPUS).shards[0]

        with pytest.raises(ConfigurationError, match="no HTTP API"):
            DeepStreamTopology().command(shard, repository="repo", http_port_base=8000)


class TestTheChildrenAreToldWhereToWrite:
    def test_the_environment_names_the_topology_the_run_and_the_config_dir(self) -> None:
        topology = DeepStreamTopology()

        env = topology.environment(ServerSettings())

        assert env[TOPOLOGY_ENV] == "deepstream"
        assert env[DEEPSTREAM_RUN_ENV] == topology.run_id and len(topology.run_id) == 12
        assert topology.run_id in env[DEEPSTREAM_CONFIG_DIR_ENV]

    def test_two_launches_never_share_a_run_id_or_a_config_dir(self) -> None:
        one, two = DeepStreamTopology(), DeepStreamTopology()

        assert one.run_id != two.run_id
        assert one.config_dir(ServerSettings()) != two.config_dir(ServerSettings())

    def test_a_configured_directory_wins_over_the_per_run_default(self, tmp_path) -> None:
        settings = ServerSettings(topology={"deepstream": {"config_dir": str(tmp_path)}})

        assert DeepStreamTopology().config_dir(settings) == tmp_path

    def test_the_config_dir_is_never_inside_the_model_repository(self) -> None:
        """`ModelRepository` refuses a stray config in a model directory; the default must not
        put one there."""
        settings = ServerSettings(model_repository=Path("model_repository"))

        directory = DeepStreamTopology().config_dir(settings)

        assert not str(directory).startswith(str(Path("model_repository").resolve()))

    def test_each_child_is_told_its_own_index(self) -> None:
        topology = DeepStreamTopology()
        plan = topology.plan(ServerSettings(), cameras=CAMERAS, gpus=GPUS, shards=4)

        assert [topology.shard_environment(s) for s in plan.shards] == [
            {DEEPSTREAM_SHARD_ENV: str(index)} for index in range(4)
        ]

    def test_describe_is_one_line_and_names_the_probe(self) -> None:
        described = DeepStreamTopology().describe()

        assert "\n" not in described
        assert "probe" in described and "nvstreammux" in described


class TestTheFleetCommandRunsIt:
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

    def test_the_dry_run_names_the_topology_and_prints_the_plan(
        self, monkeypatch, capsys
    ) -> None:
        from shipinfer.cli.commands.fleet import fleet

        self._config(monkeypatch)

        assert (
            fleet(
                Path("model_repository"),
                topology="deepstream",
                gpus="0,1,2,3",
                dry_run=True,
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "topology: deepstream" in out
        assert "4 shard(s)" in out

    def test_more_gpus_than_shards_stops_before_anything_is_spawned(self, monkeypatch) -> None:
        from shipinfer.cli.commands.fleet import fleet

        self._config(monkeypatch)

        with pytest.raises(ConfigurationError, match="exactly one GPU per shard"):
            fleet(
                Path("model_repository"),
                shards=2,
                topology="deepstream",
                gpus="0,1,2,3",
                dry_run=True,
            )
