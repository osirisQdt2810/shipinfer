"""`service` registers against the topology seam and tells its children how to find each other."""

from __future__ import annotations

import json

from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.topology import (
    SERVICE_PEERS_ENV,
    SERVICE_RUN_ENV,
    SERVICE_SHARD_ENV,
)
from shipinfer.server.launcher import Fleet
from shipinfer.server.topology import TOPOLOGIES, ServiceTopology, build_topology
from shipinfer.server.topology.base import TOPOLOGY_ENV

CAMERAS = {f"cam{i}": 20.0 for i in range(8)}


class TestTheRegistryIsTheSwitch:
    def test_service_is_a_registered_topology(self) -> None:
        assert "service" in TOPOLOGIES.names()
        assert isinstance(build_topology("service"), ServiceTopology)

    def test_the_plan_is_the_fleets(self) -> None:
        topology = ServiceTopology()
        plan = topology.plan(ServerSettings(), cameras=CAMERAS, gpus=[2, 3], shards=2)
        assert len(plan) == 2 and plan.cameras == tuple(sorted(CAMERAS))


class TestTheChildrenAreToldHowToFindEachOther:
    def test_the_fleet_wide_environment_names_the_run_and_the_peers(self) -> None:
        topology = ServiceTopology()
        topology.plan(ServerSettings(), cameras=CAMERAS, gpus=[2, 3, 4], shards=3)
        env = topology.environment(ServerSettings())
        assert env[TOPOLOGY_ENV] == "service"
        assert env[SERVICE_RUN_ENV] == topology.run_id and len(topology.run_id) == 12
        assert json.loads(env[SERVICE_PEERS_ENV]) == [0, 1, 2]

    def test_two_launches_never_share_a_run_id(self) -> None:
        assert ServiceTopology().run_id != ServiceTopology().run_id

    def test_each_child_is_told_its_own_index(self) -> None:
        topology = ServiceTopology()
        plan = topology.plan(ServerSettings(), cameras=CAMERAS, gpus=[2, 3], shards=2)
        assert [topology.shard_environment(s) for s in plan.shards] == [
            {SERVICE_SHARD_ENV: "0"},
            {SERVICE_SHARD_ENV: "1"},
        ]

    def test_the_settings_read_what_the_launcher_sets(self, monkeypatch) -> None:
        monkeypatch.setenv(TOPOLOGY_ENV, "service")
        monkeypatch.setenv(SERVICE_RUN_ENV, "abc123")
        monkeypatch.setenv(SERVICE_PEERS_ENV, "[0, 1]")
        monkeypatch.setenv(SERVICE_SHARD_ENV, "1")
        settings = ServerSettings()
        assert settings.topology.kind == "service"
        assert (settings.topology.service.run_id, settings.topology.service.peers) == (
            "abc123",
            [0, 1],
        )
        assert settings.topology.service.shard == 1

    def test_the_fleet_carries_a_per_shard_environment(self) -> None:
        """`Fleet` takes the topology's per-shard hook so `service` can name each child."""
        topology = ServiceTopology()
        plan = topology.plan(ServerSettings(), cameras=CAMERAS, gpus=[2, 3], shards=2)
        fleet = Fleet(
            plan=plan, command=lambda shard: ["true"], shard_env=topology.shard_environment
        )
        assert fleet.shard_env(plan.shards[1]) == {SERVICE_SHARD_ENV: "1"}

    def test_fleet_says_nothing_per_shard(self) -> None:
        from shipinfer.server.topology import FleetTopology

        plan = FleetTopology().plan(ServerSettings(), cameras=CAMERAS, gpus=[2], shards=1)
        assert FleetTopology().shard_environment(plan.shards[0]) == {}

    def test_describe_says_what_crosses(self) -> None:
        assert "rings" in ServiceTopology().describe()
