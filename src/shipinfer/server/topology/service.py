"""``service``: the fleet plus a cross-process inference tier for the crop-stage models."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import ClassVar

from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.topology import (
    SERVICE_PEERS_ENV,
    SERVICE_RUN_ENV,
    SERVICE_SHARD_ENV,
)
from shipinfer.scheduling.sharding import Shard, ShardPlan, plan_shards
from shipinfer.server.launcher import serve_command
from shipinfer.server.topology.base import TOPOLOGIES, Topology

__all__ = ["ServiceTopology"]


@TOPOLOGIES.register(
    "service",
    description="the fleet, plus every shard offering its crop-stage instances to its peers",
)
class ServiceTopology(Topology):
    """One process per shard, and a cross-process tier for the crop-stage models.

    This is topology **C** (ledger T3, `docs/design/topology-service.md`): the fleet's shape —
    decode, detect and track local to the GPU that decoded — with the stateless crop-stage
    models (`topology.service.shared_models`) served *symmetrically* across processes. Every
    shard keeps serving its own GPU's instances and also offers them to its peers through
    pinned shared-memory rings, so a dead process loses its K cameras and its capacity,
    nothing else; the dispatcher's candidate set is local instances plus a proxy per peer, and
    `locality_spillover` keeps work home while the local queue is shallow.

    The plan and the command are the fleet's. What differs is what the children are told: a
    run id that names the rings, every shard's index (the peers), and each child's own index.
    """

    name: ClassVar[str] = "service"

    def __init__(self) -> None:
        # One id per launch names this fleet's rings, so two fleets on one box cannot open each
        # other's memory and a restarted fleet cannot attach to a dead one's leftovers.
        self._run_id = uuid.uuid4().hex[:12]
        self._shards: int = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    def plan(
        self,
        settings: ServerSettings,
        *,
        cameras: Mapping[str, float],
        gpus: Sequence[int],
        shards: int,
    ) -> ShardPlan:
        plan = plan_shards(cameras, shards=shards, gpus=gpus)
        self._shards = len(plan)
        return plan

    def command(self, shard: Shard, *, repository: str) -> Sequence[str]:
        return serve_command(shard, repository=repository)

    def environment(self, settings: ServerSettings) -> Mapping[str, str]:
        base = dict(super().environment(settings))
        base[SERVICE_RUN_ENV] = self._run_id
        base[SERVICE_PEERS_ENV] = json.dumps(list(range(self._shards)))
        return base

    def shard_environment(self, shard: Shard) -> Mapping[str, str]:
        return {SERVICE_SHARD_ENV: str(shard.index)}

    def describe(self) -> str:
        return (
            "one process per shard; the crop-stage models are also served to peers through "
            "pinned shared-memory rings, so a busy shard borrows a quiet one's embedder"
        )
