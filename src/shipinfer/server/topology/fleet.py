"""``fleet``: one process per shard, everything local to its GPU — static balance by the plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from shipinfer.core.settings import ServerSettings
from shipinfer.scheduling.sharding import Shard, ShardPlan, plan_shards
from shipinfer.server.launcher import serve_command
from shipinfer.server.topology.base import TOPOLOGIES, Topology

__all__ = ["FleetTopology"]


@TOPOLOGIES.register("fleet", description="one process per shard; every stage local to its GPU")
class FleetTopology(Topology):
    """One process per shard; every stage of a frame runs on the GPU that decoded it.

    This is topology **B** in the design discussion (ledger Phase 7): the launcher's original
    shape, now behind the contract so the next two are a file and a decorator each. Balance is
    *static* — `plan_shards` balances offered load across shards, largest camera first — and a
    shard that becomes busy at 15:00 cannot borrow another shard's embedder. What that costs
    under skew is the measurement that sizes `service` (T2 → T3).
    """

    name: ClassVar[str] = "fleet"

    def plan(
        self,
        settings: ServerSettings,
        *,
        cameras: Mapping[str, float],
        gpus: Sequence[int],
        shards: int,
    ) -> ShardPlan:
        return plan_shards(cameras, shards=shards, gpus=gpus)

    def command(
        self, shard: Shard, *, repository: str, http_port_base: int | None = None
    ) -> Sequence[str]:
        return serve_command(shard, repository=repository, http_port_base=http_port_base)
