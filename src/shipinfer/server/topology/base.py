"""The `Topology` contract: how a deployment is laid out into processes.

WHAT A TOPOLOGY DECIDES
-----------------------
Three things, and only three. **The plan** — which cameras and which GPUs each process owns,
a :class:`~shipinfer.scheduling.sharding.ShardPlan`, pure and printable before anything is
spawned. **The command** — the argv one shard's process runs. **The environment** — what every
child must have before it imports torch, on top of what :class:`~shipinfer.launch.Fleet`
already sets per shard (``CUDA_VISIBLE_DEVICES`` and the shard's cameras).

Everything else — supervision, draining, the process group, signal forwarding — is `Fleet`,
and is the same for every topology. That is the reason the contract is this small: the
topologies differ in *where work crosses process boundaries*, and that is decided by what the
children are told, not by how they are supervised.

THE SHAPE, AND WHERE IT COMES FROM (V77)
----------------------------------------
vLLM's data-parallel serving launches one engine process per rank with the rank's devices in
its environment at process creation (`vllm/v1/engine/utils.py`, `CoreEngineProcManager`), and
a coordinator publishes each engine's queue lengths so a front-end can pick the least-loaded
one (`DPCoordinator`, `DPLBAsyncMPClient.get_core_engine_for_request`). Triton keeps one queue
per model that instances on several GPUs pull from (`instance_group`, `DynamicBatchScheduler`).
`fleet` is the first of those without the second: static balance by the plan, everything local.
`service` (ledger T3) adds the second — every shard serves its GPU's crop-stage instances to
its peers, so a crop goes to whichever GPU is free — and `deepstream` (T4) runs NVIDIA's
graph in place of this project's pipeline, same events out.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from typing import ClassVar

from shipinfer.core.registry import Registry
from shipinfer.core.settings import ServerSettings
from shipinfer.scheduling.sharding import Shard, ShardPlan

__all__ = ["TOPOLOGIES", "Topology"]

#: The environment variable that tells a child which topology it is part of. Read by the
#: child through the settings tree (``SHIPINFER_RUNNER__RUNNER``), so a shard started by hand
#: with the same variable behaves the same way — there is one switch, not two.
TOPOLOGY_ENV = "SHIPINFER_RUNNER__RUNNER"


class Topology(abc.ABC):
    """One way of laying a deployment out into processes. Subclass, register, done."""

    #: The registered name; set by the subclass and echoed into every child's environment.
    name: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def plan(
        self,
        settings: ServerSettings,
        *,
        cameras: Mapping[str, float],
        gpus: Sequence[int],
        shards: int,
    ) -> ShardPlan:
        """Who owns which cameras and which GPUs. Pure; printed before anything is spawned."""

    @abc.abstractmethod
    def command(
        self, shard: Shard, *, repository: str, http_port_base: int | None = None
    ) -> Sequence[str]:
        """The argv for ``shard``'s process.

        ``http_port_base`` set means every shard also serves HTTP, on ``base + shard.index``;
        unset means a warm shard with no ingress, which is `serve`'s own default.
        """

    def environment(self, settings: ServerSettings) -> Mapping[str, str]:
        """Extra environment for every child, on top of what `Fleet` sets per shard.

        The default is the one thing every topology needs: its own name, so the child builds
        the same topology the parent planned. A topology that needs more — the peers' ring
        names, a coordinator address — extends this.
        """
        return {TOPOLOGY_ENV: self.name}

    def adopt(self, plan: ShardPlan) -> None:
        """Be told the plan that will be launched, when someone else made it.

        :meth:`plan` calls this on its own result. A caller with an explicit plan — the
        benchmark harness splitting cameras unevenly on purpose — hands it here so a topology
        whose environment depends on the plan (`service`: the peer set) still describes the
        fleet that actually starts. The default needs nothing.
        """

    def shard_environment(self, shard: Shard) -> Mapping[str, str]:
        """Extra environment for *one* child, on top of :meth:`environment`.

        The default is nothing: `fleet` tells a shard everything it needs through the plan
        (`Fleet` sets the devices and the cameras). A topology whose children must find each
        other — `service` names its rings by shard index — extends this.
        """
        return {}

    def describe(self) -> str:
        """One line for the operator, printed above the plan."""
        return (self.__doc__ or self.name).strip().split("\n")[0]


TOPOLOGIES: Registry[Topology] = Registry("topology", Topology)
