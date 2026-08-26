"""How the deployment is laid out into processes, and how work crosses between them.

The three-plane architecture (ADR-014 and the operator's target, ledger Phase 7) puts the
stateful streaming work — decode, detect-local, track — in a process pinned to one GPU, and the
stateless crop work behind a queue whichever free instance pulls from. *Where* those processes
are and *how* a crop reaches an instance on another GPU is the **topology**, and it is a
registry (`shipinfer.server.topology.TOPOLOGIES`) for the same reason a placement policy is:
adding one is a new file and a decorator, never a branch.

This section is the switch, env-overridable like the rest of the tree — ``SHIPINFER_TOPOLOGY__KIND``.
The one per-child value that is *not* a setting — which cameras a shard reads — is declared in
`shipinfer.envs` (``SHARD_CAMERAS``) with the other non-settings variables, so it has a typed
parse and a `describe()` entry for ``shipinfer doctor``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TopologySettings"]


class TopologySettings(BaseModel):
    """Which topology ``shipinfer fleet`` runs, and its knobs."""

    model_config = ConfigDict(extra="forbid")

    #: A name registered in `shipinfer.server.topology.TOPOLOGIES`. ``fleet`` is one process
    #: per shard with everything local to its GPU — static balance by the plan, the topology
    #: the multi-process launcher shipped with. Validated against the registry when the
    #: topology is built, not here: settings import no server code.
    kind: str = "fleet"
    #: How many processes to split the cameras across. ``None`` means one per visible GPU,
    #: which is the process-per-GPU shape (ADR-006) and the right answer unless the operator
    #: knows otherwise.
    shards: int | None = Field(default=None, ge=1)
    #: Seconds a shard gets after SIGTERM before SIGKILL.
    drain_s: float = Field(default=20.0, gt=0.0)


#: The settings-tree keys a fleet launcher sets for each shard process. Defined here, beside the
#: settings, because both sides — the launcher that sets them and the settings loader that
#: reads them — must agree on the spelling.
#:
#: The shard's *logical* device list after ``CUDA_VISIBLE_DEVICES`` renumbered its GPUs —
#: ``[0, 1]`` for a two-GPU shard, whatever the physical ordinals. Set by the launcher so an
#: inherited ``SHIPINFER_DEVICES__VISIBLE_GPUS`` naming physical ordinals cannot survive the
#: remap and fail the child at start-up.
VISIBLE_GPUS_ENV = "SHIPINFER_DEVICES__VISIBLE_GPUS"
#: How many shard processes share each of the shard's devices, aligned with the logical
#: ordinals. Two shards on one GPU must each load *half* the configured instances, or the
#: device holds twice the engines and twice the VRAM for the same total throughput.
SHARED_BY_ENV = "SHIPINFER_DEVICES__SHARED_BY"
#: The shard's rank among the processes sharing each device, aligned with the ordinals. The
#: remainder of a count that does not divide evenly goes to the lowest ranks.
SHARE_RANK_ENV = "SHIPINFER_DEVICES__SHARE_RANK"
