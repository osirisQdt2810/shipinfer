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

__all__ = ["ServiceSettings", "TopologySettings"]


class ServiceSettings(BaseModel):
    """`service`: the fleet plus a cross-process inference tier for the crop-stage models.

    Every shard keeps serving its own GPU's instances of the shared models *and* offers them to
    its peers through pinned shared-memory rings (`runtime/memory/shared_ring.py`). The
    per-child keys (`shard`, `peers`, `run_id`) are set by the launcher, never by an operator.
    """

    model_config = ConfigDict(extra="forbid")

    #: The models a shard offers to its peers: crops, never frames — a 1080p frame is 6 MB and
    #: would triple the pinned footprint; the detector stays local by design.
    shared_models: list[str] = Field(
        default_factory=lambda: ["person_embedder", "ship_embedder", "ship_segmenter"]
    )
    #: Slots per (submitter, owner, model) ring. Small on purpose: the rings are pairwise, so
    #: four shards and three models are 24 rings each way, and every slot is pinned.
    slots_per_pair: int = Field(default=8, ge=1)
    #: Bytes per slot: one crop batch plus its heads. 1.5 MiB is exactly 32 crops of 3 x 128 x 64
    #: fp16; the extra 64 KiB (400 pages in all) carries the request head and the per-tensor heads,
    #: which travel in the slot ahead of the bytes. A request that does not fit is refused before a
    #: byte moves; per-model slot sizes are an open question (T3).
    slot_bytes: int = Field(default=1_638_400, ge=4096)
    #: How long a submit waits for a free slot before the ring is called full.
    submit_timeout_ms: float = Field(default=5.0, gt=0.0)
    #: How often an owner stamps its ring headers, and after how many missed stamps a peer is
    #: lost. 200 ms and 1 s: one missed stamp is a scheduler hiccup, five is a dead process.
    heartbeat_ms: float = Field(default=200.0, gt=0.0)
    lost_after_ms: float = Field(default=1000.0, gt=0.0)
    #: How long a starting shard waits for its peers' rings to appear.
    connect_timeout_s: float = Field(default=60.0, gt=0.0)

    #: Set by the launcher for each child: this shard's index, every shard's index, and the
    #: run id that names the rings. `None` / empty in a single-process `serve`, where there is
    #: no tier to join.
    shard: int | None = Field(default=None, ge=0)
    peers: list[int] = Field(default_factory=list)
    run_id: str = ""


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
    #: The `service` topology's knobs; unused by `fleet`.
    service: ServiceSettings = Field(default_factory=ServiceSettings)


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
#: The `service` topology's per-child keys, settings-tree spellings (`topology.service.*`).
SERVICE_SHARD_ENV = "SHIPINFER_TOPOLOGY__SERVICE__SHARD"
SERVICE_PEERS_ENV = "SHIPINFER_TOPOLOGY__SERVICE__PEERS"
SERVICE_RUN_ENV = "SHIPINFER_TOPOLOGY__SERVICE__RUN_ID"
