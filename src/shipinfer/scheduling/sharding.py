"""Splitting the fleet across OS processes — who owns which camera, and which GPU.

WHY THIS EXISTS
---------------
Threads were not the problem the way the design assumed. ``CLAUDE.md`` predicted the GIL
would not bind because every worker thread spends its time inside TensorRT or a CUDA memcpy,
both of which release it — and that prediction is half right. Measured on the dev box, a
saturated run uses **390-534% CPU**: five cores of forty-eight, not one, so the C extensions
really are running in parallel. But five of forty-eight, while every model queue sits empty
and only the pipeline queue grows, is the signature of the *pure-Python* share of the
per-frame path holding the GIL and capping the whole process at about five cores' worth of
frames.

Four candidates were eliminated by measurement before this one was reached:

===========================  ==========================================================
not the GPUs                 at 120 img/s offered the detector retired 119.9 with a CI
                             straddling zero; every model queue SUSTAINED
not the worker pool          24 / 96 / 192 workers -> 87.6 / 81.4 / 85.0 img/s: an 8x
                             range for under 8%, and not monotonic
not the reassembly lock      98.9% of its hold removed (770 -> 8.7 us/frame), no change
not the load generator       it delivered 100% of the offered 120 img/s while the
                             pipeline retired 77
===========================  ==========================================================

What is left is one interpreter. So the fix is more interpreters, which is what
``references/bitbucket-subfaceid/docs/new-system-architecture.md`` section 9 already
specifies — ``[Decode procs]``, plural — and what this module makes decidable.

**This was once deleted as superseded by the C++ data plane, and that was half right.** The
C++ plane removed the GIL from the *per-frame* work and took the pipeline from 77 to 390
img/s. What it did not do is let one deployment use a whole machine: that 390 is still **one
process**, and the box has 48 cores. The two fixes answer different questions — "is the
interpreter in the hot path?" and "is one process using the machine?" — and vLLM, which has
the same problem, answers the second one exactly this way: its `MultiprocExecutor` spawns
`context.Process` per GPU worker and talks over ZMQ, with threads reserved for auxiliary work
like KV offload and engine monitoring. Forty-one of its files touch `multiprocessing`; none of
its twenty-one `threading.Thread` uses is on the model-execution path.

So this is not dead code waiting for a use. It is the second half of the answer to a
measurement we already have.

WHAT A SHARD OWNS
-----------------
A shard is one OS process holding a slice of the fleet: its own cameras, its own
``InferenceServer``, its own ``PipelineRunner``, its own model instances on its own GPUs.
Nothing is shared, because nothing needs to be: cameras are independent of each other all
the way to MTMC, which is a separate plane that consumes tracklets rather than frames.

Two decisions, both pure, both here:

1. **Which cameras.** Stable, because ingest is *stateful* per camera (ADR-011): a camera's
   reconnect backoff, frame-id watermark and health history live in its actor, so moving a
   camera between shards on a restart throws that away. And balanced by offered load rather
   than by count, because the failure this project exists to fix is a busy camera starving a
   quiet one — putting twenty 30 fps cameras in one shard and twenty 5 fps cameras in another
   would rebuild that failure one level up.

2. **Which GPUs.** Round-robin, and deliberately **not** one shard per GPU. The bottleneck
   measured above is CPU, not GPU, so the useful number of shards is set by cores and by
   per-frame Python cost — it can exceed the GPU count, and then several shards share a
   device. That is why :meth:`ShardPlan.sharing_for` exists: the launcher tells each child how
   many shards share its GPUs, and the child loads its *share* of every model's instances —
   two shards on one GPU each loading the full count would give the device twice the engines
   and twice the VRAM for the same total throughput.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shipinfer.core.errors import ConfigurationError

__all__ = ["Shard", "ShardPlan", "plan_shards"]


@dataclass(frozen=True, slots=True)
class Shard:
    """One process's share of the fleet."""

    #: 0-based, and stable: shard 3 owns the same cameras on the next start-up.
    index: int
    #: Camera ids this shard reads. Sorted, so a log or a diff is comparable across runs.
    cameras: tuple[str, ...]
    #: **Physical** CUDA ordinals — what goes in ``CUDA_VISIBLE_DEVICES`` for this process.
    #: Physical rather than logical because the child sets that variable itself, before it
    #: imports torch, and at that moment no restriction is in force yet.
    gpus: tuple[int, ...]
    #: Sum of the offered fps of this shard's cameras. Carried for the report: a plan whose
    #: shards differ by 40% in offered load is a plan worth looking at again.
    offered_fps: float

    @property
    def cuda_visible_devices(self) -> str:
        """What this shard's process must export before anything imports torch."""
        return ",".join(str(g) for g in self.gpus)


@dataclass(frozen=True, slots=True)
class ShardPlan:
    """Every shard, plus the questions a launcher and a report need to ask of the whole."""

    shards: tuple[Shard, ...]
    #: How many shards share each physical GPU. Keyed by physical ordinal.
    shards_per_gpu: Mapping[int, int]

    def __len__(self) -> int:
        return len(self.shards)

    @property
    def cameras(self) -> tuple[str, ...]:
        """Every camera in the plan, sorted. Exists to be asserted against the input."""
        return tuple(sorted(c for shard in self.shards for c in shard.cameras))

    @property
    def imbalance(self) -> float:
        """``(max - min) / max`` over per-shard offered load, or 0.0 for one shard.

        The number the plan is judged by. A shard is a process and a process is a wall: the
        fleet's throughput is bounded by its *busiest* shard, so an imbalanced plan wastes
        exactly the capacity it looks like it added.
        """
        if len(self.shards) < 2:
            return 0.0
        loads = [s.offered_fps for s in self.shards]
        high = max(loads)
        return 0.0 if high <= 0 else (high - min(loads)) / high

    def sharing_for(self, shard: Shard) -> tuple[int, ...]:
        """How many shards share each of ``shard``'s GPUs, in the shard's device order.

        What the launcher hands the child as ``SHIPINFER_DEVICES__SHARED_BY``, aligned with
        the logical ordinals the child sees after ``CUDA_VISIBLE_DEVICES`` renumbers its
        devices. The child divides every model's configured instance count by it where the
        instance groups expand (``InstanceGroup.expand``): two shards on one GPU must each
        load *half* the instances, or the device ends up with twice the engines and twice the
        VRAM for the same total throughput. The division — and the refusal when a share
        rounds to zero — lives there, at the one place that knows the per-model count.
        """
        return tuple(self.shards_per_gpu.get(gpu, 1) for gpu in shard.gpus)

    def describe(self) -> str:
        """One line per shard — what the launcher prints before it spawns anything."""
        lines = [
            f"{len(self.shards)} shard(s), imbalance {self.imbalance:.1%}",
        ]
        lines.extend(
            f"  shard {shard.index}: {len(shard.cameras)} camera(s), "
            f"{shard.offered_fps:g} fps offered, gpu(s) {list(shard.gpus)}"
            for shard in self.shards
        )
        return "\n".join(lines)


def plan_shards(
    cameras: Mapping[str, float] | Sequence[str],
    *,
    shards: int,
    gpus: Sequence[int],
) -> ShardPlan:
    """Assign cameras and GPUs to ``shards`` processes.

    Args:
        cameras: camera id -> offered fps, or just the ids (then every camera counts as 1.0
            and the balance is by count, which is the right answer for a uniform fleet).
        shards: how many processes. One is legal and means "no split" — the plan still
            describes it, so a caller has one code path rather than two.
        gpus: physical CUDA ordinals available, in the order they should be handed out.

    **Cameras are balanced greedily, largest first.** Longest-processing-time-first: sort by
    offered load descending, and give each camera to the shard that is currently lightest.
    That is the standard makespan heuristic, it is within 4/3 of optimal, and it beats
    round-robin exactly where it matters — a fleet of a few 30 fps cameras and many 5 fps ones
    is the shape that made round-robin put all the busy ones together.

    Ties break on camera id so the plan is **deterministic**: the same fleet produces the
    same assignment on every start-up, which is what makes a per-camera actor's state worth
    keeping.

    Raises:
        ConfigurationError: no cameras, no GPUs, fewer than one shard, or more shards than
            cameras — the last because a shard with no cameras is a process that starts, loads
            engines, holds a CUDA context and reads nothing.
    """
    load = dict(cameras) if isinstance(cameras, Mapping) else dict.fromkeys(cameras, 1.0)
    if not load:
        raise ConfigurationError("cannot plan shards for an empty fleet")
    if shards < 1:
        raise ConfigurationError(f"a plan needs at least one shard, got {shards}")
    if not gpus:
        raise ConfigurationError("cannot plan shards with no gpus")
    if shards > len(load):
        raise ConfigurationError(
            f"{shards} shards for {len(load)} camera(s): a shard with no cameras still "
            f"loads engines and holds a CUDA context. Use at most {len(load)} shards."
        )

    buckets: list[list[str]] = [[] for _ in range(shards)]
    weights = [0.0] * shards
    # Descending by load, then by name: the first key does the balancing and the second makes
    # it reproducible.
    for name in sorted(load, key=lambda n: (-load[n], n)):
        target = min(range(shards), key=lambda i: (weights[i], i))
        buckets[target].append(name)
        weights[target] += load[name]

    ordinals = list(gpus)
    assigned: list[tuple[int, ...]] = []
    sharing: dict[int, int] = {}
    for index in range(shards):
        if shards >= len(ordinals):
            # More shards than GPUs (or exactly as many): one device each, round-robin, so
            # the sharing is as even as the two counts allow.
            mine = (ordinals[index % len(ordinals)],)
        else:
            # Fewer shards than GPUs: contiguous groups, so a shard's devices are as close
            # together as the topology allows and no device is left idle.
            per = len(ordinals) // shards
            extra = len(ordinals) % shards
            start = index * per + min(index, extra)
            mine = tuple(ordinals[start : start + per + (1 if index < extra else 0)])
        assigned.append(mine)
        for ordinal in mine:
            sharing[ordinal] = sharing.get(ordinal, 0) + 1

    return ShardPlan(
        shards=tuple(
            Shard(
                index=index,
                cameras=tuple(sorted(buckets[index])),
                gpus=assigned[index],
                offered_fps=weights[index],
            )
            for index in range(shards)
        ),
        shards_per_gpu=sharing,
    )
