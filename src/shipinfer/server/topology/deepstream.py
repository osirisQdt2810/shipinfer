"""``deepstream``: one NVIDIA DeepStream graph per shard, instead of one server per shard."""

from __future__ import annotations

import pathlib
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from shipinfer.core.errors import ConfigurationError, ModelNotFoundError
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.topology import (
    DEEPSTREAM_CONFIG_DIR_ENV,
    DEEPSTREAM_RUN_ENV,
    DEEPSTREAM_SHARD_ENV,
)
from shipinfer.scheduling.sharding import Shard, ShardPlan, plan_shards
from shipinfer.server.launcher import deepstream_command
from shipinfer.server.topology.base import TOPOLOGIES, Topology

__all__ = ["DeepStreamTopology"]


@TOPOLOGIES.register(
    "deepstream",
    description="one DeepStream graph per shard; decode, detect, track and embed on the GPU",
)
class DeepStreamTopology(Topology):
    """One process per shard, running a DeepStream graph rather than ``shipinfer serve``.

    This is topology **D** (ledger T4, `docs/design/topology-deepstream.md`): NVIDIA's graph
    in place of this project's pipeline, the same events out (V108: a first-class topology,
    not a competitor benchmark).
    `fleet` and `service` both run *this* server's scheduler over TensorRT engines; `deepstream`
    hands the whole per-frame path to NVIDIA's graph — ``nvurisrcbin -> nvstreammux ->
    nvinfer(detector) -> nvtracker -> nvinfer(embedders)`` — and keeps only the two ends: the
    model repository, which generates the nvinfer configs, and the result sink, which receives
    the same :class:`~shipinfer.pipeline.schema.PerceptionEvent` every other topology emits.
    One sink serves every topology, which is what makes the comparison a comparison.

    **One process per shard, one GPU per shard**, and deliberately not the reference's
    one-process-many-branches (`mtmc_deepstream.py`: two `nvstreammux` chains with per-element
    ``gpu-id`` in one pipeline). Three reasons, in order of how much they cost when ignored.
    :class:`~shipinfer.launch.Fleet` sets ``CUDA_VISIBLE_DEVICES`` *before the child's
    interpreter starts*, so the child sees exactly one device numbered 0 — per-element physical
    ``gpu-id`` values would then name devices it cannot see. A process that touches G devices
    holds G CUDA contexts, ~300 MiB each, for nothing. And one plugin segfault should cost K
    cameras, not fifty. A single-process many-branch variant, if it is ever wanted, is another
    file and another decorator; that is the point of the registry.

    The plan and the environment are the fleet's, plus a run id that names this launch's
    generated config directory. The *command* is the difference: a child that is not a server.
    """

    name: ClassVar[str] = "deepstream"

    def __init__(self) -> None:
        # One id per launch, as `service` does for its rings: two fleets on one box must not
        # write each other's generated nvinfer configs, and a restarted fleet must not read a
        # dead one's leftovers.
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
        self.adopt(plan)
        # The parent refuses a shard the primary GIE cannot bind ONCE, here, instead of
        # every child refusing identically at config generation (#32 round 4). The children
        # keep their own refusal — a hand-started child bypasses this method.
        limit = self._detector_batch_limit(settings)
        if limit is not None:
            for shard in plan.shards:
                if len(shard.cameras) > limit:
                    raise ConfigurationError(
                        f"shard {shard.index} muxes {len(shard.cameras)} cameras, and the "
                        f"detector's engine declares max_batch_size {limit} — nvstreammux's "
                        f"batch is the primary GIE's batch. Use more shards (at most {limit} "
                        f"cameras each), or rebuild the engine with a larger max_batch_size"
                    )
        return plan

    @staticmethod
    def _detector_batch_limit(settings: ServerSettings) -> int | None:
        """The detector's declared max batch, or None when the repository cannot say.

        Best-effort on purpose: a --dry-run on a control box may point at a repository
        whose configs exist but whose engines do not, and the *plan* is still worth
        printing — the child's config generation is the refusal of record.
        """
        if not pathlib.Path(settings.model_repository).is_dir():
            # The stated case, checked for what it is: a control box has no repository at
            # the configured path, and the plan is still worth printing.
            return None
        try:
            from shipinfer.repository import ModelRepository

            repository = ModelRepository.load(settings.model_repository)
            name = settings.topology.deepstream.detector
            return int(repository.entry(name).config.max_batch_size)
        except (ModelNotFoundError, OSError):
            # Only the other stated case — a repository directory without the configured
            # model (or one the process cannot read). A malformed config.yaml raises
            # ConfigurationError and surfaces here, once, as intended; the child's
            # generation remains the refusal of record (#32 round 6: a bare Exception
            # silently skipped the very check this method adds).
            return None

    def adopt(self, plan: ShardPlan) -> None:
        """Record the plan, and refuse one this topology cannot honour.

        Validated here rather than only in :meth:`plan` because a caller with an explicit plan
        — the benchmark harness splitting cameras unevenly on purpose — reaches the launcher
        through this method, and a plan that cannot run must fail before sixteen processes have
        started rather than inside a GStreamer element in each of them.
        """
        devices = {gpu for shard in plan.shards for gpu in shard.gpus}
        for shard in plan.shards:
            if len(shard.gpus) != 1:
                raise ConfigurationError(
                    f"the `deepstream` topology needs exactly one GPU per shard, but shard "
                    f"{shard.index} was given {list(shard.gpus)}. Every element in the graph "
                    f"takes a single `gpu-id`, and the child sees one logical device because "
                    f"CUDA_VISIBLE_DEVICES was set before its interpreter started — so a "
                    f"two-GPU shard would run the whole graph on the first of them while "
                    f"holding a context on both. Use --shards N with N >= {len(devices)} "
                    f"(at least one shard per GPU), or pass fewer --gpus"
                )
        self._shards = len(plan)

    def command(
        self, shard: Shard, *, repository: str, http_port_base: int | None = None
    ) -> Sequence[str]:
        if http_port_base is not None:
            raise ConfigurationError(
                "the `deepstream` topology has no HTTP API to expose: a shard runs a GStreamer "
                "graph, not an InferenceServer, so there is no model table and nothing for "
                f"port {http_port_base + shard.index} to serve. Drop --http-port-base, or run "
                "the `fleet` / `service` topology if the deployment needs KServe v2"
            )
        return deepstream_command(shard, repository=repository)

    def environment(self, settings: ServerSettings) -> Mapping[str, str]:
        base = dict(super().environment(settings))
        base[DEEPSTREAM_RUN_ENV] = self._run_id
        base[DEEPSTREAM_CONFIG_DIR_ENV] = str(self.config_dir(settings))
        return base

    def shard_environment(self, shard: Shard) -> Mapping[str, str]:
        return {DEEPSTREAM_SHARD_ENV: str(shard.index)}

    def config_dir(self, settings: ServerSettings) -> Path:
        """Where the children write their generated nvinfer, tracker and label files.

        Decided by the parent and handed down, so every shard's files land under one directory
        an operator can read, diff and paste into ``gst-launch-1.0``. Defaulted per *run* rather
        than to a fixed path for the reason the run id exists; never inside the model
        repository, which is scanned as a repository and refuses stray config files in a model
        directory outright.
        """
        configured = settings.topology.deepstream.config_dir
        if configured is not None:
            return Path(configured)
        return Path(tempfile.gettempdir()) / f"shipinfer-ds-{self._run_id}"

    def describe(self) -> str:
        return (
            "one DeepStream graph per shard: nvurisrcbin -> nvstreammux -> nvinfer(detector) "
            "-> nvtracker -> nvinfer(embedders); metadata leaves through a src-pad probe, "
            "frames never do"
        )
