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

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["DeepStreamSettings", "ServiceSettings", "TopologySettings"]


class ServiceSettings(BaseModel):
    """`service`: the fleet plus a cross-process inference tier for the crop-stage models.

    Every shard keeps serving its own GPU's instances of the shared models *and* offers them to
    its peers through pinned shared-memory rings (`runtime/memory/shared_ring.py`). The
    per-child keys (`shard`, `peers`, `run_id`) are set by the launcher, never by an operator.
    """

    model_config = ConfigDict(extra="forbid")

    #: The models a shard offers to its peers: crops, never frames — a 1080p frame is 6 MB and
    #: would triple the pinned footprint; the detector stays local by design. The segmenter is
    #: not in the default either: its batch is 8 x 3 x 640 x 640 fp32 = 39 MB, so its rings would
    #: dwarf the embedders' — whether to pay that is the operator's call (T3's open question).
    shared_models: list[str] = Field(
        default_factory=lambda: ["person_embedder", "ship_embedder"]
    )
    #: Slots per (submitter, owner, model) ring. Small on purpose: the rings are pairwise, so
    #: four shards and three models are 24 rings each way, and every slot is pinned.
    slots_per_pair: int = Field(default=8, ge=1)
    #: The fallback slot size, used only when a model declares a dynamic extent. Otherwise slots
    #: are sized per model *and per direction* from the model's own config
    #: (`service_mesh.wire_slot_bytes`): max_batch x the tensors' bytes plus 64 KiB for the heads,
    #: which travel in the slot ahead of the bytes, page-rounded. A request that does not fit is
    #: refused before a byte moves.
    slot_bytes: int = Field(default=1_638_400, ge=4096)
    #: How long a submit waits for a free slot before the ring is called full.
    submit_timeout_ms: float = Field(default=5.0, gt=0.0)
    #: How often an owner stamps its ring headers, and after how many missed stamps a peer is
    #: lost. 200 ms and 1 s: one missed stamp is a scheduler hiccup, five is a dead process.
    heartbeat_ms: float = Field(default=200.0, gt=0.0)
    lost_after_ms: float = Field(default=1000.0, gt=0.0)
    #: How long a pending remote request may wait for its reply before it fails with a
    #: timeout — the bound on how long a stranded WorkItem can pin its inputs.
    pending_timeout_ms: float = Field(default=60_000.0, gt=0.0)
    #: How long a starting shard waits for its peers' rings to appear.
    connect_timeout_s: float = Field(default=60.0, gt=0.0)

    #: Set by the launcher for each child: this shard's index, every shard's index, and the
    #: run id that names the rings. `None` / empty in a single-process `serve`, where there is
    #: no tier to join.
    shard: int | None = Field(default=None, ge=0)
    peers: list[int] = Field(default_factory=list)
    run_id: str = ""


class DeepStreamSettings(BaseModel):
    """`deepstream`: one DeepStream GStreamer graph per shard instead of one server.

    The fourth topology (T4, V108 — a first-class pipeline implementation, not a competitor
    benchmark). A shard's child process is not ``shipinfer serve`` at all: it
    is ``nvurisrcbin -> nvstreammux -> nvinfer -> nvtracker -> nvinfer(sgies)``, with decode,
    detection, tracking and embedding all inside one NVIDIA-owned graph and only *metadata*
    leaving it, through a pad probe. Everything below is what that graph needs to be generated
    from this repository's own model configs rather than hand-written per deployment.

    Two of the fields are set by the launcher and never by an operator (`run_id`, `shard`), the
    way :class:`ServiceSettings` does it: they name this launch's generated config directory and
    tell a child which shard it is.
    """

    model_config = ConfigDict(extra="forbid")

    #: The primary GIE — one nvinfer over every muxed frame. A model in the repository, so its
    #: engine, its input dims and its output names come from `config.yaml` rather than from a
    #: second description of the same model in a `.txt` an operator maintains by hand.
    detector: str = "ship_detector"
    #: The secondary GIEs, in order. Each runs over the objects the detector produced, on the
    #: crop the tracker is following. The segmenter and the recogniser are **not** here in PR1
    #: (see `docs/design/topology-deepstream.md`), which is why every event this topology emits
    #: names them in `missing_stages` rather than pretending the frame was complete.
    secondaries: list[str] = Field(default_factory=lambda: ["person_embedder", "ship_embedder"])
    #: Which detector **labels** each secondary runs on. Labels, not class ids: an id is a
    #: property of the checkpoint and changes when the detector is retrained, so it is resolved
    #: through `pipeline.class_labels` at config-generation time — the same map the Python DAG
    #: branches on. Two secondaries claiming one label is refused there: an object would carry
    #: two output tensors and whichever the probe read first would silently win.
    operate_on: dict[str, list[str]] = Field(
        default_factory=lambda: {"person_embedder": ["person"], "ship_embedder": ["ship"]}
    )

    #: The muxed frame every camera is scaled into. One resolution for the whole batch is
    #: nvstreammux's contract, and it is also the coordinate space every box comes back in —
    #: `pipeline.deepstream.probe.FrameGeometry` is what converts them back to source pixels.
    mux_width: int = Field(default=1920, ge=16)
    mux_height: int = Field(default=1080, ge=16)
    #: How long the muxer waits for a slow camera before pushing a short batch. 40 ms is two
    #: frames at 25 fps: long enough to fill a batch, short enough that one dark camera cannot
    #: hold the other K back for a visible fraction of a second.
    mux_batched_push_timeout_us: int = Field(default=40_000, ge=1)
    #: Letterbox into the muxed frame instead of stretching. Off by default because the fleet is
    #: 16:9 and the scale is then exact; on, the probe undoes the padding as well as the scale.
    mux_enable_padding: bool = False
    #: Live sources are never waited for: the muxer must not block the batch on a camera that
    #: has stopped delivering. False only for file replay, where dropping frames is not the point.
    live_source: bool = True

    #: nvinfer's precision selector, in *its* numbering order (0=fp32, 1=int8, 2=fp16). Only
    #: honoured when nvinfer builds the engine itself from `onnx-file`; a prebuilt `model.plan`
    #: already has its precision baked in and this is then documentation.
    network_mode: Literal["fp32", "int8", "fp16"] = "fp16"
    #: Skip inference on N of every N+1 batches. 0 is every frame, which is what a perception
    #: deployment wants; it exists because it is the first knob an operator reaches for when a
    #: GPU is short, and finding it here beats finding it in a hand-edited generated file.
    interval: int = Field(default=0, ge=0)

    #: The custom bounding-box parser for the detector's output layout, and the `.so` it lives
    #: in. Both empty by default and both required for an end-to-end detector such as yolo26:
    #: nvinfer's built-in parsers cannot read a single decoded `(300, 6)` tensor, and config
    #: generation refuses rather than emitting a config that fails at start-up.
    bbox_parser: str = ""
    custom_lib: str = ""

    #: The low-level tracker library and its config. The library ships with DeepStream; the
    #: config is generated when this is unset, and an operator who wants NVIDIA's tuned NvDCF
    #: numbers points this at the SDK's own `config_tracker_NvDCF_perf.yml`.
    tracker_lib: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    tracker_config: Path | None = None
    #: The frame the tracker works in. Multiples of 32, which is what the low-level library
    #: requires; a wrong extent is a start-up failure inside a `.so`, so it is refused here.
    tracker_width: int = Field(default=960, ge=32)
    tracker_height: int = Field(default=544, ge=32)

    #: Where the generated nvinfer/tracker/label files are written. ``None`` means a per-run
    #: directory under `$TMPDIR`. **Never** inside the model repository: that directory is
    #: scanned as a repository, and it refuses a `config.pbtxt` in a model directory outright.
    config_dir: Path | None = None

    #: Set by the launcher, like `service.run_id` / `service.shard`. The run id names this
    #: launch's config directory so two fleets on one box cannot overwrite each other's
    #: generated files, and the shard index names the subdirectory inside it.
    run_id: str = ""
    shard: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _the_parser_and_its_library_travel_together(self) -> DeepStreamSettings:
        if bool(self.bbox_parser) != bool(self.custom_lib):
            given, missing = (
                ("bbox_parser", "custom_lib")
                if self.bbox_parser
                else ("custom_lib", "bbox_parser")
            )
            raise ValueError(
                f"{given} is set but {missing} is not: nvinfer dlsym()s the parse function "
                f"out of the custom library, so one without the other fails with "
                f"NVDSINFER_CUSTOM_LIB_FAILED inside the GStreamer element on every shard, "
                f"seconds into the deploy. Set both or neither"
            )
        return self

    @model_validator(mode="after")
    def _secondaries_are_the_ones_operated_on(self) -> DeepStreamSettings:
        unknown = sorted(set(self.operate_on) - set(self.secondaries))
        if unknown:
            raise ValueError(
                f"operate_on names {unknown}, which is not in secondaries {self.secondaries}: "
                "a class filter for a GIE that does not run configures nothing"
            )
        unfiltered = sorted(set(self.secondaries) - set(self.operate_on))
        if unfiltered:
            # nvinfer runs an UNFILTERED secondary on every class, so an object would carry
            # two embeddings and the probe publishes whichever tensor comes first in the
            # user-meta list — a person with a ship's embedding, at full confidence, on
            # every frame (#32 round 1 reproduced it). Claiming labels is not optional.
            raise ValueError(
                f"secondaries {unfiltered} have no operate_on entry: an unfiltered GIE runs "
                "on every class and its embedding silently displaces the right one. Name the "
                "labels each secondary operates on"
            )
        return self

    @model_validator(mode="after")
    def _tracker_extents_are_multiples_of_32(self) -> DeepStreamSettings:
        bad = {
            name: value
            for name, value in (
                ("tracker_width", self.tracker_width),
                ("tracker_height", self.tracker_height),
            )
            if value % 32
        }
        if bad:
            raise ValueError(
                f"tracker extents must be multiples of 32, got {bad}: the low-level tracker "
                "library refuses anything else, several seconds into a deploy"
            )
        return self


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
    #: The `deepstream` topology's knobs; unused by the other three.
    deepstream: DeepStreamSettings = Field(default_factory=DeepStreamSettings)


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
#: The `deepstream` topology's per-child keys (`topology.deepstream.*`). The run id and the
#: config directory are fleet-wide; the shard index is per child. Same rule as `service`: the
#: launcher sets them, an operator does not.
DEEPSTREAM_RUN_ENV = "SHIPINFER_TOPOLOGY__DEEPSTREAM__RUN_ID"
DEEPSTREAM_SHARD_ENV = "SHIPINFER_TOPOLOGY__DEEPSTREAM__SHARD"
DEEPSTREAM_CONFIG_DIR_ENV = "SHIPINFER_TOPOLOGY__DEEPSTREAM__CONFIG_DIR"
