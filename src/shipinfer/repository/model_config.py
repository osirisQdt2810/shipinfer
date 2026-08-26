"""Per-model configuration — the ``config.yaml`` that mirrors Triton's ``config.pbtxt``.

The shape of this file is a deliberate act of compatibility: ``max_batch_size``,
``instance_group``, ``dynamic_batching`` and the input/output ``dims`` mean exactly what
they mean in Triton, so a config can be ported in either direction by hand. Where this
project adds something Triton has no word for (``locality``, ``priority_levels``) the key
is new rather than an overloaded reinterpretation of an existing one.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.types import DYNAMIC, DataType, Device, TensorSpec

__all__ = [
    "DynamicBatchingConfig",
    "EnsembleConfig",
    "EnsembleStep",
    "IOConfig",
    "InstanceGroup",
    "InstanceKind",
    "InstancePlacement",
    "ModelConfig",
    "RateLimiterConfig",
    "VersionPolicy",
    "WarmupInput",
    "WarmupSample",
    "load_model_config",
]


class InstanceKind(str, enum.Enum):
    GPU = "KIND_GPU"
    CPU = "KIND_CPU"
    #: Let the server pick: GPU if any is visible, else CPU. Useful for a model that has
    #: both a TensorRT and a numpy implementation.
    AUTO = "KIND_AUTO"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IOConfig(_Strict):
    """One declared input or output.

    ``dims`` excludes the batch dimension when ``max_batch_size > 0`` — the same rule
    Triton uses, and the reason a detector's input reads ``[3, 640, 640]`` rather than
    ``[-1, 3, 640, 640]``.
    """

    name: str
    data_type: DataType
    dims: list[int]
    optional: bool = False

    @field_validator("dims")
    @classmethod
    def _valid_dims(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("dims must have at least one entry (use [1] for a scalar)")
        for dim in value:
            if dim == 0 or dim < DYNAMIC:
                raise ValueError(f"invalid dim {dim}: use a positive extent or -1 for dynamic")
        return value

    def to_spec(self) -> TensorSpec:
        return TensorSpec(
            name=self.name, dtype=self.data_type, shape=tuple(self.dims), optional=self.optional
        )


class InstancePlacement(_Strict):
    """One concrete instance the server will create: a device plus its stream count."""

    device: Device
    #: CUDA streams (TensorRT execution contexts) inside this instance, so H2D, compute
    #: and D2H of consecutive batches overlap instead of serialising.
    streams: int = 1

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class InstanceGroup(_Strict):
    """A rule that expands into N instances — Triton's ``instance_group``.

    Semantics match Triton exactly: ``count`` is *per device*, so ``count: 2`` over
    ``gpus: [0, 1]`` creates four instances. An empty ``gpus`` means "every visible GPU",
    which is what makes one config file work unchanged on 2 GPUs and on 16.
    """

    kind: InstanceKind = InstanceKind.AUTO
    count: int = Field(default=1, ge=1)
    gpus: list[int] = Field(default_factory=list)
    streams: int = Field(default=1, ge=1)
    name: str = ""

    @model_validator(mode="after")
    def _cpu_groups_have_no_gpus(self) -> InstanceGroup:
        if self.kind is InstanceKind.CPU and self.gpus:
            raise ValueError("a KIND_CPU instance group cannot name gpus")
        return self

    def expand(
        self,
        visible_gpus: Sequence[int],
        shared_by: Mapping[int, int] | None = None,
        share_rank: Mapping[int, int] | None = None,
    ) -> list[InstancePlacement]:
        """Materialise the group against the GPUs this process can actually see.

        Args:
            visible_gpus: device indices the :class:`~shipinfer.runtime.device.DeviceManager`
                reports. Empty means the host has none.
            shared_by: how many processes share each visible device (absent means one). A
                shard of a fleet that shares its GPU loads its *share* of ``count`` there —
                divided, not repeated, because TensorRT contexts are per-process and two shards
                each loading the full count would double the device's engines and VRAM for the
                same total throughput.
            share_rank: this process's rank among those sharing each device (absent means 0).
                A count that does not divide evenly gives its remainder to the lowest ranks:
                ``count: 3`` over two processes is 2 + 1, so the device carries the three the
                config asked for rather than the two that floor division would leave it.

        Raises:
            ConfigurationError: if the group demands a GPU that is not visible — failing
                here beats a confusing CUDA error on the first inference — or if the share
                on some device rounds to zero: silently rounding down produces a shard that
                accepts frames and can never execute one, which looks like a throughput
                result rather than a misconfiguration.
        """
        kind = self.kind
        if kind is InstanceKind.AUTO:
            kind = InstanceKind.GPU if visible_gpus else InstanceKind.CPU

        if kind is InstanceKind.CPU:
            return [
                InstancePlacement(device=Device.cpu(), streams=1) for _ in range(self.count)
            ]

        targets = list(self.gpus) if self.gpus else list(visible_gpus)
        if not targets:
            raise ConfigurationError(
                "instance group needs a GPU but none are visible; "
                "set kind: KIND_CPU, or make devices visible"
            )
        missing = sorted(set(targets) - set(visible_gpus))
        if missing:
            raise ConfigurationError(
                f"instance group requests gpu(s) {missing} which are not visible "
                f"(visible: {sorted(visible_gpus)})"
            )
        placements: list[InstancePlacement] = []
        for gpu in targets:
            sharing = (shared_by or {}).get(gpu, 1)
            rank = (share_rank or {}).get(gpu, 0)
            share = self.count // sharing + (1 if rank < self.count % sharing else 0)
            if share < 1:
                raise ConfigurationError(
                    f"{sharing} processes share gpu {gpu} but only {self.count} instance(s) "
                    f"per gpu are configured, so the process ranked {rank} would get none. Use "
                    f"at most {self.count} shards per gpu, or raise the instance count"
                )
            placements.extend(
                InstancePlacement(device=Device.cuda(gpu), streams=self.streams)
                for _ in range(share)
            )
        return placements


class DynamicBatchingConfig(_Strict):
    """Server-side batching window.

    The trade is the whole point: waiting ``max_queue_delay_us`` costs every request that
    latency but lets the GPU run one large kernel launch instead of many small ones. At 50
    cameras x 20 fps a 5 ms window fills a batch of 32 without being visible end-to-end.
    """

    enabled: bool = True
    max_queue_delay_us: int = Field(default=5_000, ge=0)
    #: Sizes the batcher will stop early at (a TensorRT engine is usually profiled for a
    #: few shapes; hitting them avoids a reshape). Empty means "any size up to the max".
    preferred_batch_sizes: list[int] = Field(default_factory=list)
    #: Emit responses in arrival order. Costs a little throughput; required only when a
    #: downstream stage assumes ordering, which in this pipeline it does not.
    preserve_ordering: bool = False
    #: Number of priority classes honoured inside the queue (see core.request.Priority).
    priority_levels: int = Field(default=len(list(range(4))), ge=1, le=8)

    @field_validator("preferred_batch_sizes")
    @classmethod
    def _sorted_positive(cls, value: list[int]) -> list[int]:
        if any(v <= 0 for v in value):
            raise ValueError("preferred_batch_sizes must be positive")
        return sorted(set(value))


class RateLimiterConfig(_Strict):
    """A bound on how many of this model's instances may execute at once.

    Different from ``scheduler.max_queue_size``, and the difference is why this exists:
    the queue bound says how much work may be *waiting*, this one says how much may be
    *running*. Eight instances each holding a full queue all enter compute the moment their
    batching windows close, and they share a memory bus, a PCIe root complex and — on a
    shared box — the devices themselves.

    Triton spells this ``rate_limiter { resources [...] }`` with a general named-resource
    model. This is the same idea with the generality left out, because the only resource
    this pipeline has ever needed to bound is "an execution".

    Per model rather than per deployment because the answer differs per model: a detector
    that owns its GPU wants no bound, a segmenter sharing one with three others does.

    Args:
        kind: a name registered in :data:`shipinfer.scheduling.limits.RATE_LIMITERS`. A
            plain string, not an enum, so a limiter shipped by another package is
            selectable without editing this file — and so ``repository`` never imports
            ``scheduling``. An unknown name fails when the model is built, with the
            registry listing what it could have been.
        max_concurrent_executions: the bound. Meaningless for ``off``, and setting it there
            is refused rather than ignored: a config that states a bound the server does not
            apply is worse than one that states none.
    """

    kind: str = "off"
    max_concurrent_executions: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _bound_matches_kind(self) -> RateLimiterConfig:
        off = self.kind in ("off", "none", "unlimited")
        if off and self.max_concurrent_executions:
            raise ValueError(
                "rate_limiter.max_concurrent_executions is set but kind is "
                f"{self.kind!r}, which applies no bound; set kind: concurrency"
            )
        if not off and self.max_concurrent_executions < 1:
            raise ValueError(
                f"rate_limiter kind {self.kind!r} needs max_concurrent_executions >= 1"
            )
        return self

    @property
    def enabled(self) -> bool:
        return self.max_concurrent_executions > 0


class WarmupInput(_Strict):
    """Where one warm-up tensor's bytes come from — Triton's ``model_warmup`` input.

    Exactly one source, because the alternative is a precedence rule nobody remembers:

    * ``zero_data`` — zeros, which is what the implicit warm-up already does;
    * ``random_data`` — uniform noise, for a model whose kernels branch on the data (an NMS
      that finds no boxes in a zero image never runs its sort);
    * ``input_data_file`` — a real sample, relative to the model's **version directory**.
      This is the one Triton exists to offer and the one that matters here: the fused
      preprocessing and the detector's post-processing take a different path on a frame with
      ships in it than on a frame of zeros, and only the real path can be warmed.

    ``dims`` is needed only when the model declares a dynamic extent, because there is
    nothing to infer then. Otherwise the model's own declaration is the single source of
    truth and repeating it here is a second place to get it wrong.
    """

    zero_data: bool = False
    random_data: bool = False
    input_data_file: str | None = None
    dims: list[int] | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> WarmupInput:
        chosen = [self.zero_data, self.random_data, self.input_data_file is not None]
        if sum(chosen) != 1:
            raise ValueError(
                "a warm-up input needs exactly one of zero_data / random_data / "
                "input_data_file"
            )
        if self.dims is not None and any(dim < 1 for dim in self.dims):
            raise ValueError(f"warm-up dims must be positive extents, got {self.dims}")
        return self


class WarmupSample(_Strict):
    """One named warm-up batch, run ``count`` times before the instance reports ready.

    Triton's ``model_warmup``, same field names. The reason to prefer this over an
    iteration count is that a count only decides *how often*; the sample decides *what*,
    and what is what selects the kernels. A TensorRT engine picks tactics per shape, CUDA
    modules load lazily per kernel, and a detector that never sees a box during warm-up has
    not warmed its NMS.

    Args:
        name: what an operator sees in the log when this sample fails. Required, because
            "warm-up sample 2 failed" in a config with four samples is not a diagnosis.
        batch_size: rows in the batch. Warm the shape you serve — capturing a graph for
            batch 8 and warming at batch 1 warms the wrong tactic.
        count: how many times to run it. Two or three is enough; the first execution pays
            the lazy costs and the rest confirm they are paid.
        inputs: one entry per declared model input, keyed by tensor name.
    """

    name: str = Field(min_length=1)
    batch_size: int = Field(default=1, ge=1)
    count: int = Field(default=1, ge=1)
    inputs: dict[str, WarmupInput] = Field(default_factory=dict)


class VersionPolicy(_Strict):
    """Which versions of a model to load. Mirrors Triton's three policies."""

    #: Load the N highest-numbered versions.
    latest: int | None = 1
    #: Load every version present on disk.
    all: bool = False
    #: Load exactly these versions.
    specific: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exactly_one(self) -> VersionPolicy:
        chosen = [self.latest is not None, self.all, bool(self.specific)]
        if sum(chosen) != 1:
            raise ValueError("version_policy: set exactly one of latest / all / specific")
        return self

    def select(self, available: Sequence[int]) -> list[int]:
        ordered = sorted(available)
        if self.all:
            return ordered
        if self.specific:
            missing = sorted(set(self.specific) - set(ordered))
            if missing:
                raise ConfigurationError(
                    f"version_policy.specific names missing versions {missing}"
                )
            return sorted(self.specific)
        return ordered[-(self.latest or 1) :]


class EnsembleStep(_Strict):
    """One node of an ensemble DAG.

    ``input_map`` / ``output_map`` rename between the ensemble's own tensor namespace and
    the step model's, exactly as in Triton — which is what lets ``ship_detector`` be reused
    unchanged by two different pipelines.
    """

    model: str
    model_version: int | None = None
    input_map: dict[str, str] = Field(default_factory=dict)
    output_map: dict[str, str] = Field(default_factory=dict)
    #: Run this step only when the named boolean/count tensor is non-empty. This is how
    #: "segment only where a ship was detected" is expressed declaratively instead of in
    #: a hand-written orchestration loop.
    condition: str | None = None


class EnsembleConfig(_Strict):
    steps: list[EnsembleStep]

    @field_validator("steps")
    @classmethod
    def _non_empty(cls, value: list[EnsembleStep]) -> list[EnsembleStep]:
        if not value:
            raise ValueError("an ensemble needs at least one step")
        return value


class ModelConfig(_Strict):
    """The complete description of one servable model."""

    name: str
    #: Which backend executes it: ``tensorrt``, ``torch``, ``onnx``, ``python``, ``mock``
    #: or ``ensemble``. Resolved through :mod:`shipinfer.backends.registry`.
    platform: str
    #: 0 disables server-side batching entirely (the model handles its own batch dim).
    max_batch_size: int = Field(default=0, ge=0)
    inputs: list[IOConfig] = Field(default_factory=list)
    outputs: list[IOConfig] = Field(default_factory=list)
    instance_groups: list[InstanceGroup] = Field(default_factory=lambda: [InstanceGroup()])
    dynamic_batching: DynamicBatchingConfig = Field(default_factory=DynamicBatchingConfig)
    version_policy: VersionPolicy = Field(default_factory=VersionPolicy)
    ensemble: EnsembleConfig | None = None
    #: Backend-specific knobs (engine filename, precision, warm-up iterations, ...). Kept
    #: opaque here so adding a backend never means editing this file.
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: Batches sent through every instance at load time so the first real request does not
    #: pay for lazy CUDA module loading and TensorRT's first-call allocations.
    warmup_batches: int = Field(default=2, ge=0)
    #: Named sample batches run at load time instead of the zero-filled implicit warm-up.
    #: Present means "warm the model with *these*"; empty keeps the old behaviour, driven by
    #: ``execution.warmup_iterations``. See :class:`WarmupSample` for why the data matters.
    model_warmup: list[WarmupSample] = Field(default_factory=list)
    #: How many of this model's instances may be executing at once. Off by default.
    rate_limiter: RateLimiterConfig = Field(default_factory=RateLimiterConfig)

    @model_validator(mode="after")
    def _coherent(self) -> ModelConfig:
        if self.platform == "ensemble":
            if self.ensemble is None:
                raise ValueError("platform 'ensemble' requires an `ensemble:` section")
        elif not self.inputs or not self.outputs:
            raise ValueError(f"model {self.name!r}: declare at least one input and one output")

        if self.max_batch_size == 0 and self.dynamic_batching.enabled:
            raise ValueError(f"model {self.name!r}: dynamic_batching needs max_batch_size > 0")
        for size in self.dynamic_batching.preferred_batch_sizes:
            if size > self.max_batch_size:
                raise ValueError(
                    f"model {self.name!r}: preferred batch size {size} exceeds "
                    f"max_batch_size {self.max_batch_size}"
                )
        self._validate_warmup()
        names = [io.name for io in (*self.inputs, *self.outputs)]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"model {self.name!r}: duplicate tensor name(s) {sorted(duplicates)}"
            )
        return self

    def _validate_warmup(self) -> None:
        """Reject a warm-up sample that could not run, at load rather than at start-up.

        Every complaint here is one an operator would otherwise meet as a failed instance
        several seconds into a deploy, with the config file the last place they would look.
        """
        if not self.model_warmup:
            return
        if self.is_ensemble:
            raise ValueError(
                f"model {self.name!r}: model_warmup does not apply to an ensemble — warm "
                "the step models instead, which is where the kernels are"
            )
        declared = {io.name: io for io in self.inputs}
        seen: set[str] = set()
        for sample in self.model_warmup:
            if sample.name in seen:
                raise ValueError(
                    f"model {self.name!r}: duplicate model_warmup name {sample.name!r}"
                )
            seen.add(sample.name)
            if sample.batch_size > self.effective_max_batch_size:
                raise ValueError(
                    f"model {self.name!r}: warm-up sample {sample.name!r} asks for batch "
                    f"{sample.batch_size}, above max_batch_size "
                    f"{self.effective_max_batch_size}"
                )
            unknown = sorted(set(sample.inputs) - set(declared))
            if unknown:
                raise ValueError(
                    f"model {self.name!r}: warm-up sample {sample.name!r} names "
                    f"input(s) {unknown} the model does not declare"
                )
            missing = sorted(
                io.name
                for io in self.inputs
                if not io.optional and io.name not in sample.inputs
            )
            if missing:
                raise ValueError(
                    f"model {self.name!r}: warm-up sample {sample.name!r} is missing "
                    f"required input(s) {missing}"
                )
            for tensor_name, warmup_input in sample.inputs.items():
                dims = warmup_input.dims or declared[tensor_name].dims
                if DYNAMIC in dims:
                    raise ValueError(
                        f"model {self.name!r}: warm-up input {tensor_name!r} in sample "
                        f"{sample.name!r} has a dynamic extent {dims}; give explicit dims"
                    )

    # -- derived views ---------------------------------------------------------------

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return tuple(io.to_spec() for io in self.inputs)

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        return tuple(io.to_spec() for io in self.outputs)

    @property
    def is_ensemble(self) -> bool:
        return self.platform == "ensemble"

    @property
    def effective_max_batch_size(self) -> int:
        """1 when server-side batching is off, so callers need no special case."""
        return self.max_batch_size or 1

    def placements(
        self,
        visible_gpus: Sequence[int],
        shared_by: Mapping[int, int] | None = None,
        share_rank: Mapping[int, int] | None = None,
    ) -> list[InstancePlacement]:
        """Every instance this model wants, flattened across its groups.

        ``shared_by`` and ``share_rank`` are this process's share of each device — see
        :meth:`InstanceGroup.expand`.
        """
        out: list[InstancePlacement] = []
        for group in self.instance_groups:
            out.extend(group.expand(visible_gpus, shared_by, share_rank))
        if not out:
            raise ConfigurationError(f"model {self.name!r} expands to zero instances")
        return out

    def iter_devices(self, visible_gpus: Sequence[int]) -> Iterator[Device]:
        seen: set[Device] = set()
        for placement in self.placements(visible_gpus):
            if placement.device not in seen:
                seen.add(placement.device)
                yield placement.device


def load_model_config(path: Path, *, name_hint: str | None = None) -> ModelConfig:
    """Read and validate one ``config.yaml``.

    Args:
        path: the file to read.
        name_hint: directory name to fall back on when the file omits ``name``, matching
            Triton's behaviour where the directory is authoritative.

    Raises:
        ConfigurationError: with the file path attached — a validation error with no path
            is nearly useless when a repository holds thirty models.
    """
    try:
        raw: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigurationError(f"cannot read model config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"{path}: expected a mapping at the top level")

    data = dict(raw)
    data.setdefault("name", name_hint or path.parent.name)
    try:
        return ModelConfig(**data)
    except Exception as exc:  # pydantic ValidationError, or our own ValueError
        raise ConfigurationError(f"{path}: {exc}") from exc
