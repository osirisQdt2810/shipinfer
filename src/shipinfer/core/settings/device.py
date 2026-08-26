"""Which accelerators this process may touch."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["DeviceSettings"]


class DeviceSettings(BaseModel):
    """Device visibility and start-up validation."""

    model_config = ConfigDict(extra="forbid")

    #: Empty means "every device the driver reports". An explicit list is how one process
    #: is pinned to one GPU in the process-per-GPU deployment (ADR-006).
    visible_gpus: list[int] = Field(default_factory=list)
    #: How many processes share each visible device, aligned with ``visible_gpus`` (or with
    #: the driver's ordinals when that is empty). Empty means one process per device. Set by
    #: the fleet launcher for a shard that shares its GPU, so the shard loads its *share* of
    #: every model's ``instance_group.count`` rather than the whole count — two shards each
    #: loading the full count would double the device's engines and VRAM for the same total.
    shared_by: list[int] = Field(default_factory=list)
    #: This process's rank (0-based) among the processes sharing each visible device, aligned
    #: with ``shared_by``. A count that does not divide by the sharing gives its remainder to
    #: the lowest ranks, so the device still carries every instance the config asked for.
    share_rank: list[int] = Field(default_factory=list)
    #: Fail at start-up rather than at the first inference on a missing device.
    validate_on_start: bool = True
    #: Allow the server to come up with zero GPUs (CPU backends only). True in dev/tests.
    allow_cpu_only: bool = True
    #: Pin each worker thread to the CPU cores nearest its GPU. On a multi-socket box this
    #: is worth several percent of end-to-end latency, because a pinned-memory staging copy
    #: that crosses a NUMA node pays for it twice.
    numa_affinity: bool = False

    @field_validator("shared_by")
    @classmethod
    def _positive(cls, value: list[int]) -> list[int]:
        if any(i < 1 for i in value):
            raise ValueError("shared_by entries must be at least 1 (one process per device)")
        return value

    @field_validator("share_rank")
    @classmethod
    def _non_negative(cls, value: list[int]) -> list[int]:
        if any(i < 0 for i in value):
            raise ValueError("share_rank entries must be 0 or more")
        return value

    @field_validator("visible_gpus")
    @classmethod
    def _unique_and_sorted(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("visible_gpus contains duplicates")
        if any(i < 0 for i in value):
            raise ValueError("visible_gpus must be non-negative")
        return sorted(value)
