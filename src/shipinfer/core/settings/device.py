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
    #: Fail at start-up rather than at the first inference on a missing device.
    validate_on_start: bool = True
    #: Allow the server to come up with zero GPUs (CPU backends only). True in dev/tests.
    allow_cpu_only: bool = True
    #: Pin each worker thread to the CPU cores nearest its GPU. On a multi-socket box this
    #: is worth several percent of end-to-end latency, because a pinned-memory staging copy
    #: that crosses a NUMA node pays for it twice.
    numa_affinity: bool = False

    @field_validator("visible_gpus")
    @classmethod
    def _unique_and_sorted(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("visible_gpus contains duplicates")
        if any(i < 0 for i in value):
            raise ValueError("visible_gpus must be non-negative")
        return sorted(value)
