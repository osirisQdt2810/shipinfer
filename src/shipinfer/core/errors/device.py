"""Device-level failures."""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError

__all__ = ["DeviceError", "DeviceOutOfMemoryError"]


class DeviceError(ShipInferError):
    """A CUDA/HIP call failed, or a requested device does not exist."""


class DeviceOutOfMemoryError(DeviceError):
    """An allocation on a device failed for want of memory.

    Its own type because the operational response differs: OOM means "reduce
    max_batch_size or instance count", while a generic device error usually means
    "the driver or the engine is wrong".
    """
