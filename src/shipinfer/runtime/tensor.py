"""The bridge between :class:`shipinfer.core.types.Tensor` and ``torch.Tensor``.

``core.Tensor`` is the transport type: it crosses every seam, it is hashable-by-content for
the response cache, and it must stay importable without torch. ``torch.Tensor`` is the
*execution* type: it owns device memory out of torch's caching allocator, moves with
``non_blocking`` copies, and is what a CUDA graph captures.

This module is the only place the two meet, and it does so without copying wherever the
memory layout allows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from shipinfer.core.types import DataType, Device, MemoryKind, Tensor
from shipinfer.runtime.platform import require_torch

__all__ = [
    "batch_to_torch",
    "core_dtype",
    "from_torch",
    "to_torch",
    "to_torch_device",
    "torch_dtype",
]

#: Filled on first use so importing this module does not import torch.
_TORCH_BY_DATATYPE: dict[DataType, Any] = {}
_DATATYPE_BY_TORCH: dict[Any, DataType] = {}


def _dtype_tables() -> tuple[dict[DataType, Any], dict[Any, DataType]]:
    if not _TORCH_BY_DATATYPE:
        torch = require_torch()
        _TORCH_BY_DATATYPE.update(
            {
                DataType.BOOL: torch.bool,
                DataType.UINT8: torch.uint8,
                DataType.INT8: torch.int8,
                DataType.INT32: torch.int32,
                DataType.INT64: torch.int64,
                DataType.FP16: torch.float16,
                DataType.FP32: torch.float32,
                DataType.FP64: torch.float64,
            }
        )
        _DATATYPE_BY_TORCH.update({v: k for k, v in _TORCH_BY_DATATYPE.items()})
    return _TORCH_BY_DATATYPE, _DATATYPE_BY_TORCH


def torch_dtype(dtype: DataType) -> Any:
    return _dtype_tables()[0][dtype]


def core_dtype(dtype: Any) -> DataType:
    table = _dtype_tables()[1]
    try:
        return table[dtype]
    except KeyError:
        raise ValueError(f"unsupported torch dtype for inference I/O: {dtype}") from None


def to_torch_device(device: Device) -> Any:
    torch = require_torch()
    return torch.device("cpu") if device.kind == "cpu" else torch.device("cuda", device.index)


def to_torch(tensor: Tensor, device: Device | None = None, *, non_blocking: bool = True) -> Any:
    """Materialise a ``core.Tensor`` as a torch tensor on ``device``.

    ``torch.from_numpy`` shares the buffer, so a host tensor costs nothing until it moves.
    The device copy is issued ``non_blocking`` on the current stream, which is only
    actually asynchronous when the source is pinned — hence
    :class:`~shipinfer.runtime.memory.PinnedStagingPool`.
    """
    torch = require_torch()
    if tensor.host is None:
        raise ValueError("cannot bridge a device-resident core.Tensor without a host view")
    result = torch.from_numpy(tensor.host)
    if device is None or device.kind == "cpu":
        return result
    return result.to(to_torch_device(device), non_blocking=non_blocking)


def from_torch(tensor: Any, *, copy_to_host: bool = True) -> Tensor:
    """Wrap a torch tensor as a ``core.Tensor``.

    A CUDA tensor is brought back to the host because ``core.Tensor`` promises a numpy view
    to whoever holds it. The copy is explicit and visible here rather than hidden behind an
    attribute access — a synchronising D2H that happens implicitly at an arbitrary point is
    the classic way a pipeline loses its overlap.
    """
    detached = tensor.detach()
    if detached.is_cuda:
        if not copy_to_host:
            raise ValueError("device tensors must be copied to the host to become core.Tensors")
        detached = detached.to("cpu", non_blocking=False)
    array = np.ascontiguousarray(detached.numpy())
    return Tensor(
        dtype=core_dtype(tensor.dtype),
        shape=tuple(array.shape),
        device=Device.cpu(),
        memory_kind=MemoryKind.HOST,
        host=array,
    )


def batch_to_torch(
    inputs: Mapping[str, Tensor], device: Device, *, non_blocking: bool = True
) -> dict[str, Any]:
    """Move a whole input map to a device in one pass."""
    return {name: to_torch(t, device, non_blocking=non_blocking) for name, t in inputs.items()}
