"""The vocabulary of the whole server: data types, devices, memory kinds, tensors.

The most dependency-free package in the tree — numpy and the standard library, nothing
else. A tensor that lives in GPU memory is *described* here but never *allocated* here;
allocation belongs to :mod:`shipinfer.runtime.memory`, which this package knows only
through the :class:`MemoryHandle` protocol. That inversion is what keeps
``shipinfer.core`` importable on a host with no NVIDIA driver (ADR-001).
"""

from shipinfer.core.types.device import Device, DeviceKind
from shipinfer.core.types.dtype import DataType
from shipinfer.core.types.memory import MemoryHandle, MemoryKind
from shipinfer.core.types.spec import DYNAMIC, Shape, TensorSpec
from shipinfer.core.types.tensor import Tensor, stack_tensors, validate_against

__all__ = [
    "DYNAMIC",
    "DataType",
    "Device",
    "DeviceKind",
    "MemoryHandle",
    "MemoryKind",
    "Shape",
    "Tensor",
    "TensorSpec",
    "stack_tensors",
    "validate_against",
]
