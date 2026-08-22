"""Persistent I/O buffers for one TensorRT execution context — allocated as torch tensors.

Using ``torch.empty(..., device="cuda:i")`` rather than raw ``cudaMalloc`` is a deliberate
choice and it buys three things at once (ADR-003):

* allocation comes from torch's **caching** allocator, so creating and dropping buffers is
  a free-list operation rather than a synchronising driver call;
* the allocator is **graph-aware**, so a buffer that participates in a CUDA graph capture is
  not later handed to a different stream — the failure mode that makes hand-rolled graph
  capture produce silently wrong results;
* the buffers are ordinary tensors, so staging a batch is ``copy_(..., non_blocking=True)``
  instead of a hand-written ``cudaMemcpyAsync`` with manual size arithmetic.

TensorRT only ever sees ``tensor.data_ptr()``, so nothing is lost by letting torch own the
memory.

The buffers are allocated **once**, at load time, sized for ``max_batch_size``; a smaller
batch uses a prefix. That is not an optimisation but a requirement: CUDA graph capture
records addresses, so a graph captured against a buffer that is later reallocated would
replay into freed memory (ADR-008).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from shipinfer.core.errors import InferenceError
from shipinfer.core.logging import get_logger
from shipinfer.core.types import DataType, Device
from shipinfer.runtime.memory import PinnedStagingPool
from shipinfer.runtime.platform import require_torch
from shipinfer.runtime.stream import Stream
from shipinfer.runtime.tensor import torch_dtype

__all__ = ["Binding", "BindingSet"]

_LOG = get_logger("backends.tensorrt.bindings")


@dataclass(slots=True)
class Binding:
    """One engine tensor's persistent device buffer."""

    name: str
    is_input: bool
    device_tensor: Any  # torch.Tensor on the instance's GPU
    dtype: DataType
    shape: tuple[int, ...]

    @property
    def ptr(self) -> int:
        return int(self.device_tensor.data_ptr())

    @property
    def nbytes(self) -> int:
        return int(self.device_tensor.numel() * self.device_tensor.element_size())


class BindingSet:
    """The I/O buffers for one execution context."""

    def __init__(self, device: Device, staging: PinnedStagingPool) -> None:
        self._device = device
        self._staging = staging
        self._torch = require_torch()
        self._bindings: dict[str, Binding] = {}

    def add(
        self, name: str, *, is_input: bool, dtype: DataType, shape: tuple[int, ...]
    ) -> Binding:
        tensor = self._torch.empty(
            shape, dtype=torch_dtype(dtype), device=f"cuda:{self._device.index}"
        )
        binding = Binding(
            name=name, is_input=is_input, device_tensor=tensor, dtype=dtype, shape=shape
        )
        self._bindings[name] = binding
        _LOG.debug(
            "allocated binding %s %s %s (%d B) on %s",
            name,
            shape,
            dtype.value,
            binding.nbytes,
            self._device,
        )
        return binding

    def __getitem__(self, name: str) -> Binding:
        try:
            return self._bindings[name]
        except KeyError:
            raise InferenceError(
                f"no binding named {name!r}; engine has {sorted(self._bindings)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._bindings

    def names(self) -> list[str]:
        return list(self._bindings)

    def device_ptr(self, name: str) -> int:
        return self[name].ptr

    def device_tensor(self, name: str) -> Any:
        return self[name].device_tensor

    # -- transfers -----------------------------------------------------------------------

    def stage_input(
        self, name: str, array: np.ndarray, stream: Stream, *, async_copy: bool
    ) -> None:
        """Copy a host array into the persistent device buffer.

        The hop through pinned memory is not optional for an async copy:
        ``cudaMemcpyAsync`` from *pageable* memory silently degrades to a synchronous copy
        through a driver bounce buffer, which serialises the stream and cancels the whole
        multi-stream design with no error and no obvious symptom.
        """
        binding = self[name]
        rows = array.shape[0]
        if rows > binding.shape[0]:
            raise InferenceError(
                f"input {name!r}: batch of {rows} exceeds the allocated {binding.shape[0]}"
            )
        source = self._torch.from_numpy(np.ascontiguousarray(array))
        if async_copy:
            source = self._staging.stage(source)
        binding.device_tensor[:rows].copy_(source, non_blocking=async_copy)

    def fetch_output(
        self, name: str, batch_size: int, stream: Stream, *, async_copy: bool
    ) -> np.ndarray:
        """Bring the used prefix of an output back to the host.

        Returns a fresh host array rather than a view of a reused buffer: the device buffer
        belongs to the next batch the moment this one is done, so handing out a view would
        let a response mutate under its owner.
        """
        binding = self[name]
        rows = min(batch_size, binding.shape[0])
        host = binding.device_tensor[:rows].to("cpu", non_blocking=async_copy)
        if async_copy:
            stream.synchronize()
        return np.ascontiguousarray(host.numpy())

    # -- lifecycle -----------------------------------------------------------------------

    def close(self) -> None:
        # Dropping the tensors returns their blocks to torch's cache, not to the driver.
        self._bindings.clear()

    def total_bytes(self) -> int:
        return sum(b.nbytes for b in self._bindings.values())

    def __repr__(self) -> str:
        return f"<BindingSet {self._device} n={len(self._bindings)} {self.total_bytes()} B>"
