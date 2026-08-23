"""The tensor value object and the two operations the hot path needs on it."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from shipinfer.core.types.device import Device
from shipinfer.core.types.dtype import DataType
from shipinfer.core.types.memory import MemoryHandle, MemoryKind
from shipinfer.core.types.spec import Shape, TensorSpec

__all__ = ["Tensor", "stack_tensors", "validate_against"]


@dataclass(frozen=True, slots=True)
class Tensor:
    """A batch-major buffer plus the metadata needed to move it across a seam.

    Exactly one of ``host`` / ``handle`` is set. A host tensor carries a numpy array; a
    device tensor carries an opaque :class:`~shipinfer.core.types.memory.MemoryHandle` and
    *no* numpy view — materialising one would mean a synchronous D2H copy at an arbitrary
    point in the pipeline, which is the classic accidental stall this design exists to
    avoid.
    """

    dtype: DataType
    shape: Shape
    device: Device = field(default_factory=Device.cpu)
    memory_kind: MemoryKind = MemoryKind.HOST
    host: np.ndarray | None = None
    handle: MemoryHandle | None = None

    def __post_init__(self) -> None:
        if (self.host is None) == (self.handle is None):
            raise ValueError("Tensor takes exactly one of `host` or `handle`")
        if self.host is not None and self.host.shape != self.shape:
            raise ValueError(f"array shape {self.host.shape} != declared shape {self.shape}")

    # -- constructors ----------------------------------------------------------------

    @classmethod
    def from_numpy(cls, array: np.ndarray, *, copy: bool = False) -> Tensor:
        """Wrap a numpy array.

        Contiguity is enforced because every downstream copy (H2D, pinned staging, socket
        write) assumes a single flat span; discovering otherwise inside a CUDA memcpy is a
        much worse place to find out.
        """
        arr = np.ascontiguousarray(array) if copy or not array.flags["C_CONTIGUOUS"] else array
        return cls(
            dtype=DataType.from_numpy(arr.dtype),
            shape=tuple(arr.shape),
            device=Device.cpu(),
            memory_kind=MemoryKind.HOST,
            host=arr,
        )

    @classmethod
    def from_handle(cls, handle: MemoryHandle, dtype: DataType, shape: Shape) -> Tensor:
        """Describe memory that already lives where it needs to be (usually a GPU)."""
        return cls(
            dtype=dtype,
            shape=tuple(shape),
            device=handle.device,
            memory_kind=handle.kind,
            handle=handle,
        )

    # -- accessors -------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.shape[0] if self.shape else 1

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def nbytes(self) -> int:
        return self.element_count * self.dtype.itemsize

    @property
    def is_device_resident(self) -> bool:
        return self.memory_kind is MemoryKind.DEVICE

    def numpy(self) -> np.ndarray:
        """The host view.

        Raises:
            RuntimeError: if the tensor is device-resident. Copy it back explicitly
                through the runtime rather than hiding a synchronising transfer here.
        """
        if self.host is None:
            raise RuntimeError(
                f"tensor on {self.device} is not host-visible; "
                "copy it with runtime.memory.to_host() before reading"
            )
        return self.host

    def slice_batch(self, start: int, stop: int) -> Tensor:
        """A zero-copy view of rows ``[start, stop)``.

        How a batched response is split back into per-request responses without
        reallocating anything.
        """
        if self.host is None:
            raise RuntimeError("cannot slice a device-resident tensor on the host")
        view = self.host[start:stop]
        return Tensor(
            dtype=self.dtype,
            shape=tuple(view.shape),
            device=self.device,
            memory_kind=self.memory_kind,
            host=view,
        )

    def describe(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"<{dims}:{self.dtype.value}@{self.device}/{self.memory_kind.value}>"


def stack_tensors(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate per-request tensors along the batch axis.

    The single place a dynamic batch is materialised, so also the single place worth
    optimising: it allocates the destination once and fills it with slice assignments,
    rather than letting ``np.concatenate`` build an intermediate list of arrays.
    """
    if not tensors:
        raise ValueError("cannot stack an empty batch")
    if len(tensors) == 1:
        return tensors[0]

    head = tensors[0]
    row_shape = head.shape[1:]
    total = 0
    for t in tensors:
        if t.dtype is not head.dtype:
            raise ValueError(f"dtype mismatch in batch: {t.dtype} vs {head.dtype}")
        if t.shape[1:] != row_shape:
            raise ValueError(f"row shape mismatch in batch: {t.shape[1:]} vs {row_shape}")
        total += t.shape[0]

    out = np.empty((total, *row_shape), dtype=head.dtype.numpy_dtype)
    offset = 0
    for t in tensors:
        rows = t.shape[0]
        out[offset : offset + rows] = t.numpy()
        offset += rows
    return Tensor.from_numpy(out)


def validate_against(
    tensors: Mapping[str, Tensor], specs: Iterable[TensorSpec], *, what: str
) -> None:
    """Check a tensor map against a model's declared specs.

    Raises:
        ValueError: naming the first offending tensor. Failing here, before a batch is
            formed, keeps one malformed request from poisoning a whole batch.
    """
    by_name = {spec.name: spec for spec in specs}
    for spec in by_name.values():
        if spec.name not in tensors and not spec.optional:
            raise ValueError(f"missing required {what} tensor {spec.describe()!r}")
    for name, tensor in tensors.items():
        spec = by_name.get(name)
        if spec is None:
            raise ValueError(f"unexpected {what} tensor {name!r}; known: {sorted(by_name)}")
        if tensor.dtype is not spec.dtype:
            raise ValueError(
                f"{what} {name!r}: dtype {tensor.dtype.value} != declared {spec.dtype.value}"
            )
        if not spec.matches(tensor.shape[1:]):
            raise ValueError(
                f"{what} {name!r}: shape {tensor.shape[1:]} does not match {spec.describe()}"
            )
