"""Declared input/output contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shipinfer.core.types.dtype import DataType

__all__ = ["DYNAMIC", "Shape", "TensorSpec"]

#: Marker for a dimension whose extent is only known at request time. Matches the
#: TensorRT/Triton convention so a config copied from a ``config.pbtxt`` reads the same.
DYNAMIC = -1

Shape = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """The declared contract for one model input or output.

    ``shape`` excludes the batch dimension — exactly like Triton, where ``max_batch_size``
    owns dim 0. A ``-1`` (:data:`DYNAMIC`) entry means "any extent".
    """

    name: str
    dtype: DataType
    shape: Shape
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor spec needs a name")
        for dim in self.shape:
            if dim < 0 and dim != DYNAMIC:
                raise ValueError(f"{self.name}: invalid dimension {dim} (use -1 for dynamic)")

    @property
    def is_dynamic(self) -> bool:
        return DYNAMIC in self.shape

    def matches(self, shape: Sequence[int]) -> bool:
        """True if ``shape`` (batch dimension already stripped) satisfies this spec."""
        if len(shape) != len(self.shape):
            return False
        return all(want in (DYNAMIC, got) for want, got in zip(self.shape, shape, strict=True))

    def nbytes(self, batch_size: int) -> int:
        """Byte size of a batch of this tensor. Dynamic dims are rejected, not guessed."""
        if self.is_dynamic:
            raise ValueError(f"{self.name}: cannot size a dynamic shape {self.shape}")
        count = batch_size
        for dim in self.shape:
            count *= dim
        return count * self.dtype.itemsize

    def describe(self) -> str:
        dims = "x".join("?" if d == DYNAMIC else str(d) for d in self.shape)
        return f"{self.name}[{dims}]:{self.dtype.value}"
