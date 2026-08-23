"""Element types that may cross a seam."""

from __future__ import annotations

import enum
from typing import Any

import numpy as np

__all__ = ["DataType"]


class DataType(str, enum.Enum):
    """The element types the server can move between layers.

    The names mirror Triton's ``TYPE_*`` spelling minus the prefix, so a model config
    ported from a real Triton deployment needs no translation table.
    """

    BOOL = "BOOL"
    UINT8 = "UINT8"
    INT8 = "INT8"
    INT32 = "INT32"
    INT64 = "INT64"
    FP16 = "FP16"
    FP32 = "FP32"
    FP64 = "FP64"

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        return _NUMPY_BY_DTYPE[self]

    @property
    def itemsize(self) -> int:
        return int(self.numpy_dtype.itemsize)

    @classmethod
    def from_numpy(cls, dtype: np.dtype[Any] | type) -> DataType:
        """Map a numpy dtype back to a :class:`DataType`.

        Raises:
            ValueError: for a dtype with no server-side equivalent (object arrays, and
                anything else that could not be memcpy'd to a device).
        """
        resolved = np.dtype(dtype)
        try:
            return _DTYPE_BY_NUMPY[resolved]
        except KeyError:
            raise ValueError(
                f"unsupported numpy dtype for inference I/O: {resolved!r}"
            ) from None


_NUMPY_BY_DTYPE: dict[DataType, np.dtype[Any]] = {
    DataType.BOOL: np.dtype(np.bool_),
    DataType.UINT8: np.dtype(np.uint8),
    DataType.INT8: np.dtype(np.int8),
    DataType.INT32: np.dtype(np.int32),
    DataType.INT64: np.dtype(np.int64),
    DataType.FP16: np.dtype(np.float16),
    DataType.FP32: np.dtype(np.float32),
    DataType.FP64: np.dtype(np.float64),
}
_DTYPE_BY_NUMPY: dict[np.dtype[Any], DataType] = {v: k for k, v in _NUMPY_BY_DTYPE.items()}
