"""The device-tensor bridge, offline: the provider's retention contract.

``to_torch`` hands torch a ``_DeviceSpan`` and torch keeps *that object* alive for the view's
life — nothing else. So the span must retain the ``core.Tensor`` (and through it the handle
that owns the memory), or a caller dropping its tensor frees the buffer under a live view.
The full chain needs CUDA (``tests/server/test_remote_wire.py``, gpu tier); the retention
property itself does not.
"""

from __future__ import annotations

import ctypes
import gc
import types
import weakref

import numpy as np

from shipinfer.core.types import DataType, Device, MemoryKind, Tensor
from shipinfer.runtime.tensor import _DeviceSpan


class _Handle(types.SimpleNamespace):
    """``SimpleNamespace`` itself refuses weakrefs; the retention probe needs one."""


class TestTheSpanRetainsItsTensor:
    def test_the_tensor_and_its_bytes_outlive_the_callers_reference(self) -> None:
        backing = np.arange(16, dtype=np.float32) * 3.0
        handle = _Handle(
            ptr=backing.ctypes.data,
            nbytes=backing.nbytes,
            device=Device.cuda(0),
            kind=MemoryKind.DEVICE,
            owner=backing,  # a real DeviceBuffer owns its allocation; the fake must too
        )
        tensor = Tensor.from_handle(handle, DataType.FP32, (16,))
        span = _DeviceSpan(tensor)
        # ``Tensor`` is slotted, so the probe is the handle: only the tensor references it,
        # so it lives exactly as long as the tensor does.
        ref = weakref.ref(handle)
        del tensor, handle
        gc.collect()
        assert ref() is not None, "the span is all torch retains, so the span retains the rest"

        pointer, _read_only = span.__cuda_array_interface__["data"]
        raw = ctypes.string_at(pointer, backing.nbytes)
        np.testing.assert_array_equal(np.frombuffer(raw, dtype=np.float32), backing)

    def test_dropping_the_span_releases_the_tensor(self) -> None:
        backing = np.zeros(4, dtype=np.float32)
        handle = _Handle(
            ptr=backing.ctypes.data,
            nbytes=backing.nbytes,
            device=Device.cuda(0),
            kind=MemoryKind.DEVICE,
            owner=backing,
        )
        tensor = Tensor.from_handle(handle, DataType.FP32, (4,))
        span = _DeviceSpan(tensor)
        ref = weakref.ref(handle)
        del tensor, handle, span
        gc.collect()
        assert ref() is None, "retention is for the view's life, not a leak"
