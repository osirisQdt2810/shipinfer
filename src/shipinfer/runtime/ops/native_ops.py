"""Image operations backed by the fused kernels in ``shipvision``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from shipinfer.core.errors import ValidationError
from shipinfer.core.logging import get_logger
from shipinfer.runtime.native import require_native
from shipinfer.runtime.ops.base import ImageOps, LetterboxResult, NormalizeParams
from shipinfer.runtime.ops.registry import IMAGE_OPS
from shipinfer.runtime.platform import require_torch

__all__ = ["NativeImageOps", "check_device_output"]

_LOG = get_logger("runtime.ops.native")


def check_device_output(out: Any, *, device_index: int, rows: int, caller: str) -> None:
    """Reject an output tensor the fused kernels cannot legally write into.

    The kernels are handed a bare ``float*`` and a byte count, so every assumption the
    launch makes has to be checked here or nowhere. A byte-count check on its own is not
    enough, and each hole it leaves is silent rather than loud:

    * a channels-last ``(N, 3, H, W)`` tensor has exactly the same ``numel``, so the kernel
      writes NCHW into NHWC storage — scrambled channels, no error, a detector that just
      gets worse;
    * a float64 tensor has *twice* the bytes, so a size check passes and float32 is written
      into half of it — garbage;
    * a sliced view is written as if it were contiguous, corrupting whatever it is a view
      of;
    * a tensor on another device is a cross-device write, which ADR-002 says nothing in
      this codebase performs.

    Args:
        rows: leading rows the launch will fill — the batch size, or the box count.
        caller: the public method name, so the message says where to look.

    Raises:
        ValidationError: naming the property that was wrong.
    """
    torch = require_torch()
    if not getattr(out, "is_cuda", False):
        raise ValidationError(
            f"{caller} needs a CUDA tensor; got one on {getattr(out, 'device', 'the host')}"
        )
    index = out.device.index
    if index != device_index:
        raise ValidationError(
            f"{caller} was given a tensor on cuda:{index} but these ops are bound to "
            f"cuda:{device_index}; ADR-002 forbids the cross-device write"
        )
    if out.dtype is not torch.float32:
        raise ValidationError(f"{caller} writes float32; the output tensor is {out.dtype}")
    if out.dim() != 4 or int(out.shape[1]) != 3:
        raise ValidationError(
            f"{caller} needs an (N, 3, H, W) output tensor; got {tuple(out.shape)}"
        )
    if not out.is_contiguous():
        raise ValidationError(
            f"{caller} writes contiguous NCHW; the output tensor is not contiguous "
            f"(shape {tuple(out.shape)}, strides {tuple(out.stride())}) — a channels-last "
            "or sliced tensor would be filled as though it were NCHW"
        )
    if int(out.shape[0]) < rows:
        raise ValidationError(
            f"{caller} output holds {int(out.shape[0])} rows but the batch has {rows}"
        )


@IMAGE_OPS.register("native", "cuda")
class NativeImageOps(ImageOps):
    """Fused device kernels: one pass over the pixels instead of four.

    The kernel behind :meth:`letterbox_batch` does resize, BGR->RGB, mean/std normalise and
    NHWC->NCHW inside a single thread-per-output-pixel launch. Each of those steps is
    memory-bound, so fusing them is close to a 4x reduction in memory traffic for the same
    result — the same technique the reference ``*-trt`` services use in
    ``src/tools/imgproc/*.cu``, generalised over a batch.

    The kernels live in the ``shipvision`` repository, pinned here as the submodule
    ``3rdparty/shipvision``. This class is the adapter between that library's raw
    binding and the :class:`~shipinfer.runtime.ops.base.ImageOps` contract.

    :meth:`nms` runs on the device for the same reason: 25 000 candidate boxes is ~800 KB
    that never needs to reach the host when only 20 survive.
    """

    name = "native"
    on_device = True

    def __init__(self, device_index: int = 0, stream: int = 0) -> None:
        self._native: Any = require_native()
        # `cuda_available`, which is the name the extension actually binds. This read
        # `is_available` for as long as this class has existed, and nothing caught it: the
        # AttributeError surfaced upstream as "the fused kernels are unavailable — fetch and
        # build the submodule", so every kernel benchmark reported the native column skipped
        # and blamed a missing build. The build was there. The two repositories simply
        # disagreed about one identifier, which is the same shape as the `version()` /
        # `__version__` bug found in review — a defect that lives in neither file.
        probe = getattr(self._native, "cuda_available", None)
        if probe is None:
            raise RuntimeError(
                f"{self._native.__name__} is built but defines no `cuda_available`, so this "
                f"build predates the kernels or exports a different surface. Rebuild the "
                f"submodule; if that does not fix it, the two repositories have drifted and "
                f"`tests/runtime/test_native.py` is where the agreement is pinned"
            )
        if not probe():
            raise RuntimeError(
                "shipvision is built but reports no usable GPU kernels — either it was built "
                "without CUDA, or no device is visible to this process"
            )
        self._device_index = device_index
        self._stream = stream
        self._ops = self._native.ImageOps(device_index)

    def letterbox_to_device(
        self,
        images: Sequence[np.ndarray],
        out: Any,
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Write the fused result straight into a torch CUDA tensor. The fast path.

        ``out`` must be a contiguous float32 ``(N, 3, H, W)`` CUDA tensor on this object's
        device, with ``N`` at least ``len(images)``; see :func:`check_device_output` for
        why each of those is checked rather than trusted.

        Raises:
            ValidationError: if ``out`` does not satisfy that contract.
        """
        check_device_output(
            out,
            device_index=self._device_index,
            rows=len(images),
            caller="letterbox_to_device",
        )
        # Three values since the submodule started reporting the applied extents. The
        # two-value unpack this used to be raised `ValueError` on every call — and went
        # unnoticed for as long as it did because the path was unreachable for three
        # unrelated reasons (see `runtime/native.py`). A dead path is where a contract
        # change is invisible. The extents are not surfaced here because the ABC returns
        # `(scales, pads)` and nothing downstream re-derives `out_h` yet; the ledger tracks
        # widening the contract so they are.
        scales, pads, _extents = self._ops.letterbox_into(
            [np.ascontiguousarray(img) for img in images],
            int(out.data_ptr()),
            int(out.numel() * out.element_size()),
            int(out.shape[2]),
            int(out.shape[3]),
            list(params.mean),
            list(params.std),
            bool(params.swap_rb),
            int(pad_value),
            int(self._stream),
        )
        return scales, pads

    def letterbox_batch(
        self,
        images: Sequence[np.ndarray],
        dst_size: tuple[int, int],
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> LetterboxResult:
        """Fused preprocess returned as numpy.

        Correct, and deliberately not the production path: the device-to-host copy of a
        640x640 batch costs several times the kernel that produced it. Use
        :meth:`letterbox_to_device` for anything on the critical path.
        """
        tensor, scales, pads, extents = self._ops.letterbox_batch(
            [np.ascontiguousarray(img) for img in images],
            int(dst_size[0]),
            int(dst_size[1]),
            list(params.mean),
            list(params.std),
            bool(params.swap_rb),
            int(pad_value),
            int(self._stream),
        )
        return LetterboxResult(tensor=tensor, scales=scales, pads=pads, extents=extents)

    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dst_size: tuple[int, int],
        params: NormalizeParams,
    ) -> np.ndarray:
        return self._ops.crop_batch(
            np.ascontiguousarray(image),
            np.ascontiguousarray(boxes, dtype=np.float32),
            int(dst_size[0]),
            int(dst_size[1]),
            list(params.mean),
            list(params.std),
            bool(params.swap_rb),
            int(self._stream),
        )

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        score_threshold: float,
        max_output: int,
    ) -> np.ndarray:
        return self._ops.nms(
            np.ascontiguousarray(boxes, dtype=np.float32),
            np.ascontiguousarray(scores, dtype=np.float32),
            float(iou_threshold),
            float(score_threshold),
            int(max_output),
            int(self._stream),
        )

    def describe(self) -> str:
        return f"native fused kernels on cuda:{self._device_index}"
