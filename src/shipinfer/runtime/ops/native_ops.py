"""Image operations backed by the fused CUDA/HIP kernels in ``native/``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from shipinfer.core.logging import get_logger
from shipinfer.runtime.native import require_native
from shipinfer.runtime.ops.base import ImageOps, LetterboxResult, NormalizeParams
from shipinfer.runtime.ops.registry import IMAGE_OPS

__all__ = ["NativeImageOps"]

_LOG = get_logger("runtime.ops.native")


@IMAGE_OPS.register("native", "cuda")
class NativeImageOps(ImageOps):
    """Fused device kernels: one pass over the pixels instead of four.

    The kernel behind :meth:`letterbox_batch` does resize, BGR->RGB, mean/std normalise and
    NHWC->NCHW inside a single thread-per-output-pixel launch. Each of those steps is
    memory-bound, so fusing them is close to a 4x reduction in memory traffic for the same
    result — the same technique the reference ``*-trt`` services use in
    ``src/tools/imgproc/*.cu``, generalised over a batch.

    :meth:`nms` runs on the device for the same reason: 25 000 candidate boxes is ~800 KB
    that never needs to reach the host when only 20 survive.
    """

    name = "native"
    on_device = True

    def __init__(self, device_index: int = 0, stream: int = 0) -> None:
        self._native: Any = require_native()
        if not self._native.cuda_available():
            raise RuntimeError(
                "shipinfer._C was built without GPU support; rebuild with "
                "-DSHIPINFER_WITH_CUDA=ON (or -DSHIPINFER_WITH_HIP=ON)"
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
        """Write the fused result straight into a torch CUDA tensor. The fast path."""
        if not out.is_cuda:
            raise ValueError("letterbox_to_device needs a CUDA tensor")
        scales, pads = self._ops.letterbox_into(
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
        tensor, scales, pads = self._ops.letterbox_batch(
            [np.ascontiguousarray(img) for img in images],
            int(dst_size[0]),
            int(dst_size[1]),
            list(params.mean),
            list(params.std),
            bool(params.swap_rb),
            int(pad_value),
            int(self._stream),
        )
        return LetterboxResult(tensor=tensor, scales=scales, pads=pads)

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
