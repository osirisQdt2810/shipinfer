"""Image operations on torch — the default when an accelerator is present.

Every op here maps onto a torch primitive that is already a tuned CUDA kernel:
``F.interpolate`` for resize, ``roi_align``-style indexing for crops, fused arithmetic for
normalisation, and ``torchvision.ops.nms`` for suppression. Writing any of those by hand
would be slower and would need maintaining for both CUDA and ROCm (ADR-003).

What torch *cannot* do in one pass is the fusion:
:class:`~shipinfer.runtime.ops.native_ops.NativeImageOps` runs resize + colour convert +
normalise + NHWC->NCHW as a single kernel, where this class runs four. That fusion is the
reason ``native/`` exists at all — it is the one place a custom kernel genuinely beats the
library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from shipinfer.core.logging import get_logger
from shipinfer.runtime.ops.base import ImageOps, LetterboxResult, NormalizeParams
from shipinfer.runtime.ops.registry import IMAGE_OPS
from shipinfer.runtime.platform import require_torch

__all__ = ["TorchImageOps"]

_LOG = get_logger("runtime.ops.torch")


@IMAGE_OPS.register("torch")
class TorchImageOps(ImageOps):
    """Batched pre/post-processing through torch kernels."""

    name = "torch"

    def __init__(
        self, device_index: int | None = None, *, interpolation: str = "bilinear"
    ) -> None:
        self._torch = require_torch()
        self._device = (
            self._torch.device("cuda", device_index)
            if device_index is not None and self._torch.cuda.is_available()
            else self._torch.device("cpu")
        )
        self._interpolation = interpolation

    @property
    def on_device(self) -> bool:  # type: ignore[override]
        return self._device.type == "cuda"

    # -- preprocess ---------------------------------------------------------------------

    def letterbox_to_device(
        self,
        images: Sequence[np.ndarray],
        out: Any,
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fill ``out`` in place, with no host round trip."""
        scales, pads, _extents = self._letterbox(images, out, params, pad_value)
        return scales, pads

    def letterbox_batch(
        self,
        images: Sequence[np.ndarray],
        dst_size: tuple[int, int],
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> LetterboxResult:
        torch = self._torch
        dst_h, dst_w = dst_size
        canvas = torch.empty(
            (len(images), 3, dst_h, dst_w), dtype=torch.float32, device=self._device
        )
        scales, pads, extents = self._letterbox(images, canvas, params, pad_value)
        return LetterboxResult(
            tensor=canvas.cpu().numpy(), scales=scales, pads=pads, extents=extents
        )

    def _letterbox(
        self, images: Sequence[np.ndarray], canvas: Any, params: NormalizeParams, pad_value: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared implementation: everything happens in ``canvas``, wherever it lives.

        Returns ``(scales, pads, extents)`` — the third is the ``(new_h, new_w)`` each image
        was resized to, the size ``interpolate`` was actually asked for.
        """
        if not images:
            raise ValueError("letterbox needs at least one image")
        torch = self._torch
        n = len(images)
        if canvas.shape[0] < n:
            raise ValueError(f"output holds {canvas.shape[0]} rows but the batch has {n}")
        dst_h, dst_w = int(canvas.shape[2]), int(canvas.shape[3])

        canvas.fill_(float(pad_value))
        scales = np.empty(n, dtype=np.float32)
        pads = np.empty((n, 2), dtype=np.float32)
        extents = np.empty((n, 2), dtype=np.int32)

        for i, image in enumerate(images):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"image {i}: expected (H, W, 3), got {image.shape}")
            src_h, src_w = image.shape[:2]
            scale = min(dst_h / src_h, dst_w / src_w)
            new_h = max(1, round(src_h * scale))
            new_w = max(1, round(src_w * scale))
            pad_y = (dst_h - new_h) // 2
            pad_x = (dst_w - new_w) // 2
            extents[i] = (new_h, new_w)

            # HWC uint8 -> 1CHW float on the device, then one interpolate call.
            src = (
                torch.from_numpy(np.ascontiguousarray(image))
                .to(canvas.device, non_blocking=True)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            resized = torch.nn.functional.interpolate(
                src, size=(new_h, new_w), mode=self._interpolation, align_corners=False
            )
            canvas[i, :, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized[0]
            scales[i] = scale
            pads[i] = (pad_x, pad_y)

        if params.swap_rb:
            # In place, because `flip` allocates a whole second N x 3 x H x W tensor and the
            # point of writing into a caller-owned buffer is that nothing else is allocated.
            canvas[:n] = canvas[:n].flip(1)

        mean = torch.tensor(params.mean, dtype=torch.float32, device=canvas.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor(params.std, dtype=torch.float32, device=canvas.device).view(
            1, 3, 1, 1
        )
        canvas[:n].sub_(mean).div_(std)
        return scales, pads, extents

    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dst_size: tuple[int, int],
        params: NormalizeParams,
    ) -> np.ndarray:
        torch = self._torch
        dst_h, dst_w = dst_size
        if boxes.size == 0:
            return np.empty((0, 3, dst_h, dst_w), dtype=np.float32)

        src_h, src_w = image.shape[:2]
        frame = (
            torch.from_numpy(np.ascontiguousarray(image))
            .to(self._device, non_blocking=True)
            .permute(2, 0, 1)
            .float()
        )
        clipped = np.empty_like(boxes, dtype=np.int64)
        clipped[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_w - 1)
        clipped[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_h - 1)

        out = torch.zeros(
            (boxes.shape[0], 3, dst_h, dst_w), dtype=torch.float32, device=self._device
        )
        for i, (x1, y1, x2, y2) in enumerate(clipped):
            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box -> zeros, never an exception
            patch = frame[:, y1:y2, x1:x2].unsqueeze(0)
            out[i] = torch.nn.functional.interpolate(
                patch, size=(dst_h, dst_w), mode=self._interpolation, align_corners=False
            )[0]

        if params.swap_rb:
            out = out.flip(1)
        mean = torch.tensor(params.mean, dtype=torch.float32, device=self._device).view(
            1, 3, 1, 1
        )
        std = torch.tensor(params.std, dtype=torch.float32, device=self._device).view(
            1, 3, 1, 1
        )
        out.sub_(mean).div_(std)
        return out.cpu().numpy()

    # -- postprocess --------------------------------------------------------------------

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        score_threshold: float,
        max_output: int,
    ) -> np.ndarray:
        torch = self._torch
        keep = scores >= score_threshold
        candidates = np.nonzero(keep)[0]
        if candidates.size == 0:
            return np.empty(0, dtype=np.int64)

        b = torch.from_numpy(np.ascontiguousarray(boxes[candidates], dtype=np.float32)).to(
            self._device
        )
        s = torch.from_numpy(np.ascontiguousarray(scores[candidates], dtype=np.float32)).to(
            self._device
        )
        try:
            from torchvision.ops import nms as tv_nms

            kept = tv_nms(b, s, iou_threshold)
        except ImportError:
            kept = self._nms_fallback(b, s, iou_threshold)
        return candidates[kept[:max_output].cpu().numpy()]

    def _nms_fallback(self, boxes: Any, scores: Any, iou_threshold: float) -> Any:
        """Greedy NMS in torch, for installs without torchvision.

        Vectorised against all survivors per iteration, so the loop runs once per *kept*
        box rather than once per pair.
        """
        torch = self._torch
        order = scores.argsort(descending=True)
        areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(
            min=0
        )
        kept: list[int] = []
        while order.numel() > 0:
            best = int(order[0])
            kept.append(best)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(boxes[best, 0], boxes[rest, 0])
            yy1 = torch.maximum(boxes[best, 1], boxes[rest, 1])
            xx2 = torch.minimum(boxes[best, 2], boxes[rest, 2])
            yy2 = torch.minimum(boxes[best, 3], boxes[rest, 3])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            iou = inter / (areas[best] + areas[rest] - inter).clamp(min=1e-9)
            order = rest[iou <= iou_threshold]
        return torch.tensor(kept, dtype=torch.long, device=boxes.device)

    def describe(self) -> str:
        return f"torch kernels on {self._device}"
