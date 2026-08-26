"""Reference image operations in numpy.

Readable, correct, and slow enough that nothing should ship on it at 1000 fps. Its job is
to define what the CUDA kernels must compute, to keep the offline suite hardware-free, and
to be the other half of the parity test.

Even here the implementation avoids the obvious waste: the destination is allocated once
per batch, the resize is a gather with precomputed index arrays rather than a per-pixel
loop, and normalisation happens in the same expression as the transpose so numpy does not
materialise an extra ``(N, H, W, C)`` float array.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shipinfer.runtime.ops.base import ImageOps, LetterboxResult, NormalizeParams
from shipinfer.runtime.ops.registry import IMAGE_OPS

__all__ = ["NumpyImageOps"]


@IMAGE_OPS.register("numpy", "cpu")
class NumpyImageOps(ImageOps):
    """Host-side reference implementation."""

    name = "numpy"
    on_device = False

    # -- preprocess ---------------------------------------------------------------------

    def letterbox_batch(
        self,
        images: Sequence[np.ndarray],
        dst_size: tuple[int, int],
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> LetterboxResult:
        if not images:
            raise ValueError("letterbox_batch needs at least one image")
        dst_h, dst_w = dst_size
        n = len(images)

        canvas = np.full((n, dst_h, dst_w, 3), pad_value, dtype=np.uint8)
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

            canvas[i, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = _resize_nearest(
                image, new_h, new_w
            )
            scales[i] = scale
            pads[i] = (pad_x, pad_y)
            extents[i] = (new_h, new_w)

        if params.swap_rb:
            canvas = canvas[..., ::-1]

        mean = np.asarray(params.mean, dtype=np.float32)
        std = np.asarray(params.std, dtype=np.float32)
        # transpose first so the arithmetic writes straight into NCHW layout instead of
        # producing an NHWC float array and then copying it.
        chw = np.ascontiguousarray(canvas.transpose(0, 3, 1, 2), dtype=np.float32)
        chw -= mean[None, :, None, None]
        chw /= std[None, :, None, None]
        return LetterboxResult(tensor=chw, scales=scales, pads=pads, extents=extents)

    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dst_size: tuple[int, int],
        params: NormalizeParams,
    ) -> np.ndarray:
        if boxes.size == 0:
            return np.empty((0, 3, *dst_size), dtype=np.float32)
        dst_h, dst_w = dst_size
        src_h, src_w = image.shape[:2]
        n = boxes.shape[0]

        out = np.empty((n, dst_h, dst_w, 3), dtype=np.uint8)
        # Clip in float, then cast once. `np.clip(..., out=int_array)` on a fancy-indexed
        # view neither writes back nor casts — it raises, and only for some dtypes, which
        # is exactly the kind of bug that survives a smoke test.
        clipped = np.empty(boxes.shape, dtype=np.int32)
        clipped[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_w - 1).astype(np.int32)
        clipped[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_h - 1).astype(np.int32)

        for i in range(n):
            x1, y1, x2, y2 = clipped[i]
            if x2 <= x1 or y2 <= y1:
                out[i] = 0  # a degenerate box yields a black crop,
                continue  # never an exception that kills the batch
            out[i] = _resize_nearest(image[y1:y2, x1:x2], dst_h, dst_w)

        if params.swap_rb:
            out = out[..., ::-1]
        mean = np.asarray(params.mean, dtype=np.float32)
        std = np.asarray(params.std, dtype=np.float32)
        chw = np.ascontiguousarray(out.transpose(0, 3, 1, 2), dtype=np.float32)
        chw -= mean[None, :, None, None]
        chw /= std[None, :, None, None]
        return chw

    # -- postprocess --------------------------------------------------------------------

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        score_threshold: float,
        max_output: int,
    ) -> np.ndarray:
        keep_mask = scores >= score_threshold
        candidates = np.nonzero(keep_mask)[0]
        if candidates.size == 0:
            return np.empty(0, dtype=np.int64)

        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

        kept: list[int] = []
        while order.size and len(kept) < max_output:
            best = int(order[0])
            kept.append(best)
            if order.size == 1:
                break
            rest = order[1:]
            # Vectorised IoU of the winner against every survivor at once — the loop runs
            # once per *kept* box, not once per pair.
            inter_w = np.maximum(
                0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest])
            )
            inter_h = np.maximum(
                0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest])
            )
            inter = inter_w * inter_h
            iou = inter / np.maximum(areas[best] + areas[rest] - inter, 1e-9)
            order = rest[iou <= iou_threshold]
        return np.asarray(kept, dtype=np.int64)


def _resize_nearest(image: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """Nearest-neighbour resize by index gather.

    Nearest rather than bilinear so this file has no dependency on OpenCV and so the CUDA
    kernel has an unambiguous reference to match bit-for-bit. Production preprocessing runs
    the fused device kernel, which offers bilinear.
    """
    src_h, src_w = image.shape[:2]
    if (src_h, src_w) == (dst_h, dst_w):
        return image
    rows = (np.arange(dst_h, dtype=np.float32) + 0.5) * (src_h / dst_h) - 0.5
    cols = (np.arange(dst_w, dtype=np.float32) + 0.5) * (src_w / dst_w) - 0.5
    rows = np.clip(np.rint(rows), 0, src_h - 1).astype(np.intp)
    cols = np.clip(np.rint(cols), 0, src_w - 1).astype(np.intp)
    return image[rows[:, None], cols[None, :]]
