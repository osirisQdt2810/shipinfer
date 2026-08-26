"""The image-operations contract — the pre/post-processing seam.

Everything here is a *batched* operation on whole frames or whole crop sets. That is not a
stylistic preference: per-image Python loops around a CUDA call are the single most common
way a perception pipeline ends up CPU-bound with an idle GPU, and a batched interface makes
that mistake hard to write.

Two implementations satisfy this contract:

* :class:`~shipinfer.runtime.ops.numpy_ops.NumpyImageOps` — the readable reference, used by
  tests and on hosts with no GPU;
* :class:`~shipinfer.runtime.ops.native_ops.NativeImageOps` — fused CUDA/HIP kernels from
  ``native/``, where letterbox-resize + BGR->RGB + normalise + NHWC->NCHW is **one** pass
  over the pixels instead of four.

``tests/runtime/test_ops_parity.py`` asserts they agree, which is what makes the fast one
trustworthy.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

__all__ = ["ImageOps", "LetterboxResult", "NormalizeParams"]


@dataclass(frozen=True, slots=True)
class NormalizeParams:
    """Mean/std normalisation, in the source pixel scale (0-255)."""

    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (255.0, 255.0, 255.0)
    #: Swap channel order (BGR from OpenCV -> RGB for most checkpoints).
    swap_rb: bool = True

    def __post_init__(self) -> None:
        if any(s == 0 for s in self.std):
            raise ValueError("normalisation std must be non-zero")


@dataclass(frozen=True, slots=True)
class LetterboxResult:
    """A batched letterbox plus the affine parameters needed to undo it.

    The scales and pads are returned rather than recomputed downstream because postprocess
    *must* use exactly the numbers preprocess used. Recomputing them from the shapes is
    where off-by-one box drift comes from.
    """

    #: ``(N, C, H, W)`` float32, already normalised.
    tensor: np.ndarray
    #: Per-image resize scale.
    scales: np.ndarray
    #: Per-image ``(pad_x, pad_y)`` in destination pixels.
    pads: np.ndarray
    #: Per-image ``(out_h, out_w)`` — the resized extent the kernel *actually* wrote, before
    #: padding. Optional because only the native implementation reports it today. It exists
    #: for the reason the submodule's review gave when it added the output: ``out_h`` is what
    #: the kernel divides by, so it decides the sampling ratio, and a consumer re-deriving it
    #: from ``scale`` can disagree by a pixel while scale and pad both still match —
    #: ``pad = (T - r) / 2`` is the same for ``r`` and ``r + 1`` whenever ``T - r`` is even.
    #: Nothing in the pipeline re-derives it yet, which is why this is optional rather than
    #: required; when something does, it should read this and the field should become
    #: required for every implementation.
    extents: np.ndarray | None = None


class ImageOps(abc.ABC):
    """Batched image pre/post-processing."""

    name: ClassVar[str] = "abstract"
    #: True when the operations run on a GPU.
    on_device: ClassVar[bool] = False

    @abc.abstractmethod
    def letterbox_batch(
        self,
        images: Sequence[np.ndarray],
        dst_size: tuple[int, int],
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> LetterboxResult:
        """Resize-with-aspect-preserving-pad, colour-convert, normalise and transpose.

        One call, because these four steps read and write the same pixels: doing them
        separately means four passes over a 1080p frame, which is four times the memory
        traffic for identical output.

        Args:
            images: ``(H, W, 3)`` uint8 frames, possibly of differing sizes.
            dst_size: ``(height, width)`` of the model input.
            params: normalisation and channel order.
            pad_value: fill for the letterbox bars.
        """

    def letterbox_to_device(
        self,
        images: Sequence[np.ndarray],
        out: Any,
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess straight into a caller-owned device tensor.

        The path production should use. Preprocessing exists to feed an engine on the same
        device, so returning to the host first undoes most of what the fused kernel saved —
        and on a busy host the round trip can cost several times the kernel itself.

        Args:
            out: a pre-allocated output tensor, which an implementation writes into
                directly and therefore must be able to trust. It must be

                * **on the device this ``ImageOps`` is bound to** — a tensor from another
                  GPU is a cross-device write, which ADR-002 says never happens here;
                * **float32**, because that is what the kernels write. A wider dtype has
                  enough bytes to pass a size check and still receives garbage;
                * **contiguous** and rank 4 with shape ``(N, 3, H, W)``, ``N`` at least
                  ``len(images)``. A channels-last or sliced tensor has the same element
                  count and would be filled as though it were contiguous NCHW: scrambled
                  channels, and no error to say so;
                * **stable in address**, if a CUDA graph will capture the consumer.

                ``H`` and ``W`` define the destination size. Implementations must *check*
                these and raise, never write on the assumption that the caller got them
                right — every violation above is silent at the pixel level.

        Returns:
            ``(scales, pads)`` — the geometry needed to invert the letterbox exactly.

        Raises:
            ValidationError: if ``out`` violates any of the requirements above.
            NotImplementedError: for host-only implementations, which have no device to
                write to. Callers should branch on :attr:`on_device` rather than catching.
        """
        raise NotImplementedError(f"{type(self).__name__} has no device output path")

    @abc.abstractmethod
    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dst_size: tuple[int, int],
        params: NormalizeParams,
    ) -> np.ndarray:
        """Extract and resize N boxes from one frame into one ``(N, C, h, w)`` tensor.

        The embedding stage's hot path. Cropping on the device is what keeps the frame
        from crossing PCIe: the frame is megabytes, the crops are kilobytes, and only the
        crops need to travel (ADR-004).

        Args:
            boxes: ``(N, 4)`` float32 ``[x1, y1, x2, y2]`` in ``image`` pixel coordinates.
        """

    @abc.abstractmethod
    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        score_threshold: float,
        max_output: int,
    ) -> np.ndarray:
        """Class-agnostic non-maximum suppression, returning kept indices.

        Belongs next to the model, not on the host: copying 25 000 candidate boxes back to
        filter them down to 20 is the most common self-inflicted bottleneck in this kind of
        pipeline.
        """

    def describe(self) -> str:
        return f"{self.name} ({'device' if self.on_device else 'host'})"

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"
