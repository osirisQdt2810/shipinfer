"""Batched image pre/post-processing — the pre-processing seam.

Three implementations of one contract, in increasing order of speed and decreasing order of
portability:

* :class:`NumpyImageOps` — the readable reference. Defines what the others must compute,
  keeps the offline suite hardware-free, and is the other half of the parity test.
* :class:`TorchImageOps` — the default with an accelerator. Every op is an existing tuned
  torch kernel; nothing is hand-written (ADR-003).
* :class:`NativeImageOps` — the fused CUDA/HIP kernels in ``native/``. Resize + colour
  convert + normalise + NHWC->NCHW in **one** pass instead of four, which is the one place
  a custom kernel genuinely beats the library.

``tests/runtime/test_ops_parity.py`` asserts all three agree.
"""

from __future__ import annotations

from shipinfer.core.logging import get_logger
from shipinfer.core.settings import ExecutionProvider
from shipinfer.runtime.native import resolve_provider
from shipinfer.runtime.ops.base import ImageOps, LetterboxResult, NormalizeParams
from shipinfer.runtime.ops.native_ops import NativeImageOps
from shipinfer.runtime.ops.numpy_ops import NumpyImageOps
from shipinfer.runtime.ops.registry import IMAGE_OPS
from shipinfer.runtime.ops.torch_ops import TorchImageOps
from shipinfer.runtime.platform import is_available

__all__ = [
    "IMAGE_OPS",
    "ImageOps",
    "LetterboxResult",
    "NativeImageOps",
    "NormalizeParams",
    "NumpyImageOps",
    "TorchImageOps",
    "get_image_ops",
]

_LOG = get_logger("runtime.ops")


def get_image_ops(
    provider: ExecutionProvider = ExecutionProvider.AUTO, *, device_index: int = 0
) -> ImageOps:
    """The image ops for one device: native if built, else torch, else numpy.

    Under ``AUTO`` a construction failure degrades one step with a warning rather than
    raising — a GPU-less CI box and a box whose extension was built for the wrong
    architecture should both still run. Under ``NATIVE`` it raises, because a deployment
    that asked for the fused kernels should not silently get the unfused ones.
    """
    resolved = resolve_provider(provider)
    if resolved is ExecutionProvider.NATIVE:
        try:
            return NativeImageOps(device_index=device_index)
        except Exception as exc:
            if provider is ExecutionProvider.NATIVE:
                raise
            _LOG.warning("native image ops unavailable (%s); falling back", exc)
    if is_available():
        try:
            return TorchImageOps(device_index=device_index)
        except Exception as exc:  # pragma: no cover - a broken torch install
            _LOG.warning("torch image ops unavailable (%s); falling back to numpy", exc)
    return NumpyImageOps()
