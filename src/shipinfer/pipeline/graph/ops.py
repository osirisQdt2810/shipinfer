"""One :class:`~shipinfer.runtime.ops.ImageOps` per worker thread, spread across the GPUs.

Two defects this exists to prevent, both found by running the end-to-end test on a machine
that actually has GPUs.

**An ``ImageOps`` instance belongs to one thread.** Its own contract says so — the native
backend keeps a per-instance staging ring precisely so that a reused pinned buffer cannot be
overwritten while its DMA is in flight. Sharing one instance across four pipeline workers
produced ``GpuError: crop_kernel failed: invalid argument`` from inside a stage, which reads
like a bad box and is not: it is two threads in one staging ring.

**Preprocessing must not all land on GPU 0.** The failure this project exists to fix is
"everything ran on device 0 because nothing called ``cudaSetDevice``", and a pipeline that
letterboxes every frame on ``cuda:0`` while the detector is spread over sixteen GPUs
re-creates it one layer up. So each worker thread is assigned the next visible device in
rotation, and its ops are bound to that device for the thread's life (ADR-002).

**The known cost, stated plainly.** These ops return host arrays, so a frame is letterboxed
on a GPU, copied back to the host, and staged again to whichever device the model instance
lives on. A fully GPU-resident path would use :meth:`ImageOps.letterbox_to_device` and write
straight into the instance's binding buffer — but choosing that buffer means knowing which
instance will run the request, which is the dispatcher's decision, not this layer's. That is
the "fast-path GPU-resident pipeline" the architecture document files under Phase 2, to be
done when a measurement says the round trip is what hurts (ADR-007).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np

from shipinfer.core.logging import get_logger
from shipinfer.runtime.ops import ImageOps, LetterboxResult, NormalizeParams

__all__ = ["ThreadLocalImageOps"]

_LOG = get_logger("pipeline.ops")


class ThreadLocalImageOps(ImageOps):
    """Delegates to a per-thread instance, built on first use and bound to one device.

    Implements :class:`ImageOps` so every stage keeps taking an ``ImageOps`` and knows
    nothing about threads. Construction is lazy because the thread that will use an instance
    is the one that must create it — a CUDA context belongs to the thread that made it.

    Args:
        factory: ``(device_index) -> ImageOps``.
        devices: visible device indices, assigned to threads round-robin. Defaults to
            ``(0,)``, which is right for a host-only implementation and for a single GPU.
    """

    name: ClassVar[str] = "thread_local"

    def __init__(
        self, factory: Callable[[int], ImageOps], devices: Sequence[int] = (0,)
    ) -> None:
        self._factory = factory
        self._devices = tuple(devices) or (0,)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._assigned = 0
        self._per_device: dict[int, int] = {}

    @property
    def _ops(self) -> ImageOps:
        ops: ImageOps | None = getattr(self._local, "ops", None)
        if ops is None:
            with self._lock:
                device = self._devices[self._assigned % len(self._devices)]
                self._assigned += 1
                self._per_device[device] = self._per_device.get(device, 0) + 1
            ops = self._factory(device)
            self._local.ops = ops
            _LOG.info(
                "thread %s preprocesses on device %d with %s",
                threading.current_thread().name,
                device,
                ops.describe(),
            )
        return ops

    @property
    def on_device(self) -> bool:  # type: ignore[override]
        return self._ops.on_device

    def letterbox_batch(
        self,
        images: Sequence[np.ndarray],
        dst_size: tuple[int, int],
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> LetterboxResult:
        return self._ops.letterbox_batch(images, dst_size, params, pad_value=pad_value)

    def letterbox_to_device(
        self,
        images: Sequence[np.ndarray],
        out: Any,
        params: NormalizeParams,
        *,
        pad_value: int = 114,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._ops.letterbox_to_device(images, out, params, pad_value=pad_value)

    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dst_size: tuple[int, int],
        params: NormalizeParams,
    ) -> np.ndarray:
        return self._ops.crop_batch(image, boxes, dst_size, params)

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        score_threshold: float,
        max_output: int,
    ) -> np.ndarray:
        return self._ops.nms(boxes, scores, iou_threshold, score_threshold, max_output)

    def assignments(self) -> dict[int, int]:
        """``device -> threads assigned`` — how a test proves the spread is a spread."""
        with self._lock:
            return dict(self._per_device)

    def describe(self) -> str:
        return f"thread-local over devices {list(self._devices)}"

    def __repr__(self) -> str:
        return f"<ThreadLocalImageOps devices={list(self._devices)} threads={self._assigned}>"
