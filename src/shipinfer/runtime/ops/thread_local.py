"""One :class:`~shipinfer.runtime.ops.ImageOps` per worker thread, spread across the GPUs.

Lives in ``runtime`` because it is an ``ImageOps`` decorator and nothing else: it knows about
threads and devices, which is what this layer is for, and about no caller. It was written for
:class:`~shipinfer.pipeline.PipelineRunner` and moved here when the second and third callers
appeared -- ``shipinfer run`` and ``shipinfer-shard``, which build a
:class:`~shipinfer.runners.base.Runner` whose ``pipeline.workers`` threads walk one chain
concurrently and were each handed the *same* instance. The old import path
(``shipinfer.pipeline.graph.ops``) is a re-export shim.

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

from shipinfer.core.logging import LOG

# `..base`, not the package: `runtime.ops.__init__` imports this module, so naming the
# package here would be a cycle at the first import of `shipinfer.runtime.ops`.
from shipinfer.runtime.ops.base import ImageOps, LetterboxResult, NormalizeParams

__all__ = ["ThreadLocalImageOps", "staging_owner"]


def staging_owner(device_index: int) -> str:
    """The pinned-staging pool key for the calling thread's ops on ``device_index``.

    :meth:`~shipinfer.runtime.memory.MemoryPool.staging_for` hands one pool per owner string,
    so this string is the whole safety argument: two live threads sharing one key share one
    buffer, and the second one's copy overwrites the first's while its DMA is still reading
    it. That failure has no diagnostic — one camera's crops come back under another camera's
    tag, with plausible pixels and no error anywhere.

    The thread's name alone is not unique. Two :class:`~shipinfer.pipeline.PipelineRunner`
    instances over one server both name their workers ``pipeline-worker-0``, which is exactly
    the collision above -- and so do a chain runner's workers and a pipeline's in one process.
    The identity is therefore in the key as well, and so is the device, because a pool's
    buffers are only useful to the device they were staged for.

    A recycled identity — the interpreter hands one out again after a thread exits — is safe:
    only one live thread can hold it at a time, and the ops synchronise every staged copy
    before returning, so a thread that has exited left no DMA reading its buffers.
    """
    thread = threading.current_thread()
    return f"ops:{thread.name}#{threading.get_ident():x}:cuda:{device_index}"


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
        #: The round-robin cursor: how many device slots have been *handed out*, including to
        #: builds that then failed. Advancing it on a failure is deliberate — a thread whose
        #: construction raised retries onto the next device rather than back onto the one that
        #: just refused it — which is why it is not the ledger. See :meth:`assignments`.
        self._assigned = 0
        #: ``device -> delegates that exist``. Written after the factory returns, so a run
        #: whose builds failed does not report a spread it does not have.
        self._per_device: dict[int, int] = {}

    @property
    def _ops(self) -> ImageOps:
        ops: ImageOps | None = getattr(self._local, "ops", None)
        if ops is None:
            with self._lock:
                device = self._devices[self._assigned % len(self._devices)]
                self._assigned += 1
            # The factory is called with the lock released: it binds a CUDA context and may
            # allocate a pinned pool, and holding a process-wide lock across that serialises
            # every worker's first frame behind the slowest construction.
            ops = self._factory(device)
            # Counted *after* the build, not before: a factory that raises used to leave the
            # device credited with a delegate that does not exist, and `assignments()` is what
            # a `-m multigpu` run is read for. "The spread is even" must not be true of a
            # process where half the constructions failed.
            with self._lock:
                self._per_device[device] = self._per_device.get(device, 0) + 1
            self._local.ops = ops
            LOG.info(
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
        """``device -> live delegates`` — how a test proves the spread is a spread.

        Only successful builds are in here. A thread that has never touched the ops is not in
        it either, because construction is lazy and there is nothing on that thread yet.
        """
        with self._lock:
            return dict(self._per_device)

    def describe(self) -> str:
        return f"thread-local over devices {list(self._devices)}"

    def __repr__(self) -> str:
        with self._lock:
            built = sum(self._per_device.values())
        return f"<ThreadLocalImageOps devices={list(self._devices)} threads={built}>"
