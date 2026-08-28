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

And one decorator over any of them: :class:`ThreadLocalImageOps`, which builds one delegate
per calling thread and spreads the threads over the visible devices. It is not a fourth
implementation — it is how a *caller with several worker threads* satisfies the one-instance-
per-thread contract :func:`get_image_ops` states, and :func:`get_thread_local_image_ops` is
the one call that wires it to a device manager and a pinned pool.

``tests/runtime/test_ops_parity.py`` asserts all three agree.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from shipinfer.core.logging import LOG
from shipinfer.core.settings import ExecutionProvider
from shipinfer.core.types import Device
from shipinfer.runtime.native import resolve_provider
from shipinfer.runtime.ops.base import ImageOps, LetterboxResult, NormalizeParams
from shipinfer.runtime.ops.native_ops import NativeImageOps
from shipinfer.runtime.ops.numpy_ops import NumpyImageOps
from shipinfer.runtime.ops.registry import IMAGE_OPS
from shipinfer.runtime.ops.thread_local import ThreadLocalImageOps, staging_owner
from shipinfer.runtime.ops.torch_ops import TorchImageOps
from shipinfer.runtime.platform import is_available

if TYPE_CHECKING:  # types only; a bare `import shipinfer.runtime.ops` stays cheap
    from shipinfer.runtime.device import DeviceManager
    from shipinfer.runtime.memory import MemoryPool
    from shipinfer.runtime.memory.staging import PinnedStagingPool

__all__ = [
    "IMAGE_OPS",
    "ImageOps",
    "LetterboxResult",
    "NativeImageOps",
    "NormalizeParams",
    "NumpyImageOps",
    "ThreadLocalImageOps",
    "TorchImageOps",
    "get_image_ops",
    "get_thread_local_image_ops",
    "staging_owner",
]


def get_image_ops(
    provider: ExecutionProvider = ExecutionProvider.AUTO,
    *,
    device_index: int = 0,
    staging: PinnedStagingPool | None = None,
) -> ImageOps:
    """The image ops for one device: native if built, else torch, else numpy.

    Under ``AUTO`` a construction failure degrades one step with a warning rather than
    raising — a GPU-less CI box and a box whose extension was built for the wrong
    architecture should both still run. Under ``NATIVE`` it raises, because a deployment
    that asked for the fused kernels should not silently get the unfused ones.

    ``staging`` reaches :class:`TorchImageOps` and nothing else, and the fan-out belongs here
    rather than at the call site: :class:`NativeImageOps` keeps a staging ring of its own
    inside the extension, and :class:`NumpyImageOps` never touches a device, so neither has
    anywhere to put a pool. A caller that had to know which implementation accepts one would
    be choosing the implementation itself, which is the decision this function exists to make.

    Args:
        staging: the calling thread's private pinned pool, from
            :meth:`~shipinfer.runtime.memory.MemoryPool.staging_for`. One pool per owner,
            never a shared instance — two threads through one pool hand each other the same
            buffer.
    """
    resolved = resolve_provider(provider)
    if resolved is ExecutionProvider.NATIVE:
        try:
            return NativeImageOps(device_index=device_index)
        except Exception as exc:
            if provider is ExecutionProvider.NATIVE:
                raise
            LOG.warning("native image ops unavailable (%s); falling back", exc)
    if is_available():
        try:
            return TorchImageOps(device_index=device_index, staging=staging)
        except Exception as exc:  # pragma: no cover - a broken torch install
            LOG.warning("torch image ops unavailable (%s); falling back to numpy", exc)
    return NumpyImageOps()


def get_thread_local_image_ops(
    provider: ExecutionProvider = ExecutionProvider.AUTO,
    *,
    devices: Sequence[int] = (),
    device_manager: DeviceManager | None = None,
    memory: MemoryPool | None = None,
    claim: Callable[[Any, str], PinnedStagingPool] | None = None,
) -> ThreadLocalImageOps:
    """Image ops for a caller with several worker threads: one instance each, spread over GPUs.

    :func:`get_image_ops` answers for **one** thread, and every implementation it can return
    says so — :class:`NativeImageOps` keeps a staging ring inside the extension,
    :class:`TorchImageOps` binds a device on the constructing thread and caches an event and a
    ping-pong staging pair on the instance. A process that resolves one instance and hands it
    to four workers therefore has four threads in one ring, which is CONVENTIONS 2.8's
    overwritten-mid-DMA failure: plausible pixels, no error, invisible to a benchmark. It is
    also invisible to the offline tier, because :class:`NumpyImageOps` is stateless.

    So a caller that walks a chain on ``pipeline.workers`` threads asks for this instead. It
    is one call rather than the same twenty lines in every composition root: ``shipinfer run``,
    ``shipinfer-shard`` and :class:`~shipinfer.pipeline.PipelineRunner` all need exactly this,
    and the third one predates it — it kept its own copy for one line, which is now the
    ``claim`` hook, because two copies of this wiring drift and the drift is silent.

    Args:
        provider: the execution provider each delegate is resolved under.
        devices: the device indices to spread threads across, round-robin. Empty means "one
            device, index 0", which is the right answer both for a single GPU and for a host
            with none — there the delegate degrades to :class:`NumpyImageOps` and the index is
            never used.
        device_manager: the process's :class:`~shipinfer.runtime.device.DeviceManager`, or
            ``None``. When it reports an accelerator, each delegate is built **on the thread
            that will use it** and that thread is bound to its device first: a CUDA context
            belongs to the thread that created it, and a worker sitting on device 0 while
            holding ops built for ``cuda:1`` gets ``invalid resource handle`` from an event
            that belongs to another context (ADR-002).
        memory: the process's :class:`~shipinfer.runtime.memory.MemoryPool`, or ``None``. Each
            thread claims its own pinned staging pool from it, keyed by :func:`staging_owner`
            — per *thread*, not per device, because several workers share a device in rotation
            and one pool between two of them is one buffer between two DMAs.
        claim: how a pool is taken out of ``memory``, defaulting to
            :meth:`~shipinfer.runtime.memory.MemoryPool.staging_for`. The one thing a caller
            that stops and restarts in-process needs to change: it passes a bound method that
            calls ``staging_for`` *and records the owner key*, so ``stop()`` can hand the
            page-locked pages back (see the Note). Called ``(memory, owner)`` on the worker
            thread that owns the key, and only when a pool would have been claimed at all.

    Note:
        The pools are released when the :class:`~shipinfer.runtime.memory.MemoryPool` closes.
        A caller that stops and restarts within one process mints fresh keys each cycle and
        must release them itself: that is what ``claim`` exists for, and
        :meth:`~shipinfer.pipeline.PipelineRunner._build_ops` is the caller that passes one. A
        composition root that builds one runner for the life of the process does not need to.

        With ``device_manager=None`` nothing is bound and no pool is claimed — the delegates
        still spread over ``devices``, and each is *constructed with* its index, which is what
        :class:`TorchImageOps` and :class:`NativeImageOps` actually act on (both carry their
        own device rather than reading the ambient current one). It is safe and it is not the
        full arrangement: ADR-002's binding is missing, so a delegate that later reached for
        the current device would find the wrong one. Every caller that has an accelerator has
        a :class:`~shipinfer.runtime.device.DeviceManager` to pass, and the one path that does
        not — a chain needing ops but no model pool, in ``cli/commands/run.py`` — says so at
        its call site.
    """
    spread = tuple(devices) or (0,)
    # `is not None` and an explicit attribute read, so a test double that is not a real
    # DeviceManager is treated as "no accelerator" rather than crashing the first worker.
    accelerated = device_manager is not None and bool(
        getattr(device_manager, "has_accelerator", False)
    )

    def build(device_index: int) -> ImageOps:
        if accelerated:
            # Called lazily, on the worker thread itself: a CUDA context belongs to the thread
            # that created it, so binding from the composition root would bind the wrong one.
            device_manager.bind_current_thread(Device.cuda(device_index))
        staging = None
        if accelerated and memory is not None:
            # The owner key is per *thread*, not per device: several workers share a device in
            # rotation, and one pool between two of them is one buffer between two DMAs.
            owner = staging_owner(device_index)
            staging = memory.staging_for(owner) if claim is None else claim(memory, owner)
        return get_image_ops(provider, device_index=device_index, staging=staging)

    return ThreadLocalImageOps(build, devices=spread)
