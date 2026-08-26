"""Image operations on torch — the default when an accelerator is present.

Every op here maps onto a torch primitive that is already a tuned CUDA kernel:
``F.interpolate`` for resize, fused arithmetic for normalisation, and
``torchvision.ops.nms`` for suppression. Writing any of those by hand would be slower and
would need maintaining for both CUDA and ROCm (ADR-003).

Cropping is the one op with no torch primitive behind it, and the reason is worth writing
down so the next reader does not "simplify" it back into a loop or into a library call that
computes something else:

* ``F.interpolate`` resizes **one** source extent per call, and fifteen boxes on a frame
  have fifteen different extents. That is why this class used to run a Python loop with one
  launch per box — a cost that grew with the crowd, on the hot path of the embedder.
* ``F.grid_sample`` batches, but it clamps the far bilinear neighbour to the **frame**
  where this contract clamps it inside the **patch**: a 2-pixel-wide box upsampled to 4
  columns must end on exactly ``p[x1 + 1]``, and ``grid_sample`` returns
  ``0.75 * p[x1 + 1] + 0.25 * p[x2]`` — a real pixel from outside the box, so the error is
  plausible rather than obvious. Its ``[-1, 1]`` coordinate round trip also costs precision
  a direct index does not: one float32 ULP near 1.0 is ~1e-4 px on a 1920-wide frame.
* ``torchvision.ops.roi_align`` clamps to the feature map for the same reason, takes float
  extents, and zeroes outside the map instead of holding the edge. torchvision is an
  optional extra here, so the offline tier could not even import it.

So :meth:`TorchImageOps.crop_batch` composes the gather itself out of torch ops — index
tables, ``lerp``, and nothing hand-written per element — for a launch count that is
constant per pass instead of linear in the box count.

What torch *cannot* do in one pass is the fusion:
:class:`~shipinfer.runtime.ops.native_ops.NativeImageOps` runs resize + colour convert +
normalise + NHWC->NCHW as a single kernel, where this class runs four. That fusion is the
reason ``native/`` exists at all — it is the one place a custom kernel genuinely beats the
library.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from shipinfer.core.errors import ConfigurationError, DeviceError
from shipinfer.core.logging import get_logger
from shipinfer.runtime.ops.base import ImageOps, LetterboxResult, NormalizeParams
from shipinfer.runtime.ops.registry import IMAGE_OPS
from shipinfer.runtime.platform import require_torch

if TYPE_CHECKING:  # a type only; importing the pool here would tie every ops import to it
    from shipinfer.runtime.memory.staging import PinnedStagingPool

__all__ = ["TorchImageOps"]

_LOG = get_logger("runtime.ops.torch")


@IMAGE_OPS.register("torch")
class TorchImageOps(ImageOps):
    """Batched pre/post-processing through torch kernels."""

    name = "torch"

    #: float32 output elements one crop pass may produce. The pass holds four gathered
    #: corners and two interpolated rows, so the transient working set is about four times
    #: this — ~128 MiB at the value below. The bound is explicit because the per-box loop
    #: this replaced was O(1) in memory and a batched gather is not: 15 person crops at
    #: 256x128 (1.5M elements) run in one pass, while 40 ship crops at 512x512 (31M) run in
    #: four rather than asking the allocator for half a gigabyte at once.
    _CROP_CHUNK_ELEMENTS: ClassVar[int] = 8 * 1024 * 1024

    #: Entries a lookup table keeps before it is dropped whole. Keys come from model
    #: configs — a handful of ``(mean, std, device)`` and ``(swap_rb, device)`` tuples that
    #: live for the process — so this never trips in practice. It is here so that a caller
    #: synthesising :class:`NormalizeParams` per request turns a cache into a rebuild rather
    #: than into a leak on a 24/7 server.
    _CACHE_LIMIT: ClassVar[int] = 32

    #: float32 elements one *pinned host* staging buffer may hold. Deliberately its own
    #: bound and not derived from :attr:`_CROP_CHUNK_ELEMENTS`: that one caps device memory a
    #: caching allocator hands straight back, this one caps page-locked host memory, which
    #: the kernel can never swap and every process on the box competes for. The arithmetic,
    #: re-derivable: 2 Mi float32 elements = 8 MiB per buffer; only a call whose result
    #: spans more than one chunk stages at all (the structural rule in `_to_host`), and such
    #: a call uses a ping-pong pair, so one staged name costs at most 16 MiB. One name
    #: stages in practice ("crop", engaged by mask-sized batches: at (N, 3, 640, 640) a row
    #: is 1.17 Mi elements, so every ship is its own span) — at most 16 MiB of pinned memory
    #: per worker, released at the runner's stop and at MemoryPool.close().
    _STAGE_CHUNK_ELEMENTS: ClassVar[int] = 2 * 1024 * 1024

    def __init__(
        self,
        device_index: int | None = None,
        *,
        interpolation: str = "bilinear",
        staging: PinnedStagingPool | None = None,
    ) -> None:
        """Bind these ops to one device, and optionally to one caller's staging pool.

        Args:
            staging: the pinned host buffers this instance copies results back through, from
                the owning caller's :meth:`~shipinfer.runtime.memory.MemoryPool.staging_for`.
                A pool belongs to exactly one thread and so does an ``ImageOps``, which is
                what makes reusing a buffer safe (see :meth:`_to_host`).
        """
        self._torch = require_torch()
        self._device = (
            self._torch.device("cuda", device_index)
            if device_index is not None and self._torch.cuda.is_available()
            else self._torch.device("cpu")
        )
        self._interpolation = interpolation
        # Decided once here rather than per call, because the answer cannot change: a pool
        # only pays for itself where there is a DMA to accelerate. `.cpu()` on a tensor that
        # is already host memory is a free no-op, so a CPU-bound instance handed a pool would
        # bounce every result through an extra copy for nothing.
        self._staging = staging if staging is not None and self.on_device else None
        # Small constant tensors that depend only on the config, cached because building
        # them is a host-to-device copy: at 1000 frames a second, four synchronous copies
        # per frame for twelve floats that never change is pure latency on the worker.
        self._norm_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._channel_cache: dict[tuple[Any, ...], Any] = {}
        self._event_cache: dict[str, tuple[Any, Any]] = {}

    @property
    def on_device(self) -> bool:  # type: ignore[override]
        return self._device.type == "cuda"

    # -- constant tables ----------------------------------------------------------------

    def _normalization(self, params: NormalizeParams, device: Any) -> tuple[Any, Any]:
        """The broadcastable ``(mean, std)`` pair for ``params`` on ``device``.

        Keyed on the values rather than on the ``NormalizeParams`` object so two equal
        configs share one pair; the device is in the key because a tensor is only usable on
        the device it was built for, and this class is constructed per worker thread.
        """
        key = (tuple(params.mean), tuple(params.std), device)
        pair = self._norm_cache.get(key)
        if pair is None:
            torch = self._torch
            pair = (
                torch.tensor(params.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1),
                torch.tensor(params.std, dtype=torch.float32, device=device).view(1, 3, 1, 1),
            )
            self._remember(self._norm_cache, key, pair)
        return pair

    def _channel_order(self, swap_rb: bool, device: Any) -> Any:
        """The channel permutation as an index tensor.

        As an index it composes with the transpose out of the crop's channels-first layout,
        so BGR->RGB costs nothing beyond the copy that layout already needs — where ``flip``
        would be a second full pass over the batch.
        """
        key = (swap_rb, device)
        order = self._channel_cache.get(key)
        if order is None:
            torch = self._torch
            order = torch.tensor(
                [2, 1, 0] if swap_rb else [0, 1, 2], dtype=torch.long, device=device
            )
            self._remember(self._channel_cache, key, order)
        return order

    def _remember(self, cache: dict[Any, Any], key: Any, value: Any) -> None:
        """Insert, dropping the whole table first if it has grown past the bound.

        Dropping everything rather than evicting one entry: there is no useful recency order
        over a handful of config-derived tensors, and rebuilding them is two small copies.
        """
        if len(cache) >= self._CACHE_LIMIT:
            cache.clear()
        cache[key] = value

    # -- the trip home ------------------------------------------------------------------

    @classmethod
    def _stage_rows(cls, shape: tuple[int, ...]) -> int:
        """How many rows of ``shape`` one staging buffer holds — at least one.

        Flooring at one is the same policy as :meth:`_crop_chunks`: refusing a 512x512 mask
        because a single row exceeds the budget is a worse failure than one large buffer.
        """
        return max(1, cls._STAGE_CHUNK_ELEMENTS // max(1, math.prod(shape[1:])))

    def _to_host(self, tensor: Any, name: str) -> np.ndarray:
        """A device tensor as a host array, copied through pinned memory when there is a pool.

        A copy into *pageable* host memory never DMAs. The driver moves it in pieces through
        a bounce buffer of its own, which is why the same transfer measures around 1.4 GB/s
        pageable and around 10 GB/s pinned. The C++ plane had this exact defect in NMS —
        downloading the mask into a fresh pageable vector cost 30.8 ms a call against 1.7 ms
        through a pinned scratch (ledger C32) — and these two sites are the Python plane's
        version of it: at 1000 frames a second they carry the letterboxed batch and every
        crop back across PCIe.

        The pool key is the **fixed** shape ``(name, rows, *shape[1:])``, never the true row
        count. That is deliberate: a crowded frame's eighteen crops and a quiet frame's three
        must land in the same buffer. Keying on ``N`` would put one entry per crowd size into
        a 64-entry pool, evict, and turn steady-state calls back into ``cudaHostAlloc`` —
        slower than the pageable copy this replaces.

        Two buffers ping-pong on this thread's own stream: while the host copies chunk *k*
        out of one, chunk *k+1*'s DMA is already in flight into the other — the copy engine
        never idles behind the memcpy, and no second stream is needed (one `torch.cuda.Event`
        per buffer orders each read; ADR-002's one-thread-one-stream discipline holds).
        Single-chunk calls — the letterbox frame, a reid-sized crop set — degenerate to
        exactly the serial cost; the overlap engages where the pageable tails lived, the
        multi-chunk mask batches.

        Args:
            tensor: the result to bring home, batched along dimension 0.
            name: which buffer this is — one per call site, because two names in one pool are
                two buffers. Sharing a pool between ``"letterbox"`` and ``"crop"`` is safe
                for the stronger reason that every chunk is synchronised before this returns,
                so no DMA of this instance's is ever still in flight when it does.

        Returns:
            A freshly allocated host array the caller owns outright — never a view of the
            staging buffer, which the next call overwrites.
        """
        torch = self._torch
        shape = tuple(tensor.shape)
        if self._staging is None or not shape:
            return tensor.cpu().numpy()
        if shape[0] == 0:
            # Nothing to copy, and asking the pool would allocate a buffer for a batch that
            # does not exist.
            return torch.empty(shape, dtype=tensor.dtype).numpy()

        rows = self._stage_rows(shape)
        spans = [(lo, min(lo + rows, shape[0])) for lo in range(0, shape[0], rows)]
        if len(spans) == 1:
            # One span: no overlap to win, and the staged path would add a full-size serial
            # host memcpy that `.cpu()` never performs. The rule is structural on purpose
            # (#31 round 3): the production letterbox frame (1, 3, 640, 640) and a
            # design-sizing person-reid batch (~15, 3, 256, 128) both land here, and a
            # per-call-site human judgment already missed the second one once. What stages
            # is genuinely multi-chunk work — the mask batches, where every ship is its own
            # span and the ping-pong has something to overlap.
            return tensor.cpu().numpy()
        try:
            staged = (
                self._staging.get(f"{name}:a", (rows, *shape[1:]), tensor.dtype),
                self._staging.get(f"{name}:b", (rows, *shape[1:]), tensor.dtype),
            )
        except DeviceError:
            # A mid-capture refusal is transient by nature: take the ordinary path for THIS
            # call and ask again next time (#31 round 3 — "degrade once, forever" was
            # stronger than that failure warrants).
            return tensor.cpu().numpy()
        except RuntimeError as exc:
            # The host is out of lockable pages. Degrade once and then never ask again:
            # the array is identical either way, and an optimisation must not be able to
            # take a worker down.
            self._staging = None
            _LOG.warning(
                "pinned staging unavailable for %s on %s (%s); copying pageable from now on",
                name,
                self._device,
                exc,
            )
            return tensor.cpu().numpy()

        host = torch.empty(shape, dtype=tensor.dtype)
        tensor = tensor.contiguous()  # a no-op for the call site; a whole-batch copy if not
        stream = torch.cuda.current_stream(self._device)
        events = self._stage_events(name)
        for index, (lo, hi) in enumerate(spans):
            staged[index % 2][: hi - lo].copy_(tensor[lo:hi], non_blocking=True)
            events[index % 2].record(stream)
            if index:
                # Drain the PREVIOUS chunk while this one's DMA runs on the copy engine.
                # Waiting on the event is not optional: the buffer is reused two chunks on
                # and by the next call, so reading it before its DMA lands returns the
                # *previous* frame's pixels — no error, plausible values, invisible to
                # anything that submits one frame. The event waits on this thread's own
                # work, never the device-wide `torch.cuda.synchronize()` that would stall
                # every other worker sharing the GPU (ADR-002). Buffer reuse is safe by
                # host order: chunk k's memcpy-out finishes here before chunk k+2's DMA
                # into the same buffer is ever enqueued.
                plo, phi = spans[index - 1]
                events[(index - 1) % 2].synchronize()
                host[plo:phi].copy_(staged[(index - 1) % 2][: phi - plo])
        last = len(spans) - 1
        lo, hi = spans[last]
        events[last % 2].synchronize()
        host[lo:hi].copy_(staged[last % 2][: hi - lo])
        return host.numpy()

    def _stage_events(self, name: str) -> tuple[Any, Any]:
        """The two reusable CUDA events that order reads of ``name``'s ping-pong buffers.

        Cached per name for the same reason the buffers are: creating an event is a driver
        call, and this path runs per frame. An event is re-recorded only after its previous
        recording was synchronised, so reuse cannot observe a stale completion.
        """
        pair = self._event_cache.get(name)
        if pair is None:
            torch = self._torch
            pair = (torch.cuda.Event(), torch.cuda.Event())
            if len(self._event_cache) >= self._CACHE_LIMIT:
                self._event_cache.clear()
            self._event_cache[name] = pair
        return pair

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
            tensor=self._to_host(canvas, "letterbox"),
            scales=scales,
            pads=pads,
            extents=extents,
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

        mean, std = self._normalization(params, canvas.device)
        canvas[:n].sub_(mean).div_(std)
        return scales, pads, extents

    def crop_batch(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dst_size: tuple[int, int],
        params: NormalizeParams,
    ) -> np.ndarray:
        """Every box in a constant number of launches, whatever the crowd looks like.

        Two gathers over the whole set — one per bracketing source row, each fetching both
        bracketing columns — driven by index and weight tables built on the host, then two
        blends across and one down. The geometry is identical to the ``F.interpolate`` call
        this replaced — ``align_corners=False`` half-pixel centres, the far neighbour
        clamped inside the patch, a degenerate box blacked out before normalisation — see
        :func:`_bilinear_axis` and the module docstring for why no torch primitive does it.

        Raises:
            ConfigurationError: if this instance was built for an interpolation mode other
                than bilinear. The tables encode bilinear sampling, and silently cropping
                bilinearly for an operator who asked for nearest is the kind of mismatch
                that shows up as a slightly worse embedding and never as an error.
        """
        torch = self._torch
        dst_h, dst_w = dst_size
        if boxes.size == 0:
            return np.empty((0, 3, dst_h, dst_w), dtype=np.float32)
        if self._interpolation != "bilinear":
            raise ConfigurationError(
                f"crop_batch samples bilinearly, but these ops were built with "
                f"interpolation={self._interpolation!r}. Use a bilinear TorchImageOps for "
                f"crops, or the numpy implementation for nearest."
            )

        src_h, src_w = image.shape[:2]
        device = self._device
        # uint8 on the wire and a *view* for the transpose: only the gathered corners are
        # widened, so the frame crosses PCIe at a third of what a float32 copy would cost.
        # Pageable source, so the copy is synchronous whatever we ask; `non_blocking` would
        # claim an overlap that cannot happen (CONVENTIONS 2.4).
        planes = torch.from_numpy(np.ascontiguousarray(image)).to(device).permute(2, 0, 1)
        clipped = np.empty_like(boxes, dtype=np.int64)
        clipped[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_w - 1)
        clipped[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_h - 1)
        widths = clipped[:, 2] - clipped[:, 0]
        heights = clipped[:, 3] - clipped[:, 1]

        col0, col1, weight_x = _bilinear_axis(clipped[:, 0], widths, dst_w)
        row0, row1, weight_y = _bilinear_axis(clipped[:, 1], heights, dst_h)
        columns = torch.from_numpy(np.stack((col0, col1))).to(device)
        rows = torch.from_numpy(np.stack((row0, row1))).to(device)
        across = torch.from_numpy(weight_x).to(device)
        down = torch.from_numpy(weight_y).to(device)

        # Channels lead because the gather leaves them there: indexing dimensions 1 and 2
        # of a CHW frame puts the broadcast index shape where those dimensions were.
        count = int(boxes.shape[0])
        out = torch.empty((3, count, dst_h, dst_w), dtype=torch.float32, device=device)
        for lo, hi in self._crop_chunks(count, dst_h, dst_w):
            span = slice(lo, hi)
            column = columns[:, span].unsqueeze(2)  # (2, n, 1, dst_w)
            weight = across[span].unsqueeze(1)  # (n, 1, dst_w)
            top = self._sample_rows(planes, rows[0, span], column, weight)
            bottom = self._sample_rows(planes, rows[1, span], column, weight)
            torch.lerp(top, bottom, down[span].unsqueeze(2), out=out[:, span])

        degenerate = np.nonzero((widths <= 0) | (heights <= 0))[0]
        if degenerate.size:
            # Zeroed *before* normalisation, which is where the loop's untouched row sat: a
            # degenerate crop reads `(0 - mean) / std`. Blacking it out afterwards would
            # silently change the value for every model with a non-zero mean.
            out[:, torch.from_numpy(degenerate).to(device)] = 0.0

        # The transpose to NCHW and the BGR->RGB swap are one gather, and it is also what
        # makes the result contiguous — which `.numpy()` would otherwise not be.
        result = out.permute(1, 0, 2, 3)[:, self._channel_order(params.swap_rb, device)]
        mean, std = self._normalization(params, device)
        result.sub_(mean).div_(std)
        return self._to_host(result, "crop")

    def _sample_rows(self, planes: Any, rows: Any, column: Any, weight: Any) -> Any:
        """One bilinearly interpolated row per crop: gather both columns, blend across.

        A method rather than four inline lines so the two gathered corners die when it
        returns; holding all four alive at once would double the pass's peak footprint.

        Args:
            planes: the frame as a ``(3, H, W)`` view, any dtype the gather can widen.
            rows: ``(n, dst_h)`` int64 frame rows to read.
            column: ``(2, n, 1, dst_w)`` int64 left/right frame columns.
            weight: ``(n, 1, dst_w)`` float32 weight of the right column.
        """
        pair = planes[:, rows.unsqueeze(2), column].float()
        return self._torch.lerp(pair[:, 0], pair[:, 1], weight)

    def _crop_chunks(self, count: int, dst_h: int, dst_w: int) -> Iterator[tuple[int, int]]:
        """Half-open crop ranges whose output fits :attr:`_CROP_CHUNK_ELEMENTS`.

        At least one crop per range even when a single crop exceeds the budget: refusing a
        512x512 mask because it is large is a worse failure than a large allocation.
        """
        rows = max(1, self._CROP_CHUNK_ELEMENTS // (3 * dst_h * dst_w))
        for lo in range(0, count, rows):
            yield lo, min(lo + rows, count)

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
        # Not staged, unlike the two image paths: the survivors are a few hundred int64 at
        # most (2.4 KB at `max_output=300`), and the synchronise a staged copy needs costs
        # more than the copy itself.
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
        staged = " (pinned staging)" if self._staging is not None else ""
        return f"torch kernels on {self._device}{staged}"


def _bilinear_axis(
    origin: np.ndarray, extent: np.ndarray, dst: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bilinear sample table for one axis of every box at once, in frame coordinates.

    This is ATen's ``align_corners=False`` source index — ``scale * (d + 0.5) - 0.5`` in
    float32, clamped at zero, with ``scale = extent / dst`` — evaluated for the whole set
    and then shifted into frame coordinates by an integer add. Doing the arithmetic in
    *patch* coordinates is the point: the far neighbour is clamped to the patch's last
    row/column, so a 2-pixel box upsampled to 4 columns finishes on ``p[origin + 1]`` and
    never reads the pixel beyond the box the detector gave us. Sampling in frame
    coordinates instead — which is what ``grid_sample`` and ``roi_align`` do — bleeds the
    neighbouring object into the crop's edge.

    A non-positive extent (an empty or reversed box) collapses to ``origin`` instead of
    raising. The caller blacks those crops out, but the indices still have to be legal:
    an out-of-range gather on CUDA is a device-side assert, and that poisons the context for
    every later launch on the thread rather than failing this one call.

    Args:
        origin: ``(N,)`` int64 first row/column of each patch, already clipped to the frame.
        extent: ``(N,)`` int64 patch size along this axis, ``x2 - x1`` as the slice took it.
        dst: destination size along this axis.

    Returns:
        ``(i0, i1, w1)`` — two ``(N, dst)`` int64 frame indices bracketing each destination
        pixel, and the ``(N, dst)`` float32 weight of ``i1``.
    """
    extent_safe = np.maximum(extent, 1).astype(np.int64)
    scale = extent_safe.astype(np.float32) / np.float32(dst)
    centres = np.arange(dst, dtype=np.float32) + np.float32(0.5)
    src = scale[:, None] * centres[None, :] - np.float32(0.5)
    np.maximum(src, np.float32(0.0), out=src)

    last = (extent_safe - 1)[:, None]
    # The clamp is defensive on i0 — `floor(src) <= extent - 1` already holds for every
    # scale — and load-bearing on i1, which is where the patch clamp actually lives.
    i0 = np.minimum(src.astype(np.int64), last)
    w1 = (src - i0.astype(np.float32)).astype(np.float32)
    i1 = np.minimum(i0 + 1, last)
    return origin[:, None] + i0, origin[:, None] + i1, w1
