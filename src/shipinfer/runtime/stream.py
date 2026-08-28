"""Streams and events — thin wrappers over ``torch.cuda``.

One stream is enough to be correct and never enough to be fast: with a single stream an
instance does H2D, then compute, then D2H strictly in order, and the copy engines idle for
two of those three phases. With ``k`` streams the instance overlaps the H2D of batch *n+1*
with the compute of *n* and the D2H of *n-1*.

The wrapper is thin on purpose. ``torch.cuda.Stream`` already does the hard parts —
lifetime, the allocator's stream-awareness, ROCm parity — so this adds exactly two things
torch does not: a CPU no-op fallback so callers need no ``if device.is_cuda`` branch, and a
round-robin pool.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from shipinfer.core.errors import DeviceError
from shipinfer.core.logging import LOG
from shipinfer.core.types import Device
from shipinfer.runtime.platform import require_torch

__all__ = ["Stream", "StreamPool"]


class Stream:
    """A ``torch.cuda.Stream``, or a no-op on CPU."""

    __slots__ = ("_device", "_stream", "_torch")

    def __init__(self, device: Device, torch_stream: Any | None = None) -> None:
        self._device = device
        if device.is_cuda:
            self._torch = require_torch()
            self._stream = torch_stream or self._torch.cuda.Stream(device=device.index)
        else:
            self._torch = None
            self._stream = None

    @property
    def device(self) -> Device:
        return self._device

    @property
    def torch_stream(self) -> Any | None:
        """The underlying ``torch.cuda.Stream``, or ``None`` on CPU."""
        return self._stream

    @property
    def handle(self) -> int:
        """The raw ``cudaStream_t``, for handing to TensorRT."""
        return int(self._stream.cuda_stream) if self._stream is not None else 0

    @contextmanager
    def activate(self) -> Iterator[Stream]:
        """Make this the current stream for the block.

        Every torch op issued inside runs here, and — importantly — torch's caching
        allocator records the stream on any block allocated inside, so it will not be
        reused by another stream before this one has finished with it.
        """
        if self._stream is None:
            yield self
            return
        with self._torch.cuda.stream(self._stream):
            yield self

    def synchronize(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()

    def wait_stream(self, other: Stream) -> None:
        """Order this stream after everything currently queued on ``other``."""
        if self._stream is not None and other._stream is not None:
            self._stream.wait_stream(other._stream)

    def record_event(self) -> Any:
        if self._stream is None:
            return None
        event = self._torch.cuda.Event(enable_timing=False)
        event.record(self._stream)
        return event

    def close(self) -> None:
        """Torch owns the stream's lifetime; dropping the reference is the release."""
        self._stream = None

    def __repr__(self) -> str:
        return f"<Stream {self._device} handle=0x{self.handle:x}>"


class StreamPool:
    """A fixed set of streams for one instance, handed out round-robin.

    Round-robin rather than least-busy: querying stream state costs a driver call, and with
    a small fixed pool draining one queue the rotation is already balanced. The goal is
    overlap, not optimal assignment.
    """

    def __init__(self, device: Device, count: int) -> None:
        if count < 1:
            raise DeviceError(f"stream pool needs at least one stream, got {count}")
        self._device = device
        self._streams: list[Stream] = [Stream(device) for _ in range(count)]
        self._cursor = itertools.count()
        LOG.debug("created %d stream(s) on %s", count, device)

    @property
    def device(self) -> Device:
        return self._device

    @property
    def streams(self) -> Sequence[Stream]:
        return tuple(self._streams)

    def __len__(self) -> int:
        return len(self._streams)

    def next(self) -> Stream:
        return self._streams[next(self._cursor) % len(self._streams)]

    @contextmanager
    def borrow(self) -> Iterator[Stream]:
        """Take a stream for a block, and synchronise it on the way out.

        The synchronise is what makes the borrow safe: handing back a stream with work
        still queued would let the next borrower's copies race the previous borrower's
        kernels over the same pooled buffers.
        """
        stream = self.next()
        try:
            yield stream
        finally:
            stream.synchronize()

    def synchronize_all(self) -> None:
        for stream in self._streams:
            stream.synchronize()

    def close(self) -> None:
        for stream in self._streams:
            stream.close()
        self._streams.clear()

    def __repr__(self) -> str:
        return f"<StreamPool {self._device} n={len(self._streams)}>"
