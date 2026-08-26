"""A pinned shared-memory ring between two processes on one box — the `service` topology's wire.

WHY THIS EXISTS
---------------
Topology C (`service`, ledger T3) lets a shard hand a crop batch to an instance on another
GPU in another process. ADR-002 forbids CUDA IPC across GPUs — opening handles on every other
device would cost G contexts per process, G² x ~300 MiB on the box — so a payload crosses
through **host memory**: the producer copies device → pinned slot, the consumer copies pinned
slot → its device (~125 µs each for a 1.5 MB batch on PCIe 4). The slot has to be visible to
both processes and pinned in both, which is what this module provides:
:mod:`multiprocessing.shared_memory` for the visibility, ``cudaHostRegister`` once per process
for the pinning.

THE PROTOCOL — one writer, one reader
--------------------------------------
Taken from vLLM's ``ShmRingBuffer`` (``vllm/distributed/device_communicators/shm_broadcast.py``):
a fixed number of fixed-size slots, each with a byte of state, no locks. vLLM has one writer
and many readers; we have the opposite need — many peers submitting to one owner — and the
design decision is to keep vLLM's discipline anyway: **one ring per (submitter, owner, model)
pair, so every ring has exactly one writer**. That makes a slot claim a plain store rather
than a compare-and-swap Python cannot do on shared memory, and it costs pairwise rings, which
is why ``slots`` defaults small.

A slot's state byte::

    FREE    -> claimed by the writer   (writer fills the payload)
    CLAIMED -> written                 (writer publishes: the reader may take it)
    WRITTEN -> taken by the reader     (reader is copying out)
    TAKEN   -> free                    (reader releases)

Each transition is a single-byte store after the work it announces, so a reader that sees
``WRITTEN`` sees a complete payload (x86 does not reorder stores; the CPython bytecode
boundary is a compiler barrier). Waiting is a short spin, then ``time.sleep(0)``, then short
sleeps up to a deadline — vLLM's spin → ``sched_yield`` → timeout, without the C extension.

THE HEADER
----------
The first page carries what a :class:`~shipinfer.scheduling.policies.base.Placeable` proxy
needs to read without a lock: the owner's queue depth and EWMA latency (stamped by the owner
on every enqueue/dequeue), a heartbeat, and a ``closed`` flag. Version and layout are checked
at ``open`` so two processes cannot disagree about slot sizes and call the result a model bug.

OFFLINE
-------
Everything here except :meth:`SharedRing.pinned_tensor` works with no driver and no torch:
the layout is arithmetic, the protocol is bytes in a ``memoryview``, and the tests run the
reader as a thread in the same process. Pinning is a per-process registration of the same
pages, done lazily and only when asked.

The rings are pairwise, so an owner multiplexes: there is deliberately no select/poll primitive
here. The intended consumer shape (the proxy layer implements it) is one thread sweeping its
rings round-robin with ``take(timeout_s=0)`` and a backoff when a whole sweep is idle — at the
box's ceiling (16 GPUs x 3 shared models is ~48 rings) a sweep is 48 one-byte state scans, and
``is_closed`` is a single byte read, so an idle sweep allocates nothing.
"""

from __future__ import annotations

import contextlib
import struct
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

from shipinfer.core.errors import DeviceError, RingClosedError, RingFullError, RingProtocolError

__all__ = ["RingHeader", "RingLayout", "SharedRing", "SlotState"]

#: Layout version. Bump when the header or the slot metadata changes shape.
RING_VERSION = 1
_MAGIC = 0x5348_5249  # "SHRI"
_PAGE = 4096


class SlotState:
    """The four values a slot's state byte takes, in the order the protocol moves through them."""

    FREE = 0
    CLAIMED = 1
    WRITTEN = 2
    TAKEN = 3


@dataclass(frozen=True, slots=True)
class RingLayout:
    """Where everything lives in the block. Pure arithmetic, so a test can pin it.

    ``| header page | slot states (one byte each, padded to a page) | payloads, page-aligned |``
    """

    slots: int
    slot_bytes: int

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"a ring needs at least one slot, got {self.slots}")
        if self.slot_bytes < 1:
            raise ValueError(f"a slot needs at least one byte, got {self.slot_bytes}")

    @property
    def header_bytes(self) -> int:
        return _PAGE

    @property
    def states_offset(self) -> int:
        return self.header_bytes

    @property
    def states_bytes(self) -> int:
        return _align_up(self.slots, _PAGE)

    @property
    def payload_offset(self) -> int:
        return self.states_offset + self.states_bytes

    @property
    def padded_slot_bytes(self) -> int:
        """Each payload starts on a page: ``cudaMemcpy`` from pinned memory likes it, and a
        slot can never straddle another's page."""
        return _align_up(self.slot_bytes, _PAGE)

    @property
    def total_bytes(self) -> int:
        return self.payload_offset + self.slots * self.padded_slot_bytes

    def state_offset(self, index: int) -> int:
        self._check(index)
        return self.states_offset + index

    def slot_offset(self, index: int) -> int:
        self._check(index)
        return self.payload_offset + index * self.padded_slot_bytes

    def _check(self, index: int) -> None:
        if not 0 <= index < self.slots:
            raise IndexError(f"slot {index} out of range for a ring of {self.slots}")


#: magic u32 | version u32 | slots u32 | slot_bytes u32 | depth u32 | pad u32 |
#: ewma_latency_us f64 | heartbeat_ns u64 | closed u8 | owner (63 bytes, NUL-padded)
_HEADER = struct.Struct("<IIIIIIdQB63s")


@dataclass(frozen=True, slots=True)
class RingHeader:
    """What a peer reads off the first page without taking a lock."""

    version: int
    slots: int
    slot_bytes: int
    depth: int
    ewma_latency_us: float
    heartbeat_ns: int
    closed: bool
    owner: str


_PENDING_LOCK = threading.Lock()
_PENDING_CLOSE: list[tuple[shared_memory.SharedMemory, Any]] = []


def reap_pending_closes() -> int:
    """Close the handles whose last view has since died; return how many still wait.

    A ring closed while a zero-copy payload was still in flight keeps its mapping alive until
    that view dies (the right lifetime for the bytes). This closes the mapping once it can be
    closed. It runs on every `close`, and a caller that wants the count can call it directly.
    """
    with _PENDING_LOCK:
        still = []
        for block, unregister in _PENDING_CLOSE:
            if unregister is not None:
                # The close-time in-flight window is long past by the time a reap runs, so
                # unpinning first is safe — and it must precede the close, which unmaps.
                unregister()
                unregister = None
            try:
                block.close()
            except BufferError:
                still.append((block, unregister))
        _PENDING_CLOSE[:] = still
        return len(still)


def _attach(name: str) -> shared_memory.SharedMemory:
    """Map an existing block without registering it with the resource tracker.

    Before Python 3.13 every ``SharedMemory`` attach registers the name for cleanup at exit, so a
    process that merely *opened* a peer's ring would unlink it when it stops (bpo-38119) and warn
    about "leaked" objects that were never its to free. The owner is the one that unlinks.
    """
    try:
        return shared_memory.SharedMemory(name=name, create=False, track=False)  # 3.13+
    except TypeError:
        block = shared_memory.SharedMemory(name=name, create=False)
        from multiprocessing import resource_tracker

        resource_tracker.unregister(block._name, "shared_memory")
        return block


class SharedRing:
    """One ring in a :class:`multiprocessing.shared_memory.SharedMemory` block.

    The **owner** creates it (:meth:`create`), stamps the header, takes written slots and
    releases them, and closes it. The one **writer** opens it by name (:meth:`open`), claims a
    free slot, fills the payload, publishes. Both sides may pin the block in their own process
    (:meth:`pinned_tensor`); the block itself exists once.
    """

    def __init__(
        self,
        block: shared_memory.SharedMemory,
        layout: RingLayout,
        *,
        name: str,
        owner: str,
        is_owner: bool,
    ) -> None:
        self._block = block
        self._layout = layout
        self._name = name
        self._owner = owner
        self._is_owner = is_owner
        self._view = memoryview(block.buf)
        self._next = 0
        self._pinned: Any = None
        self._pinned_ptr = 0
        self._closed_here = False
        self._detached = False

    # -- construction ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str, layout: RingLayout, *, owner: str) -> SharedRing:
        """Create the block and write the header. The caller is the owner (the reader)."""
        if len(owner.encode()) > 63:
            raise ValueError(f"owner name too long for the header: {owner!r}")
        block = shared_memory.SharedMemory(name=name, create=True, size=layout.total_bytes)
        ring = cls(block, layout, name=name, owner=owner, is_owner=True)
        ring._write_header(
            depth=0, ewma_latency_us=0.0, heartbeat_ns=time.monotonic_ns(), closed=False
        )
        for index in range(layout.slots):
            ring._set_state(index, SlotState.FREE)
        return ring

    @classmethod
    def open(cls, name: str, layout: RingLayout) -> SharedRing:
        """Attach to an existing block as its one writer, checking version and layout first.

        Raises:
            RingProtocolError: the block was created by a different version, or with a
                different number or size of slots — the two processes must agree before a
                byte of payload moves.
        """
        block = _attach(name)
        if block.size < _HEADER.size:
            block.close()
            raise RingProtocolError(f"ring {name!r} is {block.size} bytes: no header")
        fields = _HEADER.unpack_from(block.buf, 0)
        magic, version, slots, slot_bytes = fields[0], fields[1], fields[2], fields[3]
        owner = fields[9].rstrip(b"\0").decode(errors="replace")
        if magic != _MAGIC or version != RING_VERSION:
            block.close()
            raise RingProtocolError(
                f"ring {name!r}: header version {version} (magic {magic:#x}), this process "
                f"speaks version {RING_VERSION}; the two processes are not the same build"
            )
        if (slots, slot_bytes) != (layout.slots, layout.slot_bytes):
            block.close()
            raise RingProtocolError(
                f"ring {name!r}: created with {slots} slots of {slot_bytes} bytes, opened "
                f"expecting {layout.slots} of {layout.slot_bytes}"
            )
        return cls(block, layout, name=name, owner=owner, is_owner=False)

    # -- header ------------------------------------------------------------------------

    def header(self) -> RingHeader:
        if self._detached:
            # This handle was closed: a reader thread that wakes after `close()` sees a closed
            # ring and returns, rather than a ValueError from a released view.
            return RingHeader(
                RING_VERSION,
                self._layout.slots,
                self._layout.slot_bytes,
                0,
                0.0,
                0,
                True,
                self._owner,
            )
        f = _HEADER.unpack_from(self._view, 0)
        return RingHeader(
            version=f[1],
            slots=f[2],
            slot_bytes=f[3],
            depth=f[4],
            ewma_latency_us=f[6],
            heartbeat_ns=f[7],
            closed=bool(f[8]),
            owner=f[9].rstrip(b"\0").decode(errors="replace"),
        )

    _CLOSED_OFFSET = 40  # <IIIIII d Q B...: six u32 (24) + f64 (32) + u64 (40), then the byte

    @property
    def is_closed(self) -> bool:
        """One byte: has the owner closed this ring (or this handle detached)?

        The loop a consumer writes is ``while not ring.is_closed:`` with ``take`` returning
        ``None`` for an idle window — closed and idle are different events and this is the
        one that ends the loop. Allocation-free, so a spin may read it every iteration.
        """
        return self._detached or self._view[self._CLOSED_OFFSET] == 1

    def stamp(self, *, depth: int, ewma_latency_us: float) -> None:
        """The owner publishes its load. Called on every enqueue and dequeue; cheap on purpose.

        This is what a remote proxy's ``depth`` and ``ewma_latency_us`` read, and the
        heartbeat is what tells a peer the owner is alive. On a closed handle it is a no-op:
        a metrics thread stamping during shutdown must not crash on the released view.
        """
        if self._detached:
            return
        self._write_header(
            depth=depth,
            ewma_latency_us=ewma_latency_us,
            heartbeat_ns=time.monotonic_ns(),
            closed=self._closed_here,
        )

    def _write_header(
        self, *, depth: int, ewma_latency_us: float, heartbeat_ns: int, closed: bool
    ) -> None:
        _HEADER.pack_into(
            self._view,
            0,
            _MAGIC,
            RING_VERSION,
            self._layout.slots,
            self._layout.slot_bytes,
            depth,
            0,
            float(ewma_latency_us),
            heartbeat_ns,
            1 if closed else 0,
            self._owner.encode(),
        )

    # -- the protocol ------------------------------------------------------------------

    @property
    def layout(self) -> RingLayout:
        return self._layout

    @property
    def name(self) -> str:
        return self._name

    @property
    def owner(self) -> str:
        return self._owner

    def state(self, index: int) -> int:
        return self._view[self._layout.state_offset(index)]

    def _set_state(self, index: int, value: int) -> None:
        self._view[self._layout.state_offset(index)] = value

    @property
    def depth(self) -> int:
        """Slots not currently free — what a writer sees as the ring's backlog."""
        return sum(1 for i in range(self._layout.slots) if self.state(i) != SlotState.FREE)

    def claim(self, timeout_s: float) -> int:
        """Writer: take a free slot, or refuse with the numbers.

        Raises:
            RingFullError: no slot came free within ``timeout_s``. The request is still with
                the caller; the dispatcher's spill loop excludes this peer and re-selects.
        """
        deadline = time.monotonic() + timeout_s
        spins = 0
        while True:
            if self.is_closed:
                raise RingClosedError(self._owner, self._name)
            for _ in range(self._layout.slots):
                index = self._next
                self._next = (index + 1) % self._layout.slots
                if self.state(index) == SlotState.FREE:
                    self._set_state(index, SlotState.CLAIMED)
                    return index
            if time.monotonic() >= deadline:
                raise RingFullError(self._owner, self._name, self.depth, self._layout.slots)
            spins = _backoff(spins)

    def payload(self, index: int) -> memoryview:
        """The slot's bytes, ``slot_bytes`` long, as a writable view."""
        start = self._layout.slot_offset(index)
        return self._view[start : start + self._layout.slot_bytes]

    def publish(self, index: int) -> None:
        """Writer: the payload is complete; the reader may take it.

        Inert on a closed handle: the reader is gone with the ring, so there is nobody to
        see the slot, and a writer racing `close()` must not crash on the released view.
        """
        if self._detached:
            return
        if self.state(index) != SlotState.CLAIMED:
            raise RingProtocolError(
                f"ring {self._name!r}: slot {index} is not claimed (state {self.state(index)})"
            )
        self._set_state(index, SlotState.WRITTEN)

    def abandon(self, index: int) -> None:
        """Writer: a claimed slot goes straight back to free, unpublished.

        For the caller whose payload could not be written — a request that does not fit, a
        tensor that could not be copied — so the slot is not lost to the ring for the rest of
        its life. The reader never sees it: only ``WRITTEN`` is visible to ``take``.
        Inert on a closed handle, like `release`.
        """
        if self._detached:
            return
        if self.state(index) != SlotState.CLAIMED:
            raise RingProtocolError(
                f"ring {self._name!r}: slot {index} is not claimed (state {self.state(index)})"
            )
        self._set_state(index, SlotState.FREE)

    def take(self, timeout_s: float | None) -> int | None:
        """Owner: the next written slot, or ``None`` for "no work in this window".

        ``None`` means only that — an idle window is an ordinary event, and a loop that
        breaks on it dies the first quiet 50 ms. The loop to write is::

            while not ring.is_closed:
                index = ring.take(timeout_s=0.05)
                if index is None:
                    continue
                ...

        (:attr:`is_closed` is the event that ends the loop, and it is a one-byte read.)
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        spins = 0
        while True:
            if self._detached:
                return None
            for index in range(self._layout.slots):
                if self.state(index) == SlotState.WRITTEN:
                    self._set_state(index, SlotState.TAKEN)
                    return index
            if self.is_closed:
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None
            spins = _backoff(spins)

    def release(self, index: int) -> None:
        """Owner: the payload has been copied out; the slot is free again.

        Inert on a closed handle: a completion callback settling after `close()` has nothing
        left to free and must not crash.
        """
        if self._detached:
            return
        if self.state(index) != SlotState.TAKEN:
            raise RingProtocolError(
                f"ring {self._name!r}: slot {index} is not taken (state {self.state(index)})"
            )
        self._set_state(index, SlotState.FREE)

    # -- pinning -----------------------------------------------------------------------

    def pinned_tensor(self, index: int) -> Any:
        """The slot as a torch uint8 tensor over pinned memory, registered once per process.

        Pinning is per process because ``cudaHostRegister`` registers *this* process's
        mapping with *this* process's DMA engine; the block exists once. Requires torch and a
        device — the one method here that does, kept apart so everything else runs offline.
        """
        torch = self._torch_with_registration()
        start = self._layout.slot_offset(index)
        return torch.frombuffer(
            self._block.buf, dtype=torch.uint8, count=self._layout.slot_bytes, offset=start
        )

    def _torch_with_registration(self) -> Any:
        if self._pinned is not None:
            return self._pinned
        from shipinfer.runtime.platform import require_torch

        torch = require_torch()
        whole = torch.frombuffer(self._block.buf, dtype=torch.uint8)
        cudart = torch.cuda.cudart()
        status = cudart.cudaHostRegister(whole.data_ptr(), whole.numel(), 0)
        if int(status) != 0:
            raise DeviceError(
                f"cudaHostRegister failed for ring {self._name!r}: status {int(status)}"
            )
        # The base pointer is remembered so close() can unregister without creating a new
        # export of the block — a fresh `frombuffer` would itself block the close.
        self._pinned_ptr = whole.data_ptr()
        self._pinned = torch
        return torch

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        """Owner: mark closed so writers stop claiming and takers return; then release.

        A non-owner just detaches. The owner also unlinks the block: a ring whose owner is
        gone must not be reopened by a peer that has not noticed.
        """
        if self._is_owner and not self._closed_here:
            self._closed_here = True
            header = self.header()
            self._write_header(
                depth=header.depth,
                ewma_latency_us=header.ewma_latency_us,
                heartbeat_ns=time.monotonic_ns(),
                closed=True,
            )
        unregister = None
        if self._pinned is not None:
            torch_module, pinned_ptr = self._pinned, self._pinned_ptr
            self._pinned = None

            def unregister() -> None:
                torch_module.cuda.cudart().cudaHostUnregister(pinned_ptr)

        # A payload decoded without a copy may still view the block (an in-flight tensor);
        # releasing under it would raise `BufferError` and skip the unlink. The mapping then
        # lives until the last view dies, which is the right lifetime — the *name* goes now.
        self._detached = True
        quiesced = True
        try:
            self._view.release()
        except BufferError:
            quiesced = False
        # Whether or not that worked, drop our export: the callers' slices are then the only
        # thing keeping the mapping, and the handle can close the moment the last one dies.
        self._view = memoryview(b"")
        closed_now = False
        if quiesced:
            # Nothing views the block through this handle, so no copy through it can be in
            # flight: unregister now, while the mapping is certainly alive, then try to close.
            if unregister is not None:
                unregister()
                unregister = None
            try:
                self._block.close()
                closed_now = True
            except BufferError:
                pass  # a pinned_tensor is still held somewhere; parked below
        if not closed_now:
            # Keep the handle: its finaliser would retry the close under the same view and
            # fail noisily at an arbitrary later point. It is closed at the next opportunity,
            # and an unregister that could not run safely yet rides along with it.
            with _PENDING_LOCK:
                _PENDING_CLOSE.append((self._block, unregister))
        reap_pending_closes()
        if self._is_owner:
            # `_attach` unregisters on open; in one process (the tests, and any same-process
            # pair) that removes the create-time entry too, because the tracker's cache is a
            # set. Re-register before unlink so its own unregister finds the entry instead of
            # a KeyError in the tracker daemon.
            from multiprocessing import resource_tracker

            resource_tracker.register(self._block._name, "shared_memory")
            with contextlib.suppress(FileNotFoundError):
                self._block.unlink()

    def __enter__(self) -> SharedRing:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<SharedRing {self._name!r} owner={self._owner!r} slots={self._layout.slots} slot_bytes={self._layout.slot_bytes}>"


def _align_up(value: int, to: int) -> int:
    return (value + to - 1) // to * to


def _backoff(spins: int) -> int:
    """Spin briefly, yield, then sleep in short steps — vLLM's waiting shape without the extension."""
    if spins < 64:
        pass
    elif spins < 256:
        time.sleep(0)
    else:
        time.sleep(0.00005)
    return spins + 1
