"""A pinned shared-memory ring between two processes on one box — the `service` wire.

A payload crosses through host memory, not CUDA IPC (ADR-002: G contexts per process is too
much): device → pinned slot → the peer's device. `multiprocessing.shared_memory` gives the
visibility, one `cudaHostRegister` per process the pinning.

The protocol is vLLM's `ShmRingBuffer`: fixed slots, one state byte each, no locks — but
**one ring per (submitter, owner, model)**, so every ring has exactly one writer and a claim
is a plain store. State: FREE → CLAIMED (writer fills) → WRITTEN (reader may take) → TAKEN →
FREE. The byte is stored *after* the work it announces, so WRITTEN implies a complete payload
on x86-TSO; a weakly ordered port needs a fence here. Waiting spins, yields, then sleeps to a
deadline. The header page carries queue depth, EWMA latency, a heartbeat and `closed`, read
without a lock; layout and version are checked at `open`.

Everything but :meth:`SharedRing.pinned_tensor` runs with no driver. A writer that dies
between `claim` and `publish` strands that slot for the ring's life — the heartbeat takes the
whole ring down instead (`PeerLostError`). Rings are pairwise with no poll primitive: a
consumer sweeps its rings with ``take(timeout_s=0)``. A pinned handle must be `close()`d, not
dropped, and a tensor from `pinned_tensor` must outlive every copy it feeds. `abandon` leaves
stale bytes, so a consumer frames its own payload and never trusts the slot's tail.
"""

from __future__ import annotations

import contextlib
import struct
import threading
import time
import weakref
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

from shipinfer.core.errors import DeviceError, RingClosedError, RingFullError, RingProtocolError

__all__ = [
    "RingHeader",
    "RingLayout",
    "SharedRing",
    "SlotState",
    "reap_pending_closes",
]

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
#: The stamp's in-place window: depth u32, reserved u32, ewma f64, heartbeat u64 at offset 16
#: — everything before it (magic, version, layout) and after it (closed, owner) untouched.
_STAMP = struct.Struct("<IIdQ")
_STAMP_OFFSET = 16  # after magic, version, slots, slot_bytes — four u32


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


# Reentrant for the same reason `_pin_lock` is: `_finalize_handle` and `_pinned_view_died`
# are weakref.finalize callbacks, which run synchronously in whatever thread triggers the
# collection — including a gc landing on the allocations *inside* these critical sections
# (reap's list mutations, open's set add). Re-entry is benign at every site: each mutates its
# own structure and none re-reads a value computed before a possible re-entry.
_PENDING_LOCK = threading.RLock()
_PENDING_CLOSE: list[shared_memory.SharedMemory] = []
#: Names this process has opened (not created) -> the identity token of the handle holding
#: the claim. The one-writer discipline means a second open is a wiring bug; keying by token
#: means only the holder (its close, or its finalizer) can release the claim — a dead handle
#: collected *after* a successor opened must not silently re-permit two writers.
_OPENED_CLAIMS: dict[str, object] = {}


def reap_pending_closes() -> int:
    """Close the handles whose last view has since died; return how many still wait.

    A ring closed while a zero-copy payload was still in flight keeps its mapping alive until
    that view dies (the right lifetime for the bytes). This closes the mapping once it can be
    closed. It runs on every `close`, and a caller that wants the count can call it directly.
    """
    with _PENDING_LOCK:
        still = []
        for block in _PENDING_CLOSE:
            try:
                block.close()
            except BufferError:
                still.append(block)
        _PENDING_CLOSE[:] = still
        return len(still)


def _finalize_handle(
    view: memoryview, opened_name: str | None, claim_token: object | None = None
) -> None:
    """Finalizer for a handle that was never closed.

    Drops the view's export so ``SharedMemory.__del__`` can close quietly, and — for a writer
    handle — releases the one-open-per-process claim *if this handle still holds it*: a
    closed-then-collected predecessor firing after its successor opened must not release the
    successor's claim and silently re-permit two writers over one segment.
    """
    with contextlib.suppress(Exception):
        view.release()
    if opened_name is not None:
        with _PENDING_LOCK:
            if opened_name in _OPENED_CLAIMS and _OPENED_CLAIMS[opened_name] is claim_token:
                del _OPENED_CLAIMS[opened_name]


def _attach(name: str) -> shared_memory.SharedMemory:
    """Map an existing block without registering it with the resource tracker.

    Raises:
        RingClosedError: the name does not exist — never created, or its owner unlinked it on
            the way out. Typed so a submit path that catches ``QueueFullError`` treats a peer
            restart as an excluded candidate, not an unhandled ``OSError``.

    Before Python 3.13 every ``SharedMemory`` attach registers the name for cleanup at exit, so a
    process that merely *opened* a peer's ring would unlink it when it stops (bpo-38119) and warn
    about "leaked" objects that were never its to free. The owner is the one that unlinks.
    """
    try:
        try:
            return shared_memory.SharedMemory(name=name, create=False, track=False)  # 3.13+
        except TypeError:
            block = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError as exc:
        raise RingClosedError("unknown", name, reason="absent") from exc
    except ValueError as exc:
        # The window even before the sub-header one: shm_open creates the name at size 0 and
        # the creator's ftruncate has not run yet, so the attach's own mmap refuses with
        # ValueError('cannot mmap an empty file') — seen straight out of a mesh connect as
        # CI's third spelling of the same birth race (27 Aug). Retryable for the same reason
        # as absent and unborn; a persistently empty file becomes the connect deadline's
        # configuration answer like the others.
        raise RingClosedError("unknown", name, reason="unborn") from exc
    from multiprocessing import resource_tracker

    resource_tracker.unregister(block._name, "shared_memory")
    return block


class SharedRing:
    """One ring in a :class:`multiprocessing.shared_memory.SharedMemory` block.

    The **owner** creates it (:meth:`create`), stamps the header, takes written slots and
    releases them, and closes it. The one **writer** opens it by name (:meth:`open`), claims a
    free slot, fills the payload, publishes. Both sides may pin the block in their own process
    (:meth:`pinned_tensor`); the block itself exists once.

    **The closed-handle rule, applied to the whole surface:** a closed (or detached) handle
    answers questions honestly, refuses to hand out data, and treats late transitions as
    no-ops. Questions: ``is_closed`` is True, ``header()`` answers a synthetic closed header,
    ``depth`` is 0, ``state(i)`` is FREE. Data: ``payload`` and ``pinned_tensor`` raise
    :class:`RingClosedError`. Work: ``claim`` raises :class:`RingClosedError`, ``take``
    returns ``None``. Transitions: ``publish``, ``release``, ``abandon`` and ``stamp`` return
    without effect — a completion callback or metrics thread settling after ``close()`` is an
    ordinary shutdown event, not a crash.
    """

    def __init__(
        self,
        block: shared_memory.SharedMemory,
        layout: RingLayout,
        *,
        name: str,
        owner: str,
        is_owner: bool,
        claim_token: object | None = None,
    ) -> None:
        self._block = block
        self._layout = layout
        self._name = name
        self._owner = owner
        self._is_owner = is_owner
        self._view = memoryview(block.buf)
        # Guards every access to `_view` against `close()` swapping it out from another thread
        # (the consumer loop and the closer are concurrent by design). Uncontended acquire is
        # tens of nanoseconds against a ~125 us slot copy; slices already handed out stay
        # valid regardless — they export the mapping itself, not this object.
        self._view_lock = threading.Lock()
        # The submitter's state machine (claim's test-and-set, publish, abandon) is driven
        # from many request threads by the dispatcher — the same reason the local queue wraps
        # its put in a lock. Plain, not reentrant: no finalizer touches it. Never held across
        # a backoff sleep. Ordering: _submit_lock -> _view_lock, never the reverse.
        self._submit_lock = threading.Lock()
        # A handle dropped without close() must not strand the mapping: the finalizer lets
        # `SharedMemory.__del__` succeed instead of printing an ignored BufferError.
        weakref.finalize(
            self, _finalize_handle, self._view, None if is_owner else name, claim_token
        )
        self._claim_token = claim_token
        self._next = 0
        self._take_next = 0
        self._pinned: Any = None
        self._pinned_ptr = 0
        # Reentrant, deliberately: `_pinned_view_died` is a weakref.finalize callback and
        # those run synchronously in whatever thread triggers the collection — including a
        # cyclic gc landing on the allocation *inside* `pinned_tensor`'s locked region. With a
        # plain Lock that thread deadlocks on itself; re-entry here is benign (the count was
        # already taken, and `_parked_unregister` is still None at every allocation point).
        self._pin_lock = threading.RLock()
        self._pinned_live = 0
        #: Latched under `_pin_lock` by `close()` before anything is unpinned, so a handout
        #: racing the close is refused instead of registering pages nothing will ever free.
        self._pin_closed = False
        self._parked_unregister: Any = None
        self._closed_here = False
        self._unlinked = False
        self._detached = False

    # -- construction ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str, layout: RingLayout, *, owner: str) -> SharedRing:
        """Create the block and write the header. The caller is the owner (the reader)."""
        if len(owner.encode()) > 63:
            raise ValueError(f"owner name too long for the header: {owner!r}")
        try:
            block = shared_memory.SharedMemory(name=name, create=True, size=layout.total_bytes)
        except FileExistsError as exc:
            raise RingProtocolError(
                f"ring {name!r} already exists — a previous owner's segment survived its "
                f"process; remove it or use a fresh run id"
            ) from exc
        ring = cls(block, layout, name=name, owner=owner, is_owner=True)
        # States first, header body next, magic LAST — because writing the magic IS the
        # readiness signal (`open` retries until it lands), everything a peer may touch must
        # be final before it. That includes the rest of the header itself: the one-slice
        # pack writes magic at offset 0 *first* in a forward memcpy, so a peer catching the
        # copy mid-flight could see magic set with slots still 0 and sail past the unborn
        # branch into the terminal created-with-0-slots refusal — the fourth spelling of the
        # birth race (#37 round 1). So the header is written with magic=0 (still "unborn" to
        # every reader), and the magic word is stored on its own afterwards. POSIX shm is
        # zero-filled and FREE == 0, so the state loop is belt-and-braces, but the *order*
        # is the contract — a peer that wins the open race and publishes must never have its
        # slot stamped back to FREE by the creator's own initialisation.
        for index in range(layout.slots):
            ring._set_state(index, SlotState.FREE)
        ring._write_header(
            depth=0,
            ewma_latency_us=0.0,
            heartbeat_ns=time.monotonic_ns(),
            closed=False,
            magic=0,
        )
        struct.pack_into("<I", block.buf, 0, _MAGIC)
        return ring

    @classmethod
    def open(cls, name: str, layout: RingLayout) -> SharedRing:
        """Attach to an existing block as its one writer, checking version and layout first.

        One open per ring per process: a second `_attach` of the same name would unbalance
        the resource tracker's set-based cache (a daemon-side KeyError at exit). The
        one-writer discipline makes a second open a wiring bug anyway.

        Raises:
            RingProtocolError: the block was created by a different version, or with a
                different number or size of slots — the two processes must agree before a
                byte of payload moves.
        """
        block = _attach(name)
        if block.size < _HEADER.size:
            # The name becomes visible at shm creation, BEFORE the creator sizes the block:
            # a block shorter than the header is a ring seen even earlier in its birth than
            # the all-zero header below — the same "not ready", retryable for the same
            # reason. Raising the protocol error here killed the whole mesh connect on the
            # first attach that won that race (twice on CI's runners, 27 Aug); the connect
            # loop's deadline is what turns a persistently sizeless block into its own
            # configuration answer.
            block.close()
            raise RingClosedError("unknown", name, reason="unborn")
        fields = _HEADER.unpack_from(block.buf, 0)
        magic, version, slots, slot_bytes = fields[0], fields[1], fields[2], fields[3]
        if magic == 0:
            # The name becomes visible at creation, *before* the creator writes the header:
            # an all-zero header is a ring mid-birth (or a zero-filled foreign block), and
            # both are "not ready" — retryable — not a build mismatch. The connect loop
            # retries on RingClosedError and turns a persistent one into its own error.
            block.close()
            raise RingClosedError("unknown", name, reason="unborn")
        owner = fields[9].rstrip(b"\0").decode(errors="replace")
        if magic != _MAGIC or version != RING_VERSION:
            block.close()
            raise RingProtocolError(
                f"ring {name!r}: header version {version} (magic {magic:#x}), this process "
                f"speaks version {RING_VERSION}; the two processes are not the same build"
            )
        if block.size < layout.total_bytes:
            block.close()
            raise RingProtocolError(
                f"ring {name!r}: block is {block.size} bytes, layout needs "
                f"{layout.total_bytes} — a short block would clamp payloads silently"
            )
        if (slots, slot_bytes) != (layout.slots, layout.slot_bytes):
            block.close()
            raise RingProtocolError(
                f"ring {name!r}: created with {slots} slots of {slot_bytes} bytes, opened "
                f"expecting {layout.slots} of {layout.slot_bytes}"
            )
        claim_token = object()
        with _PENDING_LOCK:
            if name in _OPENED_CLAIMS:
                block.close()
                raise RingProtocolError(
                    f"ring {name!r} is already open in this process — one writer per ring"
                )
            _OPENED_CLAIMS[name] = claim_token
        return cls(
            block, layout, name=name, owner=owner, is_owner=False, claim_token=claim_token
        )

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
        with self._view_lock:
            if self._detached:
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

    _CLOSED_OFFSET = _HEADER.size - 64  # the byte before the 63-byte owner name and its NUL

    @property
    def is_closed(self) -> bool:
        """One byte: has the owner closed this ring (or this handle detached)?

        The loop a consumer writes is ``while not ring.is_closed:`` with ``take`` returning
        ``None`` for an idle window — closed and idle are different events and this is the
        one that ends the loop. Allocation-free, so a spin may read it every iteration.
        """
        with self._view_lock:
            return self._detached or self._view[self._CLOSED_OFFSET] == 1

    def load_signal(self) -> tuple[int, float, int]:
        """``(depth, ewma_latency_us, heartbeat_ns)`` — one unpack over the stamp window.

        The cheap read for the per-request path: `header()` builds a dataclass and decodes
        the owner name, which is five allocations a policy touching every candidate per
        request does not need. A detached handle answers ``(0, 0.0, 0)``.
        """
        with self._view_lock:
            if self._detached:
                return (0, 0.0, 0)
            depth, _pad, ewma, heartbeat = _STAMP.unpack_from(self._view, _STAMP_OFFSET)
        return (depth, ewma, heartbeat)

    def stamp(self, *, depth: int, ewma_latency_us: float) -> None:
        """The owner publishes its load. Called on every enqueue and dequeue; cheap on purpose.

        This is what a remote proxy's ``depth`` and ``ewma_latency_us`` read, and the
        heartbeat is what tells a peer the owner is alive. Owner-only: a peer holds the other
        end to *read*, and a stamp that re-packed the header from a non-owner would clear the
        ``closed`` flag the owner set. Writes exactly its three fields in place — magic,
        version and layout stay whatever ``open()`` validated. On a closed handle: a no-op.
        """
        if self._detached:
            return
        if not self._is_owner:
            raise RingProtocolError(
                f"ring {self._name!r}: only the owner stamps; this handle merely reads"
            )
        depth = max(0, min(int(depth), 0xFFFFFFFF))  # the metrics path must not struct.error
        with self._view_lock:
            if self._detached:
                return
            # `pack_into` memsets its whole target region to zero *before* packing, so a
            # lock-free reader in another process can observe depth == 0 on a saturated ring
            # or heartbeat_ns == 0 on a healthy one (measured: ~24% of colliding reads torn).
            # Packing to bytes first makes the write one memcpy of naturally aligned fields:
            # a reader can see an old field next to a new one — stale, harmless — never zero.
            self._view[_STAMP_OFFSET : _STAMP_OFFSET + _STAMP.size] = _STAMP.pack(
                depth, 0, float(ewma_latency_us), time.monotonic_ns()
            )

    def _write_header(
        self,
        *,
        depth: int,
        ewma_latency_us: float,
        heartbeat_ns: int,
        closed: bool,
        magic: int = _MAGIC,
    ) -> None:
        with self._view_lock:
            if self._detached:
                return
            # Pack first, one memcpy: see `stamp` — `pack_into` memsets the region, and a
            # peer's `header()` mid-write would read slots == 0, slot_bytes == 0.
            self._view[: _HEADER.size] = _HEADER.pack(
                magic,
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
        with self._view_lock:
            if self._detached:
                return int(SlotState.FREE)
            return self._view[self._layout.state_offset(index)]

    def _set_state(self, index: int, value: int) -> None:
        with self._view_lock:
            if self._detached:
                return
            self._view[self._layout.state_offset(index)] = value

    def _swap_state(
        self, index: int, expect: int, to: int, *, expected_word: str | None = None
    ) -> bool:
        """The read-modify-write as ONE critical section, with the detached check inside.

        Returns True on the swap; False when the handle detached under the caller (the
        transition is a no-op, per the closed rule) or — scanning callers, ``expected_word``
        None — when the state differs. With ``expected_word`` set, a differing state raises
        :class:`RingProtocolError`: splitting the check and the write across two acquisitions
        let a racing ``close()`` turn an ordinary shutdown into a fake build-mismatch page.
        """
        with self._view_lock:
            if self._detached:
                return False
            offset = self._layout.state_offset(index)
            current = self._view[offset]
            if current != expect:
                if expected_word is not None:
                    raise RingProtocolError(
                        f"ring {self._name!r}: slot {index} is not {expected_word} "
                        f"(state {current})"
                    )
                return False
            self._view[offset] = to
            return True

    @property
    def depth(self) -> int:
        """Slots not currently free — what a writer sees as the ring's backlog."""
        with self._view_lock:
            if self._detached:
                return 0
            offset = self._layout.state_offset(0)
            states = self._view[offset : offset + self._layout.slots]
            return sum(1 for value in states if value != SlotState.FREE)

    def claim(self, timeout_s: float) -> int:
        """Writer: take a free slot, or refuse with the numbers.

        Raises:
            RingFullError: no slot came free within ``timeout_s``. The request is still with
                the caller; the dispatcher's spill loop excludes this peer and re-selects.
        """
        if self._is_owner:
            raise RingProtocolError(
                f"ring {self._name!r}: the owner reads; claiming is the submit handle's — an "
                f"owner-claimed slot would come back through its own take as a real payload"
            )
        deadline = time.monotonic() + timeout_s
        spins = 0
        while True:
            if self.is_closed:
                raise RingClosedError(self._owner, self._name)
            with self._submit_lock:
                # The test-and-set and the cursor bump are one critical section: two request
                # threads that both saw FREE used to both claim the same slot, and one
                # camera's payload was published over another's — the misattribution ADR-002's
                # tag rule exists to prevent, silent on the side that shipped.
                for _ in range(self._layout.slots):
                    index = self._next
                    self._next = (index + 1) % self._layout.slots
                    if self._swap_state(index, SlotState.FREE, SlotState.CLAIMED):
                        return index
            if time.monotonic() >= deadline:
                raise RingFullError(self._owner, self._name, self.depth, self._layout.slots)
            spins = _backoff(spins)

    def payload(self, index: int) -> memoryview:
        """The slot's bytes, ``slot_bytes`` long, as a writable view.

        Raises:
            RingClosedError: on a closed handle. ``memoryview`` slicing clamps, so without
                the guard this would answer a 0-byte view — and an empty buffer read as "no
                detections" is the silent-wrong-answer the house rules name as blocking.
        """
        with self._view_lock:
            if self._detached:
                raise RingClosedError(self._owner, self._name)
            start = self._layout.slot_offset(index)
            return self._view[start : start + self._layout.slot_bytes]

    def publish(self, index: int) -> None:
        """Writer: the payload is complete; the reader may take it.

        Inert on a closed handle: the reader is gone with the ring, so there is nobody to
        see the slot, and a writer racing `close()` must not crash on the released view.
        **Publishing into a ring whose owner closed between `claim` and here succeeds
        silently** — the write is harmless, but nobody will take it. A submitter that must
        not lose the batch re-checks `is_closed` after publishing, or relies on its
        heartbeat watcher failing the pending future (`PeerLostError`), which is what the
        proxy layer does.
        """
        if self._detached:
            return
        with self._submit_lock:
            self._publish_locked(index)

    def _publish_locked(self, index: int) -> None:
        self._swap_state(index, SlotState.CLAIMED, SlotState.WRITTEN, expected_word="claimed")

    def abandon(self, index: int) -> None:
        """Writer: a claimed slot goes straight back to free, unpublished.

        For the caller whose payload could not be written — a request that does not fit, a
        tensor that could not be copied — so the slot is not lost to the ring for the rest of
        its life. The reader never sees it: only ``WRITTEN`` is visible to ``take``.
        Inert on a closed handle, like `release`.
        """
        if self._detached:
            return
        with self._submit_lock:
            self._abandon_locked(index)

    def _abandon_locked(self, index: int) -> None:
        self._swap_state(index, SlotState.CLAIMED, SlotState.FREE, expected_word="claimed")

    def take(self, timeout_s: float | None) -> int | None:
        """Owner: a written slot, or ``None`` for "no work in this window".

        **The ring does not preserve submission order.** ``claim`` skips busy slots, so a
        late message can land in a lower slot than an earlier one; the rotating scan here
        bounds the damage — every written slot is taken within one rotation, so displacement
        is at most the slot count and nothing starves — but consumers must key on the payload
        (the request id, the ``(camera_id, frame_id)`` tag), never on arrival order.

        ``None`` means only that — an idle window is an ordinary event, and a loop that
        breaks on it dies the first quiet 50 ms. The loop to write is::

            while not ring.is_closed:
                index = ring.take(timeout_s=0.05)
                if index is None:
                    continue
                ...

        (:attr:`is_closed` is the event that ends the loop, and it is a one-byte read.)
        """
        if not self._is_owner:
            raise RingProtocolError(
                f"ring {self._name!r}: only the owner takes; this handle submits"
            )
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        spins = 0
        while True:
            if self._detached:
                return None
            for step in range(self._layout.slots):
                index = (self._take_next + step) % self._layout.slots
                if self._swap_state(index, SlotState.WRITTEN, SlotState.TAKEN):
                    self._take_next = (index + 1) % self._layout.slots
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
        if not self._is_owner:
            raise RingProtocolError(
                f"ring {self._name!r}: only the owner releases; this handle submits"
            )
        self._swap_state(index, SlotState.TAKEN, SlotState.FREE, expected_word="taken")

    # -- pinning -----------------------------------------------------------------------

    def pinned_tensor(self, index: int) -> Any:
        """The slot as a torch uint8 tensor over pinned memory, registered once per process.

        **The caller's obligation:** liveness is tracked on the *tensor object*, not on the
        copies it feeds. torch does not extend a host source's lifetime across an async H2D
        copy, so the caller must keep this tensor alive until every ``non_blocking`` copy from
        it has synchronised (record the stream or event and sync before dropping it) —
        otherwise ``close()`` may unpin pages a queued DMA is still reading, which is
        undefined behaviour.

        Pinning is per process because ``cudaHostRegister`` registers *this* process's
        mapping with *this* process's DMA engine; the block exists once. Requires torch and a
        device — the one method here that does, kept apart so everything else runs offline.
        """
        # The whole handout is serialised against `close()` under `_pin_lock`: the closed
        # check, the once-per-process registration, the view over the block, and the liveness
        # increment happen as one step, so a close can land before it (refused, typed) or
        # after it (the count is already up, so nothing is unpinned) — never inside it.
        with self._pin_lock:
            if self._pin_closed or self._detached:
                raise RingClosedError(self._owner, self._name)
            torch = self._torch_with_registration()
            start = self._layout.slot_offset(index)
            tensor = torch.frombuffer(
                self._block.buf, dtype=torch.uint8, count=self._layout.slot_bytes, offset=start
            )
            # Liveness is tracked, not guessed: the count comes down in a finalizer when the
            # tensor dies; `close()` reads it to decide whether unpinning is safe *now*, and
            # the last finalizer runs a parked unregister itself.
            self._pinned_live += 1
        weakref.finalize(tensor, self._pinned_view_died)
        return tensor

    def _pinned_view_died(self) -> None:
        # The whole path holds `_pin_lock` (reentrant): clearing `_parked_unregister`,
        # running the unregister, and closing the mapping are one step, so a concurrent
        # `close()` cannot see the cleared state, munmap the block, and leave this thread's
        # `cudaHostUnregister` running on an unmapped pointer.
        with self._pin_lock:
            self._pinned_live -= 1
            if self._pinned_live != 0 or self._parked_unregister is None:
                return
            unregister = self._parked_unregister
            self._parked_unregister = None
            unregister()
            try:
                self._block.close()
            except BufferError:
                # A plain payload view is still out there; the *mapping* waits for it (the
                # pages are already unpinned, which is safe — payload views are synchronous).
                with _PENDING_LOCK:
                    _PENDING_CLOSE.append(self._block)

    def _torch_with_registration(self) -> Any:
        # Called under `_pin_lock` only: two threads racing this used to both see `_pinned is
        # None` and the loser got cudaErrorHostMemoryAlreadyRegistered as a spurious failure.
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
        if not self._is_owner:
            with _PENDING_LOCK:
                if (
                    self._name in _OPENED_CLAIMS
                    and _OPENED_CLAIMS[self._name] is self._claim_token
                ):
                    del _OPENED_CLAIMS[self._name]
        if self._is_owner and not self._closed_here:
            self._closed_here = True
            header = self.header()
            self._write_header(
                depth=header.depth,
                ewma_latency_us=header.ewma_latency_us,
                heartbeat_ns=time.monotonic_ns(),
                closed=True,
            )
        # A payload decoded without a copy may still view the block (an in-flight tensor);
        # releasing under it would raise `BufferError` and skip the unlink. The mapping then
        # lives until the last view dies, which is the right lifetime — the *name* goes now.
        with self._view_lock:
            self._detached = True
            with contextlib.suppress(BufferError):
                # Slices taken from `_view` hold the *underlying* buffer, not this object, so
                # a clean release here proves nothing about in-flight views — it only drops
                # our own export so the mapping can close once the callers' slices die.
                self._view.release()
            self._view = memoryview(b"")
        # Reap earlier leftovers first — the reap only ever closes mappings whose pages are
        # already unpinned, so it can never unpin anything, let alone under a live DMA.
        reap_pending_closes()
        # One pin-lock section end to end: the latch, the decision, the unpin, and the
        # close/park. Splitting the unpin out of the lock let a concurrent close() see the
        # cleared state, fall through to block.close(), and munmap the pages this thread was
        # about to cudaHostUnregister.
        with self._pin_lock:
            # Latch first: from here no handout can start, and any that completed has already
            # counted itself, so `_pinned_live` below is the whole truth.
            self._pin_closed = True
            parked_for_finalizer = False
            if self._parked_unregister is not None:
                # A previous close already deferred to the finalizer. Without this, a second
                # close saw `_pinned is None`, fell through to `block.close()`, and parked a
                # mapping whose pages were STILL REGISTERED — a later reap would munmap them
                # and the finalizer's unregister would then run on an unmapped pointer.
                parked_for_finalizer = True
            elif self._pinned is not None:
                torch_module, pinned_ptr = self._pinned, self._pinned_ptr
                self._pinned = None
                if self._pinned_live != 0:

                    def _unregister() -> None:
                        torch_module.cuda.cudart().cudaHostUnregister(pinned_ptr)

                    # Pinned views are still held; the last one's finalizer unpins and closes.
                    self._parked_unregister = _unregister
                    parked_for_finalizer = True
                else:
                    # No pinned view is out at the latch — so, *provided callers honoured
                    # `pinned_tensor`'s contract* (a tensor outlives the copies it feeds), no
                    # DMA through this ring is in flight: unpin now, mapping certainly alive.
                    torch_module.cuda.cudart().cudaHostUnregister(pinned_ptr)
            if not parked_for_finalizer:
                closed_now = False
                try:
                    self._block.close()
                    closed_now = True
                except BufferError:
                    pass  # a zero-copy payload is still in flight; the mapping waits for it
                if not closed_now:
                    with _PENDING_LOCK:
                        _PENDING_CLOSE.append(self._block)
        if self._is_owner and not self._unlinked:
            self._unlinked = True
            # `_attach` unregisters on open; in one process (the tests, and any same-process
            # pair) that removes the create-time entry too, because the tracker's cache is a
            # set. Re-register before unlink so its own unregister finds the entry instead of
            # a KeyError in the tracker daemon — and only once: a second register+failed-unlink
            # would orphan the entry and warn about a "leak" at exit.
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
    """Spin politely: yield the GIL from the very first iteration, sleep once it drags on.

    vLLM spins hot because its ring lives in C without the GIL; a Python spinner holding the
    GIL starves the process's other threads, so even the first spins cost a ``sleep(0)``.
    """
    if spins < 256:
        time.sleep(0)
    else:
        time.sleep(0.00005)
    return spins + 1
