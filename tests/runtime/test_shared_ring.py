"""The pinned shared-memory ring, offline: layout arithmetic and the one-writer protocol.

The reader is a thread in this process — the protocol is bytes in shared memory, so a thread
and a peer process see the same thing. Pinning (`pinned_tensor`) is the one method that needs
a device and is not exercised here.
"""

from __future__ import annotations

import gc
import threading
import time
import uuid

import numpy as np
import pytest

from shipinfer.core.errors import (
    QueueFullError,
    RingClosedError,
    RingFullError,
    RingProtocolError,
)
from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing, SlotState


def _name() -> str:
    return f"shipinfer-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture()
def ring():
    layout = RingLayout(slots=4, slot_bytes=1000)
    owner = SharedRing.create(_name(), layout, owner="shard-1")
    try:
        yield owner
    finally:
        owner.close()


class TestTheLayoutIsArithmetic:
    def test_pages_and_alignment(self) -> None:
        layout = RingLayout(slots=3, slot_bytes=1000)
        assert layout.header_bytes == 4096
        assert layout.states_offset == 4096
        assert layout.payload_offset == 8192
        assert layout.padded_slot_bytes == 4096
        assert layout.total_bytes == 8192 + 3 * 4096
        assert [layout.slot_offset(i) for i in range(3)] == [8192, 12288, 16384]
        assert [layout.state_offset(i) for i in range(3)] == [4096, 4097, 4098]

    def test_a_slot_never_straddles_a_page(self) -> None:
        layout = RingLayout(slots=2, slot_bytes=4097)
        assert layout.padded_slot_bytes == 8192
        assert layout.slot_offset(1) - layout.slot_offset(0) == 8192

    @pytest.mark.parametrize(
        "bad", [{"slots": 0, "slot_bytes": 1}, {"slots": 1, "slot_bytes": 0}]
    )
    def test_degenerate_layouts_are_refused(self, bad) -> None:
        with pytest.raises(ValueError):
            RingLayout(**bad)

    def test_out_of_range_slots_are_refused(self) -> None:
        with pytest.raises(IndexError):
            RingLayout(slots=2, slot_bytes=8).slot_offset(2)


class TestCreateAndOpen:
    def test_the_writer_sees_the_owners_header(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            header = writer.header()
            assert header.owner == "shard-1"
            assert (header.slots, header.slot_bytes) == (4, 1000)
            assert header.depth == 0 and not header.closed
            assert all(writer.state(i) == SlotState.FREE for i in range(4))
        finally:
            writer.close()

    def test_a_layout_disagreement_is_refused_at_open(self, ring) -> None:
        with pytest.raises(RingProtocolError, match="created with 4 slots of 1000"):
            SharedRing.open(ring.name, RingLayout(slots=4, slot_bytes=2000))

    def test_a_foreign_block_is_not_a_ring(self) -> None:
        from multiprocessing import shared_memory

        block = shared_memory.SharedMemory(name=_name(), create=True, size=8192)
        try:
            with pytest.raises(RingProtocolError, match="version"):
                SharedRing.open(block.name, RingLayout(slots=1, slot_bytes=8))
        finally:
            block.close()
            block.unlink()

    def test_the_owner_unlinks_on_close(self) -> None:
        name = _name()
        owner = SharedRing.create(name, RingLayout(slots=1, slot_bytes=8), owner="s")
        owner.close()
        from multiprocessing import shared_memory

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)


class TestTheProtocol:
    def test_claim_publish_take_release(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            index = writer.claim(timeout_s=0.1)
            assert ring.state(index) == SlotState.CLAIMED
            writer.payload(index)[:5] = b"hello"
            assert ring.take(timeout_s=0.01) is None, "a claimed slot is not yet visible"
            writer.publish(index)
            taken = ring.take(timeout_s=0.5)
            assert taken == index and ring.state(index) == SlotState.TAKEN
            assert bytes(ring.payload(index)[:5]) == b"hello"
            ring.release(index)
            assert ring.state(index) == SlotState.FREE
        finally:
            writer.close()

    def test_publish_and_release_check_the_state(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            with pytest.raises(RingProtocolError, match="not claimed"):
                writer.publish(0)
            with pytest.raises(RingProtocolError, match="not taken"):
                ring.release(0)
        finally:
            writer.close()

    def test_a_full_ring_refuses_with_the_numbers(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            for _ in range(4):
                writer.claim(timeout_s=0.1)
            started = time.monotonic()
            with pytest.raises(RingFullError) as caught:
                writer.claim(timeout_s=0.05)
            assert time.monotonic() - started < 1.0
            assert (caught.value.depth, caught.value.capacity) == (4, 4)
            assert caught.value.owner == "shard-1" and caught.value.ring == ring.name
        finally:
            writer.close()

    def test_slots_are_reused_in_rotation(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            order = []
            for _ in range(6):
                index = writer.claim(timeout_s=0.1)
                writer.publish(index)
                taken = ring.take(timeout_s=0.5)
                order.append(taken)
                ring.release(taken)
            assert order == [0, 1, 2, 3, 0, 1]
        finally:
            writer.close()

    def test_a_reader_thread_receives_every_payload_in_order(self, ring) -> None:
        received: list[bytes] = []

        def reader() -> None:
            while (index := ring.take(timeout_s=1.0)) is not None:
                received.append(bytes(ring.payload(index)[:4]))
                ring.release(index)

        thread = threading.Thread(target=reader)
        thread.start()
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            for i in range(50):  # more than the slot count: the writer waits for releases
                index = writer.claim(timeout_s=1.0)
                writer.payload(index)[:4] = i.to_bytes(4, "little")
                writer.publish(index)
            deadline = time.monotonic() + 2.0
            while len(received) < 50 and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            ring.close()  # closes the reader's loop: take() returns None
            thread.join(timeout=2.0)
            writer.close()
        assert [int.from_bytes(b, "little") for b in received] == list(range(50))

    def test_a_closed_handle_answers_closed_instead_of_raising(self) -> None:
        """A reader thread that wakes after `close()` must see a closed ring, not a ValueError
        from a released view — the first version leaked exactly that as an unhandled thread
        exception under the reader test."""
        owner = SharedRing.create(_name(), RingLayout(slots=2, slot_bytes=8), owner="s")
        owner.close()
        assert owner.header().closed
        assert owner.take(timeout_s=0.01) is None

    def test_close_stops_the_reader_and_the_writer(self) -> None:
        owner = SharedRing.create(_name(), RingLayout(slots=2, slot_bytes=8), owner="s")
        writer = SharedRing.open(owner.name, owner.layout)
        owner.close()
        try:
            assert writer.header().closed
            with pytest.raises(RingClosedError, match="closed"):
                writer.claim(timeout_s=0.05)
        finally:
            writer.close()


class TestTheHeaderIsTheLoadSignal:
    def test_stamp_is_what_a_proxy_reads(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            before = writer.header().heartbeat_ns
            time.sleep(0.001)
            ring.stamp(depth=7, ewma_latency_us=1234.5)
            header = writer.header()
            assert header.depth == 7
            assert header.ewma_latency_us == pytest.approx(1234.5)
            assert header.heartbeat_ns > before
            assert not header.closed
        finally:
            writer.close()

    def test_depth_counts_slots_that_are_not_free(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            assert writer.depth == 0
            a = writer.claim(0.1)
            b = writer.claim(0.1)
            writer.publish(a)
            assert writer.depth == 2
            taken = ring.take(0.5)
            assert writer.depth == 2, "taken is still not free"
            ring.release(taken)
            assert writer.depth == 1
            writer.publish(b)
        finally:
            writer.close()


class TestClosingUnderALiveView:
    def test_a_live_zero_copy_view_defers_the_handle_and_the_reaper_closes_it_later(
        self,
    ) -> None:
        """An in-flight request decoded without a copy still views the block when the ring
        closes. Closing must not raise, must not let the handle's finaliser fail later, and
        must take the *name* down at once; the mapping goes when the last view does."""
        from shipinfer.runtime.memory import shared_ring as module

        layout = RingLayout(slots=2, slot_bytes=4096)
        owner = SharedRing.create(_name(), layout, owner="A")
        writer = SharedRing.open(owner.name, layout)
        index = writer.claim(timeout_s=0.1)
        writer.publish(index)
        taken = owner.take(timeout_s=0.5)
        tensor = np.frombuffer(owner.payload(taken), dtype=np.uint8)  # the in-flight view

        writer.close()
        owner.close()  # neither raises
        with pytest.raises(FileNotFoundError):
            SharedRing.open(owner.name, layout)  # the name is gone regardless
        waiting = module.reap_pending_closes()
        assert waiting >= 1, "the owner's handle waits for the view"

        del tensor
        gc.collect()
        assert module.reap_pending_closes() < waiting, "the view died, so the handle closed"


class TestAbandon:
    def test_a_claimed_slot_can_go_straight_back_to_free(self, ring) -> None:
        """A payload that cannot be written must not cost the ring a slot for its whole life."""
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            index = writer.claim(timeout_s=0.1)
            assert writer.depth == 1
            writer.abandon(index)
            assert writer.state(index) == SlotState.FREE and writer.depth == 0
            assert ring.take(timeout_s=0.02) is None, "the reader never saw it"
            # Every slot is claimable again — the abandoned one included (claims are round-robin,
            # so it is not necessarily the *next* one).
            for _ in range(ring.layout.slots):
                writer.claim(timeout_s=0.1)
            assert writer.depth == ring.layout.slots
        finally:
            writer.close()

    def test_only_a_claimed_slot_can_be_abandoned(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            with pytest.raises(RingProtocolError, match="not claimed"):
                writer.abandon(0)
            index = writer.claim(timeout_s=0.1)
            writer.publish(index)
            with pytest.raises(RingProtocolError, match="not claimed"):
                writer.abandon(index)  # published: it is the reader's now
        finally:
            writer.close()


class TestClosedIsNotFull:
    def test_a_closed_ring_says_so_and_the_dispatcher_can_still_spill_on_it(self, ring) -> None:
        """Closed and full are different events with different operator responses — but both
        are `QueueFullError`s, so the dispatcher's recovery (exclude, re-select) covers the
        race where a request passes `is_ready` just before the owner leaves."""
        writer = SharedRing.open(ring.name, ring.layout)
        ring.close()
        with pytest.raises(RingClosedError, match="closed - its owner is gone") as caught:
            writer.claim(timeout_s=0.05)
        assert isinstance(caught.value, QueueFullError)
        assert "full" not in str(caught.value)
        writer.close()

    def test_a_genuinely_full_ring_reports_its_real_depth(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            for _ in range(ring.layout.slots):
                writer.publish(writer.claim(timeout_s=0.1))
            with pytest.raises(RingFullError) as caught:
                writer.claim(timeout_s=0.05)
            assert (caught.value.depth, caught.value.capacity) == (
                ring.layout.slots,
                ring.layout.slots,
            )
        finally:
            writer.close()


class TestIsClosedIsTheLoopCondition:
    def test_an_idle_window_is_not_the_end_of_the_loop(self, ring) -> None:
        """`take` returning None means "nothing this window"; the loop the docstring
        recommends keeps running until `is_closed` — an owner must survive quiet seconds."""
        assert ring.take(timeout_s=0.06) is None
        assert not ring.is_closed, "idle is not closed"
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            assert not writer.is_closed
            writer.publish(writer.claim(timeout_s=0.1))
            assert ring.take(timeout_s=0.5) is not None, "still alive after the idle window"
        finally:
            writer.close()

    def test_close_flips_the_byte_on_both_handles(self, ring) -> None:
        writer = SharedRing.open(ring.name, ring.layout)
        ring.close()
        assert ring.is_closed, "the detached handle answers closed"
        assert writer.is_closed, "the peer reads the byte"
        writer.close()


class TestAClosedHandleIsInert:
    def test_late_transitions_and_stamps_do_not_crash(self, ring) -> None:
        """The shutdown window: a completion callback or a metrics thread touching the ring
        after `close()` must be a no-op, not an IndexError from the released view — this is
        the exact noise the first two-process bench run produced at every rung teardown."""
        writer = SharedRing.open(ring.name, ring.layout)
        index = writer.claim(timeout_s=0.1)
        writer.publish(index)
        taken = ring.take(timeout_s=0.5)
        ring.close()
        writer.close()
        ring.release(taken)  # settles late; nothing to free, nothing raised
        ring.stamp(depth=3, ewma_latency_us=1.0)  # the metrics thread's last stamp
        assert ring.is_closed and writer.is_closed


@pytest.mark.gpu
class TestPinnedForReal:
    def test_register_roundtrip_unregister(self) -> None:
        """The one device path in the module: a slot pinned once per process, H2D and back
        bit-exact, and close() unregistering without touching a new export of the block."""
        import torch

        layout = RingLayout(slots=2, slot_bytes=8192)
        ring = SharedRing.create(_name(), layout, owner="A")
        try:
            slot = ring.pinned_tensor(0)
            assert slot.numel() == layout.slot_bytes
            source = torch.arange(layout.slot_bytes, dtype=torch.uint8) % 251
            slot.copy_(source)
            device = slot.to("cuda:0")
            back = device.to("cpu")
            assert torch.equal(back, source)
            del slot, device, back
        finally:
            ring.close()  # unregisters via the remembered pointer; raising here fails the test
        with pytest.raises(FileNotFoundError):
            SharedRing.open(ring.name, layout)
