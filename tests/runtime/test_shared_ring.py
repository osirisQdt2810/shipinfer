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
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import (
    QueueFullError,
    RingClosedError,
    RingFullError,
    RingProtocolError,
)
from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing, SlotState


@pytest.fixture(autouse=True)
def _a_clean_pending_list():
    """`_PENDING_CLOSE` is process-global: another file's rings, closed under views that have
    since died, would otherwise make this file's `reap() == 0` asserts order-dependent."""
    from shipinfer.runtime.memory.shared_ring import reap_pending_closes

    gc.collect()
    reap_pending_closes()
    yield


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
        """Nonzero garbage where a header should be is a build/protocol disagreement — loud.
        (An all-zero header is different: that is a ring mid-birth, covered elsewhere.)"""
        from multiprocessing import resource_tracker, shared_memory

        name = _name()
        layout = RingLayout(slots=2, slot_bytes=4096)
        foreign = shared_memory.SharedMemory(name=name, create=True, size=layout.total_bytes)
        try:
            foreign.buf[:64] = b"\x7fELF" + bytes(range(60))  # anything but our magic
            with pytest.raises(RingProtocolError, match="not the same build"):
                SharedRing.open(name, layout)
        finally:
            resource_tracker.register(foreign._name, "shared_memory")
            foreign.close()
            foreign.unlink()

    def test_the_owner_unlinks_on_close(self) -> None:
        name = _name()
        layout = RingLayout(slots=1, slot_bytes=8)
        owner = SharedRing.create(name, layout, owner="s")
        owner.close()
        from multiprocessing import shared_memory

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)  # the raw name is gone
        with pytest.raises(RingClosedError, match="does not exist"):
            SharedRing.open(name, layout)  # and the ring API says so in its own vocabulary


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

    def test_a_reader_thread_receives_every_payload_with_bounded_reordering(self, ring) -> None:
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
        values = [int.from_bytes(b, "little") for b in received]
        assert sorted(values) == list(range(50)), "every payload is delivered exactly once"
        # The ring does not preserve submission order: `claim` skips busy slots, so a late
        # message can land in a lower slot. The rotating take bounds the displacement to the
        # slot count and nothing starves — that bound, not order, is the guarantee.
        displacement = max(abs(position - value) for position, value in enumerate(values))
        assert (
            displacement <= ring.layout.slots
        ), f"displacement {displacement} exceeds a rotation"

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
        with pytest.raises(RingClosedError, match="does not exist"):
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
        assert caught.value.reason == "closed"
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
    def test_the_whole_surface_follows_the_closed_rule(self, ring) -> None:
        """Questions answer honestly, data access refuses, transitions are no-ops — one rule,
        every public method, because the round-2 review found the guard applied piecemeal."""
        writer = SharedRing.open(ring.name, ring.layout)
        index = writer.claim(timeout_s=0.1)
        ring.close()
        writer.close()
        for handle in (ring, writer):
            assert handle.is_closed
            assert handle.depth == 0
            assert handle.state(0) == SlotState.FREE
            assert handle.header().closed
            if handle is ring:
                assert handle.take(timeout_s=0.01) is None
            else:
                # The one-reader discipline outranks the closed rule: a submit handle taking
                # is a wiring mistake and fails loudly whatever the ring's state.
                with pytest.raises(RingProtocolError, match="only the owner takes"):
                    handle.take(timeout_s=0.01)
            if handle is ring:
                with pytest.raises(RingProtocolError, match="the owner reads"):
                    handle.claim(timeout_s=0.01)
            else:
                with pytest.raises(RingClosedError):
                    handle.claim(timeout_s=0.01)
            with pytest.raises(RingClosedError):
                handle.payload(0)
            with pytest.raises(RingClosedError):
                handle.pinned_tensor(0)
            handle.stamp(depth=9, ewma_latency_us=9.0)
            handle.publish(index)
            handle.release(index)
            handle.abandon(index)

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
            ring.close()  # no views out, so close() unpins and unmaps inline
        from shipinfer.runtime.memory.shared_ring import reap_pending_closes

        assert reap_pending_closes() == 0, "nothing was parked: close freed inline"
        with pytest.raises(RingClosedError, match="does not exist"):
            SharedRing.open(ring.name, layout)


class TestStampIsTheOwners:
    def test_a_peer_cannot_stamp_and_cannot_clear_the_closed_flag(self, ring) -> None:
        """A non-owner stamping would re-pack the header and wipe `closed`; a dead peer would
        then read as back-pressure forever. The stamp is owner-only and writes in place."""
        writer = SharedRing.open(ring.name, ring.layout)
        try:
            with pytest.raises(RingProtocolError, match="only the owner stamps"):
                writer.stamp(depth=1, ewma_latency_us=1.0)
            ring.stamp(depth=5, ewma_latency_us=123.0)
            header = writer.header()
            assert (header.depth, header.ewma_latency_us) == (5, 123.0)
            assert header.slots == ring.layout.slots, "the layout fields were not re-packed"
            assert not header.closed
        finally:
            writer.close()


class TestAShortBlockIsRefused:
    def test_open_checks_the_whole_layout_size(self) -> None:
        """A block shorter than the layout would clamp payloads silently (memoryview slicing
        clamps); open refuses it as the protocol error it is."""
        from multiprocessing import resource_tracker, shared_memory

        name = _name()
        layout = RingLayout(slots=4, slot_bytes=4096)
        small = shared_memory.SharedMemory(name=name, create=True, size=8192)
        try:
            _HEADER = __import__(
                "shipinfer.runtime.memory.shared_ring", fromlist=["_HEADER"]
            )._HEADER
            _HEADER.pack_into(
                small.buf,
                0,
                __import__("shipinfer.runtime.memory.shared_ring", fromlist=["_MAGIC"])._MAGIC,
                1,
                layout.slots,
                layout.slot_bytes,
                0,
                0,
                0.0,
                0,
                0,
                b"A",
            )
            with pytest.raises(RingProtocolError, match="short block would clamp"):
                SharedRing.open(name, layout)
        finally:
            # `_attach` unregistered the name before validation refused the block; pair the
            # tracker entries back up so this unlink's own unregister finds one.
            resource_tracker.register(small._name, "shared_memory")
            small.close()
            small.unlink()


class TestTheUnpinWaitsForTheViews:
    """Liveness is tracked, not guessed: `pinned_tensor` counts its handouts, `close()` unpins
    only at zero, a reap can never unpin, and the last finalizer unpins and closes itself."""

    class _View:
        """Weakref-able stand-in for the torch tensor `pinned_tensor` hands out."""

    def _pinned_ring(self, ring, calls):
        from types import SimpleNamespace

        fake_torch = SimpleNamespace(
            uint8="uint8",
            frombuffer=lambda *a, **k: self._View(),
            cuda=SimpleNamespace(
                cudart=lambda: SimpleNamespace(cudaHostUnregister=lambda ptr: calls.append(ptr))
            ),
        )
        ring._pinned = fake_torch
        ring._pinned_ptr = 0xDEAD
        return ring

    def test_close_with_no_pinned_views_out_unpins_immediately(self, ring) -> None:
        """Round 3's finding 2: a pinned ring must free its pages at its own close, not when
        an unrelated object happens to be destroyed."""
        from shipinfer.runtime.memory import shared_ring as module

        calls: list[int] = []
        self._pinned_ring(ring, calls)
        view = ring.pinned_tensor(0)
        del view
        gc.collect()
        leftover = module.reap_pending_closes()  # other files' rings under still-live views
        ring.close()
        assert calls == [0xDEAD], "unpinned inside close(), nothing deferred"
        assert module.reap_pending_closes() <= leftover, "and THIS ring parked nothing"

    def test_a_reap_never_unpins_and_the_last_finalizer_does(self, ring) -> None:
        """Round 3's finding 1: another ring's close reaps microseconds after a park — the
        reap must be unable to unpin under the live view, and the finalizer frees instead."""
        from shipinfer.runtime.memory import shared_ring as module

        calls: list[int] = []
        self._pinned_ring(ring, calls)
        held = ring.pinned_tensor(0)
        leftover = module.reap_pending_closes()  # other files' rings under still-live views
        ring.close()
        assert calls == [], "close under a live pinned view unpins nothing"
        module.reap_pending_closes()
        assert calls == [], "a reap while the view is held cannot unpin either"
        del held
        gc.collect()
        assert calls == [0xDEAD], "the last finalizer unpinned, exactly once"
        assert module.reap_pending_closes() <= leftover


class TestPeerLostSpeaksItsTags:
    def test_the_message_names_the_first_tags_and_counts_the_rest(self) -> None:
        from shipinfer.core.errors import PeerLostError

        tags = [(f"cam{i:02d}", i) for i in range(11)]
        error = PeerLostError("2:person_embedder", tags)
        assert error.owner == "2:person_embedder" and len(error.tags) == 11
        assert "('cam00', 0)" in str(error) and "and 3 more" in str(error)


def _spawn_writer(name: str, slots: int, slot_bytes: int) -> None:
    """A real peer process: open by name, submit one payload, exit without unlinking."""
    from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing

    writer = SharedRing.open(name, RingLayout(slots=slots, slot_bytes=slot_bytes))
    index = writer.claim(timeout_s=5.0)
    writer.payload(index)[:5] = b"hello"
    writer.publish(index)
    writer.close()


class TestARealPeerProcess:
    def test_a_spawned_writer_submits_and_its_exit_unlinks_nothing(self, ring) -> None:
        """The bpo-38119 handling exists for *processes*: a spawn-context child attaches,
        writes, and exits — and the block must survive it, because `_attach` kept the child's
        resource tracker out of it. The owner still reads the payload afterwards."""
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        child = context.Process(
            target=_spawn_writer, args=(ring.name, ring.layout.slots, ring.layout.slot_bytes)
        )
        child.start()
        child.join(timeout=30)
        assert child.exitcode == 0
        taken = ring.take(timeout_s=5.0)
        assert taken is not None, "the block outlived the child"
        assert bytes(ring.payload(taken)[:5]) == b"hello"
        ring.release(taken)


class TestCloseIsConcurrentByDesign:
    def test_a_sweeping_consumer_survives_a_close_from_another_thread(self) -> None:
        """Round 4's reproduction: a consumer suspended between the detached check and the
        view read must never resume onto a released view. A few hundred close-vs-sweep races,
        zero tolerance for an exception in the consumer."""
        errors: list[BaseException] = []
        for _ in range(200):
            ring = SharedRing.create(_name(), RingLayout(slots=4, slot_bytes=4096), owner="A")
            started = threading.Event()

            def sweep(ring=ring, started=started) -> None:
                try:
                    started.set()
                    while not ring.is_closed:
                        index = ring.take(timeout_s=0)
                        if index is not None:
                            ring.payload(index)
                            ring.release(index)
                        ring.state(0)
                        ring.header()
                except RingClosedError:
                    pass  # the read raced the close and was told so, in the vocabulary
                except BaseException as exc:
                    errors.append(exc)

            writer = SharedRing.open(ring.name, ring.layout)
            submitter_started = threading.Event()

            def submit(writer=writer, started=submitter_started) -> None:
                """Round 11: a transition racing the close must be inert or typed-closed —
                never a RingProtocolError masquerading as a build mismatch."""
                try:
                    started.set()
                    while not writer.is_closed:
                        try:
                            index = writer.claim(timeout_s=0.001)
                        except (RingFullError, RingClosedError):
                            continue
                        if index % 2:
                            writer.publish(index)
                        else:
                            writer.abandon(index)
                except RingClosedError:
                    pass
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=sweep)
            pusher = threading.Thread(target=submit)
            thread.start()
            pusher.start()
            started.wait(1.0)
            submitter_started.wait(1.0)
            ring.close()
            thread.join(timeout=2.0)
            pusher.join(timeout=2.0)
            writer.close()
            assert not thread.is_alive() and not pusher.is_alive()
        assert errors == [], errors[:3]

    def test_close_is_idempotent_and_unlinks_once(self) -> None:
        """A second close must not re-register the name and orphan the tracker entry."""
        from shipinfer.runtime.memory import shared_ring as module

        ring = SharedRing.create(_name(), RingLayout(slots=1, slot_bytes=8), owner="A")
        leftover = module.reap_pending_closes()  # other files' rings under still-live views
        ring.close()
        ring.close()
        with ring:  # __exit__ closes a third time
            pass
        assert ring.is_closed
        assert module.reap_pending_closes() <= leftover


class TestARingMidBirth:
    def test_a_zero_header_is_not_ready_not_a_build_mismatch(self) -> None:
        """The shm name is visible before the creator writes the header; a fast peer must be
        told to retry (RingClosedError, which the connect loop retries on), not that the two
        processes are different builds."""
        from multiprocessing import resource_tracker, shared_memory

        name = _name()
        layout = RingLayout(slots=2, slot_bytes=4096)
        bare = shared_memory.SharedMemory(name=name, create=True, size=layout.total_bytes)
        try:
            with pytest.raises(RingClosedError) as caught:
                SharedRing.open(name, layout)
            assert caught.value.reason == "unborn"
        finally:
            resource_tracker.register(bare._name, "shared_memory")
            bare.close()
            bare.unlink()

    def test_an_empty_file_is_unborn_not_a_value_error(self) -> None:
        """The window before even the sizeless one: shm_open creates the name at size 0 and
        ftruncate has not run, so the attach's own mmap raises ValueError('cannot mmap an
        empty file') — CI's third spelling of the birth race (27 Aug), which escaped the
        mesh connect as a raw ValueError. It must be the same retryable 'unborn'."""
        name = _name()
        backing = Path("/dev/shm") / name
        backing.touch()
        try:
            with pytest.raises(RingClosedError) as caught:
                SharedRing.open(name, RingLayout(slots=2, slot_bytes=4096))
            assert caught.value.reason == "unborn"
        finally:
            backing.unlink()

    def test_a_sizeless_block_is_unborn_not_a_protocol_error(self) -> None:
        """Even earlier in the birth: the name exists at shm creation, BEFORE the creator
        sizes the block. A reader that wins that race must also be told to retry — raising
        the protocol error here killed the whole mesh connect on the first sub-header
        attach, twice on CI's runners (27 Aug), while the peer then reported 'never
        appeared'. A persistently sizeless block still errors, via the connect deadline."""
        from multiprocessing import resource_tracker, shared_memory

        name = _name()
        # A 1-byte block: the smallest thing shm will make, well under the header.
        bare = shared_memory.SharedMemory(name=name, create=True, size=1)
        try:
            with pytest.raises(RingClosedError) as caught:
                SharedRing.open(name, RingLayout(slots=2, slot_bytes=4096))
            assert caught.value.reason == "unborn"
        finally:
            resource_tracker.register(bare._name, "shared_memory")
            bare.close()
            bare.unlink()


def _stamp_forever(name: str, slots: int, slot_bytes: int, seconds: float) -> None:
    """The owner process under test: stamp a fixed load as fast as the loop turns."""
    from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing

    ring = SharedRing.create(name, RingLayout(slots=slots, slot_bytes=slot_bytes), owner="A")
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ring.stamp(depth=7, ewma_latency_us=123.0)
    ring.close()


class TestAStampIsNeverTorn:
    def test_a_cross_process_reader_never_observes_the_memset_window(self) -> None:
        """`pack_into` memsets its region before packing, so a lock-free reader in another
        process could catch depth == 0 on a saturated ring or heartbeat_ns == 0 on a healthy
        one (~24% of colliding reads, measured). The write is pack-then-memcpy now: a reader
        may see stale fields, never zeroed ones."""
        import multiprocessing

        name = _name()
        layout = RingLayout(slots=2, slot_bytes=4096)
        context = multiprocessing.get_context("spawn")
        child = context.Process(target=_stamp_forever, args=(name, 2, 4096, 1.5))
        child.start()
        try:
            deadline = time.monotonic() + 10.0
            reader = None
            while reader is None and time.monotonic() < deadline:
                try:
                    reader = SharedRing.open(name, layout)
                except RingClosedError:
                    time.sleep(0.005)
            assert reader is not None, "the child never created the ring"
            reads = 0
            while not reader.is_closed and reads < 200_000:
                header = reader.header()
                if header.closed:
                    break
                reads += 1
                assert header.depth == 7, f"read {reads}: torn depth {header.depth}"
                assert header.heartbeat_ns != 0, f"read {reads}: torn heartbeat"
                assert header.ewma_latency_us == 123.0, f"read {reads}: torn ewma"
            assert reads > 1_000, "the window overlapped the child's stamping"
            reader.close()
        finally:
            child.join(timeout=30)
            assert child.exitcode == 0


class TestPinningRacesClose:
    class _View:
        def data_ptr(self) -> int:
            return 0xBEEF

        def numel(self) -> int:
            return 4096

    def test_a_handout_racing_close_is_refused_or_counted_never_leaked(
        self, monkeypatch
    ) -> None:
        """Round 7: the handout is one step under the pin lock. A close can land before it
        (refused, typed) or after it (counted, so nothing unpins early) — never inside it. No
        TypeError from a vanished buffer, at most one registration, and every registration is
        eventually unregistered exactly once."""
        from types import SimpleNamespace

        import shipinfer.runtime.platform as platform_module

        for _ in range(150):
            registers: list[int] = []
            unregisters: list[int] = []
            cudart_calls = SimpleNamespace(
                cudaHostRegister=lambda ptr, n, flags, _r=registers: _r.append(ptr) or 0,
                cudaHostUnregister=lambda ptr, _u=unregisters: _u.append(ptr),
            )
            fake_torch = SimpleNamespace(
                uint8="uint8",
                frombuffer=lambda *a, **k: self._View(),
                cuda=SimpleNamespace(cudart=lambda _c=cudart_calls: _c),
            )
            monkeypatch.setattr(platform_module, "require_torch", lambda _ft=fake_torch: _ft)
            ring = SharedRing.create(_name(), RingLayout(slots=2, slot_bytes=4096), owner="A")
            surprises: list[BaseException] = []
            held: list = []
            started = threading.Event()

            def pin_loop(ring=ring, surprises=surprises, held=held, started=started) -> None:
                started.set()
                for i in range(64):
                    try:
                        tensor = ring.pinned_tensor(0)
                        if i % 8 == 0:
                            held.append(tensor)  # some survive past the close
                    except RingClosedError:
                        return  # the close landed first: refused, typed — correct
                    except BaseException as exc:
                        surprises.append(exc)
                        return

            thread = threading.Thread(target=pin_loop)
            thread.start()
            started.wait(1.0)
            ring.close()
            thread.join(timeout=2.0)
            assert surprises == [], surprises[:2]
            assert len(registers) <= 1, "one registration per process, ever"
            held.clear()
            gc.collect()
            from shipinfer.runtime.memory.shared_ring import reap_pending_closes

            reap_pending_closes()
            assert len(unregisters) == len(registers), "every registration unregistered, once"


def _publish_during_birth(name: str, slots: int, slot_bytes: int) -> None:
    """A peer that wins the open race the instant readiness lands, and publishes at once."""
    from shipinfer.core.errors import RingClosedError
    from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing

    deadline = time.monotonic() + 10.0
    writer = None
    while writer is None:
        try:
            writer = SharedRing.open(name, RingLayout(slots=slots, slot_bytes=slot_bytes))
        except RingClosedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0)
    index = writer.claim(timeout_s=2.0)
    writer.payload(index)[:6] = b"SURVIV"
    writer.publish(index)
    writer.close()


class TestCreatePublishesReadinessLast:
    def test_a_payload_published_the_instant_the_ring_is_ready_survives(
        self, monkeypatch
    ) -> None:
        """Round 8: the magic is the readiness signal, so every byte a peer may touch must be
        final before it lands. The creator here is slowed *after* the header write; a spawned
        peer opens in that window and publishes — and the payload must still be WRITTEN when
        create() returns. Against the old order (header before the state loop) the creator's
        initialisation stamps the slot back to FREE and the frame is silently gone."""
        import multiprocessing

        real = SharedRing._write_header

        def slow_header(self, **kwargs) -> None:
            real(self, **kwargs)
            time.sleep(0.08)

        monkeypatch.setattr(SharedRing, "_write_header", slow_header)
        name = _name()
        layout = RingLayout(slots=2, slot_bytes=4096)
        context = multiprocessing.get_context("spawn")
        child = context.Process(target=_publish_during_birth, args=(name, 2, 4096))
        child.start()
        try:
            ring = SharedRing.create(name, layout, owner="A")
        finally:
            child.join(timeout=30)
        assert child.exitcode == 0, "the peer opened and published"
        monkeypatch.undo()
        taken = ring.take(timeout_s=5.0)
        assert taken is not None, "the published slot survived the creator's initialisation"
        assert bytes(ring.payload(taken)[:6]) == b"SURVIV"
        ring.release(taken)
        ring.close()


class TestTheTypedCreateAndOpenEdges:
    def test_a_never_created_name_is_absent(self) -> None:
        with pytest.raises(RingClosedError, match="does not exist") as caught:
            SharedRing.open(_name(), RingLayout(slots=1, slot_bytes=8))
        assert caught.value.reason == "absent"

    def test_a_surviving_segment_makes_create_speak_the_vocabulary(self) -> None:
        from multiprocessing import shared_memory

        name = _name()
        stale = shared_memory.SharedMemory(name=name, create=True, size=8192)
        try:
            with pytest.raises(RingProtocolError, match="already exists"):
                SharedRing.create(name, RingLayout(slots=1, slot_bytes=8), owner="A")
        finally:
            stale.close()
            stale.unlink()

    def test_one_open_per_ring_per_process(self, ring) -> None:
        """The one-writer discipline enforced at the seam, not discovered as a tracker
        KeyError at exit."""
        first = SharedRing.open(ring.name, ring.layout)
        try:
            with pytest.raises(RingProtocolError, match="already open in this process"):
                SharedRing.open(ring.name, ring.layout)
        finally:
            first.close()
        second = SharedRing.open(ring.name, ring.layout)  # the close released the claim
        second.close()

    def test_a_handle_dropped_without_close_is_quiet_and_reclaimable(self) -> None:
        from multiprocessing import resource_tracker, shared_memory

        name = _name()
        ring = SharedRing.create(name, RingLayout(slots=1, slot_bytes=8), owner="A")
        del ring
        gc.collect()  # the finalizer releases the view; SharedMemory.__del__ can close
        leaked = shared_memory.SharedMemory(name=name, create=False)
        resource_tracker.register(leaked._name, "shared_memory")
        leaked.close()
        leaked.unlink()


class TestTheStampWindow:
    def test_the_stamp_stays_short_of_the_closed_byte(self) -> None:
        """The import-time assert was stripped under -O; the invariant lives here instead."""
        from shipinfer.runtime.memory import shared_ring as module

        assert module._STAMP_OFFSET + module._STAMP.size <= module.SharedRing._CLOSED_OFFSET


class TestTheFinalizerCannotDeadlockTheHandout:
    class _View:
        def data_ptr(self) -> int:
            return 0xF00D

        def numel(self) -> int:
            return 4096

    @pytest.mark.timeout(10)
    def test_a_cyclic_gc_inside_the_locked_region_re_enters_and_returns(
        self, ring, monkeypatch
    ) -> None:
        """Round 9: `weakref.finalize` callbacks run synchronously in whatever thread triggers
        the collection — including a cyclic gc landing on the allocation *inside*
        `pinned_tensor`'s locked region. With a plain Lock the worker thread deadlocks on
        itself; the reentrant lock lets the finalizer's decrement through."""
        from types import SimpleNamespace

        import shipinfer.runtime.platform as platform_module

        calls = {"n": 0}

        def frombuffer(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                gc.collect()  # collects the cycle-held first tensor -> finalizer re-enters
            return self._View()

        self_views = self._View  # noqa: F841 - keep the class referenced for clarity
        fake_torch = SimpleNamespace(
            uint8="uint8",
            frombuffer=frombuffer,
            cuda=SimpleNamespace(
                cudart=lambda: SimpleNamespace(
                    cudaHostRegister=lambda ptr, n, flags: 0,
                    cudaHostUnregister=lambda ptr: None,
                )
            ),
        )
        monkeypatch.setattr(platform_module, "require_torch", lambda _ft=fake_torch: _ft)
        first = ring.pinned_tensor(0)
        cycle = [first]
        cycle.append(cycle)  # reachable only through the cycle once the name is dropped
        del first, cycle
        second = ring.pinned_tensor(1)  # the collection fires inside the locked region
        assert second is not None, "the handout returned instead of deadlocking"
        del second
        gc.collect()
        ring.close()


class TestADroppedWriterHandleReleasesItsClaim:
    def test_open_works_again_after_the_handle_is_lost(self, ring) -> None:
        """A writer lost to an exception on the connect path must not make every later open
        of that ring read as terminal: the finalizer releases the one-open claim."""
        writer = SharedRing.open(ring.name, ring.layout)
        del writer
        gc.collect()
        again = SharedRing.open(ring.name, ring.layout)  # not "already open in this process"
        again.close()


class TestManySubmittingThreads:
    def test_two_hundred_claims_from_two_threads_land_exactly_once_each(self) -> None:
        """Round 10: the dispatcher drives a proxy from many request threads, so claim's
        test-and-set must be one critical section — two threads that both saw FREE used to
        both take the same slot, and one camera's payload shipped under another's framing."""
        layout = RingLayout(slots=2, slot_bytes=4096)
        owner = SharedRing.create(_name(), layout, owner="A")
        writer = SharedRing.open(owner.name, layout)
        received: list[bytes] = []
        surprises: list[BaseException] = []

        def drain() -> None:
            while True:
                index = owner.take(timeout_s=0.05)
                if index is None:
                    if owner.is_closed:
                        return
                    continue
                received.append(bytes(owner.payload(index)[:12]))
                owner.release(index)

        def submit(prefix: bytes) -> None:
            try:
                for i in range(200):
                    while True:
                        try:
                            index = writer.claim(timeout_s=1.0)
                            break
                        except RingFullError:
                            time.sleep(0)
                    writer.payload(index)[:12] = prefix + i.to_bytes(4, "little")
                    writer.publish(index)
            except BaseException as exc:
                surprises.append(exc)

        reader = threading.Thread(target=drain)
        reader.start()
        submitters = [
            threading.Thread(target=submit, args=(b"cam07-mk",)),
            threading.Thread(target=submit, args=(b"cam41-mk",)),
        ]
        for thread in submitters:
            thread.start()
        for thread in submitters:
            thread.join(timeout=30)
        deadline = time.monotonic() + 5.0
        while len(received) < 400 and time.monotonic() < deadline:
            time.sleep(0.01)
        owner.close()
        reader.join(timeout=5.0)
        writer.close()
        assert surprises == [], surprises[:3]
        assert len(received) == 400 and len(set(received)) == 400, (
            f"{len(received)} received, {len(set(received))} unique — a shared slot "
            f"published one camera's payload over another's"
        )


class TestThePendingLockIsReentrant:
    def test_a_finalizer_firing_inside_the_critical_section_returns(self) -> None:
        """Round 10, the round-9 mechanism on the other lock: `_finalize_handle` runs
        synchronously in the collecting thread, which may already hold `_PENDING_LOCK`
        (reap's own list mutations allocate). Re-entry must return, not self-deadlock."""
        from shipinfer.runtime.memory import shared_ring as module

        with module._PENDING_LOCK:
            module._finalize_handle(memoryview(b""), "never-opened-name")  # re-enters, returns
        assert True


class TestASecondCloseKeepsTheParkPinnedSafe:
    class _View:
        def data_ptr(self) -> int:
            return 0xACE0

        def numel(self) -> int:
            return 4096

    def test_double_close_under_a_live_pinned_view_parks_nothing_and_unpins_once(
        self, ring, monkeypatch
    ) -> None:
        """Round 11: the second close saw `_pinned is None`, fell through to block.close(),
        and parked a mapping whose pages were still registered — a later reap would munmap
        them and the finalizer's unregister would run on an unmapped pointer."""
        from types import SimpleNamespace

        import shipinfer.runtime.platform as platform_module
        from shipinfer.runtime.memory import shared_ring as module

        unregisters: list[int] = []
        cudart_calls = SimpleNamespace(
            cudaHostRegister=lambda ptr, n, flags: 0,
            cudaHostUnregister=lambda ptr, _u=unregisters: _u.append(ptr),
        )
        fake_torch = SimpleNamespace(
            uint8="uint8",
            frombuffer=lambda *a, **k: self._View(),
            cuda=SimpleNamespace(cudart=lambda _c=cudart_calls: _c),
        )
        monkeypatch.setattr(platform_module, "require_torch", lambda _ft=fake_torch: _ft)
        held = ring.pinned_tensor(0)
        ring.close()
        ring.close()  # the second close must defer to the finalizer, not park a pinned block
        assert unregisters == [], "nothing unpinned while the view lives"
        assert module.reap_pending_closes() == 0, "and nothing pinned was parked for a reap"
        del held
        gc.collect()
        assert unregisters == [0xACE0], "the finalizer unpinned exactly once"


class TestAStaleFinalizerCannotReleaseASuccessorsClaim:
    def test_the_dead_handles_collection_leaves_the_live_claim_standing(self, ring) -> None:
        """Round 12: the reconnect shape — close the old handle, open the new one, drop the
        old reference. The dead handle's finalizer fires *after* the successor claimed the
        name, and with a name-keyed claim it released the successor's — re-permitting two
        writers over one segment. The claim is identity-keyed now."""
        first = SharedRing.open(ring.name, ring.layout)
        first.close()
        second = SharedRing.open(ring.name, ring.layout)
        del first
        gc.collect()  # the stale finalizer fires; the live claim must stand
        with pytest.raises(RingProtocolError, match="already open in this process"):
            SharedRing.open(ring.name, ring.layout)
        second.close()
        third = SharedRing.open(ring.name, ring.layout)  # the holder's close released it
        third.close()


class TestCloseRacesClose:
    class _View:
        def data_ptr(self) -> int:
            return 0xCC1

        def numel(self) -> int:
            return 4096

    def test_two_closers_and_a_dying_pinned_view_unpin_exactly_once(self, monkeypatch) -> None:
        """Round 12: the unpin ran outside the pin lock after clearing the state that would
        tell a concurrent closer it was in flight — the second closer could munmap first and
        the unregister then ran on an unmapped pointer. Both paths hold the lock end to end
        now; two racing closers and the finalizer must produce exactly one unregister and no
        surprise, every time."""
        from types import SimpleNamespace

        import shipinfer.runtime.platform as platform_module

        for _ in range(100):
            unregisters: list[int] = []
            cudart_calls = SimpleNamespace(
                cudaHostRegister=lambda ptr, n, flags: 0,
                cudaHostUnregister=lambda ptr, _u=unregisters: _u.append(ptr),
            )
            fake_torch = SimpleNamespace(
                uint8="uint8",
                frombuffer=lambda *a, **k: self._View(),
                cuda=SimpleNamespace(cudart=lambda _c=cudart_calls: _c),
            )
            monkeypatch.setattr(platform_module, "require_torch", lambda _ft=fake_torch: _ft)
            ring = SharedRing.create(_name(), RingLayout(slots=2, slot_bytes=4096), owner="A")
            held = ring.pinned_tensor(0)
            surprises: list[BaseException] = []

            def closer(ring=ring, surprises=surprises) -> None:
                try:
                    ring.close()
                except BaseException as exc:
                    surprises.append(exc)

            threads = [threading.Thread(target=closer) for _ in range(2)]
            for thread in threads:
                thread.start()
            del held  # the view dies while the closers race; the finalizer may run anywhere
            gc.collect()
            for thread in threads:
                thread.join(timeout=5.0)
            gc.collect()
            from shipinfer.runtime.memory.shared_ring import reap_pending_closes

            reap_pending_closes()
            assert surprises == [], surprises[:2]
            assert unregisters == [0xCC1], f"unpinned {len(unregisters)} times, want exactly 1"
