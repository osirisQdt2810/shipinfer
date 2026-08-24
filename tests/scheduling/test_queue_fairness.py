"""The fairness guarantee, which is the reason this project exists.

The previous system funnelled every camera into one shared 1000-slot buffer that evicted
the *oldest* entry when full. A crowded camera therefore starved a quiet one, silently.
These tests pin the two behaviours that make that impossible now.
"""

from __future__ import annotations

import threading
import time

import pytest

from shipinfer.core.errors import QueueFullError
from shipinfer.core.request import Priority
from shipinfer.core.settings import OverflowPolicy
from shipinfer.scheduling.queues import BatchWindow, FairPriorityQueue, FifoQueue
from shipinfer.scheduling.work import summarise_fairness


class TestFairDrain:
    """Who gets the next batch slot: a turn per camera, except when priority says otherwise."""

    def test_round_robin_across_cameras(self, make_item) -> None:
        """One loud camera must not fill a batch.

        30 requests from cam_busy and 2 each from three quiet cameras. A FIFO drain would give
        the first 8 slots entirely to cam_busy; fair queueing must give every camera a turn.
        """
        queue = FairPriorityQueue("q", capacity=64)
        for i in range(30):
            queue.put(make_item(camera="cam_busy", frame=i))
        for camera in ("cam_a", "cam_b", "cam_c"):
            for i in range(2):
                queue.put(make_item(camera=camera, frame=i))

        batch = queue.get_batch(BatchWindow(max_batch_size=8))
        counts = summarise_fairness(batch)

        assert len(batch) == 8
        assert set(counts) == {"cam_busy", "cam_a", "cam_b", "cam_c"}
        assert counts["cam_busy"] == 2, f"the loud camera took {counts['cam_busy']} of 8 slots"

    def test_fifo_is_not_fair(self, make_item) -> None:
        """The baseline, so the fairness claim has something to be true *relative to*."""
        queue = FifoQueue("q", capacity=64)
        for i in range(30):
            queue.put(make_item(camera="cam_busy", frame=i))
        queue.put(make_item(camera="cam_quiet", frame=0))

        batch = queue.get_batch(BatchWindow(max_batch_size=8))
        assert summarise_fairness(batch) == {"cam_busy": 8}

    def test_priority_beats_fairness(self, make_item) -> None:
        """A tracking-critical request jumps every normal one, whatever its camera."""
        queue = FairPriorityQueue("q", capacity=64)
        for i in range(10):
            queue.put(make_item(camera="cam_a", frame=i))
        queue.put(make_item(camera="cam_z", frame=99, priority=Priority.TRACKING_CRITICAL))

        batch = queue.get_batch(BatchWindow(max_batch_size=4))
        assert batch[0].request.context.camera_id == "cam_z"


class TestOverflowPolicies:
    """What a full queue does — and which camera pays for it."""

    def test_reject_when_full_reports_depth(self, make_item) -> None:
        """Backpressure carries numbers an operator can act on."""
        queue = FairPriorityQueue("q", capacity=2, overflow=OverflowPolicy.REJECT)
        queue.put(make_item(frame=0))
        queue.put(make_item(frame=1))

        with pytest.raises(QueueFullError) as excinfo:
            queue.put(make_item(frame=2))

        assert excinfo.value.depth == 2
        assert excinfo.value.capacity == 2
        assert queue.stats().rejected == 1

    def test_drop_oldest_evicts_the_greediest_camera(self, make_item) -> None:
        """Eviction penalises the flood, not its victim.

        This is the precise inversion of the old behaviour: dropping the *globally oldest*
        entry kills the quiet camera's request, because it has been waiting longest through no
        fault of its own.
        """
        queue = FairPriorityQueue("q", capacity=5, overflow=OverflowPolicy.DROP_OLDEST)
        quiet = make_item(camera="cam_quiet", frame=0)
        queue.put(quiet)
        for i in range(4):
            queue.put(make_item(camera="cam_busy", frame=i))

        queue.put(make_item(camera="cam_busy", frame=99))

        batch = queue.get_batch(BatchWindow(max_batch_size=8))
        cameras = {item.request.context.camera_id for item in batch}
        assert "cam_quiet" in cameras, "the quiet camera's request was evicted"
        assert quiet.future.done() is False

    def test_block_policy_waits_for_space(self, make_item) -> None:
        queue = FairPriorityQueue(
            "q", capacity=1, overflow=OverflowPolicy.BLOCK, block_timeout_ms=500
        )
        queue.put(make_item(frame=0))

        def drain() -> None:
            time.sleep(0.05)
            queue.get_batch(BatchWindow(max_batch_size=1))

        threading.Thread(target=drain, daemon=True).start()
        queue.put(make_item(frame=1))  # blocks until the drain frees a slot
        assert queue.depth == 1

    def test_the_producer_wakes_when_the_slot_frees_not_when_the_deadline_expires(
        self, make_item
    ) -> None:
        """The test above passes 450 ms late and cannot tell the difference.

        It asserts only that the put eventually succeeds, so it stayed green through a
        regression that made every blocked producer sleep the entire `block_timeout_ms`: the
        drain's row-budget exit changed from `break` to `return` and jumped over the
        `notify_all`. At the design point that is a hard ceiling on the ingest thread, and
        when the timeout beats the drain, `put` raises `QueueFullError` and the camera actor
        charges a drop to a camera that did nothing wrong.

        `max_batch_size=1` is the shape that matters: it takes the `rows >= max_rows` exit,
        which is the one taken on the common path under load.
        """
        freed_at: list[float] = []
        queue = FairPriorityQueue(
            "q", capacity=1, overflow=OverflowPolicy.BLOCK, block_timeout_ms=2000
        )
        queue.put(make_item(frame=0))

        def drain() -> None:
            time.sleep(0.05)
            queue.get_batch(BatchWindow(max_batch_size=1))
            freed_at.append(time.monotonic())

        threading.Thread(target=drain, daemon=True).start()
        queue.put(make_item(frame=1))
        unblocked_at = time.monotonic()

        assert freed_at, "the drain never ran"
        latency_ms = (unblocked_at - freed_at[0]) * 1000
        assert latency_ms < 250, (
            f"the producer woke {latency_ms:.0f} ms after the slot freed; "
            "it was waiting out the timeout, not being notified"
        )


class TestQueueLifecycle:
    """Work that will never be useful is discarded, and a closed queue strands nothing."""

    def test_expired_requests_are_dropped_before_execution(self, make_item) -> None:
        """Spending GPU time on a frame that is already late is pure waste."""
        queue = FairPriorityQueue("q", capacity=8, drop_expired=True)
        queue.put(make_item(frame=0, deadline_ns=1))  # long past
        queue.put(make_item(frame=1))

        batch = queue.get_batch(BatchWindow(max_batch_size=8))
        assert [item.request.context.frame_id for item in batch] == [1]
        assert queue.stats().expired == 1

    def test_close_fails_everything_still_queued(self, make_item) -> None:
        queue = FairPriorityQueue("q", capacity=8)
        items = [make_item(frame=i) for i in range(3)]
        for item in items:
            queue.put(item)

        drained = queue.close()

        assert len(drained) == 3
        assert all(item.future.done() for item in items)
        assert all(isinstance(item.future.exception(), Exception) for item in items)
