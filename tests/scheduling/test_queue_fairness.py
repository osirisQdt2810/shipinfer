"""The fairness guarantee, which is the reason this project exists.

The previous system funnelled every camera into one shared 1000-slot buffer that evicted
the *oldest* entry when full. A crowded camera therefore starved a quiet one, silently.
These tests pin the two behaviours that make that impossible now.
"""

from __future__ import annotations

import threading
import time

import pytest

from shipinfer.core.errors import QueueFullError, RequestCancelledError
from shipinfer.core.request import Priority
from shipinfer.core.settings import OverflowPolicy
from shipinfer.scheduling.queues import (
    BatchWindow,
    FairPriorityQueue,
    FifoQueue,
    QueueStats,
)
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


class TestPerCameraAttribution:
    """Who paid for each drop — the reading that turns ADR-005 from a claim into a number.

    The totals already say a queue refused, evicted or expired work. They cannot say
    *whose*, and that is the exact question the inherited bug hid: "camera đông người được
    nhận diện đầy đủ, camera vắng người thỉnh thoảng bị miss" — the crowded cameras are
    recognised in full while the quiet ones occasionally miss — is a per-camera observation
    that a per-queue counter can never confirm or refute. Each test here drops work with a
    known owner and asserts the queue named that owner and nobody else.
    """

    def test_eviction_is_charged_to_the_greedy_camera_alone(self, make_item) -> None:
        """The loud camera pays for an eviction the quiet camera's frame triggered.

        The submitter is deliberately *not* the victim. An earlier version of this test put
        the three loud frames first and then submitted from `loud`, which makes submitter
        and victim the same camera — so a queue that charged the camera it was making room
        *for*, rather than the camera it took the slot *from*, passed it. Here `quiet` is
        the one asking for space and `loud` is the one that must be named.
        """
        queue = FairPriorityQueue("q", capacity=4, overflow=OverflowPolicy.DROP_OLDEST)
        queue.put(make_item(camera="quiet", frame=0))
        for i in range(3):
            queue.put(make_item(camera="loud", frame=i))

        queue.put(make_item(camera="quiet", frame=99))

        stats = queue.stats()
        assert stats.evicted == 1
        assert stats.evicted_by_camera == {"loud": 1}, "the flood must pay for its own flood"
        assert stats.depth_by_camera == {
            "quiet": 2,
            "loud": 2,
        }, "the quiet camera's first frame survived; the loud camera lost one"

    def test_expiry_names_only_the_camera_that_was_late(self, make_item) -> None:
        queue = FairPriorityQueue("q", capacity=8, drop_expired=True)
        queue.put(make_item(camera="late", frame=0, deadline_ns=1))  # long past
        queue.put(make_item(camera="ontime", frame=0))

        queue.get_batch(BatchWindow(max_batch_size=8))

        stats = queue.stats()
        assert stats.expired == 1
        assert stats.expired_by_camera == {"late": 1}

    def test_a_refusal_names_the_camera_that_was_refused(self, make_item) -> None:
        """Not the camera that filled the queue — the one whose frame was turned away.

        Both readings are defensible and they answer different questions; this one answers
        "which camera is losing frames right now", which is what an operator watching a gap
        in a stream is asking.
        """
        queue = FairPriorityQueue("q", capacity=2, overflow=OverflowPolicy.REJECT)
        queue.put(make_item(camera="resident", frame=0))
        queue.put(make_item(camera="resident", frame=1))

        with pytest.raises(QueueFullError):
            queue.put(make_item(camera="newcomer", frame=0))

        assert queue.stats().rejected_by_camera == {"newcomer": 1}

    def test_depth_by_camera_sums_to_depth_across_priority_bands(self, make_item) -> None:
        """The breakdown is of the whole queue, not of one lane.

        A camera whose critical frames sit in one lane and whose normal frames sit in
        another is one camera; reporting it twice, or reporting only the lane the walk
        happened to reach, would make the sum disagree with `depth` — and a breakdown that
        does not add up is worse than none.
        """
        queue = FairPriorityQueue("q", capacity=32)
        for i in range(3):
            queue.put(make_item(camera="cam_a", frame=i))
        queue.put(make_item(camera="cam_a", frame=9, priority=Priority.TRACKING_CRITICAL))
        for i in range(2):
            queue.put(make_item(camera="cam_b", frame=i, priority=Priority.BACKGROUND))

        stats = queue.stats()

        assert stats.depth_by_camera == {"cam_a": 4, "cam_b": 2}
        assert sum(stats.depth_by_camera.values()) == stats.depth == 6

    def test_a_caller_with_no_camera_lands_in_the_shared_bucket(self, make_item) -> None:
        """`"-"`, not `""` and not a private lane per caller.

        `WorkItem.fairness_key` already collapses camera-less callers into one lane so they
        queue together instead of each inventing their own; the attribution has to use the
        same key or the two views of the same queue would disagree.
        """
        queue = FairPriorityQueue("q", capacity=1, overflow=OverflowPolicy.REJECT)
        queue.put(make_item(camera="", frame=0))

        with pytest.raises(QueueFullError):
            queue.put(make_item(camera="", frame=1))

        stats = queue.stats()
        assert stats.depth_by_camera == {"-": 1}
        assert stats.rejected_by_camera == {"-": 1}

    def test_close_does_not_charge_anybody(self, make_item) -> None:
        """Shutdown loss is not a per-camera fault.

        The runner's `items_queue_closed` owns that outcome. Charging it here too would
        double-count it and, worse, make an orderly stop read like a flood in the one view
        an operator uses to find floods.
        """
        queue = FairPriorityQueue("q", capacity=8)
        for i in range(3):
            queue.put(make_item(camera="cam_a", frame=i))

        queue.close()

        stats = queue.stats()
        assert stats.evicted_by_camera == {}
        assert stats.expired_by_camera == {}
        assert stats.rejected_by_camera == {}
        assert stats.depth_by_camera == {}

    @pytest.mark.parametrize("queue_class", [FairPriorityQueue, FifoQueue])
    def test_a_producer_woken_by_close_is_cancelled_not_charged(
        self, make_item, queue_class
    ) -> None:
        """A shutdown that catches a blocked producer is still a shutdown.

        Under `BLOCK` a producer sleeps inside the make-room path until a slot frees or
        `block_timeout_ms` expires — and `close()` wakes it too. Both wakes left that path
        with the same `False`, so `put` charged `rejected_by_camera[<the blocked camera>]`
        and raised `QueueFullError("full (0/1)")`: a stopping server reported as a camera
        flooding, in the one view an operator uses to find floods, and in flat contradiction
        of `QueueStats`'s own promise that `close()` charges nobody. The timeout here is
        long enough that a timeout-shaped exit could not have produced this result.
        """
        queue = queue_class(
            "q", capacity=1, overflow=OverflowPolicy.BLOCK, block_timeout_ms=5000
        )
        queue.put(make_item(camera="resident", frame=0))
        raised: list[BaseException] = []

        def produce() -> None:
            try:
                queue.put(make_item(camera="waiting", frame=0))
            except Exception as exc:
                raised.append(exc)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        time.sleep(0.1)
        assert not raised, "the producer never blocked, so close() is not what freed it"

        queue.close()
        producer.join(timeout=5.0)

        assert not producer.is_alive(), "the producer was not woken by close()"
        assert len(raised) == 1
        assert isinstance(
            raised[0], RequestCancelledError
        ), f"a shutdown surfaced as {type(raised[0]).__name__}: {raised[0]}"
        stats = queue.stats()
        assert stats.rejected_by_camera == {}, "nobody pays for a shutdown"
        assert stats.rejected == 0

    def test_fifo_attributes_the_same_four_outcomes(self, make_item) -> None:
        """The fairness-blind control reports the same four maps.

        It has to: the benchmark compares the two queues, and a comparison where only one
        side can name a victim is not a comparison. What differs is *who* gets named under
        DROP_OLDEST — FIFO evicts the globally oldest, which is exactly the behaviour the
        fair queue exists to replace, and this test is where that shows up as data.
        """
        queue = FifoQueue("q", capacity=4, overflow=OverflowPolicy.DROP_OLDEST)
        queue.put(make_item(camera="quiet", frame=0))  # oldest, and blameless
        for i in range(3):
            queue.put(make_item(camera="loud", frame=i))

        queue.put(make_item(camera="loud", frame=99))

        stats = queue.stats()
        assert stats.evicted_by_camera == {
            "quiet": 1
        }, "FIFO sacrifices the blameless head — the inherited bug, now visible"
        assert stats.depth_by_camera == {"loud": 4}
        assert sum(stats.depth_by_camera.values()) == stats.depth

    def test_fifo_attributes_refusals_and_expiries(self, make_item) -> None:
        queue = FifoQueue("q", capacity=2, overflow=OverflowPolicy.REJECT, drop_expired=True)
        queue.put(make_item(camera="late", frame=0, deadline_ns=1))
        queue.put(make_item(camera="ontime", frame=0))

        with pytest.raises(QueueFullError):
            queue.put(make_item(camera="newcomer", frame=0))
        queue.get_batch(BatchWindow(max_batch_size=8))

        stats = queue.stats()
        assert stats.rejected_by_camera == {"newcomer": 1}
        assert stats.expired_by_camera == {"late": 1}


class TestQueueStatsIsASnapshotNotAView:
    """`stats()` and `as_dict()` hand out copies.

    `/v2/statistics` serialises this document and a health handler nests it into its own,
    and both are free to trim or re-key what they were given. If the maps were the queue's
    live counters, one such caller would be silently editing the attribution every other
    caller reads — the same class of bug as the staging pool's `stats()` iterating its live
    dict, which 500'd `/v2/statistics` with "dictionary changed size during iteration".
    """

    def test_as_dict_carries_the_four_maps(self, make_item) -> None:
        queue = FairPriorityQueue("q", capacity=8)
        queue.put(make_item(camera="cam_a", frame=0))

        body = queue.stats().as_dict()

        assert body["depth_by_camera"] == {"cam_a": 1}
        assert body["rejected_by_camera"] == {}
        assert body["evicted_by_camera"] == {}
        assert body["expired_by_camera"] == {}

    def test_mutating_the_result_does_not_reach_the_queue(self, make_item) -> None:
        queue = FairPriorityQueue("q", capacity=1, overflow=OverflowPolicy.REJECT)
        queue.put(make_item(camera="cam_a", frame=0))
        with pytest.raises(QueueFullError):
            queue.put(make_item(camera="cam_b", frame=0))

        body = queue.stats().as_dict()
        body["rejected_by_camera"]["cam_b"] = 999
        body["rejected_by_camera"]["ghost"] = 1
        body["depth_by_camera"].clear()

        assert queue.stats().rejected_by_camera == {"cam_b": 1}
        assert queue.stats().depth_by_camera == {"cam_a": 1}

    def test_a_third_queue_may_report_no_attribution_at_all(self) -> None:
        """The maps default to empty so a queue that cannot attribute still constructs.

        Reporting nothing is honest; reporting a zero it never measured is not, and a
        required field would force exactly that on the compiled adapter and on anybody's
        third queue.
        """
        stats = QueueStats(depth=1, capacity=8, accepted=1, rejected=0, evicted=0, expired=0)

        assert stats.as_dict()["evicted_by_camera"] == {}
