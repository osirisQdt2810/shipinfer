"""Dynamic batching: the delay window and the preferred sizes."""

from __future__ import annotations

import threading
import time

import pytest

from shipinfer.scheduling.batching import choose_batch_size
from shipinfer.scheduling.queues import BatchWindow, FairPriorityQueue


def test_returns_immediately_when_batch_is_already_full(make_item) -> None:
    queue = FairPriorityQueue("q", capacity=16)
    for i in range(8):
        queue.put(make_item(frame=i))

    started = time.monotonic()
    batch = queue.get_batch(BatchWindow(max_batch_size=8, max_delay_us=50_000))
    elapsed = time.monotonic() - started

    assert len(batch) == 8
    assert elapsed < 0.02, "a full batch must not wait out the delay window"


def test_waits_for_the_window_then_sends_a_partial_batch(make_item) -> None:
    queue = FairPriorityQueue("q", capacity=16)
    queue.put(make_item(frame=0))

    started = time.monotonic()
    batch = queue.get_batch(BatchWindow(max_batch_size=8, max_delay_us=30_000))
    elapsed = time.monotonic() - started

    assert len(batch) == 1
    assert elapsed >= 0.025, "a partial batch must actually wait for stragglers"


def test_late_arrivals_join_the_batch(make_item) -> None:
    """The whole point of the window: work that arrives during it rides along."""
    queue = FairPriorityQueue("q", capacity=16)
    queue.put(make_item(frame=0))

    def add_more() -> None:
        time.sleep(0.01)
        for i in range(1, 5):
            queue.put(make_item(frame=i))

    threading.Thread(target=add_more, daemon=True).start()
    batch = queue.get_batch(BatchWindow(max_batch_size=8, max_delay_us=100_000))
    assert len(batch) == 5


def test_preferred_size_short_circuits_the_wait(make_item) -> None:
    queue = FairPriorityQueue("q", capacity=16)
    for i in range(4):
        queue.put(make_item(frame=i))

    started = time.monotonic()
    batch = queue.get_batch(
        BatchWindow(max_batch_size=32, max_delay_us=200_000, preferred_sizes=(4, 8, 16))
    )
    elapsed = time.monotonic() - started

    assert len(batch) == 4
    assert elapsed < 0.05, "reaching a preferred size should stop the wait early"


@pytest.mark.parametrize(
    ("size", "preferred", "maximum", "expected"),
    [
        (31, (8, 16, 32), 32, 16),  # pick the profiled shape below, not the unprofiled 31
        (32, (8, 16, 32), 32, 32),
        (5, (8, 16), 32, 5),  # nothing fits: run what we have
        (100, (8, 16, 32), 32, 32),  # capped at the maximum
    ],
)
def test_choose_batch_size(size, preferred, maximum, expected) -> None:
    assert choose_batch_size(size, preferred, maximum) == expected


def test_window_rejects_impossible_preferred_sizes() -> None:
    with pytest.raises(ValueError, match="preferred_sizes"):
        BatchWindow(max_batch_size=8, preferred_sizes=(4, 16))
