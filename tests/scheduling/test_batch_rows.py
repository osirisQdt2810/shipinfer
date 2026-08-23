"""A batch is bounded in rows, not in requests.

Every queue test before this one used one-row requests, so item count and row count were
the same number and the distinction could not fail. The per-object stages break that: a
person-embedder request carries one row per crop in its frame, so sixteen requests can be
twenty-four rows. On the first real 50-camera run the assembler refused the batch —
``assembled batch of 24 rows exceeds max_batch_size 16`` — and every request in it failed.

The assembler was right and the queue was wrong, which is why these tests live here rather
than next to the assembler: the guard that fired is not the bug, the selection that fed it
is. They use multi-row requests throughout, because a one-row fixture cannot tell the two
quantities apart and that is exactly how this survived.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.request import InferenceRequest, RequestContext, ResponseFuture
from shipinfer.core.types import Tensor
from shipinfer.scheduling.queues import BatchWindow, FairPriorityQueue, FifoQueue
from shipinfer.scheduling.work import WorkItem


def item_of(rows: int, camera: str = "cam0") -> WorkItem:
    """A request carrying ``rows`` rows, the way a per-object stage submits its crops."""
    request = InferenceRequest(
        model_name="person_embedder",
        inputs={"crops": Tensor.from_numpy(np.zeros((rows, 3, 4, 4), dtype=np.float32))},
        context=RequestContext(camera_id=camera),
    )
    return WorkItem(request, ResponseFuture(request))


def window(max_rows: int) -> BatchWindow:
    """No delay: these tests are about what is selected, not about how long it waits."""
    return BatchWindow(max_batch_size=max_rows, max_delay_us=0, preferred_sizes=())


def total_rows(items) -> int:
    return sum(i.request.batch_size or 1 for i in items)


class TestTheRowBudgetIsRespected:
    @pytest.mark.parametrize("queue_type", [FairPriorityQueue, FifoQueue])
    def test_multi_row_requests_do_not_overfill_the_batch(self, queue_type):
        """The exact shape that failed: several requests of three crops, budget 16."""
        queue = queue_type("person_embedder", capacity=256)
        for index in range(16):
            queue.put(item_of(3, camera=f"cam{index:02d}"))

        batch = queue.get_batch(window(16))

        assert total_rows(batch) <= 16, "the assembler would refuse this batch"
        assert batch, "a batch that selects nothing would spin the worker"

    @pytest.mark.parametrize("queue_type", [FairPriorityQueue, FifoQueue])
    def test_it_still_fills_the_budget_it_is_given(self, queue_type):
        """Bounding by rows must not mean under-filling: 8 x 2 rows is exactly 16."""
        queue = queue_type("person_embedder", capacity=256)
        for index in range(16):
            queue.put(item_of(2, camera=f"cam{index:02d}"))

        batch = queue.get_batch(window(16))

        assert total_rows(batch) == 16
        assert len(batch) == 8, "eight two-row requests, not sixteen"

    @pytest.mark.parametrize("queue_type", [FairPriorityQueue, FifoQueue])
    def test_a_single_row_queue_is_unchanged(self, queue_type):
        """The old behaviour is the special case, and it still holds."""
        queue = queue_type("ship_detector", capacity=256)
        for index in range(20):
            queue.put(item_of(1, camera=f"cam{index:02d}"))

        batch = queue.get_batch(window(8))

        assert len(batch) == 8
        assert total_rows(batch) == 8

    @pytest.mark.parametrize("queue_type", [FairPriorityQueue, FifoQueue])
    def test_the_remainder_stays_queued(self, queue_type):
        """What does not fit is left for the next batch, not dropped."""
        queue = queue_type("person_embedder", capacity=256)
        for index in range(6):
            queue.put(item_of(5, camera=f"cam{index:02d}"))

        first = queue.get_batch(window(16))
        assert total_rows(first) <= 16

        second = queue.get_batch(window(16))
        assert second, "the rest must still be reachable"
        assert total_rows(first) + total_rows(second) <= 30


class TestAnOversizedRequestDoesNotStallTheModel:
    """The edge case that a naive row check turns into a hang."""

    @pytest.mark.parametrize("queue_type", [FairPriorityQueue, FifoQueue])
    def test_a_request_larger_than_the_budget_is_still_dequeued(self, queue_type):
        """Leaving it at the head of the lane would park that model forever.

        Returning it alone lets the assembler name the real problem — a request too large
        for the engine — instead of the queue silently refusing to make progress.
        """
        queue = queue_type("person_embedder", capacity=256)
        queue.put(item_of(40))
        queue.put(item_of(2))

        batch = queue.get_batch(window(16))

        assert len(batch) == 1
        assert batch[0].request.batch_size == 40
