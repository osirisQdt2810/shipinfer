"""Batch assembly and scatter — where a mis-routed output row would swap two cameras."""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.errors import InferenceError
from shipinfer.core.request import InferenceRequest, RequestContext, ResponseFuture
from shipinfer.core.types import DataType, Tensor, TensorSpec
from shipinfer.scheduling.batching import StackingBatcher
from shipinfer.scheduling.work import WorkItem

INPUTS = (TensorSpec("x", DataType.FP32, (4,)),)
OUTPUTS = (TensorSpec("y", DataType.FP32, (2,)),)


def _item(value: float, rows: int = 1, camera: str = "cam0") -> WorkItem:
    request = InferenceRequest(
        model_name="m",
        inputs={"x": Tensor.from_numpy(np.full((rows, 4), value, dtype=np.float32))},
        context=RequestContext(camera_id=camera),
    )
    return WorkItem(request, ResponseFuture(request))


class TestAssemble:
    """N requests become one batch, with a span per request recorded on the way in."""

    def test_assemble_stacks_in_order(self) -> None:
        batcher = StackingBatcher(INPUTS, OUTPUTS, max_batch_size=8)
        batch = batcher.assemble([_item(1.0), _item(2.0), _item(3.0)])

        assert batch.size == 3
        assert batch.spans == ((0, 1), (1, 2), (2, 3))
        np.testing.assert_array_equal(
            batch.inputs["x"].numpy()[:, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

    def test_assemble_skips_the_copy_for_a_single_request(self) -> None:
        """A one-request batch must not pay for a stack it does not need."""
        batcher = StackingBatcher(INPUTS, OUTPUTS, max_batch_size=8)
        item = _item(1.0)
        batch = batcher.assemble([item])
        assert batch.inputs["x"] is item.request.inputs["x"]

    def test_assemble_handles_multi_row_requests(self) -> None:
        batcher = StackingBatcher(INPUTS, OUTPUTS, max_batch_size=8)
        batch = batcher.assemble([_item(1.0, rows=2), _item(2.0, rows=3)])
        assert batch.size == 5
        assert batch.spans == ((0, 2), (2, 5))


class TestScatter:
    """Every output row goes back to the request that contributed it, as a view."""

    def test_scatter_returns_each_request_its_own_rows(self) -> None:
        """The correctness property that matters: row i goes back to the request that made it."""
        batcher = StackingBatcher(INPUTS, OUTPUTS, max_batch_size=8)
        items = [_item(1.0, camera="cam_a"), _item(2.0, rows=2, camera="cam_b")]
        batch = batcher.assemble(items)

        outputs = {"y": Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(3, 2))}
        scattered = batcher.scatter(batch, outputs)

        assert len(scattered) == 2
        np.testing.assert_array_equal(scattered[0]["y"].numpy(), [[0.0, 1.0]])
        np.testing.assert_array_equal(scattered[1]["y"].numpy(), [[2.0, 3.0], [4.0, 5.0]])

    def test_scatter_produces_views_not_copies(self) -> None:
        batcher = StackingBatcher(INPUTS, OUTPUTS, max_batch_size=8)
        batch = batcher.assemble([_item(1.0), _item(2.0)])
        source = np.arange(4, dtype=np.float32).reshape(2, 2)
        scattered = batcher.scatter(batch, {"y": Tensor.from_numpy(source)})
        assert np.shares_memory(scattered[0]["y"].numpy(), source)


class TestBatchValidation:
    """A batch that could misattribute a response is refused, not scattered."""

    def test_scatter_rejects_a_row_count_mismatch(self) -> None:
        """A backend that is not batch-major would silently misattribute every response."""
        batcher = StackingBatcher(INPUTS, OUTPUTS, max_batch_size=8)
        batch = batcher.assemble([_item(1.0), _item(2.0)])
        with pytest.raises(InferenceError, match="not batch-major"):
            batcher.scatter(batch, {"y": Tensor.from_numpy(np.zeros((5, 2), dtype=np.float32))})

    def test_assemble_rejects_an_oversized_batch(self) -> None:
        batcher = StackingBatcher(INPUTS, OUTPUTS, max_batch_size=2)
        with pytest.raises(InferenceError, match="exceeds max_batch_size"):
            batcher.assemble([_item(1.0), _item(2.0), _item(3.0)])

    def test_assemble_rejects_a_wrong_shape(self) -> None:
        batcher = StackingBatcher(INPUTS, OUTPUTS, max_batch_size=8)
        request = InferenceRequest(
            model_name="m", inputs={"x": Tensor.from_numpy(np.zeros((1, 7), dtype=np.float32))}
        )
        with pytest.raises(ValueError, match="does not match"):
            batcher.assemble([WorkItem(request, ResponseFuture(request))])
