"""The default batcher: numpy row-stacking with zero-copy scatter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from shipinfer.core.errors import InferenceError, ValidationError
from shipinfer.core.types import Tensor, TensorSpec, stack_tensors, validate_against
from shipinfer.scheduling.batching.base import AssembledBatch, Batcher
from shipinfer.scheduling.batching.registry import BATCHERS
from shipinfer.scheduling.work import WorkItem

__all__ = ["StackingBatcher"]


@BATCHERS.register("stacking", "numpy")
class StackingBatcher(Batcher):
    """Concatenate along the batch axis; slice views on the way back.

    Two deliberate optimisations, both of which matter at 15 000 requests/s:

    * a **single-request batch skips the stack entirely** — the common case for a heavy
      model, where copying the payload to "batch" it would be pure loss;
    * **scatter returns numpy views, not copies** — a 64-image batch is not copied 64 times
      on the way out.
    """

    name = "stacking"

    __slots__ = ("_input_names", "_input_specs", "_max_batch_size", "_output_specs")

    def __init__(
        self,
        input_specs: Sequence[TensorSpec],
        output_specs: Sequence[TensorSpec],
        max_batch_size: int,
    ) -> None:
        self._input_specs = tuple(input_specs)
        self._output_specs = tuple(output_specs)
        self._max_batch_size = max_batch_size
        self._input_names = tuple(spec.name for spec in input_specs)

    # -- pack ---------------------------------------------------------------------------

    def assemble(self, items: Sequence[WorkItem]) -> AssembledBatch:
        if not items:
            raise ValueError("cannot assemble an empty batch")

        spans: list[tuple[int, int]] = []
        offset = 0
        for item in items:
            rows = item.request.batch_size or 1
            spans.append((offset, offset + rows))
            offset += rows
        if offset > self._max_batch_size:
            raise InferenceError(
                f"assembled batch of {offset} rows exceeds max_batch_size {self._max_batch_size}"
            )

        if len(items) == 1:
            inputs = dict(items[0].request.inputs)
        else:
            inputs = {}
            for name in self._input_names:
                contributions = [
                    item.request.inputs[name] for item in items if name in item.request.inputs
                ]
                if not contributions:
                    continue
                if len(contributions) != len(items):
                    raise ValidationError(
                        f"optional input {name!r} is present in only "
                        f"{len(contributions)}/{len(items)} requests of a batch; "
                        "an optional input must be all-or-nothing within a batch"
                    )
                inputs[name] = stack_tensors(contributions)

        validate_against(inputs, self._input_specs, what="input")
        return AssembledBatch(items=tuple(items), inputs=inputs, spans=tuple(spans))

    # -- unpack -------------------------------------------------------------------------

    def scatter(
        self, batch: AssembledBatch, outputs: Mapping[str, Tensor]
    ) -> list[dict[str, Tensor]]:
        validate_against(dict(outputs), self._output_specs, what="output")
        expected_rows = batch.size

        for name, tensor in outputs.items():
            if tensor.shape and tensor.shape[0] != expected_rows:
                raise InferenceError(
                    f"output {name!r} has {tensor.shape[0]} rows but the batch had "
                    f"{expected_rows}; the backend is not batch-major"
                )

        if batch.request_count == 1:
            return [dict(outputs)]

        return [
            {name: tensor.slice_batch(start, stop) for name, tensor in outputs.items()}
            for start, stop in batch.spans
        ]

    # -- helpers ------------------------------------------------------------------------

    def zeros_like_outputs(self, batch_size: int) -> dict[str, Tensor]:
        """A zero-filled response matching the declared outputs.

        Used by warm-up, so it needs no knowledge of the declared shapes.
        """
        out: dict[str, Tensor] = {}
        for spec in self._output_specs:
            shape = tuple(max(dim, 1) for dim in spec.shape)
            out[spec.name] = Tensor.from_numpy(
                np.zeros((batch_size, *shape), dtype=spec.dtype.numpy_dtype)
            )
        return out
