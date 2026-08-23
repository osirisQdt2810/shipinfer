"""The batch-assembly contract.

Two operations, and they are inverses: pack N requests into one execution, then split the
one response back into N. Kept separate from the queue because the two answer different
questions — the queue decides *which* requests travel together, this decides *how* their
tensors are packed.

The scatter half is where correctness quietly lives. Every output row must return to the
request that produced it. Get it wrong and two cameras' detections swap places, and
nothing crashes.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from shipinfer.core.types import Tensor
from shipinfer.scheduling.work import WorkItem

__all__ = ["AssembledBatch", "Batcher"]


@dataclass(frozen=True, slots=True)
class AssembledBatch:
    """One execution's worth of work: stacked inputs plus the row span per request."""

    items: tuple[WorkItem, ...]
    inputs: dict[str, Tensor]
    #: ``spans[i]`` is the ``(start, stop)`` row range that ``items[i]`` contributed.
    spans: tuple[tuple[int, int], ...]

    @property
    def size(self) -> int:
        """Total rows in the batch."""
        return self.spans[-1][1] if self.spans else 0

    @property
    def request_count(self) -> int:
        return len(self.items)


class Batcher(abc.ABC):
    """Packs requests into a batch and unpacks the response.

    Stateless and reusable across the instances of one model; one is built per
    :class:`~shipinfer.server.model.Model` and shared by every instance of it.
    """

    name: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def assemble(self, items: Sequence[WorkItem]) -> AssembledBatch:
        """Stack ``items`` into one batched input map.

        Raises:
            ValidationError: if a request's tensors do not match the model's inputs.
            InferenceError: if the assembled batch exceeds ``max_batch_size``.
        """

    @abc.abstractmethod
    def scatter(
        self, batch: AssembledBatch, outputs: Mapping[str, Tensor]
    ) -> list[dict[str, Tensor]]:
        """Split one batched output map into one map per request."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"
