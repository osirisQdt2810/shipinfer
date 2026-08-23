"""Keep work on the GPU that already holds its data — until that stops being fastest."""

from __future__ import annotations

from collections.abc import Sequence

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import InferenceRequest
from shipinfer.scheduling.policies.base import Placeable, PlacementPolicy
from shipinfer.scheduling.policies.power_of_two import PowerOfTwoChoicesPolicy
from shipinfer.scheduling.policies.registry import POLICIES

__all__ = ["LocalityAwareSpilloverPolicy"]


@POLICIES.register("locality_spillover", "locality")
class LocalityAwareSpilloverPolicy(PlacementPolicy):
    """Prefer the resident GPU; spill to a shorter queue once it backs up.

    The project default, and the reason is bandwidth. A 1080p frame is ~6 MB, so moving it
    to another GPU costs a D2H plus an H2D; a crop is a few KB and can go anywhere. So:

    * if the request is already resident on a candidate GPU **and** that GPU's queue is at
      or below ``spill_threshold`` -> run it there, with no copy at all;
    * otherwise defer to ``fallback`` and accept the copy, because a queue that is
      genuinely backing up costs more than one transfer.

    ``spill_threshold=0`` degenerates to pure load balancing; a very large value degenerates
    to pinning. Neither extreme is right, which is why it is a knob rather than a constant.
    """

    name = "locality_spillover"

    def __init__(
        self, spill_threshold: int = 4, fallback: PlacementPolicy | None = None
    ) -> None:
        if spill_threshold < 0:
            raise ConfigurationError("locality_spillover: spill_threshold must be >= 0")
        self.spill_threshold = spill_threshold
        self._fallback = fallback or PowerOfTwoChoicesPolicy()

    def select(self, candidates: Sequence[Placeable], request: InferenceRequest) -> Placeable:
        resident = request.resident_device
        if resident is not None and resident.is_cuda:
            local_best: Placeable | None = None
            for candidate in candidates:
                if candidate.device == resident and (
                    local_best is None or candidate.depth < local_best.depth
                ):
                    local_best = candidate
            if local_best is not None and local_best.depth <= self.spill_threshold:
                return local_best
        return self._fallback.select(candidates, request)

    def describe(self) -> str:
        return (
            f"stay on the resident GPU while depth <= {self.spill_threshold}, "
            f"else {self._fallback.describe()}"
        )

    def __repr__(self) -> str:
        return (
            f"<LocalityAwareSpilloverPolicy threshold={self.spill_threshold} "
            f"fallback={self._fallback!r}>"
        )
