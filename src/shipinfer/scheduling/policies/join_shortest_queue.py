"""Send work to the instance with the fewest queued requests."""

from __future__ import annotations

from collections.abc import Sequence

from shipinfer.core.request import InferenceRequest
from shipinfer.scheduling.policies.base import Placeable, PlacementPolicy
from shipinfer.scheduling.policies.registry import POLICIES

__all__ = ["JoinShortestQueuePolicy"]


@POLICIES.register("join_shortest_queue", "jsq")
class JoinShortestQueuePolicy(PlacementPolicy):
    """Pick the shortest queue, scanning every candidate.

    Optimal for a small, uniform pool. Its weakness at scale is herding: with many
    dispatcher threads reading the same depths, they all pick the same "shortest" queue at
    the same instant — which is exactly what :class:`~shipinfer.scheduling.policies.power_of_two.PowerOfTwoChoicesPolicy`
    exists to avoid.
    """

    name = "join_shortest_queue"

    def select(self, candidates: Sequence[Placeable], request: InferenceRequest) -> Placeable:
        best = candidates[0]
        best_depth = best.depth
        for candidate in candidates[1:]:
            depth = candidate.depth
            if depth < best_depth:
                best, best_depth = candidate, depth
                if depth == 0:
                    break  # cannot beat an idle instance
        return best

    def describe(self) -> str:
        return "shortest queue over all instances (optimal for small pools)"
