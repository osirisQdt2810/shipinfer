"""Strict rotation across instances."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from shipinfer.core.request import InferenceRequest
from shipinfer.scheduling.policies.base import Placeable, PlacementPolicy
from shipinfer.scheduling.policies.registry import POLICIES

__all__ = ["RoundRobinPolicy"]


@POLICIES.register("round_robin", "rr")
class RoundRobinPolicy(PlacementPolicy):
    """Rotate through instances in order, ignoring load.

    The cheapest possible policy, and correct only when every instance is equally fast and
    equally loaded — which is exactly the assumption that produced the imbalance in the
    previous system. Kept as the baseline the other policies are benchmarked against.
    """

    name = "round_robin"

    def __init__(self) -> None:
        self._counter = itertools.count()

    def select(self, candidates: Sequence[Placeable], request: InferenceRequest) -> Placeable:
        return candidates[next(self._counter) % len(candidates)]

    def describe(self) -> str:
        return "rotate in order, load-blind (baseline)"
