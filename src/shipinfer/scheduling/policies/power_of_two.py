"""Two random probes, take the shorter queue."""

from __future__ import annotations

import random
from collections.abc import Sequence

from shipinfer.core.request import InferenceRequest
from shipinfer.scheduling.policies.base import Placeable, PlacementPolicy
from shipinfer.scheduling.policies.registry import POLICIES

__all__ = ["PowerOfTwoChoicesPolicy"]


@POLICIES.register("power_of_two", "p2c")
class PowerOfTwoChoicesPolicy(PlacementPolicy):
    """Sample two instances uniformly at random and send to the shorter queue.

    The classic result: two random probes capture almost all the benefit of full
    join-shortest-queue while keeping the decision O(1) in pool size and avoiding the
    herding that makes JSQ misbehave with many concurrent dispatchers. This is the default
    fallback inside :class:`~shipinfer.scheduling.policies.locality_spillover.LocalityAwareSpilloverPolicy`.
    """

    name = "power_of_two"

    def __init__(self, rng: random.Random | None = None) -> None:
        #: Injectable so a test can assert a deterministic placement sequence.
        self._rng = rng or random.Random()

    def select(self, candidates: Sequence[Placeable], request: InferenceRequest) -> Placeable:
        n = len(candidates)
        if n == 1:
            return candidates[0]
        i = self._rng.randrange(n)
        j = self._rng.randrange(n - 1)
        if j >= i:
            j += 1  # sample without replacement, no rejection loop
        a, b = candidates[i], candidates[j]
        return a if a.depth <= b.depth else b

    def describe(self) -> str:
        return "two random probes, shorter queue wins (scales, no herding)"
