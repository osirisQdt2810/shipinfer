"""Sticky routing for stateful models — Triton's sequence batcher, as a policy."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from shipinfer.core.request import InferenceRequest
from shipinfer.scheduling.policies.base import Placeable, PlacementPolicy
from shipinfer.scheduling.policies.power_of_two import PowerOfTwoChoicesPolicy
from shipinfer.scheduling.policies.registry import POLICIES

__all__ = ["SequenceAffinityPolicy"]


@POLICIES.register("sequence_affinity", "sticky", "sequence")
class SequenceAffinityPolicy(PlacementPolicy):
    """Route every request of a sequence to the same instance.

    Triton calls this the *sequence batcher*, and it exists because some models carry
    state between calls: a tracker's Kalman filter, a recurrent embedder, a KV cache. Such
    a model is only correct if consecutive requests for one sequence land on the instance
    that holds that sequence's state — load balancing them across GPUs quietly corrupts
    the result rather than merely slowing it down.

    Here the sequence key is the **camera id**, which is exactly the right granularity for
    this pipeline: per-camera MOT state is per-camera by definition.

    Affinity is *sticky, not permanent*. When the affine instance dies or leaves the ready
    set, the sequence is re-pinned through ``fallback``. Refusing to re-pin would turn one
    dead GPU into permanently dropped cameras, which is worse than a re-initialised
    tracker.
    """

    name = "sequence_affinity"

    def __init__(
        self, fallback: PlacementPolicy | None = None, max_sequences: int = 4096
    ) -> None:
        self._fallback = fallback or PowerOfTwoChoicesPolicy()
        self._max_sequences = max_sequences
        self._assigned: dict[str, Placeable] = {}
        self._lock = threading.Lock()

    def select(self, candidates: Sequence[Placeable], request: InferenceRequest) -> Placeable:
        key = request.context.camera_id
        if not key:
            return self._fallback.select(candidates, request)

        pinned = self._assigned.get(key)
        if pinned is not None and pinned.is_ready and pinned in candidates:
            return pinned

        chosen = self._fallback.select(candidates, request)
        with self._lock:
            if len(self._assigned) >= self._max_sequences:
                # Bounded on purpose: an unbounded affinity table is a slow memory leak in
                # a system where camera ids can churn. Dropping the whole table is crude
                # but self-healing, and it happens once per 4096 new sequences.
                self._assigned.clear()
            self._assigned[key] = chosen
        return chosen

    def forget(self, key: str) -> None:
        """Release one sequence's pin — called when a camera disconnects."""
        with self._lock:
            self._assigned.pop(key, None)

    def describe(self) -> str:
        return f"pin each camera to one instance (fallback: {self._fallback.describe()})"
