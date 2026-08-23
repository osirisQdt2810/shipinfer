"""The dispatcher: pick an instance, enqueue, and spill when the first choice is full.

It is intentionally small. All the intelligence is in the policy object it holds; all the
state is in the queues it enqueues into. What it owns is the *retry* behaviour, which is
the part that turns a policy decision into a delivery guarantee:

    chosen = policy.select(ready)
    try: chosen.enqueue(item)
    except QueueFull: try the next-best instance, once per remaining instance, then reject

Without that loop, a policy that guesses slightly wrong under a burst turns a transient
full queue into a dropped frame.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from shipinfer.core.errors import QueueFullError, ServerStateError
from shipinfer.core.logging import get_logger
from shipinfer.core.request import InferenceRequest
from shipinfer.scheduling.policies import Placeable, PlacementPolicy
from shipinfer.scheduling.work import WorkItem

__all__ = ["DispatchResult", "Dispatcher"]

_LOG = get_logger("scheduling.dispatcher")


class DispatchResult:
    """Where a request landed, and how much shopping it took to get there."""

    __slots__ = ("attempts", "instance", "spilled")

    def __init__(self, instance: Placeable, attempts: int, spilled: bool) -> None:
        self.instance = instance
        self.attempts = attempts
        self.spilled = spilled

    def __repr__(self) -> str:
        return (
            f"<DispatchResult device={self.instance.device} "
            f"attempts={self.attempts} spilled={self.spilled}>"
        )


class Dispatcher:
    """Routes requests for **one model** across that model's instances."""

    __slots__ = ("_instances", "_on_spill", "_policy", "model_name")

    def __init__(
        self,
        model_name: str,
        instances: Sequence[Placeable],
        policy: PlacementPolicy,
        *,
        on_spill: Callable[[Placeable, Placeable], None] | None = None,
    ) -> None:
        if not instances:
            raise ServerStateError(f"model {model_name!r} has no instances to dispatch to")
        self.model_name = model_name
        self._instances = tuple(instances)
        self._policy = policy
        self._on_spill = on_spill

    @property
    def policy(self) -> PlacementPolicy:
        return self._policy

    @property
    def instances(self) -> tuple[Placeable, ...]:
        return self._instances

    def ready_instances(self) -> list[Placeable]:
        return [inst for inst in self._instances if inst.is_ready]

    def select(self, request: InferenceRequest) -> Placeable:
        """Ask the policy for one instance. Does not enqueue.

        Raises:
            ServerStateError: when every instance is unavailable (loading, or failed).
        """
        ready = self.ready_instances()
        if not ready:
            raise ServerStateError(
                f"model {self.model_name!r} has no ready instance "
                f"({len(self._instances)} configured)"
            )
        return self._policy.select(ready, request)

    def dispatch(
        self, item: WorkItem, enqueue: Callable[[Placeable, WorkItem], None]
    ) -> DispatchResult:
        """Place ``item`` on an instance, spilling to the next-shortest queue if needed.

        Args:
            enqueue: how to hand the item to a chosen instance. Injected rather than
                called as ``instance.enqueue`` so this class keeps depending only on the
                narrow :class:`Placeable` protocol, and so tests can drive it with fakes.

        Raises:
            QueueFullError: only after every ready instance has refused, which is the
                honest signal that the *pool* is saturated rather than one GPU.
            ServerStateError: when nothing is ready.
        """
        ready = self.ready_instances()
        if not ready:
            raise ServerStateError(f"model {self.model_name!r} has no ready instance")

        first = self._policy.select(ready, item.request)
        try:
            enqueue(first, item)
            return DispatchResult(first, attempts=1, spilled=False)
        except QueueFullError as initial_error:
            last_error: QueueFullError = initial_error

        # Spill: try the remaining instances shortest-queue-first. Sorting is acceptable
        # here because this path only runs when a queue is already full — i.e. rarely, and
        # never in the steady state.
        remaining = sorted((i for i in ready if i is not first), key=lambda i: i.depth)
        for attempt, candidate in enumerate(remaining, start=2):
            try:
                enqueue(candidate, item)
            except QueueFullError as exc:
                last_error = exc
                continue
            if self._on_spill is not None:
                self._on_spill(first, candidate)
            _LOG.debug(
                "spilled %s from %s to %s", self.model_name, first.device, candidate.device
            )
            return DispatchResult(candidate, attempts=attempt, spilled=True)

        raise last_error
