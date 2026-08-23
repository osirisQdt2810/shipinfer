"""The placement-policy contract.

A policy answers exactly one question — *which instance runs this request* — and is given
exactly enough information to answer it. Anything richer would tempt an implementation
into asking a backend something expensive on the critical path.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from shipinfer.core.request import InferenceRequest
from shipinfer.core.types import Device

__all__ = ["Placeable", "PlacementPolicy"]


@runtime_checkable
class Placeable(Protocol):
    """The slice of a model instance a policy may see.

    Both numbers are plain attribute reads. ``depth`` is read without a lock on purpose:
    the policy consults it thousands of times a second, and a slightly stale value changes
    which of two near-equal GPUs wins, nothing more.
    """

    @property
    def device(self) -> Device: ...

    @property
    def depth(self) -> int: ...

    @property
    def ewma_latency_us(self) -> float: ...

    @property
    def is_ready(self) -> bool: ...


class PlacementPolicy(abc.ABC):
    """Chooses one instance out of the ready candidates for a model.

    Subclasses live one-per-file in this package and register themselves with
    :data:`shipinfer.scheduling.policies.registry.POLICIES`.
    """

    #: Registered name. Set by the ``@POLICIES.register(...)`` decorator's argument; kept
    #: as a class attribute too so a policy can identify itself in logs and metrics.
    name: str = "abstract"

    @abc.abstractmethod
    def select(self, candidates: Sequence[Placeable], request: InferenceRequest) -> Placeable:
        """Pick an instance.

        Args:
            candidates: ready instances; never empty — the dispatcher filters and raises.
            request: the work to place. ``request.resident_device`` is the locality hint.

        Returns:
            One element of ``candidates``. Returning anything else is a programming error.
        """

    def describe(self) -> str:
        """One-line description for ``shipinfer policies``/logs."""
        return type(self).__name__

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"
