"""The unit of work handed to a model."""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from shipinfer.core.request.context import RequestContext
from shipinfer.core.request.priority import Priority
from shipinfer.core.request.timings import Timings
from shipinfer.core.types import Device, Tensor

__all__ = ["InferenceRequest", "next_request_id"]

_REQUEST_COUNTER = itertools.count(1)


def next_request_id() -> int:
    """A process-unique request id.

    ``itertools.count`` is atomic under the GIL, which is both cheaper and less
    error-prone here than a lock around an integer.
    """
    return next(_REQUEST_COUNTER)


@dataclass(slots=True)
class InferenceRequest:
    """One inference to perform.

    ``inputs`` is batch-major even for a single item — shape ``(1, 3, 640, 640)``, not
    ``(3, 640, 640)``. That uniformity is what makes batch assembly a two-line hot path
    instead of a shape-normalising special case run 15 000 times a second.
    """

    model_name: str
    inputs: dict[str, Tensor]
    request_id: int = field(default_factory=next_request_id)
    model_version: int | None = None  # None == the repository's chosen version
    requested_outputs: tuple[str, ...] = ()  # empty == every declared output
    parameters: Mapping[str, Any] = field(default_factory=dict)
    context: RequestContext = field(default_factory=RequestContext)
    priority: Priority = Priority.NORMAL
    #: Absolute monotonic deadline in ns; 0 disables it. A request past its deadline is
    #: dropped *before* it reaches a GPU — spending compute on a late frame is pure waste.
    deadline_ns: int = 0
    #: Where the payload already lives. The dispatcher prefers this GPU when it can, which
    #: is the whole point of locality-aware placement (ADR-004).
    resident_device: Device | None = None
    timings: Timings = field(default_factory=Timings)

    @property
    def batch_size(self) -> int:
        for tensor in self.inputs.values():
            return tensor.batch_size
        return 0

    def is_expired(self, now_ns: int | None = None) -> bool:
        if not self.deadline_ns:
            return False
        return (now_ns if now_ns is not None else time.monotonic_ns()) > self.deadline_ns
