"""The result of one request."""

from __future__ import annotations

from dataclasses import dataclass, field

from shipinfer.core.request.context import RequestContext
from shipinfer.core.request.timings import Timings
from shipinfer.core.types import Device, Tensor

__all__ = ["InferenceResponse"]


@dataclass(slots=True)
class InferenceResponse:
    """Outputs plus the same context tag the request arrived with."""

    request_id: int
    model_name: str
    model_version: int
    outputs: dict[str, Tensor]
    context: RequestContext = field(default_factory=RequestContext)
    timings: Timings = field(default_factory=Timings)
    #: Which device actually executed it — the ground truth a load-balance dashboard needs,
    #: and the only way to prove spillover happened rather than merely being configured.
    executed_on: Device = field(default_factory=Device.cpu)
