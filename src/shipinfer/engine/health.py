"""Liveness and readiness, as separate questions.

They are separate because the answers drive different actions. *Live* means the process is
functioning and should not be restarted. *Ready* means it can serve traffic right now.
A server deserialising a 2 GB engine is live but not ready; conflating the two makes an
orchestrator kill a process that was about to become useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shipinfer.engine.pool import InferenceServer

__all__ = ["HealthReport", "HealthStatus", "check_health"]


class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"  # serving, but not at full capacity
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class HealthReport:
    live: bool
    ready: bool
    status: HealthStatus
    detail: str
    models_ready: int
    models_total: int
    instances_ready: int
    instances_total: int

    def as_dict(self) -> dict[str, object]:
        return {
            "live": self.live,
            "ready": self.ready,
            "status": self.status.value,
            "detail": self.detail,
            "models": {"ready": self.models_ready, "total": self.models_total},
            "instances": {"ready": self.instances_ready, "total": self.instances_total},
        }


def check_health(server: InferenceServer) -> HealthReport:
    """Summarise the server's state.

    ``DEGRADED`` is a real state, not a rounding of ``OK``: losing three of sixteen GPU
    instances leaves a server that still answers every request and has lost a fifth of its
    capacity. Reporting that as healthy is how a fleet quietly runs at 80% for a week.
    """
    models = list(server)
    instances = [i for m in models for i in m.instances]
    ready_models = [m for m in models if m.is_ready]
    ready_instances = [i for i in instances if i.is_ready]

    if not server.is_started:
        return HealthReport(
            live=True,
            ready=False,
            status=HealthStatus.UNAVAILABLE,
            detail="server has not been started",
            models_ready=0,
            models_total=0,
            instances_ready=0,
            instances_total=0,
        )

    if not ready_models:
        status, detail, ready = HealthStatus.UNAVAILABLE, "no model has a ready instance", False
    elif len(ready_models) < len(models) or len(ready_instances) < len(instances):
        status = HealthStatus.DEGRADED
        detail = (
            f"{len(ready_instances)}/{len(instances)} instances ready across "
            f"{len(ready_models)}/{len(models)} models"
        )
        ready = True
    else:
        status, detail, ready = HealthStatus.OK, "all instances ready", True

    return HealthReport(
        live=True,
        ready=ready,
        status=status,
        detail=detail,
        models_ready=len(ready_models),
        models_total=len(models),
        instances_ready=len(ready_instances),
        instances_total=len(instances),
    )
