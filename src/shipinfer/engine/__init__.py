"""The engine: the model pool — instances, models, ensembles, the cache and health.

Layering, top down: :class:`InferenceServer` owns :class:`Model` objects, a model owns
:class:`ModelInstance` objects, and an instance owns a backend, a queue and a worker
thread. Requests travel down; nothing travels back up except through a future.

This is arch.md §6 — the "mini-Triton" that elements of kind ``pool`` are thin clients of,
and the one thing behind the KServe side-door. It knows nothing about chains, cameras or
frames: it is handed an :class:`~shipinfer.core.request.InferenceRequest` and returns a
future. That ignorance is why it could be lifted out of ``server/`` unchanged.
"""

from shipinfer.engine.cache import RESPONSE_CACHES, LruResponseCache, ResponseCache
from shipinfer.engine.ensemble import EnsembleModel
from shipinfer.engine.health import HealthReport, HealthStatus, check_health
from shipinfer.engine.instance import ModelInstance
from shipinfer.engine.model import Model
from shipinfer.engine.pool import InferenceServer
from shipinfer.engine.statistics import DurationStat, ModelStatistics

# `InferenceServer` keeps its name: it is the KServe server — what `/v2/health/ready`
# reports on and what every caller of `shipinfer.InferenceServer` holds — and renaming a
# class in a public signature is a separate, breaking change from moving a package. No
# `Engine` alias either: `csrc/shipinfer/backends/engine_api.h` already has `class Engine`
# for the *backend contract*, and ADR-014 wants the two planes to mean the same thing by
# the same name.

__all__ = [
    "RESPONSE_CACHES",
    "DurationStat",
    "EnsembleModel",
    "HealthReport",
    "HealthStatus",
    "InferenceServer",
    "LruResponseCache",
    "Model",
    "ModelInstance",
    "ModelStatistics",
    "ResponseCache",
    "check_health",
]
