"""The server: instances, models, the engine that owns them, and health.

Layering, top down: :class:`InferenceServer` owns :class:`Model` objects, a model owns
:class:`ModelInstance` objects, and an instance owns a backend, a queue and a worker
thread. Requests travel down; nothing travels back up except through a future.
"""

from shipinfer.server.cache import RESPONSE_CACHES, LruResponseCache, ResponseCache
from shipinfer.server.engine import InferenceServer
from shipinfer.server.ensemble import EnsembleModel
from shipinfer.server.health import HealthReport, HealthStatus, check_health
from shipinfer.server.instance import ModelInstance
from shipinfer.server.model import Model
from shipinfer.server.statistics import DurationStat, ModelStatistics

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
