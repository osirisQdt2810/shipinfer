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

#: The name arch.md §6 uses for this package's subject. `InferenceServer` keeps the name it
#: has because it is the KServe server — it is what `/v2/health/ready` reports on and what
#: every caller of `shipinfer.InferenceServer` already holds — and renaming a class that is
#: in a public signature is a separate, breaking change from moving a package. The alias
#: lets code inside the new layout read the way the document does without either name being
#: a second implementation.
Engine = InferenceServer

__all__ = [
    "RESPONSE_CACHES",
    "DurationStat",
    "Engine",
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
