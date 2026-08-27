"""Compatibility shim: the model pool now lives in :mod:`shipinfer.engine` (arch.md §9).

What is left under ``server/`` is the KServe surface (``api/``), the shard launcher and the
topology-as-placement classes; those move in their own PRs and this package then disappears.
Until it does, every name this module used to export resolves to the *same object* it does
in :mod:`shipinfer.engine` — ``tests/test_architecture.py`` asserts the identity, because a
shim that re-exported a second copy of ``ResponseCache`` would make ``isinstance`` fail
across the seam and that is a far worse failure than an import error.

**Silent on purpose.** ``pyproject.toml`` turns a ``DeprecationWarning`` from ``shipinfer.*``
into an error in the offline tier, so warning here would fail the suite rather than nudge
anybody. The nudge is this docstring and the fact that the package is scheduled for deletion.

Deliberately absent: a submodule named ``engine``. ``from shipinfer.server import engine``
used to reach the pool module; it now has to be spelled ``from shipinfer.engine import
pool``, because re-exporting the *package* ``shipinfer.engine`` under the old attribute name
would make ``shipinfer.server.engine.InferenceServer`` keep working by accident and hide
every remaining caller from the grep that has to find them.
"""

from shipinfer.engine import (
    RESPONSE_CACHES,
    DurationStat,
    EnsembleModel,
    HealthReport,
    HealthStatus,
    InferenceServer,
    LruResponseCache,
    Model,
    ModelInstance,
    ModelStatistics,
    ResponseCache,
    check_health,
)

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
