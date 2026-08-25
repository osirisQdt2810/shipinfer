"""Rate limiters — one class per file, selected through :data:`RATE_LIMITERS`.

Importing this package registers the built-ins. ``off`` is the default and costs nothing;
``concurrency`` is the one that bounds a burst. See :mod:`shipinfer.scheduling.limits.base`
for why this bound is not the same as the queue depth bound.
"""

from shipinfer.scheduling.limits.base import RateLimiter
from shipinfer.scheduling.limits.concurrency import ConcurrencyRateLimiter
from shipinfer.scheduling.limits.registry import RATE_LIMITERS, build_rate_limiter
from shipinfer.scheduling.limits.unlimited import UnlimitedRateLimiter

__all__ = [
    "RATE_LIMITERS",
    "ConcurrencyRateLimiter",
    "RateLimiter",
    "UnlimitedRateLimiter",
    "build_rate_limiter",
]
