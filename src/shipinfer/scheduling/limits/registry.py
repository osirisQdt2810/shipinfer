"""Registry of rate limiters."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.scheduling.limits.base import RateLimiter

__all__ = ["RATE_LIMITERS", "build_rate_limiter"]

RATE_LIMITERS: Registry[RateLimiter] = Registry("rate limiter", RateLimiter)


def build_rate_limiter(name: str, max_concurrent_executions: int = 0) -> RateLimiter:
    """Instantiate a limiter by registered name.

    The whole family takes the same one argument (see :class:`RateLimiter`), so this is a
    plain call rather than a branch on the name — which is what keeps adding a limiter to a
    new file plus a decorator.
    """
    return RATE_LIMITERS.create(name, max_concurrent_executions)
