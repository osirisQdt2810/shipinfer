"""The default: no bound on concurrent executions, at no cost.

Registered as ``off`` so a config reads the way an operator expects, and so the *absence* of
rate limiting is a named choice in the registry rather than a `None` that every call site
has to test for. Two non-blocking method calls per batch is cheaper than the branch would be
readable.
"""

from __future__ import annotations

from typing import ClassVar

from shipinfer.scheduling.limits.base import RateLimiter
from shipinfer.scheduling.limits.registry import RATE_LIMITERS

__all__ = ["UnlimitedRateLimiter"]


@RATE_LIMITERS.register(
    "off", "none", "unlimited", description="No bound on concurrent executions (the default)"
)
class UnlimitedRateLimiter(RateLimiter):
    """Grants every request for a slot immediately."""

    name: ClassVar[str] = "off"

    def acquire(self, timeout_s: float | None = None) -> bool:
        self.granted += 1
        return True

    def release(self) -> None:
        """Nothing was taken, so nothing is given back."""

    @property
    def in_flight(self) -> int:
        return 0
