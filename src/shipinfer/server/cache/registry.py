"""Registry of response caches."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.server.cache.base import ResponseCache

__all__ = ["RESPONSE_CACHES"]

RESPONSE_CACHES: Registry[ResponseCache] = Registry("response cache", ResponseCache)
