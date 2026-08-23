"""Registry of request queues."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.scheduling.queues.base import RequestQueue

__all__ = ["QUEUES"]

QUEUES: Registry[RequestQueue] = Registry("request queue", RequestQueue)
