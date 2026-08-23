"""Registry of batchers."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.scheduling.batching.base import Batcher

__all__ = ["BATCHERS"]

BATCHERS: Registry[Batcher] = Registry("batcher", Batcher)
