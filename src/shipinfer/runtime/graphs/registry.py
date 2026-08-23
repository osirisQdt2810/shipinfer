"""Registry of graph caches."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.runtime.graphs.base import GraphCache

__all__ = ["GRAPH_CACHES"]

GRAPH_CACHES: Registry[GraphCache] = Registry("graph cache", GraphCache)
