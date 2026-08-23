"""Registry of low-level CUDA providers."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.runtime.providers.base import CudaProvider

__all__ = ["PROVIDERS"]

PROVIDERS: Registry[CudaProvider] = Registry("cuda provider", CudaProvider)
