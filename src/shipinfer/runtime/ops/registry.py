"""Registry of image-operation implementations."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.runtime.ops.base import ImageOps

__all__ = ["IMAGE_OPS"]

IMAGE_OPS: Registry[ImageOps] = Registry("image ops", ImageOps)
