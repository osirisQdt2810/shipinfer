"""Registry of allocators.

``torch_device`` / ``torch_pinned`` are the defaults. The ``custom_*`` entries are the
readable re-implementations — same contract, raw driver calls, selectable for comparison.
"""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.runtime.memory.base import Allocator

__all__ = ["ALLOCATORS"]

ALLOCATORS: Registry[Allocator] = Registry("allocator", Allocator)
