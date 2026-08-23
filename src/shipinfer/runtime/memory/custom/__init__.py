"""Hand-written allocators — the readable explanation of the default path.

None of these is used unless a config asks for it. They exist so that
"torch's caching allocator handles it" is a statement you can go and *read*, and so the
parity tests can prove the default and the reference agree.
"""

from shipinfer.runtime.memory.custom.caching import CustomCachingAllocator
from shipinfer.runtime.memory.custom.device import CustomDeviceAllocator
from shipinfer.runtime.memory.custom.pinned import CustomPinnedAllocator

__all__ = ["CustomCachingAllocator", "CustomDeviceAllocator", "CustomPinnedAllocator"]
