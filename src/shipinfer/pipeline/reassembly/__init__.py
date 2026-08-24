"""Joining one frame's stage results back together — bounded, fair, and never silent.

Two files, two concerns. :mod:`collector` owns the buffer, the completion rule and the
timeout; :mod:`policy` owns the one decision that the previous generation got wrong, and it
is a registry so the inherited behaviour can be run beside the fix and lose.
"""

from shipinfer.pipeline.reassembly.collector import (
    COMPLETE,
    EVICTED,
    INCOMPLETE,
    SHUTDOWN,
    TIMEOUT,
    FrameCollector,
    FrameResult,
    PendingFrame,
)
from shipinfer.pipeline.reassembly.policy import (
    EVICTION_POLICIES,
    EvictionPolicy,
    GreediestCameraEviction,
    OldestFrameEviction,
    PendingIndex,
    PendingKey,
)

__all__ = [
    "COMPLETE",
    "EVICTED",
    "EVICTION_POLICIES",
    "INCOMPLETE",
    "SHUTDOWN",
    "TIMEOUT",
    "EvictionPolicy",
    "FrameCollector",
    "FrameResult",
    "GreediestCameraEviction",
    "OldestFrameEviction",
    "PendingFrame",
    "PendingIndex",
    "PendingKey",
]
