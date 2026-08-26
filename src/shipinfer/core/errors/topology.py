"""Failures of a multi-process deployment."""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError

__all__ = ["ShardExitedError"]


class ShardExitedError(ShipInferError):
    """A shard process exited, so its cameras are dark.

    Raised by the fleet supervisor after it has stopped the rest of the fleet: three shards
    up and one down is a deployment reporting healthy while a quarter of the cameras go
    unread, which is exactly the state a supervisor exists to refuse to sit in.
    """
