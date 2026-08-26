"""Failures of a multi-process deployment."""

from __future__ import annotations

from collections.abc import Sequence

from shipinfer.core.errors.base import ShipInferError

__all__ = ["PeerLostError", "RingFullError", "RingProtocolError", "ShardExitedError"]


class ShardExitedError(ShipInferError):
    """A shard process exited, so its cameras are dark.

    Raised by the fleet supervisor after it has stopped the rest of the fleet: three shards
    up and one down is a deployment reporting healthy while a quarter of the cameras go
    unread, which is exactly the state a supervisor exists to refuse to sit in.
    """


class RingFullError(ShipInferError):
    """A shared ring had no free slot within the submit timeout.

    Carries depth and capacity (ADR-005): the dispatcher that receives it excludes the peer
    and re-selects, and an operator paged on it can tell a ring of 8 that is momentarily full
    from one that has been full for a minute. Nothing is dropped — the request is still with
    the caller.
    """

    def __init__(self, owner: str, ring: str, depth: int, capacity: int) -> None:
        super().__init__(
            f"ring {ring!r} to {owner!r} is full: {depth}/{capacity} slots claimed and not "
            f"yet released"
        )
        self.owner = owner
        self.ring = ring
        self.depth = depth
        self.capacity = capacity


class PeerLostError(ShipInferError):
    """A peer process stopped stamping its ring header, so its in-flight requests are gone.

    Carries every ``(camera_id, frame_id)`` that was waiting on the peer, because the tag
    must survive every path including this one (ADR-002): reassembly fails exactly those
    frames, names them, and the rest of the fleet keeps serving.
    """

    def __init__(self, owner: str, tags: Sequence[tuple[str, int]]) -> None:
        shown = ", ".join(f"({camera!r}, {frame})" for camera, frame in list(tags)[:8])
        more = "" if len(tags) <= 8 else f" and {len(tags) - 8} more"
        super().__init__(
            f"peer {owner!r} is lost (no heartbeat); {len(tags)} in-flight request(s) failed: "
            f"{shown}{more}"
        )
        self.owner = owner
        self.tags = tuple(tags)


class RingProtocolError(ShipInferError):
    """Two processes disagree about a ring's layout or version.

    Raised at ``open`` rather than at the first corrupt read: a header written by one
    version and read by another would otherwise become a payload of the wrong size, which is
    the kind of failure that looks like a model producing garbage.
    """
