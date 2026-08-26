"""Failures of a multi-process deployment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from shipinfer.core.errors.base import ShipInferError
from shipinfer.core.errors.inference import QueueFullError

__all__ = [
    "PeerLostError",
    "RingClosedError",
    "RingFullError",
    "RingProtocolError",
    "ShardExitedError",
]


class ShardExitedError(ShipInferError):
    """A shard process exited, so its cameras are dark.

    Raised by the fleet supervisor after it has stopped the rest of the fleet: three shards
    up and one down is a deployment reporting healthy while a quarter of the cameras go
    unread, which is exactly the state a supervisor exists to refuse to sit in.
    """


class RingFullError(QueueFullError):
    """A shared ring had no free slot within the submit timeout.

    A :class:`QueueFullError`, so the dispatcher's spill loop treats a full ring exactly like a
    full local queue: the request is still with the caller and the next candidate is tried.
    Carries depth and capacity (ADR-005), so an operator paged on it can tell a ring of 8 that
    is momentarily full from one that has been full for a minute. Nothing is dropped.
    """

    def __init__(self, owner: str, ring: str, depth: int, capacity: int) -> None:
        super().__init__(f"ring {ring} to {owner}", depth, capacity)
        self.owner = owner
        self.ring = ring
        self.depth = depth
        self.capacity = capacity


class RingClosedError(QueueFullError):
    """The ring was closed by its owner — the peer is gone or leaving, not overloaded.

    A ``QueueFullError`` subclass so the dispatcher's existing recovery — exclude the candidate
    and re-select — still applies at the races (a proxy's ``is_ready`` goes false on close, but
    a request already past that check lands here). Distinct from :class:`RingFullError` because
    the operator's responses differ: a full ring wants back-pressure tuning, a closed one wants
    to know why the peer left.
    """

    _MESSAGES: ClassVar[dict[str, str]] = {
        "closed": "ring {ring} to {owner} is closed - its owner is gone or leaving",
        "absent": "ring {ring} does not exist - never created, or its owner unlinked it",
        "unborn": "ring {ring} is mid-birth - created, its header not yet written",
    }

    def __init__(self, owner: str, ring: str, reason: str = "closed") -> None:
        # Not QueueFullError.__init__: its message says "is full", and closed is exactly not
        # that. The base's attributes are still set so shape-generic handlers read zeros.
        # (Hand-set, so a field added to QueueFullError later must be mirrored here — the
        # message override is the whole reason for bypassing its __init__.)
        ShipInferError.__init__(self, self._MESSAGES[reason].format(ring=ring, owner=owner))
        self.queue_name = ring
        self.depth = 0
        self.capacity = 0
        self.owner = owner
        self.ring = ring
        #: Why: ``closed`` (the owner marked it), ``absent`` (no such name — retry during a
        #: connect window, terminal after it), ``unborn`` (created, header not yet written —
        #: always retryable). A connect loop keys on this, never on the message.
        self.reason = reason


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
