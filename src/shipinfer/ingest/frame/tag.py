"""The per-camera ``frame_id`` counter.

Small enough to look pointless and important enough to be its own file: **the counter
belongs to the camera actor, not to the source**. A source is destroyed and rebuilt on
every reconnect; if it owned the counter, a camera that dropped out would restart at zero
and hand a downstream tracker a second frame 0 for the same camera — the same
``(camera_id, frame_id)`` key twice, which is the one thing ADR-002 relies on never
happening.
"""

from __future__ import annotations

import time

import numpy as np

from shipinfer.ingest.frame.frame import Frame

__all__ = ["FrameCounter"]


class FrameCounter:
    """Stamps decoded images with a monotonic, per-camera frame id.

    Not thread-safe, and deliberately so: exactly one thread — the camera's own actor —
    ever stamps a given camera's frames, so a lock here would be pure cost. One actor per
    camera for the actor's whole life is what makes that true (ADR-002).

    Args:
        camera_id: the camera this counter belongs to.
        start_at: the first frame id to hand out. Non-zero when a restarted process must
            not reuse tags a downstream tracker has already seen.
    """

    __slots__ = ("_next", "camera_id", "stamped")

    def __init__(self, camera_id: str, start_at: int = 0) -> None:
        if start_at < 0:
            raise ValueError("start_at must be >= 0")
        self.camera_id = camera_id
        self._next = start_at
        #: How many frames this counter has stamped, across every reconnect.
        self.stamped = 0

    @property
    def next_frame_id(self) -> int:
        """The id the next :meth:`stamp` will use."""
        return self._next

    def stamp(self, image: np.ndarray) -> Frame:
        """Wrap ``image`` in a :class:`~shipinfer.ingest.frame.frame.Frame` and advance.

        Both clocks are read here, at the moment of decode, because this is the last place
        that knows when the frame actually existed. A timestamp taken later measures the
        queue, not the camera.
        """
        frame = Frame(
            camera_id=self.camera_id,
            frame_id=self._next,
            image=image,
            captured_ns=time.monotonic_ns(),
            captured_unix_ns=time.time_ns(),
        )
        self._next += 1
        self.stamped += 1
        return frame

    def __repr__(self) -> str:
        return f"<FrameCounter {self.camera_id} next={self._next} stamped={self.stamped}>"
