"""One decoded frame, and the tag that must survive everything downstream."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shipinfer.core.request import RequestContext

__all__ = ["Frame"]


@dataclass(frozen=True, slots=True)
class Frame:
    """A decoded image plus the ``(camera_id, frame_id, timestamp)`` stamp from ADR-002.

    ``slots`` and ``frozen`` because one of these exists per decoded frame — 1000 a second
    across the fleet — and because nothing downstream may rewrite the tag. Batching,
    spillover to another GPU and out-of-order completion are all safe *precisely* because
    reassembly keys on this tag rather than on arrival order; a stage that "fixes up" a
    frame id breaks every one of those guarantees at once.

    The image is host memory in the decoder's native layout: HWC, BGR, ``uint8``. It is not
    normalised, letterboxed or moved to a device here. That is the runtime's job
    (``runtime/ops``), which is what keeps this package importable with no torch installed.
    """

    camera_id: str
    frame_id: int
    image: np.ndarray
    #: Monotonic nanoseconds at the moment of decode — for latency arithmetic.
    captured_ns: int
    #: Wall-clock nanoseconds at the moment of decode — for anything a human reads.
    captured_unix_ns: int

    @property
    def context(self) -> RequestContext:
        """The tag, in the form every request in the system carries."""
        return RequestContext(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            captured_ns=self.captured_ns,
            captured_unix_ns=self.captured_unix_ns,
        )

    @property
    def key(self) -> tuple[str, int]:
        """``(camera_id, frame_id)`` — the reassembly key."""
        return (self.camera_id, self.frame_id)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.image.nbytes)

    def as_batch(self) -> np.ndarray:
        """A batch-major ``(1, H, W, C)`` view.

        A view, not a copy: every request in this system is batch-major even at batch 1,
        and ``image[None]`` of a contiguous array stays contiguous, so the uniformity is
        free.
        """
        return self.image[None, ...]

    def __repr__(self) -> str:
        return (
            f"<Frame cam={self.camera_id} frame={self.frame_id} "
            f"{self.width}x{self.height} {self.image.dtype}>"
        )
