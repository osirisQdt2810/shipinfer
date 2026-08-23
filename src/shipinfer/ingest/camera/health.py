"""What an operator needs to know about one camera, and about the fleet.

A camera is not up or down. It is connecting, streaming, dropping frames because the
inference pool is saturated, or retrying every thirty seconds against a dead switch — and
the right operator response differs for each. The reference system had a single
``_online[cam_id]`` boolean, which is why "camera vắng người thỉnh thoảng bị miss" took a
week to diagnose.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = ["CameraHealth", "CameraState", "IngestSummary"]


class CameraState(str, enum.Enum):
    """The lifecycle of one camera actor."""

    #: Constructed, not started.
    IDLE = "idle"
    #: Trying to open the stream. Normal for a second or two at start-up.
    CONNECTING = "connecting"
    #: Frames are arriving.
    STREAMING = "streaming"
    #: Was streaming, has stopped delivering, is retrying. Recoverable.
    DEGRADED = "degraded"
    #: Has failed to connect ``failures_before_unhealthy`` times in a row. Still retrying
    #: at the capped delay — a camera down overnight must come back on its own — but this
    #: is the state a dashboard should be paging on.
    UNHEALTHY = "unhealthy"
    #: The source reported end of stream and will not produce more (a finite replay file).
    EXHAUSTED = "exhausted"
    #: Stopped on request.
    STOPPED = "stopped"

    @property
    def is_healthy(self) -> bool:
        return self in (CameraState.STREAMING, CameraState.CONNECTING)


@dataclass(frozen=True, slots=True)
class CameraHealth:
    """An immutable snapshot of one camera, safe to read from another thread.

    Frozen because it crosses a thread boundary: the actor builds it under its own lock and
    hands it out, so a caller cannot see half of an update.
    """

    camera_id: str
    state: CameraState
    #: Frames the source produced.
    frames_read: int
    #: Frames actually accepted by the queue. The gap to ``frames_read`` is backpressure.
    frames_published: int
    #: Frames the queue refused. Counted, never silent — that is the whole point of ADR-005.
    frames_dropped: int
    #: Reads that returned nothing (a timeout on a live stream, EOF on a file).
    empty_reads: int
    #: Successful (re)connections since start.
    connects: int
    #: Failed connection attempts since start.
    connect_failures: int
    #: Consecutive failures right now; resets on the first frame after a reconnect.
    consecutive_failures: int
    #: Measured over the last window, not since start: an average since start-up hides the
    #: camera that stopped ten minutes ago.
    fps: float
    last_frame_unix_ns: int
    last_error: str

    @property
    def is_healthy(self) -> bool:
        return self.state.is_healthy

    @property
    def drop_ratio(self) -> float:
        return self.frames_dropped / self.frames_read if self.frames_read else 0.0

    def as_dict(self) -> dict[str, object]:
        """Flat mapping for a health endpoint or a log line."""
        return {
            "camera_id": self.camera_id,
            "state": self.state.value,
            "frames_read": self.frames_read,
            "frames_published": self.frames_published,
            "frames_dropped": self.frames_dropped,
            "empty_reads": self.empty_reads,
            "connects": self.connects,
            "connect_failures": self.connect_failures,
            "consecutive_failures": self.consecutive_failures,
            "fps": round(self.fps, 2),
            "last_frame_unix_ns": self.last_frame_unix_ns,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True)
class IngestSummary:
    """The fleet in one line, for ``/v2/health`` and the stats log."""

    cameras: int
    streaming: int
    unhealthy: int
    total_fps: float
    frames_read: int
    frames_published: int
    frames_dropped: int

    @property
    def is_healthy(self) -> bool:
        """True when every configured camera is delivering.

        Deliberately strict: with 50 cameras, "most of them work" is the state the previous
        system lived in for months.
        """
        return self.cameras > 0 and self.streaming == self.cameras

    def as_dict(self) -> dict[str, object]:
        return {
            "cameras": self.cameras,
            "streaming": self.streaming,
            "unhealthy": self.unhealthy,
            "total_fps": round(self.total_fps, 2),
            "frames_read": self.frames_read,
            "frames_published": self.frames_published,
            "frames_dropped": self.frames_dropped,
        }
