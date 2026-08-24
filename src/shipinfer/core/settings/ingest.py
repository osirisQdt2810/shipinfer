"""The ingest plane: which cameras exist, how they are decoded, how a drop-out is handled.

Two objects, one file, because they are one concern: :class:`CameraConfig` is the record
for a single camera and :class:`IngestSettings` is the fleet-wide default every camera
inherits. Keeping them together is what makes ``None`` meaningful on a camera field — it
reads as "inherit", and the resolution order (camera -> settings -> environment) is stated
once, here.

Nothing in this module knows how to decode anything: the vocabulary lives in ``core`` so
the ingest plane can be configured, validated and unit-tested on a host with no GStreamer,
no PyAV and no camera (ADR-001).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shipinfer.core.request.priority import Priority

__all__ = ["CameraConfig", "Codec", "IngestSettings", "RtspTransport"]

#: ``auto`` builds a ``decodebin`` pipeline that negotiates the codec at connect time. It
#: is the safe choice for a mixed fleet and the slightly slower one, because the decoder is
#: then chosen by plugin rank rather than by us.
Codec = Literal["h264", "h265", "auto"]

RtspTransport = Literal["tcp", "udp", "auto"]


class CameraConfig(BaseModel):
    """One camera: where it is, how to decode it, and what to tag its frames with.

    Field names are ours, not the reference system's: the previous generation's
    ``cameradb.json`` used ``cameraID`` / ``videoSource`` / ``codecType``, and
    :func:`shipinfer.ingest.load_camera_db` translates that shape into this one so an
    existing fleet database can be pointed at this server unchanged.

    Every ``None`` means **inherit from** :class:`IngestSettings`. That is deliberate: 50
    cameras that all want TCP transport should say so once.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Stable identity. Half of the ``(camera_id, frame_id)`` tag that rides every request
    #: through the whole system, so it must not change when a camera is re-added (ADR-002).
    camera_id: str
    #: ``rtsp://…`` for a camera, a file path or a directory of frames for ``replay``.
    uri: str
    #: A name registered in :data:`shipinfer.ingest.SOURCES`; ``None`` inherits.
    source: str | None = None
    codec: Codec = "h264"
    #: Scale in the decoder rather than in Python. ``None`` keeps the native resolution,
    #: which is what the fused letterbox kernel wants anyway.
    width: int | None = Field(default=None, ge=16)
    height: int | None = Field(default=None, ge=16)
    #: Target frame rate. 0 means "whatever the source delivers"; for ``replay`` it is the
    #: pacing target, which is how one video file simulates a 20 fps camera.
    fps: float = Field(default=0.0, ge=0.0)
    latency_ms: int | None = Field(default=None, ge=0)
    transport: RtspTransport | None = None
    hwaccel: bool | None = None
    #: A camera watching a restricted area can outrank the rest of the fleet in the
    #: scheduler's priority lanes. This is the customisation a generic server has no word
    #: for (ADR-005). Read by the `FrameSink` adapter, not by `shipinfer.ingest`: a frame is
    #: data, a priority is policy, and the decode path should not know about lanes.
    priority: Priority = Priority.NORMAL
    #: Override the model these frames are submitted to; ``None`` inherits. Also read by the
    #: sink adapter rather than by this package.
    model: str | None = None
    #: Where this camera's ``frame_id`` sequence starts. Non-zero when a restarted process
    #: must not reuse tags a downstream tracker has already seen.
    first_frame_id: int = Field(default=0, ge=0)
    #: ``replay`` only: restart the file at EOF. True keeps a stress test running; False
    #: makes a finite fixture terminate, which is what a test wants.
    loop: bool = True
    #: Set False to keep a camera in the database but out of the fleet.
    enabled: bool = True
    #: Escape hatch for backend-specific options (extra AVOptions, extra GStreamer
    #: properties). Anything used by more than one deployment belongs in a real field.
    options: dict[str, str] = Field(default_factory=dict)

    @field_validator("camera_id")
    @classmethod
    def _camera_id_is_usable(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("camera_id must not be empty")
        # It becomes a metric label, a log field and a fairness key; whitespace in any of
        # those is a debugging session nobody enjoys.
        if stripped != value or any(c.isspace() for c in stripped):
            raise ValueError(f"camera_id {value!r} must not contain whitespace")
        return stripped

    @field_validator("uri")
    @classmethod
    def _uri_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("uri must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _scale_needs_both_dimensions(self) -> CameraConfig:
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be set together, or neither")
        return self


class IngestSettings(BaseModel):
    """Fleet-wide ingest defaults, plus the camera list itself.

    The reconnect numbers are the part worth reading. The reference implementation counted
    to ten and then retried every twenty seconds forever, from a separate monitor thread;
    fifty cameras behind one switch therefore reconnected in lockstep every twenty seconds
    for as long as the switch was down. Here the backoff is exponential, jittered and
    capped, and it lives in the camera's own thread.
    """

    model_config = ConfigDict(extra="forbid")

    #: ``None`` defers to ``$SHIPINFER_INGEST_BACKEND`` (see :mod:`shipinfer.envs`), which
    #: is what makes the backend switchable without editing a config file.
    backend: str | None = None
    hwaccel: bool | None = None
    transport: RtspTransport | None = None
    latency_ms: int | None = Field(default=None, ge=0)

    # The three fields below are ingest *configuration* but not ingest *behaviour*: they
    # describe how a frame becomes an inference request, which is dispatch policy and is
    # therefore read by the `FrameSink` adapter in `pipeline`, never by `shipinfer.ingest`
    # itself (see `shipinfer.ingest.sink`). They live here because an operator configuring
    # the video path expects to find them in one place, and because they are per-fleet
    # rather than per-model.

    #: Where decoded frames are submitted. The first stage of the perception DAG.
    target_model: str = "ship_detector"
    #: The input tensor name that model declares for a frame.
    input_name: str = "images"
    #: Drop a frame this many milliseconds after capture instead of inferring on it. 0
    #: disables it; a real-time deployment should set it, because spending a GPU on a frame
    #: that is already too late to act on is pure waste.
    frame_deadline_ms: int = Field(default=0, ge=0)

    #: How long one frame read may block before it counts as an empty read. Also the worst
    #: case for an actor to notice a stop request. ``None`` defers to
    #: ``$SHIPINFER_INGEST_READ_TIMEOUT_S``.
    read_timeout_ms: int | None = Field(default=None, ge=1)
    #: ``None`` defers to ``$SHIPINFER_INGEST_OPEN_TIMEOUT_S``.
    open_timeout_ms: int | None = Field(default=None, ge=1)
    #: Consecutive empty reads before the source is torn down and reopened. An RTSP source
    #: that has stopped delivering never says so; it simply times out forever.
    empty_reads_before_reconnect: int = Field(default=5, ge=1)
    #: Pause between empty reads, so a source that returns immediately (a file at EOF, a
    #: fake in a test) cannot spin a core at 100%.
    empty_read_sleep_ms: int = Field(default=5, ge=0)

    reconnect_initial_ms: int = Field(default=500, ge=1)
    reconnect_max_ms: int = Field(default=30_000, ge=1)
    reconnect_factor: float = Field(default=2.0, gt=1.0)
    #: Fraction of each delay to remove at random, in ``[0, 1)``. Non-zero on purpose: 50
    #: cameras that fail together must not retry together.
    reconnect_jitter: float = Field(default=0.2, ge=0.0, lt=1.0)
    #: Consecutive failed connection attempts before the camera is reported UNHEALTHY. It
    #: keeps retrying at the capped delay — a camera down for an hour must still come back
    #: on its own — but health, and therefore the operator's dashboard, says so.
    failures_before_unhealthy: int = Field(default=3, ge=1)

    #: The fleet. Loaded from a file with ``camera_db``, or given inline.
    cameras: list[CameraConfig] = Field(default_factory=list)
    #: Optional path to a camera database in the reference system's ``cameradb.json``
    #: shape, merged with :attr:`cameras` at start-up.
    camera_db: Path | None = None

    @model_validator(mode="after")
    def _camera_ids_are_unique(self) -> IngestSettings:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for camera in self.cameras:
            if camera.camera_id in seen:
                duplicates.add(camera.camera_id)
            seen.add(camera.camera_id)
        if duplicates:
            raise ValueError(f"duplicate camera_id(s) in ingest.cameras: {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def _reconnect_cap_is_above_the_floor(self) -> IngestSettings:
        if self.reconnect_max_ms < self.reconnect_initial_ms:
            raise ValueError(
                f"reconnect_max_ms ({self.reconnect_max_ms}) must be >= "
                f"reconnect_initial_ms ({self.reconnect_initial_ms})"
            )
        return self
