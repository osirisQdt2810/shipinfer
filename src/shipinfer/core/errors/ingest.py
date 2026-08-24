"""Errors raised by the ingest plane — everything between a camera and a frame.

Four distinct events, four types, because the operator response to each is different: fix
the install, fix the network, fix the camera, or wait. A single ``RuntimeError`` for all
four is why the previous generation logged "Can not read frame" and nothing else.
"""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError
from shipinfer.core.redact import redact, redact_in

__all__ = [
    "CameraUnavailableError",
    "FrameDecodeError",
    "IngestError",
    "SourceOpenError",
    "SourceUnavailableError",
]


class IngestError(ShipInferError):
    """Base for every ingest-plane failure, so a camera actor can catch one thing."""


class SourceUnavailableError(IngestError):
    """The decode runtime this source needs is not installed on this host.

    Distinct from :class:`SourceOpenError` on purpose: a missing PyGObject is fixed by an
    install and will never fix itself, whereas an unreachable camera might come back in a
    second. The actor must not burn a reconnect budget on the former.
    """

    def __init__(self, source: str, hint: str) -> None:
        super().__init__(f"video source {source!r} is unavailable: {hint}")
        self.source = source
        self.hint = hint


class SourceOpenError(IngestError):
    """The stream could not be opened, or negotiated no usable video.

    Carries the camera id and the URI because an ingest log with fifty cameras in it is
    useless without them.
    """

    def __init__(self, camera_id: str, uri: str, reason: str) -> None:
        # Redacted in the *message*, kept intact on the attribute. The message is what gets
        # logged and what becomes `CameraHealth.last_error` in the health API, so a fleet
        # password would otherwise be served to every reader of that payload on every retry.
        #
        # `reason` is redacted too, and that is not belt-and-braces: the decoders put the
        # URI inside it. PyAV renders `[Errno 111] Connection refused: '<uri>'` and
        # `gst_parse` reports `could not set property "location" ... to "<uri>"`, so
        # redacting only the argument named `uri` left the credential in the message by the
        # other door.
        super().__init__(
            f"camera {camera_id!r}: cannot open {redact(uri)!r}: {redact_in(str(reason))}"
        )
        self.camera_id = camera_id
        self.uri = uri
        self.reason = reason


class FrameDecodeError(IngestError):
    """A frame read failed in a way that ends the stream (EOS, decoder error).

    Raised rather than returning ``None``: "no frame yet" and "this stream is over" demand
    opposite responses — keep waiting, or reconnect — and returning an empty result for
    both is exactly the ambiguity that makes a stalled camera invisible.
    """

    def __init__(self, camera_id: str, reason: str) -> None:
        # Redacted for the same reason `SourceOpenError` redacts its own reason: the
        # decoders put the URI inside it. `av.FFmpegError.__str__` embeds the container
        # name, which is the full RTSP URI — and the actor logs this on every reconnect and
        # stores it as `CameraHealth.last_error`, which the health endpoint serves.
        super().__init__(f"camera {camera_id!r}: decode failed: {redact_in(str(reason))}")
        self.camera_id = camera_id
        self.reason = reason


class CameraUnavailableError(IngestError):
    """One or more cameras never produced a frame within the start-up window.

    Raised by :meth:`shipinfer.ingest.IngestManager.wait_ready`, so a deploy against a
    mistyped camera database fails at start-up instead of looking healthy and producing no
    detections.
    """

    def __init__(self, camera_ids: list[str], timeout_s: float) -> None:
        super().__init__(
            f"{len(camera_ids)} camera(s) produced no frame within {timeout_s:g}s: "
            f"{sorted(camera_ids)}"
        )
        self.camera_ids = sorted(camera_ids)
        self.timeout_s = timeout_s
