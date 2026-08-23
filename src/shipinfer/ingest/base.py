"""The video-source contract: open a stream, pull frames, close it.

A :class:`FrameSource` does one job. It does not own the frame id, does not know about
queues, models or GPUs, and does not decide when to retry — that is
:class:`~shipinfer.ingest.camera.actor.CameraActor`'s job. Keeping the split there is what
lets the whole ingest plane be tested with a source that returns a numpy array.

``open`` / ``read`` / ``close`` are **template methods**, not abstract. The subclass hooks
are ``_do_open`` / ``_do_read`` / ``_do_close``, and the wrappers own three invariants that
would otherwise have to be re-implemented (and eventually mis-implemented) per backend:

1. every frame leaving any source is stamped by the actor's counter, so the
   ``(camera_id, frame_id)`` tag cannot be forgotten by a new backend;
2. ``open`` is idempotent and cleans up after a partial failure;
3. ``close`` is idempotent, so a reconnect path and a shutdown path can both call it.
"""

from __future__ import annotations

import abc
import contextlib
from typing import ClassVar

import numpy as np

from shipinfer.core.errors import ConfigurationError, SourceOpenError
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest.frame.frame import Frame
from shipinfer.ingest.frame.tag import FrameCounter
from shipinfer.ingest.resolve import (
    resolve_hwaccel,
    resolve_open_timeout_s,
    resolve_read_timeout_s,
)

__all__ = ["FrameSource"]


class FrameSource(abc.ABC):
    """One camera's decode path: open it, pull frames, close it.

    A source is **not** expected to survive an error. The actor throws it away and builds a
    new one, which is why reconnect state (the backoff, the frame counter) lives outside.

    Args:
        config: the camera to open.
        counter: the actor's frame counter. A source given ``None`` makes its own, which is
            convenient for a CLI or a one-shot script but means frame ids restart if the
            source is rebuilt — the reason the actor always passes its own.
        settings: fleet-wide defaults for anything the camera leaves unset.
    """

    #: Registered name, set by the ``@SOURCES.register`` decorator's target class.
    name: ClassVar[str] = "abstract"
    #: Whether this backend can decode on a GPU at all. ``False`` makes
    #: :attr:`hwaccel` always resolve to False, so a log line says "software decode"
    #: instead of implying an NVDEC path that does not exist.
    supports_hwaccel: ClassVar[bool] = False

    def __init__(
        self,
        config: CameraConfig,
        counter: FrameCounter | None = None,
        *,
        settings: IngestSettings | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.counter = counter or FrameCounter(config.camera_id, config.first_frame_id)
        if self.counter.camera_id != config.camera_id:
            raise ConfigurationError(
                f"frame counter belongs to camera {self.counter.camera_id!r}, "
                f"not {config.camera_id!r}"
            )
        self._is_open = False
        self._height = 0
        self._width = 0
        self._fps = 0.0

    # -- resolved configuration --------------------------------------------------------

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    @property
    def hwaccel(self) -> bool:
        """Whether this source will try to decode on the GPU."""
        return self.supports_hwaccel and resolve_hwaccel(self.config, self.settings)

    @property
    def read_timeout_s(self) -> float:
        return resolve_read_timeout_s(self.settings)

    @property
    def open_timeout_s(self) -> float:
        return resolve_open_timeout_s(self.settings)

    # -- negotiated format -------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def height(self) -> int:
        """Negotiated frame height; 0 until :meth:`open` has run."""
        return self._height

    @property
    def width(self) -> int:
        """Negotiated frame width; 0 until :meth:`open` has run."""
        return self._width

    @property
    def fps(self) -> float:
        """Negotiated frame rate; 0.0 when the source does not advertise one."""
        return self._fps

    @property
    def is_exhausted(self) -> bool:
        """True when this source will never produce another frame.

        Only a finite source (a replay file with ``loop=False``) ever says True. It is what
        lets a bench or a test terminate on its own instead of being reconnected forever —
        a live camera at end-of-stream is a fault, a finished file is not.
        """
        return False

    def _set_format(self, height: int, width: int, fps: float) -> None:
        """Record what the stream actually negotiated. Called from ``_do_open``."""
        self._height = int(height)
        self._width = int(width)
        self._fps = float(fps)

    # -- lifecycle ---------------------------------------------------------------------

    def open(self) -> None:
        """Connect and negotiate. Idempotent.

        Raises:
            SourceUnavailableError: the decode runtime is not installed. Not retryable.
            SourceOpenError: the stream could not be opened or carries no video.
        """
        if self._is_open:
            return
        try:
            self._do_open()
        except BaseException:
            # A half-open source leaks a socket and a decoder thread. The subclass may not
            # be able to tell how far it got, so unwind unconditionally and best-effort —
            # suppressing, because the *original* failure is the one worth propagating.
            with contextlib.suppress(Exception):
                self._do_close()
            raise
        self._is_open = True

    def read(self) -> Frame | None:
        """One frame, or ``None`` if none arrived within the read timeout.

        ``None`` means "not yet" — a live stream that has gone quiet, or a paced replay
        between frames. It never means "broken": that is a
        :class:`~shipinfer.core.errors.FrameDecodeError`, so the actor can reconnect
        immediately instead of waiting out an empty-read budget.

        Raises:
            SourceOpenError: called before :meth:`open`.
            FrameDecodeError: the stream ended or the decoder failed.
        """
        if not self._is_open:
            raise SourceOpenError(self.camera_id, self.config.uri, "read() before open()")
        image = self._do_read()
        if image is None:
            return None
        return self.counter.stamp(image)

    def close(self) -> None:
        """Release everything. Idempotent, and safe after a failed :meth:`open`."""
        if not self._is_open:
            return
        self._is_open = False
        self._do_close()

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- subclass hooks ----------------------------------------------------------------

    @abc.abstractmethod
    def _do_open(self) -> None:
        """Connect, and call :meth:`_set_format` with what was negotiated."""

    @abc.abstractmethod
    def _do_read(self) -> np.ndarray | None:
        """One image as HWC BGR ``uint8``, or ``None`` if none is available yet."""

    @abc.abstractmethod
    def _do_close(self) -> None:
        """Release resources. Must tolerate being called after a partial ``_do_open``."""

    def __repr__(self) -> str:
        state = "open" if self._is_open else "closed"
        return (
            f"<{type(self).__name__} {self.camera_id} {state} "
            f"{self._width}x{self._height}@{self._fps:g}>"
        )
