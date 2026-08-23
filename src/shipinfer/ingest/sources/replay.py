"""Replay a local video file or a directory of frames, paced at a target frame rate.

This is the backend that makes the ingest plane *testable*. Everything above it — the
per-camera actor, the frame tag, the reconnect policy, the fair queue's per-camera lanes,
the 50-camera skew that reproduces the inherited starvation bug — is exercised by a source
that needs no camera, no network and no GPU. A design where that is impossible is a design
whose ingest path is only ever tested in production.

The pacing is the part that is easy to get wrong; see
:class:`~shipinfer.ingest.timing.pacing.DeadlinePacer` for why ``sleep(1/fps)`` is not it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from shipinfer.core.errors import FrameDecodeError, SourceOpenError, SourceUnavailableError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.registry import SOURCES
from shipinfer.ingest.timing.pacing import DeadlinePacer

__all__ = ["FRAME_SUFFIXES", "ReplaySource"]

_LOG = get_logger("ingest.replay")

#: Image suffixes recognised when the URI is a directory of frames. A directory is the more
#: robust fixture of the two: it needs no container format and no codec, so it cannot fail
#: because a given OpenCV build lacks a writer.
FRAME_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: Used when neither the camera config nor the container says anything. 25 rather than 20 so
#: an accidental default is visible in a report instead of looking like the real fleet rate.
_FALLBACK_FPS = 25.0


def _load_cv2() -> Any:
    """Import OpenCV, or explain what is missing. Never at module import time."""
    try:
        import cv2
    except ImportError as exc:
        raise SourceUnavailableError(
            "replay",
            f"OpenCV is not importable ({exc}). "
            "Install it with `pip install 'shipinfer[video]'`",
        ) from exc
    return cv2


@SOURCES.register("replay", "file", "video")
class ReplaySource(FrameSource):
    """A file or frame directory, delivered at ``config.fps``.

    ``config.loop`` decides what end-of-input means. ``True`` (the default) rewinds, which
    is what a long-running stress test wants; ``False`` marks the source
    :attr:`is_exhausted`, which is how a test that wrote six frames asserts it received
    exactly six and then terminated on its own.
    """

    name: ClassVar[str] = "replay"
    supports_hwaccel: ClassVar[bool] = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cv2: Any = None
        self._capture: Any = None
        self._frame_paths: list[Path] = []
        self._index = 0
        self._exhausted = False
        self._pacer = DeadlinePacer(0.0)

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    @property
    def pacer(self) -> DeadlinePacer:
        """Exposed so a bench can report how often the consumer fell behind."""
        return self._pacer

    # -- lifecycle ---------------------------------------------------------------------

    def _do_open(self) -> None:
        self._cv2 = _load_cv2()
        path = Path(self.config.uri).expanduser()
        if not path.exists():
            raise SourceOpenError(self.camera_id, self.config.uri, "path does not exist")

        if path.is_dir():
            self._open_directory(path)
        else:
            self._open_video(path)

        if self.config.width is not None and self.config.height is not None:
            # Report what the caller will actually receive, not what is on disk: a
            # negotiated size that silently differs from the configured one is how a
            # letterbox ends up scaling twice.
            self._set_format(self.config.height, self.config.width, self.fps)

        self._exhausted = False
        self._index = 0
        self._pacer = DeadlinePacer(self.fps)
        self._pacer.reset()
        _LOG.info(
            "camera %s replaying %s: %dx%d @ %g fps, loop=%s",
            self.camera_id,
            path,
            self.width,
            self.height,
            self.fps,
            self.config.loop,
            extra=log_context(camera_id=self.camera_id),
        )

    def _open_directory(self, path: Path) -> None:
        self._frame_paths = sorted(
            p for p in path.iterdir() if p.suffix.lower() in FRAME_SUFFIXES
        )
        if not self._frame_paths:
            raise SourceOpenError(
                self.camera_id,
                self.config.uri,
                f"directory holds no images with suffixes {list(FRAME_SUFFIXES)}",
            )
        probe = self._cv2.imread(str(self._frame_paths[0]), self._cv2.IMREAD_COLOR)
        if probe is None:
            raise SourceOpenError(
                self.camera_id, self.config.uri, f"cannot decode {self._frame_paths[0].name}"
            )
        self._set_format(probe.shape[0], probe.shape[1], self.config.fps or _FALLBACK_FPS)

    def _open_video(self, path: Path) -> None:
        capture = self._cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise SourceOpenError(
                self.camera_id, self.config.uri, "OpenCV could not open the container"
            )
        self._capture = capture
        container_fps = float(capture.get(self._cv2.CAP_PROP_FPS) or 0.0)
        self._set_format(
            int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
            self.config.fps or container_fps or _FALLBACK_FPS,
        )

    # -- reading -----------------------------------------------------------------------

    def _do_read(self) -> np.ndarray | None:
        if self._exhausted:
            return None
        self._pacer.wait()
        image = self._next_image()
        if image is None:
            if self.config.loop:
                self._rewind()
                image = self._next_image()
            if image is None:
                self._exhausted = True
                _LOG.info(
                    "camera %s: replay source exhausted after %d frame(s)",
                    self.camera_id,
                    self.counter.stamped,
                    extra=log_context(camera_id=self.camera_id),
                )
                return None
        return self._resize(image)

    def _next_image(self) -> np.ndarray | None:
        if self._capture is not None:
            ok, image = self._capture.read()
            return image if ok and image is not None else None
        if self._index >= len(self._frame_paths):
            return None
        path = self._frame_paths[self._index]
        self._index += 1
        image = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
        if image is None:
            raise FrameDecodeError(self.camera_id, f"cannot decode {path.name}")
        return image

    def _rewind(self) -> None:
        self._index = 0
        if self._capture is not None:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Honour ``config.width``/``height``, which a real decoder would do in-pipeline."""
        if self.config.width is None or self.config.height is None:
            return image
        if image.shape[1] == self.config.width and image.shape[0] == self.config.height:
            return image
        return self._cv2.resize(
            image,
            (self.config.width, self.config.height),
            interpolation=self._cv2.INTER_LINEAR,
        )

    def _do_close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._frame_paths = []
        self._index = 0
