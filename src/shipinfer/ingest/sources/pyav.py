"""RTSP ingest through PyAV (FFmpeg) — the portable fallback.

Why keep a second backend at all? Because GStreamer's dependency chain is an operating
system decision: PyGObject, the typelibs, and the right plugin packages for the codec. PyAV
is a wheel with FFmpeg statically linked, so ``pip install av`` gets a machine ingesting
without touching the distribution's package manager. On a box that has both, GStreamer wins
— it has the better NVMM path and the pipeline is inspectable — but "the default backend
does not install here" must never mean "no video".

Two things are set explicitly and both matter:

* ``rtsp_transport=tcp``. UDP silently loses packets under load and the resulting decode
  artefacts look exactly like a model regression.
* a socket timeout. Without one, ``av.open`` against an unreachable camera blocks its actor
  thread indefinitely, and the camera reports healthy because nothing ever failed.
"""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import urlsplit

import numpy as np

from shipinfer.core.errors import FrameDecodeError, SourceOpenError, SourceUnavailableError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.settings.ingest import CameraConfig
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.resolve import resolve_latency_ms, resolve_transport
from shipinfer.ingest.registry import SOURCES

__all__ = ["PyAvSource", "build_open_options", "is_network_uri"]

_LOG = get_logger("ingest.pyav")

_NETWORK_SCHEMES = frozenset({"rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "udp", "rtp"})


def is_network_uri(uri: str) -> bool:
    """Whether ``uri`` names a stream rather than a local file.

    Used to decide whether the RTSP/network AVOptions apply at all: passing
    ``rtsp_transport`` to a local file is harmless but tells a reader of the log something
    untrue about what is happening.
    """
    return urlsplit(uri).scheme.lower() in _NETWORK_SCHEMES


def build_open_options(
    config: CameraConfig,
    *,
    transport: str = "tcp",
    latency_ms: int = 200,
    read_timeout_s: float = 5.0,
) -> dict[str, str]:
    """The AVOption dictionary handed to ``av.open``.

    A pure function so the offline tier can assert the options without PyAV installed —
    which matters more than it sounds, because a mistyped AVOption is *silently ignored* by
    FFmpeg. ``rtsp_transport=tpc`` gives you UDP and no error.

    Args:
        config: the camera; its ``options`` mapping is merged last and therefore wins.
        transport: ``tcp``, ``udp``, or ``auto`` to leave FFmpeg's default alone.
        latency_ms: mapped to ``max_delay`` (microseconds), FFmpeg's reordering budget.
        read_timeout_s: mapped to ``stimeout`` (microseconds), the socket read timeout.
    """
    options: dict[str, str] = {}
    if is_network_uri(config.uri):
        if transport in ("tcp", "udp"):
            options["rtsp_transport"] = transport
        # `stimeout` is the socket I/O timeout in microseconds. Named `timeout` in recent
        # FFmpeg, with `stimeout` kept as an alias, so this spelling works on both; PyAV's
        # own `timeout=` argument covers the connect phase.
        options["stimeout"] = str(int(read_timeout_s * 1_000_000))
        options["max_delay"] = str(int(latency_ms * 1_000))
        # Live perception wants the newest frame, not a complete one: no demuxer buffering,
        # and a decoder that does not wait to reorder frames it will never need.
        options["fflags"] = "nobuffer"
        options["flags"] = "low_delay"
    options.update(config.options)
    return options


def _load_av() -> Any:
    """Import PyAV, or explain what is missing. Never at module import time."""
    try:
        import av
    except ImportError as exc:
        raise SourceUnavailableError(
            "pyav",
            f"PyAV is not importable ({exc}). Install it with `pip install 'shipinfer[video]'`",
        ) from exc
    return av


@SOURCES.register("pyav", "av", "ffmpeg")
class PyAvSource(FrameSource):
    """One camera, demuxed and decoded by FFmpeg through PyAV.

    Hardware decode is attempted when available and *falls back* to software with a warning
    rather than failing: PyAV's ``hwaccel`` argument only exists from version 13, and a CUDA
    device that is busy or absent must not take a camera offline.
    """

    name: ClassVar[str] = "pyav"
    supports_hwaccel: ClassVar[bool] = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._av: Any = None
        self._container: Any = None
        self._frames: Any = None
        self._using_hwaccel = False

    @property
    def using_hwaccel(self) -> bool:
        """Whether the open stream actually got a hardware decoder."""
        return self._using_hwaccel

    # -- lifecycle ---------------------------------------------------------------------

    def _do_open(self) -> None:
        av = self._av = _load_av()
        options = build_open_options(
            self.config,
            transport=resolve_transport(self.config, self.settings),
            latency_ms=resolve_latency_ms(self.config, self.settings),
            read_timeout_s=self.read_timeout_s,
        )
        timeout = (self.open_timeout_s, self.read_timeout_s)

        container = None
        if self.hwaccel:
            container = self._try_open(av, options, timeout, hwaccel=True)
            self._using_hwaccel = container is not None
        if container is None:
            container = self._try_open(av, options, timeout, hwaccel=False)
        if container is None:  # pragma: no cover - _try_open raises on the software path
            raise SourceOpenError(self.camera_id, self.config.uri, "no container was opened")
        self._container = container

        streams = container.streams.video
        if not streams:
            raise SourceOpenError(self.camera_id, self.config.uri, "no video stream")
        stream = streams[0]
        # Let FFmpeg use its own threads for decode. Without this a 1080p H.264 stream is
        # single-threaded and a fifty-camera host runs out of cores before it runs out of GPU.
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else self.config.fps
        self._set_format(stream.codec_context.height, stream.codec_context.width, fps)
        self._frames = container.decode(video=0)
        _LOG.info(
            "camera %s opened via pyav: %dx%d @ %.3g fps, hwaccel=%s, options=%s",
            self.camera_id,
            self.width,
            self.height,
            self.fps,
            self._using_hwaccel,
            options,
            extra=log_context(camera_id=self.camera_id),
        )

    def _try_open(
        self, av: Any, options: dict[str, str], timeout: tuple[float, float], *, hwaccel: bool
    ) -> Any:
        """Open the container, returning ``None`` when an optional attempt fails.

        The hardware attempt is optional and returns ``None`` on any failure; the software
        attempt is the last resort and raises, because at that point there is nothing left
        to fall back to.
        """
        kwargs: dict[str, Any] = {"options": options, "timeout": timeout}
        if hwaccel:
            accel = self._hwaccel(av)
            if accel is None:
                return None
            kwargs["hwaccel"] = accel
        try:
            return av.open(self.config.uri, **kwargs)
        except Exception as exc:  # noqa: BLE001 - PyAV's error tree varies across versions
            if hwaccel:
                _LOG.warning(
                    "camera %s: hardware decode unavailable (%s); falling back to software",
                    self.camera_id,
                    exc,
                    extra=log_context(camera_id=self.camera_id),
                )
                return None
            raise SourceOpenError(self.camera_id, self.config.uri, str(exc)) from exc

    def _hwaccel(self, av: Any) -> Any:
        """A CUDA hwaccel descriptor, or ``None`` if this PyAV cannot make one."""
        try:
            return av.codec.hwaccel.HWAccel(device_type="cuda", allow_software_fallback=True)
        except Exception as exc:  # noqa: BLE001 - AttributeError on PyAV < 13, or no device
            _LOG.debug(
                "camera %s: no PyAV hwaccel support (%s)",
                self.camera_id,
                exc,
                extra=log_context(camera_id=self.camera_id),
            )
            return None

    def _do_read(self) -> np.ndarray | None:
        try:
            frame = next(self._frames)
        except StopIteration as exc:
            # For a live camera this is the stream ending, not a quiet moment: FFmpeg only
            # stops iterating at EOF or after a fatal demuxer error.
            raise FrameDecodeError(self.camera_id, "stream ended") from exc
        except Exception as exc:  # noqa: BLE001 - PyAV's error tree varies across versions
            raise FrameDecodeError(self.camera_id, str(exc)) from exc
        return np.asarray(frame.to_ndarray(format="bgr24"))

    def _do_close(self) -> None:
        self._frames = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception as exc:  # noqa: BLE001 - closing a dead socket can raise
                _LOG.debug("camera %s: error closing container: %s", self.camera_id, exc)
        self._container = None
