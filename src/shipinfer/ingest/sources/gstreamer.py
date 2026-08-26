"""RTSP ingest through GStreamer, with NVDEC where the box has it.

**The pipeline is built as a string and logged.** That is the single most useful debugging
decision available here: an operator with a camera that will not connect can paste the
logged line into ``gst-launch-1.0``, add ``-v``, and see the negotiation fail for
themselves — without a Python interpreter, without this server, and without asking anyone.
Building the same graph element-by-element through ``Gst.ElementFactory.make`` is more
"correct" and produces nothing anyone can reproduce by hand.

The elements are **probed, not assumed**. ``nvv4l2decoder`` exists on a DeepStream install
and nowhere else; ``nvh264dec`` comes with ``gst-plugins-bad``'s nvcodec; a plain Ubuntu box
has neither. The reference implementation hard-coded ``avdec_h264`` — software decode for
fifty 4K streams — and then commented the whole pipeline out in favour of
``cv2.VideoCapture``, which is the same software decode with less control. Probing is why
this one can use the video engine when it is there and still start when it is not.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar

import numpy as np

from shipinfer import envs
from shipinfer.core.errors import (
    ConfigurationError,
    FrameDecodeError,
    SourceOpenError,
    SourceUnavailableError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.redact import redact_in
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.registry import SOURCES
from shipinfer.ingest.resolve import resolve_latency_ms, resolve_transport

__all__ = [
    "APPSINK_NAME",
    "GStreamerSource",
    "build_pipeline",
    "select_converter",
    "select_decoder",
]

_LOG = get_logger("ingest.gstreamer")

#: The name given to the appsink in the generated pipeline, so it can be looked up again.
APPSINK_NAME = "shipinfer_sink"

#: RTP depayloader and parser per codec. ``auto`` has neither: it uses ``decodebin``.
_DEPAY = {"h264": "rtph264depay", "h265": "rtph265depay"}
_PARSE = {"h264": "h264parse", "h265": "h265parse"}

#: Hardware decoders in preference order. ``nvv4l2decoder`` first because on a DeepStream
#: host it is the one that keeps frames in NVMM memory; the nvcodec elements are the
#: desktop/`gst-plugins-bad` equivalent.
_HW_DECODERS: dict[str, tuple[str, ...]] = {
    "h264": ("nvv4l2decoder", "nvh264dec"),
    "h265": ("nvv4l2decoder", "nvh265dec"),
}
_SW_DECODERS: dict[str, tuple[str, ...]] = {
    "h264": ("avdec_h264",),
    "h265": ("avdec_h265",),
}

#: Colour-space converters, in preference order. The first two are NVIDIA's and can take
#: Serialises GStreamer's one-time initialisation across camera threads (see `_load_gst`).
_GST_INIT_LOCK = threading.Lock()

#: Decoders that can output GL memory and open a GL display to do it. See `build_pipeline`.
_GL_CAPABLE_DECODERS: frozenset[str] = frozenset({"nvh264dec", "nvh265dec"})

#: NVMM/CUDA memory straight out of a hardware decoder; ``videoconvert`` is the portable
#: system-memory fallback and is always present.
_CONVERTERS: tuple[str, ...] = ("nvvideoconvert", "nvvidconv", "videoconvert")


def build_pipeline(
    uri: str,
    *,
    codec: str = "h264",
    latency_ms: int = 200,
    transport: str = "tcp",
    decoder: str | None = None,
    converter: str = "videoconvert",
    width: int | None = None,
    height: int | None = None,
    max_buffers: int = 2,
    appsink_name: str = APPSINK_NAME,
) -> str:
    """The ``gst-launch``-compatible pipeline description for one camera.

    A pure function of its arguments, which is what makes the exact strings assertable in
    the offline tier — no GStreamer, no camera, no network. The element *choices* are made
    by :func:`select_decoder` / :func:`select_converter`, which need a plugin registry and
    therefore cannot be pure.

    Args:
        uri: the ``rtsp://`` location.
        codec: ``h264``, ``h265``, or ``auto`` for a ``decodebin`` that negotiates it.
        latency_ms: ``rtspsrc``'s jitter buffer. A direct latency cost, so keep it small.
        transport: ``tcp`` (default), ``udp``, or ``auto`` to omit the property and let
            ``rtspsrc`` choose.
        decoder: the decoder element name, from :func:`select_decoder`. ``None`` falls back
            to the software decoder for ``codec``, so a hand-written call cannot silently
            pair an H.265 stream with an H.264 decoder.
        converter: the colour converter element name, from :func:`select_converter`.
        width, height: scale in the pipeline instead of in Python. Both or neither.
        max_buffers: appsink queue depth. ``drop=true`` plus a depth of 2 means the newest
            frame wins, which for live perception is the only sane policy — a 5-second-old
            frame is not worth a GPU.

    Raises:
        ConfigurationError: on an unknown codec, so a typo in ``config.codec`` fails at
            start-up instead of producing a pipeline that never negotiates.
    """
    if codec not in ("h264", "h265", "auto"):
        raise ConfigurationError(
            f"unsupported codec {codec!r}; expected one of ['auto', 'h264', 'h265']"
        )
    if (width is None) != (height is None):
        raise ConfigurationError("width and height must be given together, or neither")

    source = f"rtspsrc location={uri} latency={latency_ms}"
    if transport in ("tcp", "udp"):
        source += f" protocols={transport}"

    if codec == "auto":
        # decodebin picks the decoder by plugin rank, so it will use NVDEC when the
        # nvcodec/DeepStream plugins are installed and fall back on its own when they are
        # not. The cost is that we no longer know which decoder ran.
        decode = "decodebin"
    else:
        element = decoder or _SW_DECODERS[codec][0]
        decode = f"{_DEPAY[codec]} ! {_PARSE[codec]} ! {element}"
    if codec == "auto" or (decoder or "") in _GL_CAPABLE_DECODERS:
        # System memory, stated. nvcodec's `nvh264dec` / `nvh265dec` can output GL memory,
        # and when downstream leaves the choice open they create a GL display first — which
        # a headless container does not have: `gst_gl_display_gbm_new: could not find or open
        # DRM device`, then a segfault, on the first RTSP benchmark run. A `video/x-raw`
        # filter with no memory feature makes the decoder negotiate plain system memory and
        # never touch GL. Not applied to the DeepStream pair (`nvv4l2decoder` +
        # `nvvideoconvert`), whose NVMM hand-off is the point of choosing them.
        decode = f"{decode} ! video/x-raw"

    caps = "video/x-raw,format=BGR"
    if width is not None and height is not None:
        scale = f"videoscale ! {caps},width={width},height={height}"
    else:
        scale = caps

    return (
        f"{source} ! {decode} ! {converter} ! {scale} ! "
        f"appsink name={appsink_name} emit-signals=false sync=false "
        f"drop=true max-buffers={max_buffers}"
    )


def select_decoder(codec: str, *, hwaccel: bool, available: Any) -> str:
    """Pick the best installed decoder element for ``codec``.

    Args:
        codec: ``h264`` or ``h265``. ``auto`` never reaches here — ``decodebin`` does its
            own selection.
        hwaccel: try the hardware decoders first.
        available: predicate ``(element_name) -> bool``, normally a probe of the GStreamer
            plugin registry. Injected so the selection logic is testable without GStreamer.

    Raises:
        SourceUnavailableError: no decoder for this codec is installed at all, hardware or
            software. That is an install problem, not a camera problem, and the actor must
            not spend its reconnect budget on it.
    """
    candidates: tuple[str, ...] = ()
    if hwaccel:
        candidates += _HW_DECODERS.get(codec, ())
    candidates += _SW_DECODERS.get(codec, ())
    for element in candidates:
        if available(element):
            return element
    raise SourceUnavailableError(
        "gstreamer",
        f"no {codec} decoder found (tried {list(candidates)}); "
        "install gstreamer1.0-libav for software decode, or the nvcodec/DeepStream plugins",
    )


def select_converter(available: Any) -> str:
    """Pick a colour converter, preferring the ones that can read decoder memory."""
    for element in _CONVERTERS:
        if available(element):
            return element
    raise SourceUnavailableError(
        "gstreamer",
        f"none of {list(_CONVERTERS)} is installed; "
        "install gstreamer1.0-plugins-base for videoconvert",
    )


def _load_gst() -> tuple[Any, Any]:
    """Import and initialise GStreamer, or explain what is missing.

    Deliberately inside a function: importing this module must work on a host with no
    PyGObject, so the whole offline test tier can exercise the pipeline builder and the
    camera actor.
    """
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst
    except (ImportError, ValueError) as exc:
        raise SourceUnavailableError(
            "gstreamer",
            "PyGObject with GStreamer 1.0 typelibs is not importable "
            f"({redact_in(str(exc))}). Install python3-gi and gstreamer1.0-plugins-{{base,good,bad}}, "
            "or select the 'pyav' backend with SHIPINFER_INGEST_BACKEND=pyav",
        ) from exc
    # Fifty camera actors call this from fifty threads at start-up. `Gst.is_initialized()`
    # turns true as soon as *some* thread has begun initialising, before the plugin
    # registry is populated — so a thread that saw "initialised" and went straight to
    # `ElementFactory.find` got `None` for every decoder and gave its camera up as
    # "no h264 decoder found" on an image that has three. One lock, one init, and every
    # caller returns only after the registry exists.
    with _GST_INIT_LOCK:
        if not Gst.is_initialized():
            Gst.init(None)
    return Gst, GLib


@SOURCES.register("gstreamer", "gst")
class GStreamerSource(FrameSource):
    """One RTSP camera, decoded by a GStreamer pipeline into BGR frames.

    Hardware decode is a *preference*, not a requirement: the decoder is chosen by probing
    the plugin registry, and the resolved pipeline is logged at INFO so the choice is
    visible in production rather than inferred from ``nvidia-smi``.
    """

    name: ClassVar[str] = "gstreamer"
    supports_hwaccel: ClassVar[bool] = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gst: Any = None
        self._pipeline: Any = None
        self._appsink: Any = None
        self._pipeline_description = ""

    @property
    def pipeline_description(self) -> str:
        """The exact ``gst-launch-1.0`` string in use; empty before :meth:`open`."""
        return self._pipeline_description

    # -- lifecycle ---------------------------------------------------------------------

    def _do_open(self) -> None:
        gst, glib = _load_gst()
        self._gst = gst

        def available(element: str) -> bool:
            return gst.ElementFactory.find(element) is not None

        codec = self.config.codec
        override = envs.GST_DECODER_OVERRIDE.get()
        if codec == "auto":
            decoder = "decodebin"
        elif override:
            decoder = override
        else:
            decoder = select_decoder(codec, hwaccel=self.hwaccel, available=available)

        description = build_pipeline(
            self.config.uri,
            codec=codec,
            latency_ms=resolve_latency_ms(self.config, self.settings),
            transport=resolve_transport(self.config, self.settings),
            decoder=decoder,
            converter=select_converter(available),
            width=self.config.width,
            height=self.config.height,
            max_buffers=envs.GST_APPSINK_MAX_BUFFERS.get(),
        )
        self._pipeline_description = description
        _LOG.info(
            # The description embeds `location=<uri>`, so it is redacted before logging
            # even though that makes it not quite copy-pasteable: a fleet credential in a
            # log is a worse outcome than retyping a password into a debug command.
            "camera %s pipeline (paste into gst-launch-1.0, password redacted): %s",
            self.camera_id,
            redact_in(description),
            extra=log_context(camera_id=self.camera_id),
        )

        try:
            self._pipeline = gst.parse_launch(description)
        except glib.Error as exc:
            raise SourceOpenError(
                self.camera_id,
                self.config.uri,
                f"pipeline would not parse: {redact_in(str(exc))}",
            ) from exc

        self._appsink = self._pipeline.get_by_name(APPSINK_NAME)
        if self._appsink is None:  # pragma: no cover - only reachable if the string breaks
            raise SourceOpenError(
                self.camera_id, self.config.uri, f"pipeline has no appsink {APPSINK_NAME!r}"
            )

        if self._pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise SourceOpenError(
                self.camera_id, self.config.uri, "pipeline refused to enter PLAYING"
            )
        # PLAYING is asynchronous: rtspsrc has not even sent DESCRIBE yet. Block until the
        # state change actually completes so a wrong URI or a bad credential is a failed
        # open() — countable, backed off — instead of a stream that silently never
        # delivers and looks like a slow camera.
        state_return, _, _ = self._pipeline.get_state(int(self.open_timeout_s * gst.SECOND))
        if state_return != gst.StateChangeReturn.SUCCESS:
            raise SourceOpenError(
                self.camera_id,
                self.config.uri,
                f"stream did not start within {self.open_timeout_s:g}s ({state_return!r})",
            )
        self._negotiate_from_appsink()

    def _negotiate_from_appsink(self) -> None:
        """Record the caps the appsink actually negotiated.

        Best effort: the pad may not have caps yet even in PLAYING, in which case the first
        decoded frame fills them in. Reporting the *negotiated* size rather than the
        requested one is what makes a silently-ignored ``width``/``height`` visible.
        """
        pad = self._appsink.get_static_pad("sink")
        caps = pad.get_current_caps() if pad is not None else None
        if caps is None or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        ok_h, height = structure.get_int("height")
        ok_w, width = structure.get_int("width")
        fps = self.config.fps
        ok_fps, num, den = structure.get_fraction("framerate")
        if ok_fps and den:
            fps = num / den
        if ok_h and ok_w:
            self._set_format(height, width, fps)

    def _do_read(self) -> np.ndarray | None:
        gst = self._gst
        sample = self._appsink.try_pull_sample(int(self.read_timeout_s * gst.SECOND))
        if sample is None:
            # Nothing within the timeout. Distinguish "quiet" from "over" by asking the bus:
            # an EOS or ERROR message means reconnect, a timeout means keep waiting.
            self._raise_if_stream_ended()
            return None
        return self._sample_to_array(sample)

    def _raise_if_stream_ended(self) -> None:
        gst = self._gst
        bus = self._pipeline.get_bus()
        message = bus.pop_filtered(gst.MessageType.ERROR | gst.MessageType.EOS)
        if message is None:
            return
        if message.type == gst.MessageType.EOS:
            raise FrameDecodeError(self.camera_id, "end of stream")
        error, debug = message.parse_error()
        raise FrameDecodeError(self.camera_id, f"{error.message} ({debug})")

    def _sample_to_array(self, sample: Any) -> np.ndarray:
        """Copy one BGR sample out of GStreamer's buffer pool into numpy.

        The copy is not optional: ``unmap`` returns the buffer to the decoder's pool, which
        will overwrite it while a zero-copy view is still being read downstream — a bug
        that produces plausible-looking frames and is invisible to a test that submits the
        same image twice.
        """
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        if (height, width) != (self._height, self._width):
            self._set_format(height, width, self._fps or self.config.fps)
        buffer = sample.get_buffer()
        ok, info = buffer.map(self._gst.MapFlags.READ)
        if not ok:  # pragma: no cover - a mapping failure needs a broken allocator
            raise FrameDecodeError(self.camera_id, "could not map the frame buffer")
        try:
            # GStreamer pads each row of raw video to a multiple of 4 bytes. For 3-byte BGR
            # that only matters at widths not divisible by 4 — which is exactly the case a
            # naive reshape gets wrong, and only for some cameras.
            stride = ((width * 3) + 3) & ~3
            flat = np.frombuffer(info.data, dtype=np.uint8, count=stride * height)
            image = flat.reshape(height, stride)[:, : width * 3].reshape(height, width, 3)
            return image.copy()
        finally:
            buffer.unmap(info)

    def _do_close(self) -> None:
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._pipeline = None
        self._appsink = None
