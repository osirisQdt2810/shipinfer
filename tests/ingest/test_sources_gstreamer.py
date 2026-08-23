"""The GStreamer pipeline, asserted as text.

Every string here is one an operator can paste into ``gst-launch-1.0``, which is exactly why
they are worth pinning: a silent change to element order or to ``drop=true`` is a behaviour
change nobody would otherwise notice until a camera fell behind.

Runs with no GStreamer installed, by design.
"""

from __future__ import annotations

import pytest

from shipinfer.core.errors import ConfigurationError, SourceUnavailableError
from shipinfer.ingest.sources.gstreamer import (
    APPSINK_NAME,
    GStreamerSource,
    build_pipeline,
    select_converter,
    select_decoder,
)

URI = "rtsp://operator:REDACTED@10.0.0.100/stream"
APPSINK = f"appsink name={APPSINK_NAME} emit-signals=false sync=false drop=true max-buffers=2"


class TestPipelineString:
    """The exact `gst-launch-1.0` line, for every codec and both decode paths."""

    def test_h264_with_nvdec(self):
        assert build_pipeline(
            URI, codec="h264", decoder="nvv4l2decoder", converter="nvvideoconvert"
        ) == (
            f"rtspsrc location={URI} latency=200 protocols=tcp ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! "
            f"video/x-raw,format=BGR ! {APPSINK}"
        )

    def test_h265_with_nvdec(self):
        assert build_pipeline(
            URI, codec="h265", decoder="nvv4l2decoder", converter="nvvideoconvert"
        ) == (
            f"rtspsrc location={URI} latency=200 protocols=tcp ! "
            "rtph265depay ! h265parse ! nvv4l2decoder ! nvvideoconvert ! "
            f"video/x-raw,format=BGR ! {APPSINK}"
        )

    def test_h264_software_fallback(self):
        assert build_pipeline(URI, codec="h264") == (
            f"rtspsrc location={URI} latency=200 protocols=tcp ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            f"video/x-raw,format=BGR ! {APPSINK}"
        )

    def test_h265_software_fallback_uses_the_h265_decoder(self):
        """A default that paired H.265 with `avdec_h264` would build a pipeline that never links."""
        assert " h265parse ! avdec_h265 ! " in build_pipeline(URI, codec="h265")

    def test_auto_codec_delegates_to_decodebin(self):
        pipeline = build_pipeline(URI, codec="auto")
        assert "decodebin" in pipeline
        assert "depay" not in pipeline and "parse" not in pipeline

    def test_scaling_and_transport_and_latency(self):
        assert build_pipeline(
            URI,
            codec="h264",
            decoder="nvh264dec",
            latency_ms=100,
            transport="udp",
            width=1280,
            height=720,
            max_buffers=4,
        ) == (
            f"rtspsrc location={URI} latency=100 protocols=udp ! "
            "rtph264depay ! h264parse ! nvh264dec ! videoconvert ! "
            "videoscale ! video/x-raw,format=BGR,width=1280,height=720 ! "
            f"appsink name={APPSINK_NAME} emit-signals=false sync=false drop=true max-buffers=4"
        )

    def test_auto_transport_omits_the_property(self):
        """`protocols=auto` is not a GStreamer value; leaving it out is what "let rtspsrc decide" is."""
        assert "protocols=" not in build_pipeline(URI, transport="auto")

    def test_an_unknown_codec_fails_loudly(self):
        with pytest.raises(ConfigurationError, match="unsupported codec"):
            build_pipeline(URI, codec="vp9")

    def test_half_a_scale_is_rejected(self):
        with pytest.raises(ConfigurationError, match="together"):
            build_pipeline(URI, width=640)


class TestElementSelection:
    """Decoders are probed, not assumed: NVDEC when installed, software otherwise."""

    def test_hardware_decoder_is_preferred_when_present(self):
        installed = {"nvv4l2decoder", "avdec_h264"}
        assert select_decoder("h264", hwaccel=True, available=installed.__contains__) == (
            "nvv4l2decoder"
        )

    def test_the_second_hardware_choice_is_tried_before_software(self):
        installed = {"nvh265dec", "avdec_h265"}
        assert (
            select_decoder("h265", hwaccel=True, available=installed.__contains__)
            == "nvh265dec"
        )

    def test_hwaccel_off_goes_straight_to_software(self):
        installed = {"nvv4l2decoder", "avdec_h264"}
        assert select_decoder("h264", hwaccel=False, available=installed.__contains__) == (
            "avdec_h264"
        )

    def test_software_decode_is_the_fallback_when_no_nvidia_plugin_exists(self):
        installed = {"avdec_h264", "videoconvert"}
        assert (
            select_decoder("h264", hwaccel=True, available=installed.__contains__)
            == "avdec_h264"
        )

    def test_no_decoder_at_all_is_an_install_problem_not_a_camera_problem(self):
        with pytest.raises(SourceUnavailableError) as excinfo:
            select_decoder("h264", hwaccel=True, available=lambda _: False)
        assert "gstreamer1.0-libav" in str(excinfo.value)

    def test_converter_prefers_the_nvidia_one(self):
        assert (
            select_converter({"nvvideoconvert", "videoconvert"}.__contains__)
            == "nvvideoconvert"
        )
        assert select_converter({"nvvidconv", "videoconvert"}.__contains__) == "nvvidconv"
        assert select_converter({"videoconvert"}.__contains__) == "videoconvert"
        with pytest.raises(SourceUnavailableError):
            select_converter(lambda _: False)


class TestImportSafety:
    """The module imports and the object constructs on a host with no GStreamer."""

    def test_the_source_constructs_without_gstreamer(self, make_camera):
        """The module must import and the object must exist even with no PyGObject on the host."""
        source = GStreamerSource(make_camera())
        assert source.is_open is False
        assert source.pipeline_description == ""
        assert source.supports_hwaccel is True
        assert (source.height, source.width) == (0, 0)

    def test_opening_without_gstreamer_raises_a_typed_error(self, make_camera):
        try:
            import gi  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("PyGObject is installed on this host; the failure path is unreachable")

        source = GStreamerSource(make_camera())
        with pytest.raises(SourceUnavailableError) as excinfo:
            source.open()
        message = str(excinfo.value)
        assert "PyGObject" in message
        assert (
            "SHIPINFER_INGEST_BACKEND=pyav" in message
        ), "the error must say what to do instead"
        assert source.is_open is False

    def test_reading_before_open_is_a_typed_error(self, make_camera):
        from shipinfer.core.errors import SourceOpenError

        source = GStreamerSource(make_camera())
        with pytest.raises(SourceOpenError, match="before open"):
            source.read()

    def test_close_before_open_is_a_no_op(self, make_camera):
        source = GStreamerSource(make_camera())
        source.close()
        source.close()
        assert source.is_open is False
