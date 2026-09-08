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
        assert "decodebin ! video/x-raw !" in pipeline, (
            "decodebin must be told to negotiate system memory, or NVDEC opens a GL display "
            "the headless container does not have"
        )
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
            "rtph264depay ! h264parse ! nvh264dec ! video/x-raw ! videoconvert ! "
            "videoscale ! video/x-raw,format=BGR,width=1280,height=720 ! "
            f"appsink name={APPSINK_NAME} emit-signals=false sync=false drop=true max-buffers=4"
        )

    def test_the_deepstream_pair_keeps_its_nvmm_handoff(self):
        """`nvv4l2decoder ! nvvideoconvert` passes NVMM memory between them; forcing system
        memory there would undo the reason for choosing them. Only the GL-capable nvcodec
        decoders and the open-ended `decodebin` get the filter."""
        pipeline = build_pipeline(
            URI, codec="h264", decoder="nvv4l2decoder", converter="nvvideoconvert"
        )
        assert "nvv4l2decoder ! nvvideoconvert" in pipeline
        assert "video/x-raw ! nvvideoconvert" not in pipeline

    def test_the_software_decoder_needs_no_filter(self):
        pipeline = build_pipeline(URI, codec="h264", decoder="avdec_h264")
        assert "avdec_h264 ! videoconvert" in pipeline

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


class TestInitialisationIsSerialised:
    """Fifty camera threads call `_load_gst` at once. `Gst.is_initialized()` turns true before
    the registry is populated, so an unguarded caller could probe an empty registry and give
    its camera up as "no decoder found" on an image that has three."""

    def test_concurrent_callers_initialise_once_and_all_see_the_registry(self, monkeypatch):
        import sys
        import threading
        import time
        import types

        from shipinfer.ingest.sources import gstreamer

        state = {"initialised": False, "inits": 0, "ready": False}

        class FakeGst:
            @staticmethod
            def is_initialized():
                return state["initialised"]

            @staticmethod
            def init(_argv):
                state["inits"] += 1
                state["initialised"] = True  # visible before the registry is ready...
                time.sleep(0.05)  # ...which takes a while
                state["ready"] = True

        gi = types.ModuleType("gi")
        gi.require_version = lambda *_a: None
        repository = types.ModuleType("gi.repository")
        repository.Gst = FakeGst
        repository.GLib = object()
        gi.repository = repository
        monkeypatch.setitem(sys.modules, "gi", gi)
        monkeypatch.setitem(sys.modules, "gi.repository", repository)

        seen_ready: list[bool] = []
        lock = threading.Lock()

        def worker():
            _gst, _glib = gstreamer._load_gst()
            with lock:
                seen_ready.append(state["ready"])

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert state["inits"] == 1, "one init, however many threads arrive at once"
        assert all(seen_ready), "every caller returned only after the registry existed"

    def test_gio_is_told_not_to_use_libproxy_unless_the_operator_chose_one(self, monkeypatch):
        """libproxy throws a C++ exception through GIO when it has no configuration to read —
        `terminate` for the whole process, seen on the first containerised RTSP run."""
        import os
        import sys
        import types

        from shipinfer.ingest.sources import gstreamer

        class FakeGst:
            @staticmethod
            def is_initialized():
                return True

        gi = types.ModuleType("gi")
        gi.require_version = lambda *_a: None
        repository = types.ModuleType("gi.repository")
        repository.Gst = FakeGst
        repository.GLib = object()
        gi.repository = repository
        monkeypatch.setitem(sys.modules, "gi", gi)
        monkeypatch.setitem(sys.modules, "gi.repository", repository)

        monkeypatch.delenv("GIO_USE_PROXY_RESOLVER", raising=False)
        gstreamer._load_gst()
        assert os.environ["GIO_USE_PROXY_RESOLVER"] == "dummy"

        monkeypatch.setenv("GIO_USE_PROXY_RESOLVER", "gnome")  # an operator's real choice stays
        gstreamer._load_gst()
        assert os.environ["GIO_USE_PROXY_RESOLVER"] == "gnome"


class TestTheReadTimeoutIsSpentInSlices:
    """An EOS must be noticed on the FIRST empty slice, not after the whole read timeout.

    The C++ source's first version of this slicing asked the bus only after the deadline ran
    out, so a stream that had ended sat out `read_timeout_ms` — spinning through fifty pulls —
    where the single full-timeout pull used to report it at once. Slower to detect and busier
    while detecting. Both planes ask per slice now, and this pins the Python one.
    """

    def _source(self, monkeypatch, pulls: list[int], on_bus, timeout_s: float = 5.0):
        """A `GStreamerSource` with only what `_do_read` touches, so no GStreamer is needed.

        `read_timeout_s` is a property on the base class, so it is patched there rather than
        assigned -- which is also the point: the seam under test reads the same knob a
        deployment sets.
        """
        from shipinfer.core.errors import FrameDecodeError
        from shipinfer.ingest.sources import gstreamer

        # A FAKE CLOCK THE SINK ADVANCES, because a real `try_pull_sample` BLOCKS for its
        # timeout and this one returns at once: with the wall clock the loop spun 333 000
        # times in 0.35 s, which measures the fake rather than the code. Advancing the clock
        # by exactly the timeout the pull was given is what a blocking pull does.
        now = [1000.0]
        monkeypatch.setattr(gstreamer.time, "monotonic", lambda: now[0])

        class Sink:
            def try_pull_sample(self, timeout):
                pulls.append(timeout)
                now[0] += timeout / 1_000_000_000
                return

        source = object.__new__(gstreamer.GStreamerSource)
        source._gst = type("Gst", (), {"SECOND": 1_000_000_000})()
        source._appsink = Sink()
        monkeypatch.setattr(
            gstreamer.GStreamerSource, "read_timeout_s", property(lambda self: timeout_s)
        )
        monkeypatch.setattr(
            gstreamer.GStreamerSource, "_raise_if_stream_ended", on_bus, raising=True
        )
        return source, FrameDecodeError

    def test_an_ended_stream_raises_on_the_first_empty_slice(self, monkeypatch):
        pulls: list[int] = []

        def ended(self):
            from shipinfer.core.errors import FrameDecodeError

            raise FrameDecodeError("cam", "end of stream")

        source, FrameDecodeError = self._source(monkeypatch, pulls, ended)

        with pytest.raises(FrameDecodeError):
            source._do_read()

        assert len(pulls) == 1, (
            f"the bus is asked after the FIRST empty slice, not after the deadline: "
            f"{len(pulls)} pulls"
        )
        assert pulls[0] == int(
            0.1 * 1_000_000_000
        ), "and that slice is 100 ms rather than the whole 5 s read timeout"

    def test_a_stop_ends_the_read_without_waiting_out_the_timeout(self, monkeypatch):
        """The half that had no Python counterpart until `PY-SOURCE-HAS-NO-STOP-SIGNAL`.

        A read that spends its whole `read_timeout_s` is a camera the fleet abandons: the actor
        only learns of a stop when `_do_read` returns, and `stop()`'s budget is shorter than
        five seconds. Checked every slice, so the read ends within one.
        """
        import threading

        pulls: list[int] = []
        stop = threading.Event()
        source, _ = self._source(monkeypatch, pulls, lambda self: None)
        source._stop = stop

        stop.set()
        assert source._do_read() is None
        assert pulls == [], "a stop already set means no pull at all"

        stop.clear()
        assert source._do_read() is None  # the ordinary timeout path, for contrast
        before = len(pulls)
        assert before > 0, "and a running camera does pull"

        # Set mid-read: the loop must notice on its next pass rather than at the deadline.
        source._stop = stop
        original = source._appsink.try_pull_sample

        def stop_after_one(timeout):
            stop.set()
            return original(timeout)

        source._appsink.try_pull_sample = stop_after_one
        assert source._do_read() is None
        assert (
            len(pulls) == before + 1
        ), f"one more slice, not the whole timeout: {len(pulls) - before} pulls after the stop"

    def test_a_quiet_camera_still_spends_its_whole_timeout(self, monkeypatch):
        """The other half: an empty bus means keep waiting, and the DEADLINE is what bounds
        the read. Sliced or not, `read()` must not come back early and burn an empty read."""
        pulls: list[int] = []
        source, _ = self._source(monkeypatch, pulls, lambda self: None, timeout_s=0.35)

        assert source._do_read() is None
        assert 3 <= len(pulls) <= 5, f"~0.35 s in 100 ms slices: {len(pulls)} pulls"
        assert sum(pulls) <= int(0.36 * 1_000_000_000), "and no slice overruns the deadline"


class TestPullingASampleDoesNotDependOnTheTypelib:
    """`GstApp.AppSink.try_pull_sample` is a method only when the GstApp typelib is loaded;
    the `try-pull-sample` signal always exists. The first containerised RTSP run that reached
    a read failed on every camera with "'GstAppSink' object has no attribute 'try_pull_sample'".
    """

    def test_the_method_is_used_when_present(self):
        from shipinfer.ingest.sources.gstreamer import _try_pull_sample

        class Sink:
            def try_pull_sample(self, timeout):
                return ("method", timeout)

        assert _try_pull_sample(Sink(), 5) == ("method", 5)

    def test_the_signal_is_the_fallback(self):
        from shipinfer.ingest.sources.gstreamer import _try_pull_sample

        class BareElement:
            def emit(self, name, *args):
                return (name, args)

        assert _try_pull_sample(BareElement(), 7) == ("try-pull-sample", (7,))
