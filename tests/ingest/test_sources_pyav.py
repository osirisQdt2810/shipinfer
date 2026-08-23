"""The PyAV backend's option building and import safety.

The options matter more than they look: FFmpeg **silently ignores** an AVOption it does not
recognise, so ``rtsp_transport=tpc`` gets you UDP, no error, and decode artefacts that look
like a model regression. Pinning the dictionary is the only way that stays fixed.

Runs with no PyAV installed, by design.
"""

from __future__ import annotations

import pytest

from shipinfer.core.errors import SourceOpenError, SourceUnavailableError
from shipinfer.ingest.sources.pyav import PyAvSource, build_open_options, is_network_uri


class TestUriClassification:
    """Network options apply to streams and not to local files."""

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("rtsp://cam/stream", True),
            ("rtsps://cam/stream", True),
            ("http://host/stream.m3u8", True),
            ("/data/clip.mp4", False),
            ("clip.mp4", False),
            ("file:///data/clip.mp4", False),
        ],
    )
    def test_network_detection(self, uri, expected):
        assert is_network_uri(uri) is expected


class TestOpenOptions:
    """The AVOptions FFmpeg would otherwise ignore silently."""

    def test_rtsp_options_pin_tcp_and_a_timeout(self, make_camera):
        options = build_open_options(
            make_camera(uri="rtsp://cam/stream"),
            transport="tcp",
            latency_ms=200,
            read_timeout_s=5.0,
        )
        assert options == {
            "rtsp_transport": "tcp",
            "stimeout": "5000000",
            "max_delay": "200000",
            "fflags": "nobuffer",
            "flags": "low_delay",
        }

    def test_udp_is_selectable_but_not_the_default(self, make_camera):
        options = build_open_options(make_camera(uri="rtsp://cam/x"), transport="udp")
        assert options["rtsp_transport"] == "udp"
        assert build_open_options(make_camera(uri="rtsp://cam/x"))["rtsp_transport"] == "tcp"

    def test_auto_transport_leaves_ffmpegs_default_alone(self, make_camera):
        options = build_open_options(make_camera(uri="rtsp://cam/x"), transport="auto")
        assert "rtsp_transport" not in options
        assert "stimeout" in options

    def test_a_local_file_gets_no_rtsp_options(self, make_camera):
        assert build_open_options(make_camera(uri="/data/clip.mp4")) == {}

    def test_camera_options_win(self, make_camera):
        """The escape hatch has to be able to override, or it is not one."""
        camera = make_camera(
            uri="rtsp://cam/x", options={"rtsp_transport": "udp", "buffer_size": "64"}
        )
        options = build_open_options(camera, transport="tcp")
        assert options["rtsp_transport"] == "udp"
        assert options["buffer_size"] == "64"


class TestImportSafety:
    """The module imports and fails usefully on a host with no PyAV."""

    def test_the_source_constructs_without_pyav(self, make_camera):
        source = PyAvSource(make_camera())
        assert source.is_open is False
        assert source.using_hwaccel is False
        assert source.supports_hwaccel is True

    def test_opening_without_pyav_raises_a_typed_error(self, make_camera):
        try:
            import av  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("PyAV is installed on this host; the failure path is unreachable")

        source = PyAvSource(make_camera())
        with pytest.raises(SourceUnavailableError) as excinfo:
            source.open()
        assert "shipinfer[video]" in str(excinfo.value)
        assert source.is_open is False

    def test_a_failed_open_is_reported_against_the_camera(self, monkeypatch, make_camera):
        """With a stub `av`, a refused connection must surface as SourceOpenError, not ImportError."""

        class FakeAv:
            @staticmethod
            def open(*_args, **_kwargs):
                raise OSError("Connection refused")

        monkeypatch.setattr(
            "shipinfer.ingest.sources.pyav._load_av", lambda: FakeAv, raising=True
        )
        source = PyAvSource(make_camera("cam7", uri="rtsp://nowhere/stream", hwaccel=False))
        with pytest.raises(SourceOpenError) as excinfo:
            source.open()
        message = str(excinfo.value)
        assert "cam7" in message and "Connection refused" in message
        assert source.is_open is False
