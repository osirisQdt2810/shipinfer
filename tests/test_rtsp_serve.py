"""The benchmark's RTSP server: the launch line is a contract the measurement depends on."""

from __future__ import annotations

from pathlib import Path

from scripts import rtsp_serve


class TestTheStreamIsPacedByTheClock:
    """`h264parse` stamps each access unit from the framerate in the caps and `identity
    sync=true` holds it until the pipeline clock reaches that stamp; `single-segment=true` keeps
    running time continuous across the loop, or the second pass restarts at zero and goes out
    unpaced. Without the sync the packetiser pushes the file as fast as the socket accepts: the
    first containerised RTSP measurement offered 170% of its target; with sync alone, 127%."""

    def test_identity_sync_sits_between_the_parser_and_the_payloader(
        self, tmp_path: Path
    ) -> None:
        server = rtsp_serve.RtspFixtureServer(
            tmp_path / "fixture.h264", streams=1, port=18554, fps=5
        )
        line = server.launch_line()
        assert "framerate=5/1" in line
        assert (
            "h264parse config-interval=1 ! identity single-segment=true sync=true ! rtph264pay"
            in line
        )

    def test_the_frame_rate_is_the_one_asked_for(self, tmp_path: Path) -> None:
        line = rtsp_serve.RtspFixtureServer(
            tmp_path / "f.h264", streams=1, port=18554, fps=20
        ).launch_line()
        assert "framerate=20/1" in line


class TestTheFixtureCacheIsKeyedByFrameRate:
    """The stream's own SPS timing paces playback, so a fixture encoded at one rate cannot be
    served as another: the first 12 x 5 measurement offered 20 fps per camera from a fixture
    cached by an earlier 20 fps run."""

    def test_two_rates_are_two_files(self, tmp_path: Path) -> None:
        data = tmp_path / "person_2K"
        assert rtsp_serve.default_fixture_path(data, 5) != rtsp_serve.default_fixture_path(
            data, 20
        )
        assert rtsp_serve.default_fixture_path(data, 5).name == "person_2K-5fps.h264"
        assert rtsp_serve.default_fixture_path(data, 5).parent == tmp_path / ".rtsp"
