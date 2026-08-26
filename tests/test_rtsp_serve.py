"""The benchmark's RTSP server: the launch line is a contract the measurement depends on."""

from __future__ import annotations

from pathlib import Path

from scripts import rtsp_serve


class TestTheStreamIsPacedByTheClock:
    """`h264parse` stamps each access unit from the framerate in the caps and `identity
    sync=true` holds it until the pipeline clock reaches that stamp. Without the sync the
    packetiser pushes the file as fast as the socket accepts: the first containerised RTSP
    measurement offered 170% of its target — twelve "5 fps" cameras delivered 101.8 img/s,
    limited only by how fast the client could decode."""

    def test_identity_sync_sits_between_the_parser_and_the_payloader(
        self, tmp_path: Path
    ) -> None:
        server = rtsp_serve.RtspFixtureServer(
            tmp_path / "fixture.h264", streams=1, port=18554, fps=5
        )
        line = server.launch_line()
        assert "framerate=5/1" in line
        assert "h264parse config-interval=1 ! identity sync=true ! rtph264pay" in line

    def test_the_frame_rate_is_the_one_asked_for(self, tmp_path: Path) -> None:
        line = rtsp_serve.RtspFixtureServer(
            tmp_path / "f.h264", streams=1, port=18554, fps=20
        ).launch_line()
        assert "framerate=20/1" in line
