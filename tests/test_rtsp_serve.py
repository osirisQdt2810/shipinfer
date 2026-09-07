"""The benchmark's RTSP server: the launch line is a contract the measurement depends on."""

from __future__ import annotations

from pathlib import Path

import pytest

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

    def test_a_b_frame_fixture_is_its_own_file(self, tmp_path: Path) -> None:
        """Same reason as the rate: two streams, one cache key, and the second run gets the
        first's bitstream. A B-frame fixture is a DIFFERENT stream -- reordered, deeper
        reference depth -- and serving it to a check that asserts decode order is display
        order would fail for the wrong reason."""
        data = tmp_path / "person_2K"
        plain = rtsp_serve.default_fixture_path(data, 20)
        reordered = rtsp_serve.default_fixture_path(data, 20, 3)
        assert plain.name == "person_2K-20fps.h264", "the default spelling is unchanged"
        assert reordered.name == "person_2K-20fps-bf3.h264"
        assert plain != reordered

    def test_bframes_drops_zerolatency_because_that_tune_forces_them_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one line that would make the flag a no-op. ``-tune zerolatency`` sets
        ``bframes=0`` inside x264 regardless of ``-bf``, so asking for three and leaving the
        tune in place produces the same one-reference stream and the check that needed a deep
        DPB passes on a fixture that never had one."""
        data = tmp_path / "frames"
        data.mkdir()
        (data / "a.jpg").write_bytes(b"")
        seen: list[list[str]] = []
        monkeypatch.setattr(rtsp_serve.shutil, "which", lambda _: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            rtsp_serve.subprocess, "run", lambda command, **_: seen.append(command)
        )

        rtsp_serve.encode_fixture(data, tmp_path / "plain.h264", fps=20)
        rtsp_serve.encode_fixture(data, tmp_path / "deep.h264", fps=20, bframes=3)

        plain, deep = seen
        assert "zerolatency" in plain and plain[plain.index("-bf") + 1] == "0"
        assert "-refs" not in plain, "the default stream is unchanged, flag or no flag"
        assert "zerolatency" not in deep, "the tune that would silently cancel -bf"
        assert deep[deep.index("-bf") + 1] == "3"
        assert deep[deep.index("-refs") + 1] == "3", "depth, which is what a DPB is sized for"
