#!/usr/bin/env python
"""Serve N looping RTSP streams built from real JPEGs, so ingest can be tested for real.

WHY THIS EXISTS

``src/shipinfer/ingest/sources/gstreamer.py`` shipped a pipeline builder and a set of tests
that asserted the *strings* it produced. Nothing had ever connected to a camera, because
there was no camera to connect to and no way to make one. Every reconnect test therefore
ran against a scripted double, which proves the actor's arithmetic and nothing about RTSP.

This is the missing half: a real RTSP server, on a real socket, speaking real
DESCRIBE/SETUP/PLAY, carrying a real H.264 elementary stream made from the real 1920x1080
frames in ``benchmarks/baseline/data/``.

WHY GST-RTSP-SERVER AND NOT FFMPEG

``ffmpeg -f rtsp`` cannot serve. In FFmpeg 4.4 ``rtsp_flags listen`` is a **demuxer**
option: it makes ffmpeg wait for someone to ANNOUNCE *to* it. The RTSP muxer only knows how
to push, so an ffmpeg output can publish to a server and never be one. Confirmed on this
host with ``ffmpeg -h muxer=rtsp``, which lists no ``listen``.

So the server is ``gst-rtsp-server`` (the reference implementation, an apt package) and
ffmpeg's only job is the one it is good at: turning the JPEGs into an H.264 bitstream, once.
That split also keeps the test honest — the bitstream under test is x264's, not GStreamer's,
so the client is not being validated against its own encoder.

WHY ONE ENCODE AND N PACKETISERS

Fifty streams at 1080p20 is 1000 encoded frames a second. An A5000 has one NVENC engine and
x264 would need most of the box, so encoding per stream would make the *harness* the
bottleneck and the measurement meaningless. Instead the JPEGs are encoded once into an
elementary stream and each RTSP path re-packetises those bytes: no encoder in the loop, 50
streams cost almost nothing, and every camera carries identical, byte-comparable frames.

USAGE

    python scripts/rtsp_serve.py --streams 50           # rtsp://127.0.0.1:8554/cam0 ...
    python scripts/rtsp_serve.py --streams 2 --port 18554 --fps 20
    python scripts/rtsp_serve.py --print-uris --streams 4    # just the URIs, then exit

Add ``--user admin --password pass`` to require credentials, which is how the
``rtsp://user:pass@host/path`` form in ``CameraConfig.uri`` gets exercised at all.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The real frames. 10 x 1920x1080 JPEGs of people, which is what the benchmark measures on.
DEFAULT_DATA = REPO / "benchmarks" / "baseline" / "data" / "person_2K"
DEFAULT_PORT = 8554
DEFAULT_FPS = 20

__all__ = ["RtspFixtureServer", "encode_fixture", "stream_uri"]


def encode_fixture(
    data_dir: Path, out_path: Path, *, fps: int = DEFAULT_FPS, force: bool = False
) -> Path:
    """Encode a directory of JPEGs into one H.264 elementary stream. Cached.

    ``-g fps`` puts a keyframe every second, so a client that joins mid-loop starts
    decoding within a frame or two instead of showing nothing for the length of the GOP —
    which for a reconnect test is the difference between a two-second reconnect and a
    ten-second one. ``-bf 0`` removes B-frames for the same reason: no reordering delay, and
    the decode order is the display order, so a test can assert on frame content.

    Raises:
        FileNotFoundError: no JPEGs, or no ffmpeg.
    """
    frames = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.jpeg"))
    if not frames:
        raise FileNotFoundError(f"no JPEGs in {data_dir}")
    if out_path.exists() and not force:
        return out_path
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg is not on PATH; it is needed once, to build the fixture"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(data_dir / "*.jpg"),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
        "-bf",
        "0",
        "-f",
        "h264",
        str(out_path),
    ]
    subprocess.run(command, check=True)
    return out_path


def stream_uri(
    index: int,
    *,
    port: int = DEFAULT_PORT,
    host: str = "127.0.0.1",
    user: str | None = None,
    password: str | None = None,
) -> str:
    """The URI for stream ``index``, credentials included when the server requires them."""
    credentials = f"{user}:{password}@" if user else ""
    return f"rtsp://{credentials}{host}:{port}/cam{index}"


class RtspFixtureServer:
    """N RTSP paths, all serving the same looping H.264 fixture.

    Runs a GLib main loop, so it lives in its own thread or its own process. The tests use
    the process form (see ``tests/ingest/rtsp_fixture.py``) because killing a process is the
    only faithful way to simulate a camera dropping off the network: an in-process server
    could be asked to stop politely, and a polite stop is not what a switch losing power
    looks like.

    Args:
        fixture: the H.264 elementary stream every path serves.
        streams: how many paths to mount, named ``cam0`` .. ``cam{streams-1}``.
        port: the RTSP port.
        fps: the framerate advertised in the caps, which is what paces playback.
        user, password: when both are given, every path requires basic auth.
    """

    def __init__(
        self,
        fixture: Path,
        *,
        streams: int = 1,
        port: int = DEFAULT_PORT,
        fps: int = DEFAULT_FPS,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        if streams < 1:
            raise ValueError("streams must be >= 1")
        self.fixture = Path(fixture)
        self.streams = streams
        self.port = port
        self.fps = fps
        self.user = user
        self.password = password
        self._loop = None
        self._server = None

    def launch_line(self) -> str:
        """The media pipeline each path runs.

        ``multifilesrc ... loop=true`` on a single filename is what makes the stream
        endless: multifilesrc re-reads the same file forever, so the ten real frames repeat
        like a camera pointed at a slowly changing scene.

        Pacing is ``identity sync=true``, not the caps alone. ``h264parse`` gives each access
        unit a timestamp from the framerate in the caps, and ``identity sync=true`` holds a
        buffer until the pipeline clock reaches that timestamp — so the stream runs at ``fps``
        by the clock. Without it the packetiser pushes the file as fast as the socket accepts,
        and the first containerised RTSP measurement offered 170% of its target: twelve "5 fps"
        cameras delivered 101.8 img/s, limited only by how fast the client could decode.
        """
        return (
            f"( multifilesrc location={self.fixture} loop=true "
            f"caps=video/x-h264,framerate={self.fps}/1,stream-format=byte-stream,alignment=au "
            f"! h264parse config-interval=1 ! identity sync=true ! rtph264pay name=pay0 pt=96 )"
        )

    def start(self) -> None:
        """Mount every path and attach to the default main context. Does not block."""
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import GLib, Gst, GstRtspServer

        if not Gst.is_initialized():
            Gst.init(None)

        self._server = GstRtspServer.RTSPServer()
        self._server.set_service(str(self.port))
        mounts = self._server.get_mount_points()

        auth = None
        if self.user and self.password:
            auth = GstRtspServer.RTSPAuth()
            token = GstRtspServer.RTSPToken()
            token.set_string(GstRtspServer.RTSP_TOKEN_MEDIA_FACTORY_ROLE, "user")
            basic = GstRtspServer.RTSPAuth.make_basic(self.user, self.password)
            auth.add_basic(basic, token)
            self._server.set_auth(auth)

        for index in range(self.streams):
            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_launch(self.launch_line())
            # Shared, so ten clients on one path cost one pipeline. That is also what an IP
            # camera does, and it is what keeps 50 paths affordable.
            factory.set_shared(True)
            if auth is not None:
                permissions = GstRtspServer.RTSPPermissions()
                permissions.add_permission_for_role("user", "media.factory.access", True)
                permissions.add_permission_for_role("user", "media.factory.construct", True)
                factory.set_permissions(permissions)
            mounts.add_factory(f"/cam{index}", factory)

        if self._server.attach(None) == 0:
            raise RuntimeError(f"could not bind the RTSP server to port {self.port}")
        self._loop = GLib.MainLoop()

    def run_forever(self) -> None:
        """Start, then run the main loop until the process is killed."""
        if self._server is None:
            self.start()
        assert self._loop is not None
        self._loop.run()

    def uris(self, host: str = "127.0.0.1") -> list[str]:
        return [
            stream_uri(i, port=self.port, host=host, user=self.user, password=self.password)
            for i in range(self.streams)
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--streams", type=int, default=1, help="how many cameras to serve")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="directory of JPEGs")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="where to cache the encoded H.264 (default: <data>/../.rtsp/<name>.h264)",
    )
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--print-uris", action="store_true", help="print the URIs and exit")
    args = parser.parse_args(argv)

    fixture = args.fixture or (args.data.parent / ".rtsp" / f"{args.data.name}.h264")
    encode_fixture(args.data, fixture, fps=args.fps)

    server = RtspFixtureServer(
        fixture,
        streams=args.streams,
        port=args.port,
        fps=args.fps,
        user=args.user,
        password=args.password,
    )
    if args.print_uris:
        print("\n".join(server.uris()))
        return 0

    server.start()
    # Line-buffered and flushed: a test that waits for this line knows the socket is bound,
    # which is the alternative to sleeping for an arbitrary second and hoping.
    print(f"serving {args.streams} stream(s) on port {args.port}", flush=True)
    for uri in server.uris():
        print(f"  {uri}", flush=True)
    server.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
