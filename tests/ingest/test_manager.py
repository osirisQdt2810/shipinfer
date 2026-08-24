"""The fleet manager: lifecycle, runtime add/remove, health, and the camera database."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from shipinfer.core.errors import CameraUnavailableError, ConfigurationError
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest import (
    BoundedSink,
    CameraState,
    IngestManager,
    IngestMetrics,
    load_camera_db,
)

from .conftest import FRAME_COUNT, synthetic_image

pytestmark = pytest.mark.timeout(20)

REFERENCE_DB = (
    Path(__file__).resolve().parents[2]
    / "references"
    / "bitbucket-subfaceid"
    / "config"
    / "cameradb.json"
)


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


class TestFleetLifecycle:
    """One actor per enabled camera, started and stopped exactly once."""

    def test_start_runs_one_actor_per_enabled_camera(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        factory, created = scripted_factory(script=[synthetic_image(0)])
        settings = fast_settings(
            cameras=[
                make_camera("cam0"),
                make_camera("cam1"),
                make_camera("cam2", enabled=False),
            ]
        )
        with IngestManager(sink, settings=settings, source_factory=factory) as manager:
            assert manager.camera_ids == ["cam0", "cam1"]
            assert "cam2" not in manager
            assert _wait_for(lambda: sink.depth >= 2)
        assert len(manager) == 0
        assert all(source.closes >= 1 for source in created)

    def test_start_and_stop_are_idempotent(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        factory, _ = scripted_factory(script=[synthetic_image(0)])
        settings = fast_settings(cameras=[make_camera("cam0")])
        manager = IngestManager(sink, settings=settings, source_factory=factory)
        manager.stop()  # before start
        manager.start()
        manager.start()
        assert len(manager) == 1
        manager.stop()
        manager.stop()
        assert len(manager) == 0


class TestRuntimeMembership:
    """Cameras are added and removed while the server runs."""

    def test_add_and_remove_a_camera_at_runtime(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        """The reference service exposed this over REST, and a fleet genuinely needs it."""
        factory, _ = scripted_factory(script=[synthetic_image(0)])
        manager = IngestManager(sink, settings=fast_settings(), source_factory=factory)
        manager.start()
        try:
            assert manager.camera_ids == []
            actor = manager.add_camera(make_camera("cam9"))
            assert manager.camera_ids == ["cam9"]
            assert _wait_for(lambda: actor.health.frames_read >= 1)

            manager.remove_camera("cam9")
            assert manager.camera_ids == []
            assert actor.is_running is False
            assert actor.state is CameraState.STOPPED
        finally:
            manager.stop()

    def test_adding_a_camera_twice_is_refused(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        factory, _ = scripted_factory(script=[None])
        manager = IngestManager(sink, settings=fast_settings(), source_factory=factory)
        try:
            manager.add_camera(make_camera("cam0"))
            with pytest.raises(ConfigurationError, match="already running"):
                manager.add_camera(make_camera("cam0"))
        finally:
            manager.stop()

    def test_removing_an_unknown_camera_names_what_is_running(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        factory, _ = scripted_factory(script=[None])
        manager = IngestManager(sink, settings=fast_settings(), source_factory=factory)
        try:
            manager.add_camera(make_camera("cam0"))
            with pytest.raises(ConfigurationError) as excinfo:
                manager.remove_camera("cam7")
            assert "cam7" in str(excinfo.value) and "cam0" in str(excinfo.value)
            with pytest.raises(ConfigurationError):
                manager.actor("cam7")
        finally:
            manager.stop()

    def test_re_adding_a_camera_can_continue_the_tag_sequence(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        """A camera that comes back must not reissue frame ids a tracker has already seen."""
        factory, _ = scripted_factory(
            script=[synthetic_image(i) for i in range(2)], finite=True
        )
        manager = IngestManager(sink, settings=fast_settings(), source_factory=factory)
        try:
            first = manager.add_camera(make_camera("cam0"))
            assert _wait_for(lambda: not first.is_running)
            manager.remove_camera("cam0")

            resumed = manager.add_camera(
                make_camera("cam0", first_frame_id=first.frame_counter.next_frame_id)
            )
            assert _wait_for(lambda: not resumed.is_running)
        finally:
            manager.stop()

        assert [f.frame_id for f in sink.drain()] == [0, 1, 2, 3]


class TestFleetHealth:
    """The fleet reports per-camera state, and start-up fails on a dead camera."""

    def test_health_and_summary_report_the_fleet(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        factory, _ = scripted_factory(script=[synthetic_image(0)])
        metrics = IngestMetrics()
        settings = fast_settings(cameras=[make_camera("cam0"), make_camera("cam1")])
        with IngestManager(
            sink, settings=settings, metrics=metrics, source_factory=factory
        ) as manager:
            assert _wait_for(lambda: manager.summary().streaming == 2)
            health = manager.health()
            summary = manager.summary()

            assert sorted(health) == ["cam0", "cam1"]
            assert summary.cameras == 2
            assert summary.is_healthy is True
            assert summary.frames_read >= 2
            assert set(summary.as_dict()) >= {"cameras", "streaming", "unhealthy", "total_fps"}
            assert metrics.cameras_total.value() == 2
            assert metrics.cameras_streaming.value() == 2

    def test_an_unhealthy_camera_shows_up_in_the_summary(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        factory, _ = scripted_factory(open_failures=1_000)
        metrics = IngestMetrics()
        settings = fast_settings(cameras=[make_camera("cam0")], failures_before_unhealthy=2)
        manager = IngestManager(
            sink, settings=settings, metrics=metrics, source_factory=factory
        )
        # A real sleep would make this test slow; the manager's own actors use time.sleep, so
        # keep the backoff tiny instead of injecting a clock through two layers.
        manager.start()
        try:
            assert _wait_for(lambda: manager.summary().unhealthy == 1, timeout_s=10.0)
            summary = manager.summary()
            assert summary.is_healthy is False
            assert summary.streaming == 0
            assert metrics.cameras_unhealthy.value() == 1
        finally:
            manager.stop()

    def test_wait_ready_returns_once_every_camera_has_delivered(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        factory, _ = scripted_factory(script=[synthetic_image(0)])
        settings = fast_settings(cameras=[make_camera("cam0"), make_camera("cam1")])
        with IngestManager(sink, settings=settings, source_factory=factory) as manager:
            manager.wait_ready(timeout_s=5.0)
            assert all(h.frames_read >= 1 for h in manager.health().values())

    def test_wait_ready_names_the_cameras_that_never_started(
        self, sink, fast_settings, scripted_factory, make_camera
    ):
        """A mistyped camera database must fail the deploy, not look healthy and detect nothing."""
        factory, _ = scripted_factory(script=[None])
        settings = fast_settings(cameras=[make_camera("cam_dead")])
        with IngestManager(sink, settings=settings, source_factory=factory) as manager:
            with pytest.raises(CameraUnavailableError) as excinfo:
                manager.wait_ready(timeout_s=0.05)
            assert excinfo.value.camera_ids == ["cam_dead"]
            assert "cam_dead" in str(excinfo.value)


class TestCameraDatabase:
    """The existing fleet database loads, in the reference shape and in ours."""

    def test_configured_cameras_merges_inline_and_file(
        self, tmp_path, sink, fast_settings, make_camera
    ):
        database = tmp_path / "cameras.json"
        database.write_text(
            json.dumps({"cameras": [{"camera_id": "cam_file", "uri": "rtsp://b/stream"}]})
        )
        settings = fast_settings(cameras=[make_camera("cam_inline")], camera_db=database)
        manager = IngestManager(sink, settings=settings)
        assert [c.camera_id for c in manager.configured_cameras()] == ["cam_inline", "cam_file"]

    def test_a_camera_declared_twice_across_sources_is_refused(
        self, tmp_path, sink, fast_settings, make_camera
    ):
        database = tmp_path / "cameras.json"
        database.write_text(json.dumps([{"camera_id": "cam0", "uri": "rtsp://b/stream"}]))
        settings = fast_settings(cameras=[make_camera("cam0")], camera_db=database)
        with pytest.raises(ConfigurationError, match="declared both inline"):
            IngestManager(sink, settings=settings).configured_cameras()

    def test_the_reference_camera_database_shape_translates(self, tmp_path):
        """`cameradb.json` plus `gstconfig.ini`, exactly as the previous generation shipped them."""
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "gstconfig.ini").write_text(
            "[source0]\nlatency=300\nprotocols=7\n"
        )
        database = tmp_path / "config" / "cameradb.json"
        database.write_text(
            json.dumps(
                {
                    "contents": [
                        {
                            "camIP": "10.0.0.100",
                            "cameraHeight": 2160,
                            "cameraID": "104100",
                            "cameraWidth": 3840,
                            "codecType": "GST_NVV4l2",
                            "configFile": "gstconfig.ini",
                            "sourceType": "RTSP_SOURCE",
                            "streamType": "MAIN_STREAM",
                            # Shape-faithful, credential invented. The reference `cameradb.json` ships a real
                            # fleet password inline; copying it here would have put it in git history.
                            "videoSource": "rtsp://operator:REDACTED%40@10.0.0.100",
                        },
                        {
                            "cameraID": "clip1",
                            "sourceType": "VIDEO",
                            "codecType": "avdec",
                            "videoSource": "/data/clip.mp4",
                        },
                    ]
                }
            )
        )

        cameras = load_camera_db(database)
        assert [c.camera_id for c in cameras] == ["104100", "clip1"]

        rtsp = cameras[0]
        assert rtsp.uri == "rtsp://operator:REDACTED%40@10.0.0.100"
        assert rtsp.source is None, "which RTSP backend to use is a deployment decision"
        assert (rtsp.width, rtsp.height) == (3840, 2160)
        assert rtsp.hwaccel is True, "GST_NVV4l2 means the hardware decoder"
        assert rtsp.latency_ms == 300
        assert (
            rtsp.transport == "auto"
        ), "protocols=7 is UDP|UDP_MCAST|TCP, i.e. rtspsrc decides"

        video = cameras[1]
        assert video.source == "replay"
        assert video.hwaccel is False

    def test_the_native_and_reference_shapes_can_be_mixed(self, tmp_path):
        database = tmp_path / "mixed.json"
        database.write_text(
            json.dumps(
                [
                    {"cameraID": "old", "videoSource": "rtsp://a/s"},
                    {"camera_id": "new", "uri": "rtsp://b/s", "codec": "h265"},
                ]
            )
        )
        cameras = load_camera_db(database)
        assert [c.camera_id for c in cameras] == ["old", "new"]
        assert cameras[1].codec == "h265"

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ('{"contents": {}}', "must hold a list"),
            ('[{"cameraID": "x"}]', "videoSource"),
            (
                '[{"cameraID": "a", "videoSource": "rtsp://a", "sourceType": "MAGIC"}]',
                "sourceType",
            ),
            ('[{"camera_id": "a"}]', "invalid"),
            ("[1]", "not an object"),
            (
                '[{"cameraID": "a", "videoSource": "rtsp://a"},'
                ' {"cameraID": "a", "videoSource": "rtsp://b"}]',
                "more than once",
            ),
        ],
    )
    def test_a_broken_camera_database_fails_at_load(self, tmp_path, payload, match):
        database = tmp_path / "broken.json"
        database.write_text(payload)
        with pytest.raises(ConfigurationError, match=match):
            load_camera_db(database)

    def test_a_missing_or_unparseable_database_is_a_typed_error(self, tmp_path):
        with pytest.raises(ConfigurationError, match="does not exist"):
            load_camera_db(tmp_path / "absent.json")
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(ConfigurationError, match="not valid JSON"):
            load_camera_db(bad)

    @pytest.mark.skipif(
        not REFERENCE_DB.exists(), reason="the reference checkout is not present"
    )
    def test_the_real_reference_database_loads(self):
        """Evidence rather than a claim: the actual production fleet file, translated."""
        cameras = load_camera_db(REFERENCE_DB)
        assert len(cameras) >= 10
        assert all(camera.uri.startswith("rtsp://") for camera in cameras)
        assert all(camera.hwaccel is True for camera in cameras)
        assert len({camera.camera_id for camera in cameras}) == len(cameras)


class TestIngestPlaneEndToEnd:
    """The whole plane, real threads and real sources, with no hardware present."""

    def test_the_manager_runs_replay_cameras_with_no_camera_present(
        self, sink, fast_settings, frame_dir
    ):
        """The 50-camera stress test in miniature: real sources, real threads, no hardware."""
        cameras = [
            CameraConfig(
                camera_id=f"cam{index}",
                uri=str(frame_dir),
                source="replay",
                loop=False,
                fps=0.0,
            )
            for index in range(4)
        ]
        settings = IngestSettings(cameras=cameras, empty_read_sleep_ms=0)
        sink = BoundedSink(capacity=FRAME_COUNT * len(cameras), name="ingest")
        with IngestManager(sink, settings=settings) as manager:
            assert _wait_for(
                lambda: all(
                    h.state is CameraState.EXHAUSTED for h in manager.health().values()
                ),
                timeout_s=10.0,
            )
            summary = manager.summary()

        assert summary.frames_read == FRAME_COUNT * len(cameras)
        assert summary.frames_dropped == 0
        assert sink.counts() == {f"cam{i}": FRAME_COUNT for i in range(4)}
