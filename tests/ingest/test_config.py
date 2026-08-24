"""Camera configuration: validation, the inheritance chain, and the source switch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import Priority
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest import create_source
from shipinfer.ingest.resolve import (
    resolve_hwaccel,
    resolve_latency_ms,
    resolve_read_timeout_s,
    resolve_source_name,
    resolve_transport,
)
from shipinfer.ingest.sources.gstreamer import GStreamerSource
from shipinfer.ingest.sources.pyav import PyAvSource
from shipinfer.ingest.sources.replay import ReplaySource


class TestCameraConfigValidation:
    """A camera record is validated at construction, not at first frame."""

    def test_camera_id_must_be_usable_as_a_label(self):
        with pytest.raises(ValidationError):
            CameraConfig(camera_id="  ", uri="rtsp://x")
        with pytest.raises(ValidationError, match="whitespace"):
            CameraConfig(camera_id="cam 0", uri="rtsp://x")

    def test_scale_needs_both_dimensions(self):
        with pytest.raises(ValidationError, match="together"):
            CameraConfig(camera_id="cam0", uri="rtsp://x", width=1920)

    def test_unknown_camera_field_is_rejected(self):
        """``extra=forbid`` is what turns a mistyped key into a start-up failure."""
        with pytest.raises(ValidationError):
            CameraConfig(camera_id="cam0", uri="rtsp://x", hwaccell=True)

    def test_priority_rides_on_the_camera(self):
        camera = CameraConfig(camera_id="gate", uri="rtsp://x", priority=Priority.HIGH)
        assert camera.priority is Priority.HIGH


class TestIngestSettingsValidation:
    """The fleet section is part of the settings tree and validates as one."""

    def test_duplicate_camera_ids_are_rejected(self):
        with pytest.raises(ValidationError, match="duplicate"):
            IngestSettings(
                cameras=[
                    CameraConfig(camera_id="cam0", uri="rtsp://a"),
                    CameraConfig(camera_id="cam0", uri="rtsp://b"),
                ]
            )

    def test_reconnect_cap_must_not_be_below_the_floor(self):
        with pytest.raises(ValidationError, match="reconnect_max_ms"):
            IngestSettings(reconnect_initial_ms=5_000, reconnect_max_ms=1_000)

    def test_ingest_is_part_of_the_settings_tree(self):
        settings = ServerSettings(model_repository="model_repository")
        assert isinstance(settings.ingest, IngestSettings)
        assert settings.ingest.target_model == "ship_detector"

    def test_settings_tree_env_override_reaches_ingest(self, monkeypatch):
        monkeypatch.setenv("SHIPINFER_INGEST__TARGET_MODEL", "person_embedder")
        assert ServerSettings().ingest.target_model == "person_embedder"


class TestSettingsInheritance:
    """Camera beats fleet beats environment — one precedence rule, asserted once."""

    def test_camera_beats_settings_beats_environment(self, monkeypatch, make_camera):
        monkeypatch.setenv("SHIPINFER_INGEST_BACKEND", "pyav")
        monkeypatch.setenv("SHIPINFER_INGEST_HWACCEL", "0")
        monkeypatch.setenv("SHIPINFER_INGEST_RTSP_TRANSPORT", "udp")
        monkeypatch.setenv("SHIPINFER_INGEST_LATENCY_MS", "700")

        bare = make_camera()
        assert resolve_source_name(bare, None) == "pyav"
        assert resolve_hwaccel(bare, None) is False
        assert resolve_transport(bare, None) == "udp"
        assert resolve_latency_ms(bare, None) == 700

        fleet = IngestSettings(backend="replay", hwaccel=True, transport="tcp", latency_ms=300)
        assert resolve_source_name(bare, fleet) == "replay"
        assert resolve_hwaccel(bare, fleet) is True
        assert resolve_transport(bare, fleet) == "tcp"
        assert resolve_latency_ms(bare, fleet) == 300

        specific = make_camera(
            source="gstreamer", hwaccel=False, transport="udp", latency_ms=50
        )
        assert resolve_source_name(specific, fleet) == "gstreamer"
        assert resolve_hwaccel(specific, fleet) is False
        assert resolve_transport(specific, fleet) == "udp"
        assert resolve_latency_ms(specific, fleet) == 50

    def test_timeouts_fall_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("SHIPINFER_INGEST_READ_TIMEOUT_S", "1.5")
        assert resolve_read_timeout_s(None) == 1.5
        assert resolve_read_timeout_s(IngestSettings()) == 1.5
        assert resolve_read_timeout_s(IngestSettings(read_timeout_ms=250)) == 0.25


class TestBackendSelection:
    """The env-var switch picks a backend, and an unknown name is an error."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("gstreamer", GStreamerSource),
            ("gst", GStreamerSource),
            ("pyav", PyAvSource),
            ("av", PyAvSource),
            ("replay", ReplaySource),
            ("file", ReplaySource),
        ],
    )
    def test_every_registered_name_and_alias_resolves(self, make_camera, name, expected):
        assert isinstance(create_source(make_camera(source=name)), expected)

    def test_env_var_alone_selects_the_backend(self, monkeypatch, make_camera):
        monkeypatch.setenv("SHIPINFER_INGEST_BACKEND", "pyav")
        assert isinstance(create_source(make_camera()), PyAvSource)
        monkeypatch.setenv("SHIPINFER_INGEST_BACKEND", "gstreamer")
        assert isinstance(create_source(make_camera()), GStreamerSource)

    def test_an_unknown_backend_raises_and_lists_the_options(self, make_camera):
        with pytest.raises(ConfigurationError) as excinfo:
            create_source(make_camera(source="deepstream"))
        message = str(excinfo.value)
        assert "deepstream" in message
        assert "gstreamer" in message and "pyav" in message and "replay" in message

    def test_a_source_refuses_a_counter_from_another_camera(self, make_camera):
        from shipinfer.ingest import FrameCounter

        with pytest.raises(ConfigurationError, match="belongs to camera"):
            create_source(make_camera("cam0", source="replay"), FrameCounter("cam1"))
