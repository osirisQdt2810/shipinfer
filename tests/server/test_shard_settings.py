"""A shard reads the whole fleet's configuration and keeps only its own slice.

Every shard is started from the *same* configuration. That is the point rather than a
convenience: a fleet described in two places is a fleet that can disagree with itself, and the
disagreement is silent — a camera in the plan and not in the config is simply never read, with
every shard reporting healthy.

So the split travels as one environment variable, ``SHIPINFER_SHARD_CAMERAS``, set by
:class:`~shipinfer.server.launcher.Fleet` beside ``CUDA_VISIBLE_DEVICES``, and
``build_settings`` narrows the fleet to it. Unset, it is the identity — which is what a
single-process run is, and why nothing had to change for one.
"""

from __future__ import annotations

import pytest

from shipinfer.cli.common import build_settings
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings.topology import SHARD_CAMERAS_ENV, SHARED_BY_ENV

FLEET = [
    {"camera_id": "quay-1", "uri": "rtsp://host/1"},
    {"camera_id": "quay-2", "uri": "rtsp://host/2"},
    {"camera_id": "quay-3", "uri": "rtsp://host/3"},
]


def settings_for(monkeypatch, shard: str | None):
    if shard is None:
        monkeypatch.delenv(SHARD_CAMERAS_ENV, raising=False)
    else:
        monkeypatch.setenv(SHARD_CAMERAS_ENV, shard)
    return build_settings(ingest={"cameras": FLEET})


def camera_ids(settings) -> list[str]:
    return [camera.camera_id for camera in settings.ingest.cameras]


class TestNarrowingToAShard:
    def test_only_the_named_cameras_survive(self, monkeypatch) -> None:
        settings = settings_for(monkeypatch, "quay-1,quay-3")

        assert camera_ids(settings) == ["quay-1", "quay-3"]

    def test_the_order_is_the_shards_not_the_configs(self, monkeypatch) -> None:
        """The plan sorts a shard's cameras, and that order is what its logs and its
        per-camera metrics are keyed by. Re-imposing the config's order here would make two
        shards' output orderings disagree for no reason."""
        settings = settings_for(monkeypatch, "quay-3,quay-1")

        assert camera_ids(settings) == ["quay-3", "quay-1"]

    def test_whitespace_is_tolerated(self, monkeypatch) -> None:
        """It is written by a launcher but read by hand often enough."""
        settings = settings_for(monkeypatch, " quay-1 , quay-2 ")

        assert camera_ids(settings) == ["quay-1", "quay-2"]

    def test_one_camera_is_a_legal_shard(self, monkeypatch) -> None:
        settings = settings_for(monkeypatch, "quay-2")

        assert camera_ids(settings) == ["quay-2"]

    def test_the_rest_of_the_settings_are_untouched(self, monkeypatch) -> None:
        """Narrowing the fleet must not quietly rebuild anything else."""
        whole = settings_for(monkeypatch, None)
        narrowed = settings_for(monkeypatch, "quay-1")

        assert narrowed.scheduler == whole.scheduler
        assert narrowed.devices == whole.devices
        assert narrowed.ingest.model_dump(exclude={"cameras"}) == whole.ingest.model_dump(
            exclude={"cameras"}
        )


class TestWithoutTheVariableNothingChanges:
    def test_an_unset_variable_serves_the_whole_fleet(self, monkeypatch) -> None:
        settings = settings_for(monkeypatch, None)

        assert camera_ids(settings) == ["quay-1", "quay-2", "quay-3"]

    def test_a_single_process_run_is_the_identity(self, monkeypatch) -> None:
        """The reason no existing caller had to change."""
        monkeypatch.delenv(SHARD_CAMERAS_ENV, raising=False)

        assert (
            build_settings(ingest={"cameras": FLEET}).ingest.cameras
            == build_settings(ingest={"cameras": FLEET}).ingest.cameras
        )


class TestADisagreementIsRefused:
    """The plan and the config are two views of one fleet."""

    def test_a_camera_the_config_does_not_define_is_an_error(self, monkeypatch) -> None:
        """Skipping it would leave that camera unread with nothing to say so — and the shard
        that would have read it is the only thing in the system able to notice."""
        with pytest.raises(ConfigurationError, match=r"\['quay-9'\]"):
            settings_for(monkeypatch, "quay-1,quay-9")

    def test_the_message_lists_what_is_available(self, monkeypatch) -> None:
        with pytest.raises(ConfigurationError, match="quay-1"):
            settings_for(monkeypatch, "quay-9")

    def test_an_empty_variable_is_refused_rather_than_read_as_the_whole_fleet(
        self, monkeypatch
    ) -> None:
        """Set-but-empty is a launcher bug, and reading it as "everything" would give every
        shard the whole fleet — N processes each decoding fifty cameras, which is the failure
        this design exists to prevent, arrived at silently."""
        with pytest.raises(ConfigurationError, match="still loads engines"):
            settings_for(monkeypatch, "")

    def test_a_variable_of_only_separators_is_refused_too(self, monkeypatch) -> None:
        with pytest.raises(ConfigurationError, match="still loads engines"):
            settings_for(monkeypatch, " , , ")


class TestTheSharingRidesBesideTheCameras:
    def test_shared_by_is_read_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(SHARED_BY_ENV, "[2, 1]")
        settings = settings_for(monkeypatch, None)
        assert settings.devices.shared_by == [2, 1]

    def test_unset_means_one_process_per_device(self, monkeypatch) -> None:
        monkeypatch.delenv(SHARED_BY_ENV, raising=False)
        assert settings_for(monkeypatch, None).devices.shared_by == []

    def test_a_zero_share_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv(SHARED_BY_ENV, "[0]")
        with pytest.raises(Exception, match="at least 1"):
            settings_for(monkeypatch, None)
