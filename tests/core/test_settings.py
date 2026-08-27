"""The settings tree validates what this build can honour — no more, no less.

CONVENTIONS 2.6: validate at start-up, not at first use. The first split PR relaxed a start-up
guard for an API the branch did not have, so a server could validate, load zero models, report
healthy and answer every request with `ModelNotFoundError` with no way to recover; that piece
refused `EXPLICIT` outright. This piece adds the endpoints, and the refusal becomes the
relaxation it was standing in for.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.enums import ModelControlMode


class TestExplicitModelControlIsHonouredNow:
    """The piece that introduced `ModelControlMode` refused `EXPLICIT`, because it had no
    endpoints to honour it. This piece adds `/v2/repository/models/*/load` and `*/unload`, so
    the refusal becomes the relaxation it was standing in for — each piece's settings say
    exactly what its server can do, and the review's repro flips from refused to accepted on
    purpose."""

    def test_the_reviewers_repro_now_validates(self) -> None:
        settings = ServerSettings(
            load_all_models=False, startup_models=[], model_control="explicit"
        )

        assert settings.model_control is ModelControlMode.EXPLICIT
        assert settings.startup_models == []

    def test_explicit_with_a_selection_validates_too(self) -> None:
        settings = ServerSettings(model_control="explicit", startup_models=["ship_detector"])

        assert settings.startup_models == ["ship_detector"]

    def test_the_default_mode_still_needs_a_selection_when_not_loading_all(self) -> None:
        """The relaxation is scoped to EXPLICIT. Under the default mode an empty start-up is
        still a server that can never be given a model, and is still refused."""
        with pytest.raises(ValidationError, match="non-empty startup_models"):
            ServerSettings(load_all_models=False, startup_models=[])

    def test_the_default_mode_validates(self) -> None:
        settings = ServerSettings()

        assert settings.model_control is ModelControlMode.NONE
        assert settings.load_all_models is True


class TestAServerThatLoadsNothingMustHaveSaidSo:
    """The guard `EXPLICIT` used to relax. It is unconditional now, because the relaxation
    belongs with the endpoints that make an empty start-up recoverable."""

    def test_no_models_and_no_selection_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="non-empty startup_models"):
            ServerSettings(load_all_models=False, startup_models=[])

    def test_a_named_selection_is_enough(self) -> None:
        settings = ServerSettings(load_all_models=False, startup_models=["ship_detector"])

        assert settings.startup_models == ["ship_detector"]

    def test_loading_everything_needs_no_selection(self) -> None:
        assert ServerSettings(load_all_models=True, startup_models=[]).startup_models == []


class TestTheRunnerSectionAndTheOneItReplaced:
    """`SHIPINFER_TOPOLOGY__*` is gone; a stale export must not be silently ignored.

    "Topology" means the element chain now (arch.md section 1), so the settings section that
    used to name process placement is `runner` and its environment prefix is
    `SHIPINFER_RUNNER__*` (documented in arch.md section 2). An operator who exported
    `SHIPINFER_TOPOLOGY__SHARDS=4` for the previous release and forgot it would otherwise get
    a fleet quietly running the default number of processes, with nothing anywhere saying the
    knob they set had stopped being read. `extra="forbid"` on the tree is what makes that a
    refusal at start-up with the key named, and this is what says it is not decoration.
    """

    def test_the_runner_section_is_environment_overridable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SHIPINFER_RUNNER__RUNNER", "inprocess")
        monkeypatch.setenv("SHIPINFER_RUNNER__SHARDS", "4")
        monkeypatch.setenv("SHIPINFER_RUNNER__DRAIN_S", "45")

        settings = ServerSettings()

        assert settings.runner.runner == "inprocess"
        assert settings.runner.shards == 4
        assert settings.runner.drain_s == 45.0

    def test_the_old_topology_prefix_fails_loudly_rather_than_being_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`extra="forbid"` does NOT catch this on its own — pydantic-settings' environment
        source only emits keys for fields that exist, so the model never sees the variable and
        nothing forbids it. `_refuse_retired_environment` is what makes it loud, and the
        message names the section that replaced it."""
        monkeypatch.setenv("SHIPINFER_TOPOLOGY__SHARDS", "4")

        with pytest.raises(ValidationError, match="SHIPINFER_TOPOLOGY__SHARDS") as excinfo:
            ServerSettings()

        assert "SHIPINFER_RUNNER__*" in str(excinfo.value)

    def test_the_bare_old_section_name_is_refused_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`SHIPINFER_TOPOLOGY=fleet` was how the placement class was chosen; it is as unread
        as the nested spelling, so it is refused the same way."""
        monkeypatch.setenv("SHIPINFER_TOPOLOGY", "fleet")

        with pytest.raises(ValidationError, match="SHIPINFER_TOPOLOGY"):
            ServerSettings()

    def test_a_section_that_still_exists_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard is a table of retired names, not a prefix blacklist."""
        monkeypatch.setenv("SHIPINFER_INGEST__INPUT_NAME", "images")

        assert ServerSettings().ingest.input_name == "images"

    def test_an_unknown_key_inside_the_runner_section_is_refused_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`extra="forbid"` is on the section as well as the tree, so `runner.kind` — what
        `runner.runner` used to be called — is refused rather than stored and never read."""
        monkeypatch.setenv("SHIPINFER_RUNNER__KIND", "fleet")

        with pytest.raises(ValidationError, match="kind"):
            ServerSettings()
