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
