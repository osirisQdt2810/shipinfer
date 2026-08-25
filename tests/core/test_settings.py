"""The settings tree refuses a configuration this build cannot honour.

CONVENTIONS 2.6: validate at start-up, not at first use. The case pinned here is the one review
found on the first split PR — a start-up guard relaxed for an API the branch did not have, so
a server could validate, load zero models, report healthy and answer every request with
`ModelNotFoundError` with no way to recover.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.enums import ModelControlMode


class TestAModeTheServerCannotHonourFailsLoudly:
    """`ModelControlMode` is vocabulary this piece introduces; the endpoints that give
    `EXPLICIT` a meaning arrive with the server piece. Until they do, the only honest answer to
    `model_control='explicit'` is a refusal that says why — not a silently widened rule."""

    def test_the_reviewers_repro_is_refused(self) -> None:
        """Verbatim from the review. Before the fix this validated, and the resulting server
        logged `ready: 0 model(s)`, reported healthy, and could never be given a model."""
        with pytest.raises(ValidationError, match="not honoured by this build"):
            ServerSettings(load_all_models=False, startup_models=[], model_control="explicit")

    def test_explicit_is_refused_even_with_models_to_load(self) -> None:
        """Not just the empty case. A mode with no endpoints behind it is wrong whatever else
        the configuration says, and refusing only the dangerous combination would let the
        setting look supported."""
        with pytest.raises(ValidationError, match="load or \\*/unload endpoints"):
            ServerSettings(model_control="explicit", startup_models=["ship_detector"])

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
