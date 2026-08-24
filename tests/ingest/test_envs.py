"""The environment contract: one place, typed, and loud about a typo."""

from __future__ import annotations

import pytest

from shipinfer import envs
from shipinfer.core.errors import ConfigurationError
from shipinfer.ingest import SOURCES


class TestEnvironmentDefaults:
    """An unset variable takes its declared default, and says so."""

    def test_defaults_apply_when_nothing_is_set(self, monkeypatch):
        for var in envs.ALL:
            monkeypatch.delenv(var.name, raising=False)
        assert envs.INGEST_BACKEND.get() == "gstreamer"
        assert envs.INGEST_HWACCEL.get() is True
        assert envs.INGEST_RTSP_TRANSPORT.get() == "tcp"
        assert all(not var.is_set() for var in envs.ALL)

    def test_an_empty_value_is_not_an_override(self, monkeypatch):
        """``SHIPINFER_INGEST_BACKEND=`` in a compose file means "unset", not "invalid"."""
        monkeypatch.setenv("SHIPINFER_INGEST_BACKEND", "   ")
        assert envs.INGEST_BACKEND.get() == "gstreamer"
        assert envs.INGEST_BACKEND.is_set() is False

    def test_describe_reports_resolved_values(self, monkeypatch):
        monkeypatch.setenv("SHIPINFER_INGEST_BACKEND", "pyav")
        rows = {name: (value, was_set) for name, value, was_set, _ in envs.describe()}
        assert rows["SHIPINFER_INGEST_BACKEND"] == ("pyav", True)
        assert rows["SHIPINFER_INGEST_RTSP_TRANSPORT"] == ("tcp", False)
        assert all(
            doc for *_, doc in envs.describe()
        ), "every variable needs a documented reason"


class TestBooleanParsing:
    """`SHIPINFER_INGEST_HWACCEL` accepts what an operator would actually type."""

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "y"])
    def test_truthy_boolean_spellings(self, monkeypatch, raw):
        monkeypatch.setenv("SHIPINFER_INGEST_HWACCEL", raw)
        assert envs.INGEST_HWACCEL.get() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "n"])
    def test_falsy_boolean_spellings(self, monkeypatch, raw):
        monkeypatch.setenv("SHIPINFER_INGEST_HWACCEL", raw)
        assert envs.INGEST_HWACCEL.get() is False

    def test_a_bad_boolean_raises_rather_than_defaulting(self, monkeypatch):
        """The whole reason this module exists: a typo must not silently become the default."""
        monkeypatch.setenv("SHIPINFER_INGEST_HWACCEL", "maybe")
        with pytest.raises(ConfigurationError, match="SHIPINFER_INGEST_HWACCEL"):
            envs.INGEST_HWACCEL.get()


class TestClosedVocabularies:
    """A typo in a closed vocabulary fails loudly, listing the valid options."""

    def test_an_unknown_backend_names_the_valid_options(self, monkeypatch):
        monkeypatch.setenv("SHIPINFER_INGEST_BACKEND", "gstremaer")
        with pytest.raises(ConfigurationError) as excinfo:
            envs.INGEST_BACKEND.get()
        message = str(excinfo.value)
        assert "gstremaer" in message
        for option in envs.INGEST_BACKENDS:
            assert option in message

    def test_a_bad_number_raises(self, monkeypatch):
        monkeypatch.setenv("SHIPINFER_INGEST_LATENCY_MS", "soon")
        with pytest.raises(ConfigurationError, match="SHIPINFER_INGEST_LATENCY_MS"):
            envs.INGEST_LATENCY_MS.get()

        monkeypatch.setenv("SHIPINFER_INGEST_LATENCY_MS", "-5")
        with pytest.raises(ConfigurationError, match="positive"):
            envs.INGEST_LATENCY_MS.get()

    def test_declared_backends_match_the_registry(self):
        """The env var's choices are a literal (no upward import), so pin them to the registry.

        Without this, adding a fourth source would leave ``SHIPINFER_INGEST_BACKEND`` rejecting a
        name that is perfectly valid everywhere else.
        """
        assert set(envs.INGEST_BACKENDS) == set(SOURCES.names())
