"""The registry primitive every extension point is built on."""

from __future__ import annotations

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.registry import Registry


class Base:
    pass


def test_register_and_create() -> None:
    registry: Registry[Base] = Registry("thing", Base)

    @registry.register("alpha", "a", description="the first one")
    class Alpha(Base):
        pass

    assert registry.get("alpha") is Alpha
    assert registry.get("a") is Alpha
    assert isinstance(registry.create("alpha"), Alpha)
    assert ("alpha", "the first one") in registry.describe()


def test_description_defaults_to_the_docstring_summary() -> None:
    registry: Registry[Base] = Registry("thing", Base)

    @registry.register("beta")
    class Beta(Base):
        """Does the beta thing.

        And a longer paragraph nobody wants in a table.
        """

    assert dict(registry.describe())["beta"] == "Does the beta thing."


def test_registration_enforces_the_base_class() -> None:
    """Duck typing failures should surface at import, not at request time."""
    registry: Registry[Base] = Registry("thing", Base)

    with pytest.raises(TypeError, match="does not subclass"):

        @registry.register("gamma")
        class Gamma:  # not a Base
            pass


def test_duplicate_names_are_refused() -> None:
    registry: Registry[Base] = Registry("thing", Base)

    @registry.register("dup")
    class One(Base):
        pass

    with pytest.raises(ConfigurationError, match="already registered"):

        @registry.register("dup")
        class Two(Base):
            pass


def test_unknown_name_lists_the_alternatives() -> None:
    registry: Registry[Base] = Registry("thing", Base)
    registry.register("known")(type("Known", (Base,), {}))

    with pytest.raises(ConfigurationError) as excinfo:
        registry.get("unknown")
    assert "known" in str(excinfo.value)


def test_lazy_registration_defers_the_import() -> None:
    """A registry listing a TensorRT backend must not import TensorRT to list it."""
    registry: Registry[Base] = Registry("thing", Base)
    registry.register_lazy("late", "shipinfer.core.registry:Registry")

    entry = registry.entry("late")
    assert entry.is_loaded is False
    assert registry.names() == ["late"]
    assert entry.resolve() is Registry
    assert entry.is_loaded is True


def test_lazy_registration_reports_a_missing_module_usefully() -> None:
    registry: Registry[Base] = Registry("thing", Base)
    registry.register_lazy("absent", "shipinfer.does_not_exist:Thing")
    with pytest.raises(ConfigurationError, match="optional extra"):
        registry.get("absent")
