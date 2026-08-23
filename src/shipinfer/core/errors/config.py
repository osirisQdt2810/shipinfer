"""Configuration and repository errors — everything wrong before a request exists."""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError

__all__ = ["ConfigurationError", "ModelNotFoundError", "ModelVersionNotFoundError"]


class ConfigurationError(ShipInferError):
    """A model config, server setting or instance group is self-contradictory."""


class ModelNotFoundError(ShipInferError):
    """No model with this name is loaded."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        known = f"; loaded: {sorted(available)}" if available else ""
        super().__init__(f"model {name!r} is not loaded{known}")
        self.name = name


class ModelVersionNotFoundError(ShipInferError):
    """The model exists but not at the requested version."""

    def __init__(self, name: str, version: int, available: list[int] | None = None) -> None:
        known = f"; available: {sorted(available)}" if available else ""
        super().__init__(f"model {name!r} has no version {version}{known}")
        self.name = name
        self.version = version
