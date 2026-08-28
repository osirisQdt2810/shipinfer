"""Configuration and repository errors — everything wrong before a request exists."""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError

__all__ = [
    "ConfigurationError",
    "DuplicateCameraError",
    "ModelControlError",
    "ModelNotFoundError",
    "ModelVersionNotFoundError",
]


class ConfigurationError(ShipInferError):
    """A model config, server setting or instance group is self-contradictory."""


class DuplicateCameraError(ConfigurationError):
    """A camera with this id is already being read, so the add was refused.

    Split out of the plain :class:`ConfigurationError` its two raise sites used to share --
    :meth:`shipinfer.ingest.IngestManager.add_camera` and
    :meth:`shipinfer.runners.fleet.FleetRunner.add_camera` -- because one caller has to tell
    "that name is taken" from every *other* reason an add is refused. ``POST /streams`` with
    no ``camera_id`` mints one from a health report and acts on it later, so the name it
    picked can be taken in the window between; the router answers that by minting again
    against a fresh report (``api/streams.py``). Retrying on a bare ``ConfigurationError``
    made it do the whole add twice for an unrelated refusal -- an unregistered source, say --
    which is a second placement attempt and a second health report for a request that was
    always going to be a 400.

    Still a :class:`ConfigurationError`, so nothing that catches the base has to change and
    the HTTP mapping stays 400: the name will be taken on the next try too, and a control
    plane that retries it forever is the failure ``api/errors.py`` names.
    """


class ModelControlError(ShipInferError):
    """A load/unload request the server will not carry out.

    Two causes, and both are the *client's* mistake rather than a fault: asking to load a
    model on a server that was not started with explicit model control, and asking to
    unload a model an ensemble still composes. Distinct from
    :class:`~shipinfer.core.errors.inference.ServerStateError` because that one is
    retryable — a saturated pool recovers — and this one never is. Retrying it forever is
    how a control-plane script turns its own bug into a load on the server.
    """


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
