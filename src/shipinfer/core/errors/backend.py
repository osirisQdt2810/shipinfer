"""Backend errors: the runtime is missing, or it refused this model."""

from __future__ import annotations

from shipinfer.core.errors.base import ShipInferError

__all__ = ["BackendLoadError", "BackendUnavailableError"]


class BackendUnavailableError(ShipInferError):
    """The platform is known but its runtime is not installed on this host.

    Deliberately distinct from :class:`BackendLoadError`: this one is fixed by installing
    an extra (``pip install "shipinfer[tensorrt]"``), that one by fixing the model.
    """


class BackendLoadError(ShipInferError):
    """The runtime is present but refused to load this particular model."""
