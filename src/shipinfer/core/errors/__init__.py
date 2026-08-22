"""The typed failure vocabulary, grouped by the layer that raises it.

Split by domain rather than kept flat so that a new subsystem adds a module here instead
of appending to a 300-line file every layer has to read.
"""

from shipinfer.core.errors.backend import BackendLoadError, BackendUnavailableError
from shipinfer.core.errors.base import ShipInferError
from shipinfer.core.errors.config import (
    ConfigurationError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
)
from shipinfer.core.errors.device import DeviceError, DeviceOutOfMemoryError
from shipinfer.core.errors.inference import (
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    RequestTimeoutError,
    ServerStateError,
    ValidationError,
)

__all__ = [
    "BackendLoadError",
    "BackendUnavailableError",
    "ConfigurationError",
    "DeviceError",
    "DeviceOutOfMemoryError",
    "InferenceError",
    "ModelNotFoundError",
    "ModelVersionNotFoundError",
    "QueueFullError",
    "RequestCancelledError",
    "RequestTimeoutError",
    "ServerStateError",
    "ShipInferError",
    "ValidationError",
]
