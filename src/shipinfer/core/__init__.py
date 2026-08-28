"""Pure core: types, errors, requests, settings, logging, metrics, the registry primitive.

Nothing in this package imports a GPU runtime, a network client, a model file or the
compiled ``shipinfer._C`` extension. That is enforced by ``scripts/hooks/check_layers.py``
and by ``tests/test_architecture.py``, and it is the reason the scheduler's behaviour is
testable on a laptop with no NVIDIA driver (ADR-001).

Every sub-package here is a package rather than a module because each is an extension
point that will grow: sinks, exporters, settings sections, error domains.
"""

from shipinfer.core.errors import (
    BackendLoadError,
    BackendUnavailableError,
    ConfigurationError,
    DeviceError,
    DeviceOutOfMemoryError,
    InferenceError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    QueueFullError,
    RequestCancelledError,
    RequestTimeoutError,
    ServerStateError,
    ShipInferError,
    ValidationError,
)
from shipinfer.core.logging import LOG
from shipinfer.core.logging import configure as configure_logging
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.metrics import MetricsRegistry, ServerMetrics
from shipinfer.core.registry import Registry
from shipinfer.core.request import (
    InferenceRequest,
    InferenceResponse,
    Priority,
    RequestContext,
    ResponseFuture,
    Timings,
)
from shipinfer.core.settings import (
    DeviceSettings,
    ExecutionProvider,
    ExecutionSettings,
    HttpSettings,
    MemorySettings,
    ObservabilitySettings,
    OverflowPolicy,
    SchedulerSettings,
    ServerSettings,
)
from shipinfer.core.types import (
    DYNAMIC,
    DataType,
    Device,
    MemoryHandle,
    MemoryKind,
    Tensor,
    TensorSpec,
    stack_tensors,
    validate_against,
)

__all__ = [
    "DYNAMIC",
    "LOG",
    "BackendLoadError",
    "BackendUnavailableError",
    "ConfigurationError",
    "DataType",
    "Device",
    "DeviceError",
    "DeviceOutOfMemoryError",
    "DeviceSettings",
    "ExecutionProvider",
    "ExecutionSettings",
    "HttpSettings",
    "InferenceError",
    "InferenceRequest",
    "InferenceResponse",
    "MemoryHandle",
    "MemoryKind",
    "MemorySettings",
    "MetricsRegistry",
    "ModelNotFoundError",
    "ModelVersionNotFoundError",
    "ObservabilitySettings",
    "OverflowPolicy",
    "Priority",
    "QueueFullError",
    "Registry",
    "RequestCancelledError",
    "RequestContext",
    "RequestTimeoutError",
    "ResponseFuture",
    "SchedulerSettings",
    "ServerMetrics",
    "ServerSettings",
    "ServerStateError",
    "ShipInferError",
    "Tensor",
    "TensorSpec",
    "Timings",
    "ValidationError",
    "configure_logging",
    "get_logger",
    "log_context",
    "stack_tensors",
    "validate_against",
]
