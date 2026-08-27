"""The typed failure vocabulary, grouped by the layer that raises it.

Split by domain rather than kept flat so that a new subsystem adds a module here instead
of appending to a 300-line file every layer has to read.
"""

from shipinfer.core.errors.backend import BackendLoadError, BackendUnavailableError
from shipinfer.core.errors.base import ShipInferError
from shipinfer.core.errors.chain import (
    CapsMismatchError,
    CapsSyntaxError,
    ChainCycleError,
    ChainSpecError,
    ChainStructureError,
    ConditionSyntaxError,
    TopologyError,
    UnknownElementError,
    UnknownElementImplError,
    UnknownElementKindError,
)
from shipinfer.core.errors.config import (
    ConfigurationError,
    ModelControlError,
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
from shipinfer.core.errors.ingest import (
    CameraUnavailableError,
    FrameDecodeError,
    IngestError,
    SourceOpenError,
    SourceUnavailableError,
)
from shipinfer.core.errors.sink import SinkDeliveryError
from shipinfer.core.errors.topology import (
    PeerLostError,
    RingClosedError,
    RingFullError,
    RingProtocolError,
    ShardExitedError,
    WireRefusedError,
)
from shipinfer.core.errors.tracking import TrackingError

__all__ = [
    "BackendLoadError",
    "BackendUnavailableError",
    "CameraUnavailableError",
    "CapsMismatchError",
    "CapsSyntaxError",
    "ChainCycleError",
    "ChainSpecError",
    "ChainStructureError",
    "ConditionSyntaxError",
    "ConfigurationError",
    "DeviceError",
    "DeviceOutOfMemoryError",
    "FrameDecodeError",
    "InferenceError",
    "IngestError",
    "ModelControlError",
    "ModelNotFoundError",
    "ModelVersionNotFoundError",
    "PeerLostError",
    "QueueFullError",
    "RequestCancelledError",
    "RequestTimeoutError",
    "RingClosedError",
    "RingFullError",
    "RingProtocolError",
    "ServerStateError",
    "ShardExitedError",
    "ShipInferError",
    "SinkDeliveryError",
    "SourceOpenError",
    "SourceUnavailableError",
    "TopologyError",
    "TrackingError",
    "UnknownElementError",
    "UnknownElementImplError",
    "UnknownElementKindError",
    "ValidationError",
    "WireRefusedError",
]
