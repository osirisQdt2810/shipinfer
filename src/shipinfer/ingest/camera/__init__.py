"""One camera: the actor that runs it, the health it reports, the database it comes from.

The actor is the stateful half of the system (ADR-002): one thread, one stream, one decoder,
one frame counter, for its whole life. It performs no inference — it hands tagged frames to
the stateless GPU pool through a queue and never touches a device itself.
"""

from shipinfer.ingest.camera.actor import CameraActor, SourceFactory
from shipinfer.ingest.camera.db import load_camera_db, translate_reference_record
from shipinfer.ingest.camera.health import CameraHealth, CameraState, IngestSummary

__all__ = [
    "CameraActor",
    "CameraHealth",
    "CameraState",
    "IngestSummary",
    "SourceFactory",
    "load_camera_db",
    "translate_reference_record",
]
