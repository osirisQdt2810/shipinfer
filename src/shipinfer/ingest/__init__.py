"""The ingest plane: fifty cameras in, tagged frames out. No inference happens here.

One stateful actor per camera (RTSP pull, hardware decode, monotonic ``frame_id``,
reconnect schedule) feeds the stateless GPU pool: that state cannot be pooled.
Layout: ``base`` (FrameSource contract), ``registry`` (factory), ``resolve`` (config
precedence), ``sink`` (FrameSink protocol), ``manager`` (fleet control), ``frame/``,
``timing/``, ``sources/`` (one backend per file), ``camera/`` (the actor).

Invariants:
- No queue and no scheduler here: frames go to a FrameSink; ``pipeline`` supplies the
  fair, bounded, per-camera-lane one that fixed the shared-buffer starvation (ADR-005).
- A frame is tagged at decode and never retagged; reassembly keys on
  ``(camera_id, frame_id)``, not arrival order (ADR-002).
- Importing pulls in no decode runtime; backends load in ``_do_open`` (offline tier).
"""

from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.camera import (
    CameraActor,
    CameraHealth,
    CameraState,
    IngestSummary,
    SourceFactory,
    load_camera_db,
)
from shipinfer.ingest.frame import Frame, FrameCounter
from shipinfer.ingest.manager import IngestManager, configured_cameras
from shipinfer.ingest.metrics import IngestMetrics
from shipinfer.ingest.registry import SOURCES, create_source
from shipinfer.ingest.resolve import (
    resolve_hwaccel,
    resolve_latency_ms,
    resolve_source_name,
    resolve_transport,
)

# Importing the backends is what registers them. Each is import-safe by design: none of
# them imports a decode runtime at module level, so this costs nothing on a host that has
# none of them.
from shipinfer.ingest.sink import BoundedSink, CountingSink, FrameSink
from shipinfer.ingest.sources import GStreamerSource, PyAvSource, ReplaySource
from shipinfer.ingest.timing import DeadlinePacer, ExponentialBackoff

__all__ = [
    "SOURCES",
    "BoundedSink",
    "CameraActor",
    "CameraHealth",
    "CameraState",
    "CountingSink",
    "DeadlinePacer",
    "ExponentialBackoff",
    "Frame",
    "FrameCounter",
    "FrameSink",
    "FrameSource",
    "GStreamerSource",
    "IngestManager",
    "IngestMetrics",
    "IngestSummary",
    "PyAvSource",
    "ReplaySource",
    "SourceFactory",
    "configured_cameras",
    "create_source",
    "load_camera_db",
    "resolve_hwaccel",
    "resolve_latency_ms",
    "resolve_source_name",
    "resolve_transport",
]
