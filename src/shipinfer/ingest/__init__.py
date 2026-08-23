"""The ingest plane: fifty cameras in, tagged frames out. No inference happens here.

This is PLANE 1 of the architecture in
``references/bitbucket-subfaceid/docs/new-system-architecture.md``: one **stateful** actor
per camera, owning the RTSP pull, the hardware decode, a monotonic ``frame_id`` counter and
the reconnect schedule, feeding the **stateless** GPU pool through a queue. Fifty cameras at
20 fps is 1000 frames a second, and the reason ingest gets its own plane is that none of
that state can be pooled: a decoder belongs to one stream, and a frame counter belongs to
one camera, forever.

Layout — each directory has one reason to exist::

    base.py       the FrameSource contract (open / read / close), nothing else
    registry.py   SOURCES, and the factory that picks a backend for a camera
    resolve.py    camera -> settings -> environment precedence, in one place
    metrics.py    the fleet's metric handles, labelled by camera
    manager.py    IngestManager: start/stop/add/remove, and fleet health
    frame/        what flows: the Frame value and the FrameCounter that stamps it
    timing/       when: the reconnect backoff and the fps pacer, as testable policies
    sources/      how: one decode backend per file, registered on import
    camera/       who: the per-camera actor, its health vocabulary, its database

Three properties are worth knowing before changing anything here:

**Nothing in this package owns a queue.** Frames are published into
:mod:`shipinfer.scheduling.queues`, which already implements per-camera lanes and, when it
must shed load, drops the oldest request of the *greediest* camera. The system this replaces
funnelled every camera into one 1000-slot buffer that evicted the globally oldest entry, so
a crowded camera silently starved a quiet one (ADR-005). A queue written here would bring
that back.

**Every frame is tagged at the moment of decode** and the tag is never rewritten. Batching,
spillover between GPUs and out-of-order completion are all safe because reassembly keys on
``(camera_id, frame_id)`` rather than on arrival order (ADR-002).

**Importing this package pulls in no decode runtime.** GStreamer, PyAV and OpenCV are loaded
inside ``_do_open``, so the offline test tier — pipeline strings, AVOptions, the actor's
reconnect policy, the fair queue's behaviour under overload — runs with none of them
installed and no camera anywhere.
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
from shipinfer.ingest.manager import IngestManager
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
from shipinfer.ingest.sources import GStreamerSource, PyAvSource, ReplaySource
from shipinfer.ingest.timing import DeadlinePacer, ExponentialBackoff

__all__ = [
    "SOURCES",
    "CameraActor",
    "CameraHealth",
    "CameraState",
    "DeadlinePacer",
    "ExponentialBackoff",
    "Frame",
    "FrameCounter",
    "FrameSource",
    "GStreamerSource",
    "IngestManager",
    "IngestMetrics",
    "IngestSummary",
    "PyAvSource",
    "ReplaySource",
    "SourceFactory",
    "create_source",
    "load_camera_db",
    "resolve_hwaccel",
    "resolve_latency_ms",
    "resolve_source_name",
    "resolve_transport",
]
