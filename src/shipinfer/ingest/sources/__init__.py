"""The decode backends: one file per implementation, registered on import.

Importing this package registers every source and imports **no** decode runtime: GStreamer,
PyAV and OpenCV are all loaded inside ``_do_open``. That is what lets the offline test tier
exercise the pipeline builders, the option builders and the whole camera actor on a host
with none of them installed, and it is why a misconfigured backend fails with
:class:`~shipinfer.core.errors.SourceUnavailableError` — naming the package to install —
rather than "unknown video source", which would be a lie.
"""

from shipinfer.ingest.sources.gstreamer import GStreamerSource, build_pipeline
from shipinfer.ingest.sources.pyav import PyAvSource, build_open_options
from shipinfer.ingest.sources.replay import ReplaySource

__all__ = [
    "GStreamerSource",
    "PyAvSource",
    "ReplaySource",
    "build_open_options",
    "build_pipeline",
]
