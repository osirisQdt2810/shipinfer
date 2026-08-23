"""Log sinks — one per file, registered into :data:`shipinfer.core.logging.registry.SINKS`.

Importing this package registers every built-in sink.
"""

from shipinfer.core.logging.sinks.asynchronous import AsyncSink
from shipinfer.core.logging.sinks.file import RotatingFileSink
from shipinfer.core.logging.sinks.null import NullSink
from shipinfer.core.logging.sinks.stream import StreamSink

__all__ = ["AsyncSink", "NullSink", "RotatingFileSink", "StreamSink"]
