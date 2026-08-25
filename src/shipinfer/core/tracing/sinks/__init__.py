"""Trace sinks — one class per file, each registering itself with :data:`TRACE_SINKS`.

Importing this package is what makes the built-ins selectable by name.
"""

from shipinfer.core.tracing.sinks.jsonlines import JsonLinesTraceSink
from shipinfer.core.tracing.sinks.null import NullTraceSink

__all__ = ["JsonLinesTraceSink", "NullTraceSink"]
