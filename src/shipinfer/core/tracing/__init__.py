"""Request tracing: Triton's seven named timestamps, and a sink that writes them.

``pytest`` note for anyone extending this: importing this package registers the built-in
sinks, which is the only import-time side effect allowed by the conventions.
"""

from shipinfer.core.tracing.base import TRACE_EVENTS, RequestTrace, TraceSink
from shipinfer.core.tracing.registry import TRACE_SINKS
from shipinfer.core.tracing.sinks import JsonLinesTraceSink, NullTraceSink

__all__ = [
    "TRACE_EVENTS",
    "TRACE_SINKS",
    "JsonLinesTraceSink",
    "NullTraceSink",
    "RequestTrace",
    "TraceSink",
    "build_trace_sink",
]


def build_trace_sink(name: str, **options: object) -> TraceSink:
    """Instantiate a trace sink by registered name.

    Args:
        name: e.g. ``"jsonlines"``. Aliases resolve too; ``"none"`` is the default and
            costs nothing.
        **options: forwarded to the sink's constructor. An unknown keyword raises from the
            constructor rather than being dropped, so a typo in a settings file fails at
            start-up instead of producing a server that quietly traces nothing.
    """
    return TRACE_SINKS.create(name, **options)
