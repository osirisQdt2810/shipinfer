"""Registry of request-trace sinks."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.core.tracing.base import TraceSink

__all__ = ["TRACE_SINKS"]

TRACE_SINKS: Registry[TraceSink] = Registry("trace sink", TraceSink)
