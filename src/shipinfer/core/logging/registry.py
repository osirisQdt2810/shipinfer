"""Registry of log sinks."""

from __future__ import annotations

from shipinfer.core.logging.base import LogSink
from shipinfer.core.registry import Registry

__all__ = ["SINKS"]

SINKS: Registry[LogSink] = Registry("log sink", LogSink)
