"""Registry of result sinks."""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.pipeline.sinks.base import ResultSink

__all__ = ["RESULT_SINKS"]

RESULT_SINKS: Registry[ResultSink] = Registry("result sink", ResultSink)
