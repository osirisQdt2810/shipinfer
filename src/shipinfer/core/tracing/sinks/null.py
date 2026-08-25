"""The default: tracing off, at no cost.

Registered as ``none`` so ``observability.trace_sink`` reads the way an operator expects.
:meth:`should_record` is a constant ``False``, which is what keeps a server with tracing
disabled from building a :class:`RequestTrace` it would immediately throw away — the
completion path may not allocate per request.
"""

from __future__ import annotations

from typing import ClassVar

from shipinfer.core.tracing.base import RequestTrace, TraceSink
from shipinfer.core.tracing.registry import TRACE_SINKS

__all__ = ["NullTraceSink"]


@TRACE_SINKS.register("none", "null", "off", description="Tracing disabled (the default)")
class NullTraceSink(TraceSink):
    """Discards everything, and says so cheaply."""

    name: ClassVar[str] = "none"

    def should_record(self) -> bool:
        return False

    def _do_record(self, trace: RequestTrace) -> None:
        """Unreachable while :meth:`should_record` is False, and harmless if it is called."""
