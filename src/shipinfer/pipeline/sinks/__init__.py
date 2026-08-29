"""Re-export of the result sinks, which now live under ``topology``.

The family moved to :mod:`shipinfer.topology.sinks` when the ``output`` element arrived:
arch.md §9 says ``sinks/{kafka,jsonlines,null}`` become ``output`` element
implementations, and ``topology`` may import ``core`` and nothing else, so an element could
not have reached them here. This package stays because ``pipeline/`` remains the working application until the
chain has replaced it (arch.md §9 again) and its runner, its DeepStream path and their tests
all name the sinks through this path.

Nothing is redefined and nothing is re-registered: ``RESULT_SINKS`` below *is* the registry
under ``topology``, so a sink resolved by name from either import path is the same class.
New code should import :mod:`shipinfer.topology.sinks` directly.
"""

from __future__ import annotations

from shipinfer.topology.sinks import (
    RESULT_SINKS,
    JsonLinesResultSink,
    KafkaResultSink,
    NullResultSink,
    ResultSink,
)

__all__ = [
    "RESULT_SINKS",
    "JsonLinesResultSink",
    "KafkaResultSink",
    "NullResultSink",
    "ResultSink",
]
