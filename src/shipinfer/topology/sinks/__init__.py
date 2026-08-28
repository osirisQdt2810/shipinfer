"""The result sinks: where a finished event goes — a file, a broker, or deliberately nowhere.

One implementation per module, registered in ``RESULT_SINKS`` (`registry.py`). The chain's
``output`` element and the old ``pipeline/`` graph publish through the same three classes,
which is why they live at this layer: ``topology`` may import them, ``pipeline`` re-exports
them. ``kafka.py`` imports its client inside ``__init__`` — the laziness is enforced by
``tests/test_architecture.py`` (reachability, ``sys.modules``, and the module-scope AST scan),
so a chain naming ``output: {impl: kafka}`` validates on a host that never had librdkafka.
"""

from __future__ import annotations

from shipinfer.topology.sinks.base import ResultSink
from shipinfer.topology.sinks.jsonlines import JsonLinesResultSink
from shipinfer.topology.sinks.kafka import KafkaResultSink
from shipinfer.topology.sinks.null import NullResultSink
from shipinfer.topology.sinks.registry import RESULT_SINKS

__all__ = [
    "RESULT_SINKS",
    "JsonLinesResultSink",
    "KafkaResultSink",
    "NullResultSink",
    "ResultSink",
]
