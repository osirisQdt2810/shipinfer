"""Where perception events go — one implementation per file, selected by name.

``null`` is the default: a pipeline should start without a broker, and a deployment that has
not decided where results go is better off producing none loudly than failing to boot.
``jsonlines`` is what makes the whole DAG testable end to end with no Kafka and no camera.
``kafka`` is the production path, and the one that keeps the contract the tracking tier
already consumes.

Registration is eager because none of these modules imports a heavy dependency at import
time — ``confluent_kafka`` is imported inside the Kafka sink's constructor, so
``shipinfer registries`` lists every sink on a host that has none of them installed.

**Why this sits under ``topology`` and not under ``pipeline`` any more.** arch.md §9 says it
outright: ``sinks/{kafka,jsonlines,null}`` become ``output`` element implementations, and the
elements live here. The layering rule then decides the rest — ``topology`` may import ``core``
and nothing else, so an ``output`` element that reached into ``pipeline`` for a sink would be
the one edge ``check_layers.py`` refuses by name. What moved is the *transport*, not the
element: a :class:`~shipinfer.topology.sinks.base.ResultSink` still knows nothing about caps,
chains or items, and the ``output`` element that turns a
:class:`~shipinfer.topology.base.ChainItem` into an event and hands it over is a thin thing
on top of these, landing in the slice that follows this move. The move comes first and alone
because it is the one that edits the layer tables.

``shipinfer.pipeline.sinks`` re-exports every name below, so the previous generation's DAG
keeps running off the moved code for the coexistence arch.md §9 describes — the same shape
``pipeline/graph/tracking.py`` already uses for the moved tracker shard.

**``confluent_kafka`` is the one heavy dependency ``topology`` is allowed to name**, and the
allowance is deliberately narrow: :mod:`shipinfer.topology.sinks.kafka` imports it inside
``KafkaResultSink.__init__`` and nowhere else, exactly as ``shipvision`` is reached only
through the function-scope loaders in ``topology/bridge.py``. A static row in
``check_layers.py`` cannot tell those apart from a module-scope import — it walks the AST —
so what enforces the laziness is ``tests/test_architecture.py::TestImportIsCheap``, which
imports ``shipinfer.topology`` in a subprocess and fails if a broker client came with it.
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
