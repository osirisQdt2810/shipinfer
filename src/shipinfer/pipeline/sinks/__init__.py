"""Where perception events go — one implementation per file, selected by name.

``null`` is the default: a pipeline should start without a broker, and a deployment that has
not decided where results go is better off producing none loudly than failing to boot.
``jsonlines`` is what makes the whole DAG testable end to end with no Kafka and no camera.
``kafka`` is the production path, and the one that keeps the contract the tracking tier
already consumes.

Registration is eager because none of these modules imports a heavy dependency at import
time — ``confluent_kafka`` is imported inside the Kafka sink's constructor, so
``shipinfer registries`` lists every sink on a host that has none of them installed.
"""

from shipinfer.pipeline.sinks.base import ResultSink
from shipinfer.pipeline.sinks.jsonlines import JsonLinesResultSink
from shipinfer.pipeline.sinks.kafka import KafkaResultSink
from shipinfer.pipeline.sinks.null import NullResultSink
from shipinfer.pipeline.sinks.registry import RESULT_SINKS

__all__ = [
    "RESULT_SINKS",
    "JsonLinesResultSink",
    "KafkaResultSink",
    "NullResultSink",
    "ResultSink",
]
