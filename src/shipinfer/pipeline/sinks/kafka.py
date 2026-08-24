"""Publish to Kafka — the bus the tracking tier already listens on.

This is PLANE 3's feed in ``references/bitbucket-subfaceid/docs/new-system-architecture.md``:
perception publishes small results (boxes, embeddings, ship ids) and the stateful services
(``motservice`` per camera, ``mtmcservice`` globally) consume them. Frames never go on the
bus — they are megabytes and they stay in shared memory; this carries metadata only, which is
why the segmenter's masks are reduced to an area before they reach here.

Two decisions are worth reading before changing anything.

**The message key is the camera id.** Kafka guarantees order within a partition, so keying on
the camera puts one camera's frames in one partition, in order. A per-camera tracker is
stateful and consumes them in sequence; keying on the frame id, or not keying at all, would
spread one camera's timeline across partitions and hand the tracker its frames out of order —
which looks exactly like a tracking bug and is not one.

**``confluent_kafka`` is imported inside the constructor.** Nothing at import time, so
``shipinfer registries`` lists this sink on a host that has never had librdkafka, and a
deployment that asks for it without installing it fails at **start-up** with the install
command in the message rather than on the first frame.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from shipinfer.core.errors import BackendUnavailableError, ConfigurationError
from shipinfer.core.logging import get_logger
from shipinfer.pipeline.schema import PerceptionEvent
from shipinfer.pipeline.sinks.base import ResultSink
from shipinfer.pipeline.sinks.registry import RESULT_SINKS

__all__ = ["KafkaResultSink"]

_LOG = get_logger("pipeline.sinks.kafka")


@RESULT_SINKS.register("kafka", "broker")
class KafkaResultSink(ResultSink):
    """Produces one message per frame onto one topic.

    Args:
        topic: where events go. The reference deployment used one topic per message type and
            switched on the payload's ``type`` field, which v2 keeps.
        brokers: ``host:port`` list, comma separated — ``bootstrap.servers``.
        legacy: publish the v1 ``Det2MOT`` payload instead of v2. For a consumer that
            validates its input strictly and has not been rebuilt yet; the v2 payload is
            additive, so this should not normally be needed.
        config: extra librdkafka settings, merged last so a deployment can tune
            ``linger.ms``, ``compression.type`` or SASL without a code change.

    Raises:
        BackendUnavailableError: ``confluent_kafka`` is not installed. Distinct from a
            configuration error on purpose: one is fixed by ``pip install
            "shipinfer[kafka]"``, the other by fixing the broker list.
    """

    name: ClassVar[str] = "kafka"

    def __init__(
        self,
        *,
        topic: str = "perception.results",
        brokers: str = "localhost:9092",
        legacy: bool = False,
        queue_buffering_max_messages: int = 100_000,
        linger_ms: int = 20,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not topic.strip():
            raise ConfigurationError("kafka sink needs a topic")
        if not brokers.strip():
            raise ConfigurationError("kafka sink needs at least one broker")
        self.topic = topic.strip()
        self.brokers = brokers.strip()
        self.legacy = legacy
        try:
            from confluent_kafka import Producer
        except ImportError as exc:
            raise BackendUnavailableError(
                f"the kafka result sink needs confluent-kafka ({exc}). "
                'Install it with `pip install "shipinfer[kafka]"`'
            ) from exc
        settings: dict[str, Any] = {
            "bootstrap.servers": self.brokers,
            # Bound the producer's own queue. Unbounded buffering turns a broker outage into
            # host memory exhaustion, which takes the perception tier down with it; a bounded
            # queue turns it into a counted publish failure instead.
            "queue.buffering.max.messages": queue_buffering_max_messages,
            # Batch on the producer's thread rather than the pipeline's. 20 ms is well inside
            # a 50 ms frame period and is what makes emission cost a memcpy here.
            "linger.ms": linger_ms,
        }
        settings.update(config or {})
        self._producer = Producer(settings)
        _LOG.info("kafka result sink -> %s on %s", self.topic, self.brokers)

    def _do_emit(self, event: PerceptionEvent) -> None:
        payload = event.as_det2mot() if self.legacy else event.as_dict()
        self._producer.produce(
            self.topic,
            key=event.camera_id.encode("utf-8"),
            value=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
        # Non-blocking: serves delivery callbacks for messages already sent and returns.
        # `flush()` is what waits, and it belongs at shutdown, not on the frame path.
        self._producer.poll(0)

    def flush(self, timeout_s: float = 5.0) -> None:
        """Wait for the producer's queue to drain, up to ``timeout_s``."""
        remaining = self._producer.flush(timeout_s)
        if remaining:
            _LOG.warning("kafka sink closed with %d message(s) undelivered", remaining)

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "topic": self.topic, "brokers": self.brokers}
