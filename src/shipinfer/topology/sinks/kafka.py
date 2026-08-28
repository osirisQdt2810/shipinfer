"""Publish to Kafka — the bus the tracking tier already listens on.

This is PLANE 3's feed in ``references/bitbucket-subfaceid/docs/new-system-architecture.md``:
perception publishes small results (boxes, embeddings, ship ids) and the stateful services
(``motservice`` per camera, ``mtmcservice`` globally) consume them. Frames never go on the
bus — they are megabytes and they stay in shared memory; this carries metadata only, which is
why the segmenter's masks are reduced to an area before they reach here.

Three decisions are worth reading before changing anything.

**The message key is the camera id.** Kafka guarantees order within a partition, so keying on
the camera puts one camera's frames in one partition, in order. A per-camera tracker is
stateful and consumes them in sequence; keying on the frame id, or not keying at all, would
spread one camera's timeline across partitions and hand the tracker its frames out of order —
which looks exactly like a tracking bug and is not one.

**A ``produce()`` that returns is not a publish.** It only copies the message into
librdkafka's own queue; the broker's verdict arrives later, on a delivery callback serviced
by ``poll()``. Without ``on_delivery`` that verdict is *never read*, so a topic that does not
exist, an ACL denial or a broker that acknowledges nothing is completely invisible: every
``emit()`` returns ``True``, ``frames_emitted`` climbs at full rate and
``pipeline_sink_failures_total`` stays at zero through total publish loss — the exact failure
:mod:`shipinfer.topology.sinks.base` says the ``bool`` return exists to prevent. So the
callback is registered and every failure it reports is charged back to an ``emit()``.

The attribution is a few frames late and the count is exact, which is the honest trade for
not blocking a worker thread on an acknowledgement per frame: the broker's answer for frame
*n* lands while frame *n+k* is being produced, so one ``emit()`` returns ``False`` per
message the broker refused, just not the one that produced it. Anything still in flight at
shutdown is counted by :meth:`KafkaResultSink.flush` instead, because it has no later
``emit()`` to be reported against.

**``confluent_kafka`` is imported inside the constructor.** Nothing at import time, so
``shipinfer registries`` lists this sink on a host that has never had librdkafka, and a
deployment that asks for it without installing it fails at **start-up** with the install
command in the message rather than on the first frame.
"""

from __future__ import annotations

import json
import threading
from functools import partial
from typing import Any, ClassVar

from shipinfer.core.errors import (
    BackendUnavailableError,
    ConfigurationError,
)
from shipinfer.core.events.schema import PerceptionEvent
from shipinfer.core.logging import get_logger, log_context
from shipinfer.topology.sinks.base import ResultSink
from shipinfer.topology.sinks.registry import RESULT_SINKS

__all__ = ["KafkaResultSink"]

_LOG = get_logger("topology.sinks.kafka")

#: One delivery failure in this many is logged at ERROR; the rest go to DEBUG. A rejected
#: topic fails *every* message, so at 1000 frames a second an unthrottled log line would
#: cost more CPU than the publishing did — and the counter is the number an operator acts
#: on anyway. Same split :class:`shipinfer.pipeline.reassembly.FrameCollector` makes for an
#: eviction.
_LOG_EVERY = 256


def _camera_of(message: Any) -> str:
    """The camera a message belongs to, from its key. ``"unknown"`` when there is none.

    Best effort on purpose: this only feeds a log line, and a driver that hands back no
    message on some error path must not turn a publish failure into a second failure.
    """
    key = message.key() if message is not None else None
    if isinstance(key, (bytes, bytearray)):
        return bytes(key).decode("utf-8", "replace")
    return "unknown"


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
        #: Messages the broker acknowledged, and the ones it refused. Distinct from the
        #: base class's ``emitted``, which only counts what librdkafka accepted for sending.
        self.delivered = 0
        self.delivery_failures = 0
        # The delivery callback runs on whichever thread is inside `poll()`/`flush()`, and
        # several pipeline workers emit concurrently, so the hand-off is guarded. One
        # uncontended acquire per frame, against a JSON encode in the same method.
        self._delivery = threading.Lock()
        self._pending_tags: list[tuple[str, int]] = []
        self._last_delivery_error = ""
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
            # `partial`, so the callback knows *which* frame it is answering for. Without it
            # the verdict carried only the key, and the failure was charged to whichever frame
            # happened to be mid-emit when `poll(0)` ran it.
            on_delivery=partial(self._on_delivery, event.camera_id, event.frame_id),
        )
        # Non-blocking: serves delivery callbacks for messages already sent and returns.
        # `flush()` is what waits, and it belongs at shutdown, not on the frame path.
        self._producer.poll(0)

    def _on_delivery(self, camera: str, frame: int, error: Any, message: Any) -> None:
        """librdkafka's verdict on one message. Runs inside ``poll()`` or ``flush()``.

        The only place the broker's answer is available, which is why the sink cannot be
        thinner than this: a message that is queued and then refused looks identical to a
        published one from the frame path.

        ``camera`` and ``frame`` are bound at ``produce`` time rather than read back out of
        ``message``, because the key carries the camera and nothing carries the frame — and the
        frame is the half that made the old attribution wrong.
        """
        if error is None:
            with self._delivery:
                self.delivered += 1
            return
        with self._delivery:
            self.delivery_failures += 1
            self._pending_tags.append((camera, frame))
            self._last_delivery_error = str(error)
            failures = self.delivery_failures
        log = _LOG.error if failures == 1 or failures % _LOG_EVERY == 0 else _LOG.debug
        log(
            "kafka sink: broker refused camera %s frame %d on topic %s (%s) "
            "[%d delivery failure(s) so far]",
            camera,
            frame,
            self.topic,
            self._last_delivery_error,
            failures,
            extra=log_context(camera_id=camera),
        )

    def drain_delivery_failures(self) -> tuple[tuple[str, int], ...]:
        """See :meth:`ResultSink.drain_delivery_failures`.

        This replaced `_report_one_delivery_failure`, which raised `SinkDeliveryError` into
        whichever `emit()` happened to be running when the broker's answer arrived. Its
        docstring defended that by saying the *rate* was exact even though the message was not
        named — and the rate was. But `emit()`'s bool does more than feed a counter: the runner
        treats `False` as "this event was dropped", fails the current frame's future and
        returns before recording it. So a refusal of `(cam03, 100)` landing inside the emit of
        `(cam07, 412)` deleted `cam07/412` — which the broker had accepted — from the
        pipeline's records and errored its caller, while `cam03/100` was already recorded as a
        success.

        The rate is still exact; the attribution is no longer wrong.
        """
        with self._delivery:
            drained = tuple(self._pending_tags)
            self._pending_tags.clear()
        return drained

    def flush(self, timeout_s: float = 5.0) -> None:
        """Wait for the producer's queue to drain, up to ``timeout_s``.

        The delivery callbacks for everything still in flight run here, so this is also the
        last chance to count their failures: they have no later ``emit()`` to be charged to,
        and a shutdown that dropped them silently would put the sink's own totals back to
        under-reporting publish loss.
        """
        remaining = self._producer.flush(timeout_s)
        with self._delivery:
            undrained = tuple(self._pending_tags)
            self._pending_tags.clear()
            failures = self.delivery_failures
        if remaining:
            _LOG.warning("kafka sink closed with %d message(s) undelivered", remaining)
        if undrained:
            # These arrived after the pipeline stopped draining, so no metric will ever carry
            # them. Naming the frames is the whole point: at shutdown the log is the only
            # record, and "some messages were refused" is not something an operator can act on.
            _LOG.error(
                "kafka sink: %d refused message(s) were never counted, for %s "
                "(%d delivery failure(s) in total, last: %s)",
                len(undrained),
                ", ".join(f"{camera}/{frame}" for camera, frame in undrained[:10]),
                failures,
                self._last_delivery_error,
            )

    def stats(self) -> dict[str, Any]:
        with self._delivery:
            delivered = self.delivered
            failures = self.delivery_failures
            undrained = len(self._pending_tags)
        return {
            **super().stats(),
            "topic": self.topic,
            "brokers": self.brokers,
            # `emitted` is what librdkafka accepted; these are what the broker did with it.
            # Both are needed: the gap between them *is* the publish loss.
            "delivered": delivered,
            "delivery_failures": failures,
            # Refusals the broker has reported and the pipeline has not yet drained. Steadily
            # non-zero means nobody is calling `drain_delivery_failures`, so publish loss is
            # being counted here and nowhere the operator looks.
            "undrained_delivery_failures": undrained,
        }
