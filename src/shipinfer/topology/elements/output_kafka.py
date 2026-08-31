"""The ``kafka`` output implementation, kept in its own module so it is imported lazily.

This is Plane 3's feed (arch.md §2): perception publishes small results — boxes, embeddings,
track ids, ship ids — and the stateful services consume them. Frames never go on the bus.

**Why this is not in ``elements/output.py``.** The sink it builds imports
``confluent_kafka`` inside its constructor, and ``topology`` names that client in exactly one
place. Registering this class lazily (``register_lazy`` in ``elements/output.py``) means the
module is imported when a chain *declares* ``impl: kafka`` and not because
``shipinfer.topology`` was imported — so ``shipinfer repo ls``, a chain validation and the
whole offline tier stay free of a broker client, and a deployment that asks for the sink
without installing librdkafka still fails at **start-up**, with the ``pip install`` in the
message, rather than on the first frame.

Everything about assembling the event is inherited and deliberately not overridden: one
schema, one version, one place it is built.
"""

from __future__ import annotations

from typing import ClassVar

from shipinfer.topology.elements.output import SinkOutput

__all__ = ["KafkaOutput"]


class KafkaOutput(SinkOutput):
    """One message per frame onto one topic, keyed by camera id.

    Params (all forwarded to :class:`~shipinfer.topology.sinks.kafka.KafkaResultSink`):
        topic: where events go. Default ``perception.results``.
        brokers: ``host:port``, comma separated — librdkafka's ``bootstrap.servers``.
        legacy: publish the v1 ``Det2MOT`` payload instead of the current one, for a consumer
            that validates its input strictly and has not been rebuilt. The current payload is
            additive, so this should not normally be needed.
        config: extra librdkafka settings, merged last, so ``linger.ms``, ``compression.type``
            or SASL can be tuned without a code change.

    **The message key is the camera id**, which is what keeps one camera's frames in one
    partition and therefore in order — a per-camera consumer that got its frames out of order
    would look exactly like a tracking bug and not be one. That decision lives in the sink,
    with the argument, and is restated here because it is the one property a deployment can
    break from the chain file (by pointing two chains at one topic with different keys).

    The element is otherwise the base class: it assembles the event and hands it over, and it
    drains the broker's late refusals into ``shipinfer_output_events_dropped_total`` — the
    counter that is the difference between a broker outage an operator sees and one that
    shows up as a green dashboard through total publish loss.
    """

    #: Not registered by a decorator: `elements/output.py` registers this name lazily, so the
    #: `impl` string is set by `create_element` at build time. `test_chain.py`'s lazy-registration
    #: tests (`TestRegistries`) walk the same path for exactly this reason.
    sink_name: ClassVar[str] = "kafka"
