"""Result sinks: a sink never raises into the pipeline, and the file one is readable back."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest

from shipinfer.core.errors import BackendUnavailableError, ConfigurationError
from shipinfer.core.events import PerceptionEvent
from shipinfer.topology.sinks import (
    RESULT_SINKS,
    JsonLinesResultSink,
    NullResultSink,
    ResultSink,
)

pytestmark = pytest.mark.timeout(30)


def event(frame: int = 0, camera: str = "cam0") -> PerceptionEvent:
    return PerceptionEvent.build(
        camera_id=camera, frame_id=frame, source_id="p", objects=(), captured_ns=1
    )


class BrokenSink(ResultSink):
    name = "broken"

    def _do_emit(self, event: PerceptionEvent) -> None:
        raise RuntimeError("the broker went away")


class TestASinkNeverRaisesIntoThePipeline:
    """It is called from the sweeper thread, whose survival is the timeout guarantee."""

    def test_a_failing_sink_counts_instead_of_raising(self):
        sink = BrokenSink()
        sink.emit(event())
        sink.emit(event(1))

        assert sink.failed == 2
        assert sink.emitted == 0

    def test_a_closed_sink_drops_and_counts(self):
        sink = NullResultSink()
        sink.close()
        sink.emit(event())
        assert sink.emitted == 0
        assert sink.failed == 1

    def test_close_is_idempotent(self):
        sink = NullResultSink()
        sink.close()
        sink.close()
        assert sink.is_closed


class TestTheDefaultSink:
    """``null`` is the default because a pipeline must start without a broker."""

    def test_it_counts_and_keeps_nothing(self):
        sink = NullResultSink()
        for frame in range(5):
            sink.emit(event(frame))
        assert sink.emitted == 5
        assert sink.events() == ()

    def test_keep_last_is_bounded(self):
        """A long run must not turn a smoke test into an out-of-memory kill."""
        sink = NullResultSink(keep_last=3)
        for frame in range(100):
            sink.emit(event(frame))
        assert [e.frame_id for e in sink.events()] == [97, 98, 99]


class TestTheJsonLinesSink:
    """The sink that makes the whole DAG testable with no broker and no camera."""

    def test_every_event_is_one_readable_line(self, tmp_path: Path):
        path = tmp_path / "nested" / "events.jsonl"
        with JsonLinesResultSink(path, flush_every=0) as sink:
            for frame in range(3):
                sink.emit(event(frame))

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(line)["image_id"] for line in lines] == [0, 1, 2]

    def test_it_creates_its_parent_directory(self, tmp_path: Path):
        """Failing to start over a missing directory is a worse deploy than creating it."""
        path = tmp_path / "a" / "b" / "c.jsonl"
        JsonLinesResultSink(path).close()
        assert path.exists()

    def test_buffered_writes_are_flushed_on_close(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        sink = JsonLinesResultSink(path, flush_every=1000)
        sink.emit(event())
        sink.close()
        assert path.read_text(encoding="utf-8").strip() != ""

    def test_append_is_the_default_and_truncation_is_opt_in(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        JsonLinesResultSink(path, flush_every=0).emit(event(0))
        JsonLinesResultSink(path, flush_every=0).emit(event(1))
        assert len(path.read_text().strip().splitlines()) == 2

        JsonLinesResultSink(path, flush_every=0, append=False).emit(event(2))
        assert len(path.read_text().strip().splitlines()) == 1

    def test_a_negative_flush_interval_is_refused(self, tmp_path: Path):
        with pytest.raises(ConfigurationError):
            JsonLinesResultSink(tmp_path / "x.jsonl", flush_every=-1)


class FakeKafkaError:
    """The subset of ``confluent_kafka.KafkaError`` the delivery callback reads."""

    def __init__(self, message: str) -> None:
        self._message = message

    def __str__(self) -> str:
        return self._message


class FakeMessage:
    """The subset of ``confluent_kafka.Message`` the delivery callback reads."""

    def __init__(self, key: bytes | None) -> None:
        self._key = key

    def key(self) -> bytes | None:
        return self._key


class FakeProducer:
    """librdkafka's asymmetry, which is the whole reason the defect existed.

    ``produce()`` always succeeds — it only copies the message into a local queue — and the
    broker's verdict arrives later, on a callback serviced by ``poll()``. A broker that
    connects and then rejects the topic looks *exactly* like a healthy one from the produce
    call, so a sink that reads only that call cannot tell them apart.

    The one-message lag is deliberate too: the callback for the message just queued is never
    ready inside the same ``poll()``, which is why the sink's failure attribution is a frame
    late and its failure *count* has to be exact.
    """

    def __init__(self, settings: dict, *, error: str | None = None) -> None:
        self.settings = settings
        self.error = error
        self.produced: list[tuple[str, bytes | None]] = []
        self.polls = 0
        self._inflight: list[tuple[object, bytes | None]] = []

    def produce(self, topic, key=None, value=None, on_delivery=None, **kwargs) -> None:
        self.produced.append((topic, key))
        self._inflight.append((on_delivery, key))

    def poll(self, timeout: float = 0.0) -> int:
        self.polls += 1
        ready, self._inflight = self._inflight[:-1], self._inflight[-1:]
        return self._deliver(ready)

    def flush(self, timeout: float = 5.0) -> int:
        ready, self._inflight = self._inflight, []
        self._deliver(ready)
        return 0

    def _deliver(self, ready) -> int:
        for on_delivery, key in ready:
            if on_delivery is not None:
                error = FakeKafkaError(self.error) if self.error else None
                on_delivery(error, FakeMessage(key))
        return len(ready)


def install_fake_kafka(monkeypatch, *, error: str | None = None) -> list[FakeProducer]:
    """Put a fake ``confluent_kafka`` in ``sys.modules`` and return the producers built.

    The sink imports the client inside its constructor precisely so a host with no
    librdkafka can still list it, and that same seam is what lets the offline tier drive a
    rejecting broker without one.
    """
    built: list[FakeProducer] = []
    module = types.ModuleType("confluent_kafka")

    class Producer(FakeProducer):
        def __init__(self, settings: dict) -> None:
            super().__init__(settings, error=error)
            built.append(self)

    module.Producer = Producer
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)
    return built


class TestTheKafkaSinkReportsDeliveryFailures:
    """A ``produce()`` that returns is not a publish.

    The failure this pins is the one :mod:`shipinfer.topology.sinks.base` says the ``bool``
    return exists to prevent: point the sink at a broker that connects and then rejects the
    topic — an unknown topic, an ACL denial — and every ``emit()`` used to report success
    while nothing whatsoever was published.
    """

    UNKNOWN_TOPIC = "Broker: Unknown topic or partition"

    def _sink(self, monkeypatch, *, error: str | None):
        producers = install_fake_kafka(monkeypatch, error=error)
        sink = RESULT_SINKS.create("kafka", topic="perception.results", brokers="broker:9092")
        return sink, producers[0]

    def test_a_rejected_topic_is_a_counted_failure_not_a_silent_success(self, monkeypatch):
        """The count is exact. What changed is *where* it is reported.

        This used to assert `published == [True, False, False, False, False]` — every frame
        after the first reporting failure because an *earlier* message had been refused. That
        was the bug, written down as an expectation: `emit()`'s bool decides the current
        frame's future, so charging it an earlier message's loss deletes a frame the broker
        accepted and errors its caller, while the frame genuinely lost was already recorded a
        success.

        `produce` accepted all five, so all five emits are true; all five refusals are
        reported through `drain_delivery_failures`, each with its own tag.
        """
        sink, producer = self._sink(monkeypatch, error=self.UNKNOWN_TOPIC)

        published = [sink.emit(event(frame)) for frame in range(5)]

        assert published == [True] * 5, "the broker accepted every produce; emit reports that"
        assert sink.emitted == 5
        assert sink.failed == 0
        assert sink.delivered == 0
        assert len(producer.produced) == 5
        # Four verdicts have come back by the fifth produce; the last needs a flush.
        assert sink.delivery_failures == 4

    def test_each_refusal_carries_the_frame_it_belongs_to(self, monkeypatch):
        """The whole point of the fix. Four refused messages, four tags, and they are *its*
        tags — not the tag of whichever frame happened to be mid-emit."""
        sink, _ = self._sink(monkeypatch, error=self.UNKNOWN_TOPIC)
        for frame in range(5):
            sink.emit(event(frame, camera="quay_west"))

        drained = sink.drain_delivery_failures()

        assert drained == tuple(("quay_west", frame) for frame in range(4))
        assert sink.drain_delivery_failures() == (), "drained twice would double-count"

    def test_two_cameras_are_not_charged_each_others_losses(self, monkeypatch):
        """The concrete failure the review described: a refusal for one camera landing inside
        another camera's emit."""
        sink, _ = self._sink(monkeypatch, error=self.UNKNOWN_TOPIC)
        sink.emit(event(100, camera="cam03"))
        sink.emit(event(412, camera="cam07"))
        sink.emit(event(413, camera="cam07"))

        # The fake broker answers on the *next* poll, so by the third emit two verdicts are
        # back: cam03/100 (arrived inside cam07/412's emit) and cam07/412 (inside 413's).
        # Under the old code the first of those was charged to cam07/412.
        assert sink.drain_delivery_failures() == (("cam03", 100), ("cam07", 412))

    def test_the_last_failures_are_counted_at_close_not_discarded(self, monkeypatch):
        """Flush is the last chance: those messages have nobody left to drain them."""
        sink, _ = self._sink(monkeypatch, error=self.UNKNOWN_TOPIC)
        for frame in range(5):
            sink.emit(event(frame))

        sink.close()

        stats = sink.stats()
        assert stats["delivery_failures"] == 5
        assert stats["delivered"] == 0
        assert stats["undrained_delivery_failures"] == 0

    def test_the_failure_names_the_camera_an_operator_has_to_look_at(self, monkeypatch, caplog):
        sink, _ = self._sink(monkeypatch, error=self.UNKNOWN_TOPIC)
        with caplog.at_level(logging.ERROR, logger="shipinfer.topology.sinks.kafka"):
            sink.emit(event(0, camera="quay_west"))
            sink.emit(event(1, camera="quay_west"))

        refusals = [r.getMessage() for r in caplog.records if "refused" in r.getMessage()]
        assert refusals and "quay_west" in refusals[0]
        assert self.UNKNOWN_TOPIC in refusals[0]

    def test_a_healthy_broker_still_publishes_every_frame(self, monkeypatch):
        """The other half of the contract: the callback must not invent failures."""
        sink, _ = self._sink(monkeypatch, error=None)

        published = [sink.emit(event(frame)) for frame in range(5)]
        sink.close()

        assert published == [True] * 5
        assert sink.failed == 0
        assert sink.emitted == 5
        assert sink.delivered == 5
        assert sink.delivery_failures == 0


class TestTheRegistry:
    """Adding a sink is a file and a decorator."""

    def test_all_three_sinks_are_registered_with_aliases(self):
        assert set(RESULT_SINKS.names()) == {"null", "jsonlines", "kafka"}
        assert RESULT_SINKS.canonical("jsonl") == "jsonlines"
        assert RESULT_SINKS.canonical("none") == "null"

    def test_listing_the_registry_needs_no_broker_installed(self):
        """``shipinfer registries`` must work on a host that has never had librdkafka."""
        assert dict(RESULT_SINKS.describe())["kafka"]

    def test_the_kafka_sink_fails_at_construction_with_the_install_command(self):
        """At construction, not at the first frame: a missing extra should fail a deploy."""
        try:
            import confluent_kafka  # noqa: F401
        except ImportError:
            with pytest.raises(BackendUnavailableError, match=r"shipinfer\[kafka\]"):
                RESULT_SINKS.create("kafka")
        else:  # pragma: no cover - only on a host that has the client
            pytest.skip("confluent-kafka is installed; the unavailable path cannot be taken")
