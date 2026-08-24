"""Result sinks: a sink never raises into the pipeline, and the file one is readable back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shipinfer.core.errors import BackendUnavailableError, ConfigurationError
from shipinfer.pipeline.schema import PerceptionEvent
from shipinfer.pipeline.sinks import (
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
