"""Request tracing: the six stamps, Triton's seven names, and a sink that writes them.

`Timings` already stamped six points along a request's journey and the metrics registry
already histogrammed the spans. What was missing was any way to read *one* request's stamps:
a histogram says "p99 queue wait is 4 ms", it cannot say "frame 8213 from cam07 waited 40 ms
because it was batched late", which is the question an operator has when one camera is
behaving badly.

The names are Triton's, verbatim, for the same reason `runtime/profiling.py` uses Triton's
phase names — a trace from this server should diff against one from Triton without a
translation table. That alignment is what the first test pins.

Pure-layer tests only. That a served request actually reaches the sink is a server property
and lives in ``tests/server/test_request_tracing.py``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import (
    InferenceRequest,
    InferenceResponse,
    RequestContext,
    Timings,
)
from shipinfer.core.tracing import (
    TRACE_EVENTS,
    TRACE_SINKS,
    JsonLinesTraceSink,
    NullTraceSink,
    RequestTrace,
    build_trace_sink,
)
from shipinfer.core.types import Device, Tensor


def _response() -> InferenceResponse:
    request = InferenceRequest(
        model_name="echo",
        inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
        context=RequestContext(camera_id="cam07", frame_id=8213, trace_id="t-1"),
    )
    return InferenceResponse(
        request_id=request.request_id,
        model_name="echo",
        model_version=1,
        outputs={},
        context=request.context,
        timings=Timings(
            received_ns=1_000,
            queued_ns=2_000,
            batched_ns=5_000,
            compute_start_ns=6_000,
            compute_end_ns=9_000,
            completed_ns=10_000,
        ),
        executed_on=Device.cpu(),
    )


class TestTheVocabularyIsTritons:
    """A trace from this server has to read in a tool built for Triton's."""

    def test_the_seven_event_names_are_tritons_in_order(self) -> None:
        assert TRACE_EVENTS == (
            "REQUEST_START",
            "QUEUE_START",
            "COMPUTE_START",
            "COMPUTE_INPUT_END",
            "COMPUTE_OUTPUT_START",
            "COMPUTE_END",
            "REQUEST_END",
        )

    def test_profiling_re_exports_the_same_tuple_rather_than_a_copy(self) -> None:
        """Two copies of a vocabulary is how the two drift and a trace stops diffing."""
        from shipinfer.runtime import profiling

        assert profiling.TRACE_EVENTS is TRACE_EVENTS


class TestRequestTrace:
    """Six stamps onto seven names, and the arithmetic over them."""

    def test_every_event_carries_the_stamp_it_names(self) -> None:
        trace = RequestTrace.from_response(_response())

        assert dict(trace.timestamps) == {
            "REQUEST_START": 1_000,
            "QUEUE_START": 2_000,
            "COMPUTE_START": 5_000,
            "COMPUTE_INPUT_END": 6_000,
            "COMPUTE_OUTPUT_START": 9_000,
            "COMPUTE_END": 10_000,
            "REQUEST_END": 10_000,
        }

    def test_it_carries_the_identity_needed_to_find_the_request_again(self) -> None:
        trace = RequestTrace.from_response(_response())

        assert (trace.camera_id, trace.frame_id) == ("cam07", 8213)
        assert trace.trace_id == "t-1"
        assert trace.model_name == "echo"

    def test_a_span_between_two_named_events_is_microseconds(self) -> None:
        trace = RequestTrace.from_response(_response())

        assert trace.span_us("QUEUE_START", "COMPUTE_START") == 3.0
        assert trace.span_us("REQUEST_START", "REQUEST_END") == 9.0

    def test_an_unknown_event_raises_rather_than_reading_zero(self) -> None:
        """0.0 for a typo looks exactly like a span that took no time."""
        trace = RequestTrace.from_response(_response())

        with pytest.raises(ConfigurationError, match="QUEUE_END"):
            trace.span_us("QUEUE_START", "QUEUE_END")

    def test_the_wire_shape_is_tritons_timestamps_array(self) -> None:
        body = RequestTrace.from_response(_response()).as_dict()

        assert [entry["name"] for entry in body["timestamps"]] == list(TRACE_EVENTS)
        assert body["timestamps"][0] == {"name": "REQUEST_START", "ns": 1_000}
        assert body["camera_id"] == "cam07"


class TestSinkRegistry:
    """Selected by name, so a deployment picks one without editing code."""

    def test_the_builtins_are_registered(self) -> None:
        assert set(TRACE_SINKS.names()) == {"none", "jsonlines"}

    def test_none_is_the_default_and_costs_nothing(self) -> None:
        sink = build_trace_sink("none")

        assert isinstance(sink, NullTraceSink)
        assert sink.should_record() is False

    def test_an_unknown_sink_names_what_was_available(self) -> None:
        with pytest.raises(ConfigurationError, match="jsonlines"):
            build_trace_sink("otel")


class TestSampling:
    """At 1000 frames a second, tracing every request makes the instrument the bottleneck."""

    def test_rate_n_records_one_request_in_n(self, tmp_path: Path) -> None:
        sink = JsonLinesTraceSink(tmp_path / "t.jsonl", rate=3, flush_every=0)

        recorded = sum(sink.should_record() for _ in range(30))

        assert recorded == 10
        assert sink.sampled_out == 20
        sink.close()

    def test_a_rate_below_one_is_a_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="rate"):
            JsonLinesTraceSink(tmp_path / "t.jsonl", rate=0)


class TestJsonLinesSink:
    """One JSON object per line, and a write that never fails a request."""

    def test_one_line_per_trace(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        with JsonLinesTraceSink(path, flush_every=0) as sink:
            assert sink.record(RequestTrace.from_response(_response()))
            assert sink.record(RequestTrace.from_response(_response()))

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["model_name"] == "echo"

    def test_close_flushes_what_was_buffered(self, tmp_path: Path) -> None:
        """A sink that buffered and then lost the buffer would be worse than no sink: the
        operator would be reading an incomplete file believing it was complete."""
        path = tmp_path / "traces.jsonl"
        sink = JsonLinesTraceSink(path, flush_every=1_000)
        sink.record(RequestTrace.from_response(_response()))

        sink.close()

        assert len(path.read_text().strip().splitlines()) == 1

    def test_a_closed_sink_reports_failure_rather_than_pretending(self, tmp_path: Path) -> None:
        sink = JsonLinesTraceSink(tmp_path / "t.jsonl")
        sink.close()

        assert sink.record(RequestTrace.from_response(_response())) is False
        assert sink.failed == 1

    def test_a_broken_sink_never_raises_into_the_serving_path(self, tmp_path: Path) -> None:
        """`record` runs on the worker thread that just finished a batch. An exception there
        would fail a request that already succeeded."""

        class Exploding(JsonLinesTraceSink):
            def _do_record(self, trace: RequestTrace) -> None:
                raise OSError("disk full")

        sink = Exploding(tmp_path / "t.jsonl")
        try:
            assert sink.record(RequestTrace.from_response(_response())) is False
            assert sink.failed == 1
            assert sink.recorded == 0
        finally:
            sink.close()

    def test_concurrent_writers_do_not_interleave_a_line(self, tmp_path: Path) -> None:
        """Every model instance's worker records concurrently, and a torn line corrupts
        every record in the file rather than one."""
        path = tmp_path / "traces.jsonl"
        trace = RequestTrace.from_response(_response())
        with JsonLinesTraceSink(path, flush_every=0) as sink:
            threads = [
                threading.Thread(target=lambda: [sink.record(trace) for _ in range(50)])
                for _ in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(20)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 400
        assert all(json.loads(line)["frame_id"] == 8213 for line in lines)
