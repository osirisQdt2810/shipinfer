"""A served request reaches the trace sink, with its tag and its stamps intact.

The vocabulary and the sinks themselves are pinned in ``tests/core/test_tracing.py``. What
is checked here is the wiring, which is where a feature like this fails silently: one sink
per server, handed to every model, recorded after the future resolves so tracing can never
delay a caller's answer, and closed on shutdown so the last records are not lost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.tracing import TRACE_EVENTS, NullTraceSink
from shipinfer.core.types import Tensor
from shipinfer.server import InferenceServer

_ECHO = """
platform: mock
max_batch_size: 4
inputs: [{name: x, data_type: FP32, dims: [2]}]
outputs: [{name: y, data_type: FP32, dims: [2]}]
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.05}
"""


class TestTheServerWritesTraces:
    """The wiring: a served request has to reach the sink, with its tag intact."""

    def _server(self, tmp_path: Path, sink: str, **options) -> InferenceServer:
        root = tmp_path / "repo"
        (root / "echo" / "1").mkdir(parents=True)
        (root / "echo" / "config.yaml").write_text(_ECHO.lstrip())
        return InferenceServer(
            ServerSettings(
                model_repository=root,
                devices={"visible_gpus": []},
                execution={"warmup_iterations": 0},
                observability={"trace_sink": sink, "trace_sink_options": options},
            )
        )

    def _request(self, frame: int = 0) -> InferenceRequest:
        return InferenceRequest(
            model_name="echo",
            inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
            context=RequestContext(camera_id="cam07", frame_id=frame),
        )

    def test_tracing_is_off_by_default(self, tmp_path: Path) -> None:
        with self._server(tmp_path, "none") as server:
            server.infer_sync(self._request(), timeout=10)

            assert isinstance(server.traces, NullTraceSink)
            assert server.stats()["tracing"]["recorded"] == 0

    def test_a_served_request_is_written_with_its_camera_and_frame(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "traces.jsonl"
        server = self._server(tmp_path, "jsonlines", path=str(path), flush_every=0)
        with server:
            for frame in range(3):
                server.infer_sync(self._request(frame), timeout=10)

        records = [json.loads(line) for line in path.read_text().strip().splitlines()]
        assert len(records) == 3
        assert [record["frame_id"] for record in records] == [0, 1, 2]
        assert {record["camera_id"] for record in records} == {"cam07"}
        assert records[0]["device"] == "cpu"

    def test_the_written_stamps_are_ordered_and_real(self, tmp_path: Path) -> None:
        """A trace whose stamps are all zero would look plausible and answer nothing."""
        path = tmp_path / "traces.jsonl"
        server = self._server(tmp_path, "jsonlines", path=str(path), flush_every=0)
        with server:
            server.infer_sync(self._request(), timeout=10)

        stamps = json.loads(path.read_text().strip())["timestamps"]
        values = [entry["ns"] for entry in stamps]
        assert [entry["name"] for entry in stamps] == list(TRACE_EVENTS)
        assert all(value > 0 for value in values)
        assert values == sorted(values)

    def test_sampling_reaches_the_serving_path(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        server = self._server(tmp_path, "jsonlines", path=str(path), flush_every=0, rate=4)
        with server:
            for frame in range(8):
                server.infer_sync(self._request(frame), timeout=10)
            stats = server.traces.stats()

        assert stats["recorded"] == 2
        assert stats["sampled_out"] == 6


_PIPE = """
platform: ensemble
max_batch_size: 0
inputs: [{name: x, data_type: FP32, dims: [2]}]
outputs: [{name: y, data_type: FP32, dims: [2]}]
dynamic_batching: {enabled: false}
ensemble:
  steps:
    - model: echo
      input_map: {x: x}
      output_map: {y: y}
"""


class TestSamplingReachesTheEnsemblePath:
    """The ensemble recorded every completion, whatever rate the operator set.

    `ModelInstance._complete` guards its record with `should_record()` — the sampling gate,
    the thing that advances the sampler and counts `sampled_out`. `EnsembleModel` called
    `record()` directly, and `record()` applies no rate. So `rate: 1000` on the 50-camera
    deployment meant one JSONL line per second for plain models and one per completion for
    every ensemble — and even the default `NullTraceSink` had a `RequestTrace` built per DAG
    completion only to be thrown away. Review found it on the third round; this path had no
    test, which is why it survived two.
    """

    def _server(self, tmp_path: Path, **options) -> InferenceServer:
        root = tmp_path / "repo"
        (root / "echo" / "1").mkdir(parents=True)
        (root / "echo" / "config.yaml").write_text(_ECHO.lstrip())
        (root / "pipe").mkdir()
        (root / "pipe" / "config.yaml").write_text(_PIPE.lstrip())
        return InferenceServer(
            ServerSettings(
                model_repository=root,
                devices={"visible_gpus": []},
                execution={"warmup_iterations": 0},
                observability={"trace_sink": "jsonlines", "trace_sink_options": options},
            )
        )

    def _request(self, frame: int) -> InferenceRequest:
        return InferenceRequest(
            model_name="pipe",
            inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
            context=RequestContext(camera_id="cam07", frame_id=frame),
        )

    def test_the_operators_rate_applies_to_ensembles_too(self, tmp_path: Path) -> None:
        """Mirror of `test_sampling_reaches_the_serving_path`, through the DAG.

        One request through `pipe` passes the gate **twice** — once when the `echo` step's
        instance completes, once when the DAG completes — so eight requests are sixteen gate
        decisions, and at `rate=4` that is four recorded and twelve sampled out. The invariant
        this pins is that *every* trace went through the gate: recorded plus sampled_out equals
        the number of completions. Before the guard the ensemble's eight bypassed it — eight
        recorded unconditionally on top of the two the step's gate let through — and the
        numbers read ten recorded, six sampled out, with the sixteen no longer adding up.
        """
        path = tmp_path / "traces.jsonl"
        server = self._server(tmp_path, path=str(path), flush_every=0, rate=4)
        with server:
            for frame in range(8):
                server.infer_sync(self._request(frame), timeout=10)
            stats = server.traces.stats()

        assert stats["recorded"] + stats["sampled_out"] == 16, "a completion skipped the gate"
        assert stats["recorded"] == 4
        assert stats["sampled_out"] == 12
