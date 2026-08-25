"""Per-model statistics: the numbers, and the endpoint that serves one model's.

`/v2/statistics` returns the whole server, and the metrics registry holds histograms with
no per-model cumulative count in them at all. Neither answers the question an operator has
when one camera is slow: *what has this one model done since it loaded*. Triton answers it
at `/v2/models/{name}/stats`, so this server does too, in Triton's shape.

The arithmetic is where this can go quietly wrong, so it is pinned first: Triton charges
every request in a batch the whole batch's compute span, and keys `batch_stats` on rows
rather than requests. Both look like bugs to a reader who does not know the convention, and
both are the convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.server import InferenceServer
from shipinfer.server.statistics import DurationStat, ModelStatistics

_ECHO = """
platform: mock
max_batch_size: 4
inputs: [{name: x, data_type: FP32, dims: [2]}]
outputs: [{name: y, data_type: FP32, dims: [2]}]
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.05}
"""


@pytest.fixture()
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "echo" / "1").mkdir(parents=True)
    (root / "echo" / "config.yaml").write_text(_ECHO.lstrip())
    return root


@pytest.fixture()
def server(repository: Path):
    settings = ServerSettings(
        model_repository=repository,
        devices={"visible_gpus": []},
        execution={"warmup_iterations": 0},
    )
    with InferenceServer(settings) as running:
        yield running


def _request() -> InferenceRequest:
    return InferenceRequest(
        model_name="echo",
        inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
        context=RequestContext(camera_id="cam0", frame_id=1),
    )


class TestDurationStat:
    """Triton's ``(count, ns)`` pair, and the one thing it refuses to do."""

    def test_a_repeated_observation_is_added_once_per_repeat(self) -> None:
        """``count=3`` means three requests saw that span, so it contributes three times.
        Adding it once would divide the reported latency by the batch size — an error in the
        flattering direction."""
        stat = DurationStat()
        stat.observe(1_000)
        stat.observe(2_000, count=3)

        assert (stat.count, stat.ns) == (4, 7_000)

    def test_a_negative_span_contributes_zero_rather_than_going_backwards(self) -> None:
        """Two stamps from different clocks must not make a cumulative total shrink —
        there is no recovering the real number once it has drifted downwards."""
        stat = DurationStat()
        stat.observe(5_000)
        stat.observe(-4_000)

        assert stat.ns == 5_000
        assert stat.count == 2


class TestTritonArithmetic:
    """The two conventions that look like bugs and are not."""

    def test_every_request_in_a_batch_is_charged_the_whole_batch_span(self) -> None:
        stats = ModelStatistics()

        stats.record_execution(
            requests=4,
            batch_size=4,
            queue_ns=400,
            compute_input_ns=100,
            compute_infer_ns=1_000,
            compute_output_ns=50,
            total_ns=8_000,
        )

        body = stats.as_dict("m", 1)
        infer = body["inference_stats"]["compute_infer"]
        # 4 requests, each charged the batch's 1000 ns: this reads as "the compute latency a
        # request experienced", which is what a latency investigation wants.
        assert infer == {"count": 4, "ns": 4_000}
        assert body["inference_count"] == 4
        assert body["execution_count"] == 1

    def test_batch_stats_count_executions_and_are_keyed_on_rows(self) -> None:
        """A request may carry several rows; a batch of 4 rows costs what 4 rows cost
        however many callers contributed them."""
        stats = ModelStatistics()

        stats.record_execution(
            requests=2,
            batch_size=4,
            queue_ns=0,
            compute_input_ns=0,
            compute_infer_ns=1_000,
            compute_output_ns=0,
            total_ns=0,
        )

        (entry,) = stats.as_dict("m", 1)["batch_stats"]
        assert entry["batch_size"] == 4
        assert entry["count"] == 1
        assert entry["compute_infer"] == {"count": 1, "ns": 1_000}

    def test_failures_are_not_folded_into_success(self) -> None:
        stats = ModelStatistics()
        stats.record_execution(
            requests=1,
            batch_size=1,
            queue_ns=0,
            compute_input_ns=0,
            compute_infer_ns=1,
            compute_output_ns=0,
            total_ns=10,
        )

        stats.record_failure(3)

        body = stats.as_dict("m", 1)["inference_stats"]
        assert body["success"]["count"] == 1
        assert body["fail"]["count"] == 3

    def test_every_key_is_rendered_even_at_zero(self) -> None:
        """An absent key and a zero are different claims: a dashboard that hides "0
        failures" cannot tell "nothing failed" from "this server does not report it"."""
        body = ModelStatistics().as_dict("m", 2)

        assert set(body["inference_stats"]) == {
            "success",
            "fail",
            "queue",
            "compute_input",
            "compute_infer",
            "compute_output",
            "cache_hit",
            "cache_miss",
        }
        assert body["version"] == "2"
        assert body["last_inference"] == 0


class TestServingRecordsStatistics:
    """The wiring: serving a request has to move these numbers, or they are decoration."""

    def test_a_served_request_is_counted_with_its_spans(self, server: InferenceServer) -> None:
        server.infer_sync(_request(), timeout=10)

        body = server.model("echo").model_stats()
        assert body["name"] == "echo"
        assert body["inference_count"] == 1
        assert body["execution_count"] == 1
        assert body["last_inference"] > 0
        infer = body["inference_stats"]["compute_infer"]
        # The mock backend sleeps 50 us per execution, so a zero here means the span was
        # never measured rather than that the model is fast.
        assert infer["count"] == 1 and infer["ns"] > 0
        assert body["inference_stats"]["success"]["count"] == 1

    def test_the_batch_that_ran_appears_in_batch_stats(self, server: InferenceServer) -> None:
        server.infer_sync(_request(), timeout=10)

        (entry,) = server.model("echo").model_stats()["batch_stats"]
        assert entry["batch_size"] == 1
        assert entry["count"] == 1


class TestPerModelStatsEndpoint:
    """The HTTP surface, which is the point of the feature."""

    @pytest.fixture()
    def client(self, server: InferenceServer):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from shipinfer.server.api import create_app

        with TestClient(create_app(server)) as test_client:
            yield test_client

    def test_it_returns_tritons_model_stats_array(self, client, server) -> None:
        server.infer_sync(_request(), timeout=10)

        body = client.get("/v2/models/echo/stats").json()

        assert list(body) == ["model_stats"]
        (entry,) = body["model_stats"]
        assert entry["name"] == "echo"
        assert entry["inference_count"] == 1

    def test_the_versioned_spelling_works_too(self, client, server) -> None:
        server.infer_sync(_request(), timeout=10)

        body = client.get("/v2/models/echo/versions/1/stats").json()

        assert body["model_stats"][0]["version"] == "1"

    def test_a_version_that_is_not_loaded_is_a_404(self, client) -> None:
        """Answering with the loaded version's numbers under another version's URL is how a
        rollout gets declared healthy on the old build's data."""
        assert client.get("/v2/models/echo/versions/7/stats").status_code == 404

    def test_an_unknown_model_is_a_404(self, client) -> None:
        assert client.get("/v2/models/nope/stats").status_code == 404

    def test_one_models_numbers_are_not_the_servers(self, client, server) -> None:
        """The regression this feature exists for: the per-model view must be readable
        without parsing a fleet-wide document."""
        server.infer_sync(_request(), timeout=10)

        per_model = client.get("/v2/models/echo/stats").json()["model_stats"][0]
        whole_server = client.get("/v2/statistics").json()

        assert per_model["inference_count"] == 1
        assert "models" in whole_server  # the server view still exists, unchanged
