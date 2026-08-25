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


class TestABatchedRequestIsNotChargedItsOwnWaitTwice:
    """Two of the five spans handed to `record_execution` are already summed across the
    batch, and `observe` multiplies by the request count — so `queue` and `success` came out
    **exactly `batch_size` times too large**.

    Measured by review against the mock backend at `max_batch_size: 8` with eight concurrent
    requests forming one batch: reported queue 1335 us/request against an actual mean of 167,
    reported end-to-end 43 822 against 5 478. Both ratios exactly 8.0.

    At the design point (`max_batch_size: 32`) that is a mean queue wait of ~5 ms where the
    truth is ~167 us. The natural response — add instances to a pool that is not backed up —
    takes GPU from the stage that really is behind, and this is the one number in the
    endpoint an autoscaler or a pager keys on.

    The old suite could not see it: the tests with `requests > 1` did not assert on `queue`,
    and the serving-path test used a batch of one, where `n == n * 1`.
    """

    def _stats(self):
        from shipinfer.server.statistics import ModelStatistics

        return ModelStatistics()

    def test_a_summed_span_is_not_multiplied_again(self) -> None:
        stats = self._stats()
        # Four requests, 100 us of queue wait each: 400 us summed, as `_execute` accumulates.
        stats.record_execution(
            requests=4,
            batch_size=4,
            queue_ns=400_000,
            compute_input_ns=1_000,
            compute_infer_ns=8_000,
            compute_output_ns=1_000,
            total_ns=4_000_000,
        )

        queue = stats.as_dict("m", "1")["inference_stats"]["queue"]
        assert queue["count"] == 4
        assert queue["ns"] == 400_000, "the summed wait was multiplied by the batch size"
        assert queue["ns"] / queue["count"] == 100_000, "reported per-request wait is wrong"

    def test_end_to_end_is_the_same_shape(self) -> None:
        stats = self._stats()
        stats.record_execution(
            requests=8,
            batch_size=8,
            queue_ns=0,
            compute_input_ns=0,
            compute_infer_ns=0,
            compute_output_ns=0,
            total_ns=8 * 5_478_000,
        )

        success = stats.as_dict("m", "1")["inference_stats"]["success"]
        assert success["ns"] / success["count"] == 5_478_000

    def test_a_shared_phase_span_is_still_fanned_out(self) -> None:
        """The other three are one span the whole batch shared, so every request is credited
        the whole thing — Triton's convention, and the reason `observe` exists at all. Fixing
        the two must not break the three."""
        stats = self._stats()
        stats.record_execution(
            requests=4,
            batch_size=4,
            queue_ns=0,
            compute_input_ns=0,
            compute_infer_ns=2_000,
            compute_output_ns=0,
            total_ns=0,
        )

        infer = stats.as_dict("m", "1")["inference_stats"]["compute_infer"]
        assert infer["count"] == 4
        assert infer["ns"] == 8_000, "a batch-wide phase must be credited to every request"

    def test_a_batch_of_one_cannot_tell_the_two_apart(self) -> None:
        """Pinned because it is why the bug survived: at `requests=1` both spellings agree,
        so every existing serving-path assertion was satisfied by the wrong one."""
        summed = self._stats()
        summed.record_execution(
            requests=1,
            batch_size=1,
            queue_ns=100_000,
            compute_input_ns=0,
            compute_infer_ns=100_000,
            compute_output_ns=0,
            total_ns=100_000,
        )

        stats = summed.as_dict("m", "1")["inference_stats"]
        assert stats["queue"]["ns"] == stats["compute_infer"]["ns"] == 100_000
