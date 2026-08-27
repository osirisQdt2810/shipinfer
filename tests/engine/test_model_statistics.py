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

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.engine import InferenceServer
from shipinfer.engine.statistics import DurationStat, ModelStatistics

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

        from shipinfer.api import create_app

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
        from shipinfer.engine.statistics import ModelStatistics

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


class TestADeploymentWideGraphSpecIsAFilterNotAnAssertion:
    """`execution.cuda_graph_batch_sizes` applies to every model in the repository, so on a
    mixed one there is no single list that fits all of them: `ship_detector` at
    `max_batch_size: 32` beside `person_embedder` at 8.

    Treating it as an assertion made the setting unusable at all — setting `[1, 8, 32]`
    failed to *construct* the 8-batch model rather than capturing 1 and 8 for it. A per-model
    `parameters.graph_spec` stays an assertion, because that one is a claim about that model
    and an impossible size in it is a typo that should stop the deploy.
    """

    def _spec(self, *, max_batch: int, override, per_model: bool):
        from shipinfer.runtime.graphs import resolve_graph_spec

        if per_model:
            return resolve_graph_spec(
                max_batch_size=max_batch,
                preferred=(),
                override=override,
                override_source="model/config.yaml parameters.graph_spec",
            )
        fits = [s for s in override if 1 <= s <= max_batch]
        return resolve_graph_spec(
            max_batch_size=max_batch,
            preferred=(),
            override=fits or None,
            override_source="settings execution.cuda_graph_batch_sizes",
        )

    def test_a_per_model_override_still_refuses_an_impossible_size(self) -> None:
        with pytest.raises(ConfigurationError):
            self._spec(max_batch=8, override=[1, 8, 32], per_model=True)

    def test_a_deployment_wide_override_keeps_only_what_fits(self) -> None:
        spec = self._spec(max_batch=8, override=[1, 8, 32], per_model=False)

        assert tuple(spec.batch_sizes) == (1, 8)

    def test_a_deployment_wide_override_that_fits_nothing_captures_nothing_by_hand(
        self,
    ) -> None:
        """`[64]` against a model whose window tops out at 8 is not an error — it is "no
        graphs for this model", which is what an empty capture set means. The derivation
        takes over so the field is still populated for `stats()`."""
        spec = self._spec(max_batch=8, override=[64], per_model=False)

        assert spec.batch_sizes, "the spec should fall back to the derivation, not be empty"
        assert all(size <= 8 for size in spec.batch_sizes)


class TestAnInvalidCudaGraphsOverrideIsRefusedEverywhere:
    """`_graphs_enabled` is read at `Model.__init__` now, not per CUDA instance, so a typo'd
    `SHIPINFER_CUDA_GRAPHS` fails on a CPU-only host too — `shipinfer repo ls` on a laptop
    included.

    That is the right direction and it is worth a test rather than an argument. An operator
    who typed the variable is asking whether the graph path is what hurts; a deployment that
    ignores the answer on some hosts and honours it on others makes the experiment worthless.
    """

    def test_a_bad_value_is_refused(self, monkeypatch) -> None:
        from shipinfer.core.errors import ConfigurationError
        from shipinfer.core.settings import ExecutionSettings
        from shipinfer.engine.model import _graphs_enabled

        monkeypatch.setenv("SHIPINFER_CUDA_GRAPHS", "1")

        with pytest.raises(ConfigurationError, match="expected 'on' or 'off'"):
            _graphs_enabled(ExecutionSettings())

    def test_on_and_off_both_win_over_the_setting(self, monkeypatch) -> None:
        from shipinfer.core.settings import ExecutionSettings
        from shipinfer.engine.model import _graphs_enabled

        monkeypatch.setenv("SHIPINFER_CUDA_GRAPHS", "on")
        assert _graphs_enabled(ExecutionSettings(cuda_graphs=False)) is True

        monkeypatch.setenv("SHIPINFER_CUDA_GRAPHS", "off")
        assert _graphs_enabled(ExecutionSettings(cuda_graphs=True)) is False

    def test_unset_leaves_the_models_own_setting_alone(self, monkeypatch) -> None:
        from shipinfer.core.settings import ExecutionSettings
        from shipinfer.engine.model import _graphs_enabled

        monkeypatch.delenv("SHIPINFER_CUDA_GRAPHS", raising=False)

        assert _graphs_enabled(ExecutionSettings(cuda_graphs=True)) is True
        assert _graphs_enabled(ExecutionSettings(cuda_graphs=False)) is False


class TestTheTierRetrySeam:
    """`admit_local` / `try_dispatch_local` / `count_local_rejection` (#26 round 3).

    Admission is first-entry work; a dispatch retry re-runs none of it and records
    nothing, and only the tier's final give-up counts as the rejection `_infer` records.
    """

    def test_a_dispatch_retry_records_nothing_and_the_give_up_records_once(
        self, server: InferenceServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shipinfer.core.errors import QueueFullError

        model = server.model("echo")
        item = model.admit_local(_request())
        assert not item.future.done()

        class Refusing:
            @staticmethod
            def dispatch(*_args, **_kwargs):
                raise QueueFullError("echo", 8, 8)

        monkeypatch.setattr(model, "_local_dispatcher", Refusing())

        def failures() -> int:
            return model.statistics.as_dict("echo", 1)["inference_stats"]["fail"]["count"]

        before = failures()
        for _ in range(50):  # a saturated queue probed 50 times
            assert model.try_dispatch_local(item) is False
        assert failures() == before, "retries record nothing"
        model.count_local_rejection()
        assert failures() == before + 1, "the give-up records once"

    def test_an_admitted_item_dispatches_and_answers(self, server: InferenceServer) -> None:
        model = server.model("echo")
        item = model.admit_local(_request())
        assert model.try_dispatch_local(item) is True
        response = item.future.result(timeout=5.0)
        assert response.model_name == "echo"

    def test_a_wire_stamped_received_ns_survives_admission(
        self, server: InferenceServer
    ) -> None:
        """The wire carries the submitter's stamp; the owner keeps it, so a borrowed
        request's end-to-end latency is not rewritten as owner-side latency."""
        model = server.model("echo")
        stamped = _request()
        stamped.timings.received_ns = 123_456
        model.admit_local(stamped)
        assert stamped.timings.received_ns == 123_456

        fresh = _request()
        model.admit_local(fresh)
        assert fresh.timings.received_ns > 0, "a local request is stamped on arrival"
