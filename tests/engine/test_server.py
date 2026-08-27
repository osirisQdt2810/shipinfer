"""End-to-end server behaviour, on the mock backend and no GPU."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import ModelNotFoundError, ServerStateError, ValidationError
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.engine import HealthStatus, InferenceServer, check_health


@pytest.fixture()
def server(settings: ServerSettings):
    with InferenceServer(settings) as running:
        yield running


def _request(model: str = "echo", camera: str = "cam0", frame: int = 0, width: int = 4):
    return InferenceRequest(
        model_name=model,
        inputs={"x": Tensor.from_numpy(np.ones((1, width), dtype=np.float32))},
        context=RequestContext(camera_id=camera, frame_id=frame),
    )


class TestServerLifecycle:
    """Start, selective start, stop: what the server will and will not do at each stage."""

    def test_start_loads_every_model(self, server: InferenceServer) -> None:
        assert server.models() == ["echo", "slow"]
        assert server.is_ready

    def test_startup_models_can_be_selected(self, tmp_repository: Path) -> None:
        settings = ServerSettings(
            model_repository=tmp_repository, load_all_models=False, startup_models=["echo"]
        )
        with InferenceServer(settings) as server:
            assert server.models() == ["echo"]

    def test_infer_before_start_is_refused(self, settings: ServerSettings) -> None:
        server = InferenceServer(settings)
        with pytest.raises(ServerStateError):
            server.infer(_request())

    def test_stop_is_idempotent(self, settings: ServerSettings) -> None:
        server = InferenceServer(settings).start()
        server.stop()
        server.stop()
        assert not server.is_started


class TestInferenceResponse:
    """What comes back from one successful inference, tag and timings included."""

    def test_infer_returns_the_declared_outputs(self, server: InferenceServer) -> None:
        response = server.infer_sync(_request(), timeout=10)
        assert set(response.outputs) == {"y"}
        assert response.outputs["y"].shape == (1, 4)
        assert response.model_name == "echo"
        assert response.model_version == 1

    def test_the_context_tag_survives_the_round_trip(self, server: InferenceServer) -> None:
        """The invariant the whole reassembly design rests on."""
        response = server.infer_sync(_request(camera="cam42", frame=1234), timeout=10)
        assert response.context.camera_id == "cam42"
        assert response.context.frame_id == 1234

    def test_timings_are_populated(self, server: InferenceServer) -> None:
        response = server.infer_sync(_request(), timeout=10)
        assert response.timings.total_us > 0
        assert response.timings.compute_us > 0
        assert response.timings.queue_us >= 0


class TestConcurrentLoad:
    """Under bounded concurrency nothing is lost, and dynamic batching demonstrably happens."""

    def test_concurrent_requests_all_complete(self, server: InferenceServer) -> None:
        """Bounded in-flight, because the pool is bounded on purpose.

        Firing 200 requests at a 64-slot queue without waiting would be refused, and that
        refusal is the feature — see test_backpressure_rejects_rather_than_evicting.
        """
        pending: set = set()
        completed = 0
        for i in range(200):
            if len(pending) >= 32:
                done, pending = wait(pending, return_when=FIRST_COMPLETED, timeout=60)
                completed += len(done)
                assert all(f.exception() is None for f in done)
            pending.add(server.infer(_request(camera=f"cam{i % 5}", frame=i)))

        done, not_done = wait(pending, timeout=60)
        assert not not_done
        assert all(f.exception() is None for f in done)
        assert completed + len(done) == 200

    def test_requests_are_batched(self, server: InferenceServer) -> None:
        """Evidence that dynamic batching actually happened, not just that it is configured."""
        futures = [server.infer(_request(frame=i)) for i in range(32)]
        wait(futures, timeout=60)

        metrics = server.metrics
        batches = metrics.batches_total.value(model="echo", device="cpu")
        assert batches > 0
        assert (
            batches < 32
        ), f"32 requests were executed as {batches} batches — no batching happened"


class TestRequestRejection:
    """A request the server cannot serve is refused up front, by a typed error."""

    def test_validation_failures_are_synchronous(self, server: InferenceServer) -> None:
        """A malformed request must be refused before it consumes a queue slot."""
        with pytest.raises(ValidationError):
            server.infer(_request(width=99))

    def test_unknown_model(self, server: InferenceServer) -> None:
        with pytest.raises(ModelNotFoundError):
            server.infer(_request(model="nope"))


class TestObservability:
    """Health, stats and the metrics rendering an operator would page on."""

    def test_health_is_ok_when_every_instance_is_up(self, server: InferenceServer) -> None:
        report = check_health(server)
        assert report.status is HealthStatus.OK
        assert report.ready
        assert report.instances_ready == report.instances_total == 3  # echo x2 + slow x1

    def test_stats_and_metrics_render(self, server: InferenceServer) -> None:
        server.infer_sync(_request(), timeout=10)
        stats = server.stats()
        assert stats["ready"] is True
        assert {m["name"] for m in stats["models"]} == {"echo", "slow"}

        text = server.render_metrics()
        assert "shipinfer_requests_total" in text
        assert "# TYPE shipinfer_batch_size histogram" in text
