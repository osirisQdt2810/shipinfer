"""The server actually holds the model's concurrency bound.

The limiter's own properties are pinned in ``tests/scheduling/test_rate_limits.py``. What
is checked here is the wiring, which is where this kind of feature usually fails silently:
one limiter per *model*, shared by its instances, acquired around the execution and released
on every path out of it. A limiter that each instance built for itself would bound nothing
and every one of the unit tests would still pass.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import RequestCancelledError
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.server import InferenceServer

# Four instances, a bound of one, and a per-execution latency large enough that a burst
# genuinely overlaps: without the limiter all four would be in compute at once.
_MODEL = """
platform: mock
max_batch_size: 2
inputs: [{{name: x, data_type: FP32, dims: [2]}}]
outputs: [{{name: y, data_type: FP32, dims: [2]}}]
instance_groups: [{{kind: KIND_CPU, count: 4}}]
dynamic_batching: {{enabled: false}}
parameters: {{latency_ms: 5.0}}
{limits}
"""


def _server(tmp_path: Path, name: str, limits: str) -> InferenceServer:
    root = tmp_path / "repo"
    (root / name / "1").mkdir(parents=True)
    (root / name / "config.yaml").write_text(_MODEL.format(limits=limits).lstrip())
    return InferenceServer(
        ServerSettings(
            model_repository=root,
            devices={"visible_gpus": []},
            execution={"warmup_iterations": 0},
        )
    )


def _request(model: str) -> InferenceRequest:
    return InferenceRequest(
        model_name=model,
        inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
        context=RequestContext(camera_id="cam0", frame_id=0),
    )


class TestABoundedModel:
    """Four instances, a bound of one."""

    @pytest.fixture()
    def server(self, tmp_path: Path):
        with _server(
            tmp_path,
            "limited",
            "rate_limiter: {kind: concurrency, max_concurrent_executions: 1}",
        ) as running:
            yield running

    def test_a_burst_never_puts_more_than_one_instance_into_compute(self, server) -> None:
        futures = [server.infer(_request("limited")) for _ in range(12)]
        for future in futures:
            future.result(timeout=30)

        stats = server.model("limited").stats()
        limiter = stats["rate_limiter"]
        assert limiter["limit"] == 1
        assert limiter["peak_in_flight"] == 1
        # One slot per execution, and every execution took one: the batcher decides how many
        # of the 12 requests share a batch, so this counts batches rather than requests.
        assert limiter["granted"] == sum(i["batches"] for i in stats["instances"])
        assert limiter["granted"] > 0
        assert limiter["in_flight"] == 0

    def test_the_limiter_shapes_the_burst_and_never_sheds_it(self, server) -> None:
        """Shedding is the queue's job, at the edge, where the caller learns about it. A
        limiter that dropped work would be a second, invisible eviction policy."""
        futures = [server.infer(_request("limited")) for _ in range(12)]

        assert all(f.result(timeout=30).outputs["y"].shape == (1, 2) for f in futures)

    def test_the_bound_actually_bound(self, server) -> None:
        """`waited` is the number that distinguishes a limiter that is shaping a burst from
        one that is configured and never reached."""
        futures = [server.infer(_request("limited")) for _ in range(12)]
        for future in futures:
            future.result(timeout=30)

        assert server.model("limited").stats()["rate_limiter"]["waited"] > 0


class TestShutdownWhileWaiting:
    """A worker blocked on a slot must not hold the shutdown."""

    def test_stopping_the_server_does_not_wait_for_a_slot_that_never_frees(
        self, tmp_path: Path
    ) -> None:
        """Without the running check inside the wait loop the worker loops forever, the join
        times out, and `stop` abandons the backend after the full grace period — a leak, and
        a ten-second shutdown, caused by telemetry-shaped configuration.

        The limiter is reached through the model here on purpose: holding its only slot from
        outside is the one way to put a worker in that state deterministically.
        """
        server = _server(
            tmp_path,
            "limited",
            "rate_limiter: {kind: concurrency, max_concurrent_executions: 1}",
        )
        server.start()
        limiter = server.model("limited")._limiter
        assert limiter.acquire(1.0)
        try:
            future = server.infer(_request("limited"))
            time.sleep(0.2)  # let the worker reach the limiter and start waiting

            started = time.monotonic()
            server.stop()
            elapsed = time.monotonic() - started
        finally:
            limiter.release()

        assert elapsed < 5.0, "shutdown waited for a slot instead of giving up on it"
        with pytest.raises(RequestCancelledError):
            future.result(timeout=5)


class TestAnUnboundedModel:
    """The default costs nothing and claims nothing."""

    def test_a_model_without_the_section_reports_no_bound(self, tmp_path: Path) -> None:
        with _server(tmp_path, "free", "") as server:
            server.infer(_request("free")).result(timeout=30)
            limiter = server.model("free").stats()["rate_limiter"]

        assert limiter["limiter"] == "off"
        assert limiter["limit"] == 0
        assert limiter["in_flight"] == 0
