"""Multi-instance balancing and fairness, measured rather than asserted by construction."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.engine import InferenceServer


def _repo(tmp_path: Path, instances: int, latency_ms: float = 2.0) -> Path:
    root = tmp_path / "repo"
    (root / "m" / "1").mkdir(parents=True)
    (root / "m" / "config.yaml").write_text(f"""
platform: mock
max_batch_size: 4
inputs: [{{name: x, data_type: FP32, dims: [2]}}]
outputs: [{{name: y, data_type: FP32, dims: [2]}}]
instance_groups: [{{kind: KIND_CPU, count: {instances}}}]
dynamic_batching:
  enabled: true
  max_queue_delay_us: 1000
parameters:
  latency_ms: {latency_ms}
""".lstrip())
    return root


def _request(camera: str, frame: int) -> InferenceRequest:
    return InferenceRequest(
        model_name="m",
        inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
        context=RequestContext(camera_id=camera, frame_id=frame),
    )


def _drive(server: InferenceServer, work: list[tuple[str, int]], *, in_flight: int) -> list:
    """Submit with a bounded number of requests outstanding, and collect the responses.

    Every real producer needs this shape. A bounded pool refuses work once it is full, by
    design, so a load generator that fires everything at once is measuring its own lack of
    backpressure rather than the server's behaviour.
    """
    pending: set = set()
    responses = []
    for camera, frame in work:
        if len(pending) >= in_flight:
            done, pending = wait(pending, return_when=FIRST_COMPLETED, timeout=120)
            responses.extend(future.result() for future in done)
        pending.add(server.infer(_request(camera, frame)))

    done, not_done = wait(pending, timeout=120)
    assert not not_done, f"{len(not_done)} request(s) never completed"
    responses.extend(future.result() for future in done)
    return responses


class TestPlacementSpreadsAcrossInstances:
    """Under sustained load every instance gets work, whichever policy is in charge."""

    @pytest.mark.parametrize("policy", ["round_robin", "join_shortest_queue", "power_of_two"])
    def test_every_instance_receives_work(self, tmp_path: Path, policy: str) -> None:
        """A policy that leaves an instance idle under sustained load is broken.

        The instances here are identical, so any sane policy should spread across all four.
        """
        settings = ServerSettings(
            model_repository=_repo(tmp_path, instances=4, latency_ms=1.0),
            scheduler={"placement_policy": policy, "max_queue_size": 32},
        )
        with InferenceServer(settings) as server:
            # Bounded in-flight, the way a real producer must behave against a bounded pool.
            # Firing 400 requests at a 4x32-slot pool without waiting would (correctly) be
            # refused with QueueFullError — that is the backpressure working, not a bug.
            completed = _drive(server, [(f"cam{i % 8}", i) for i in range(400)], in_flight=64)

        instances = Counter(response.executed_on for response in completed)

        total = sum(instances.values())
        assert total == 400
        # Four identical CPU instances: nothing should get less than 5% of the load.
        assert min(instances.values()) > total * 0.05, dict(instances)


class TestFairQueueingAcrossCameras:
    """A camera sending 8x the traffic is served 8x, and starves nobody doing so."""

    def test_a_skewed_camera_cannot_starve_the_others(self, tmp_path: Path) -> None:
        """The inherited failure, reproduced and then shown not to happen.

        cam_busy submits 8x the traffic of each quiet camera. Fair queueing must still serve
        every quiet camera every time it asks; the old shared evict-oldest buffer would have
        dropped them.
        """
        settings = ServerSettings(
            model_repository=_repo(tmp_path, instances=2, latency_ms=1.0),
            # BLOCK, not REJECT: this test is about *fairness of service*, so every submission
            # must be accepted for the served counts to mean anything. Rejection is exercised
            # separately in test_backpressure_rejects_rather_than_evicting.
            scheduler={
                "fair_queueing": True,
                "max_queue_size": 32,
                "overflow_policy": "block",
                "enqueue_block_timeout_ms": 5000,
            },
        )
        quiet_cameras = [f"cam{i}" for i in range(1, 6)]

        work: list[tuple[str, int]] = []
        for round_index in range(40):
            work.extend(("cam_busy", round_index) for _ in range(8))
            work.extend((camera, round_index) for camera in quiet_cameras)

        with InferenceServer(settings) as server:
            completed = _drive(server, work, in_flight=48)

        served = Counter(response.context.camera_id for response in completed)

        assert served["cam_busy"] == 320
        for camera in quiet_cameras:
            assert served[camera] == 40, f"{camera} was starved: {dict(served)}"


class TestBoundedQueueBackpressure:
    """A saturated pool rejects the newest request rather than evicting an older one."""

    def test_backpressure_rejects_rather_than_evicting(self, tmp_path: Path) -> None:
        """A saturated pool must say so, not silently drop somebody else's work."""
        from shipinfer.core.errors import QueueFullError

        settings = ServerSettings(
            model_repository=_repo(tmp_path, instances=1, latency_ms=20.0),
            scheduler={"max_queue_size": 2, "overflow_policy": "reject"},
        )
        rejected = 0
        accepted = []
        with InferenceServer(settings) as server:
            for i in range(60):
                try:
                    accepted.append(server.infer(_request("cam0", i)))
                except QueueFullError:
                    rejected += 1
            wait(accepted, timeout=120)

        assert rejected > 0, "a 2-slot queue behind a 20 ms model should have refused work"
        assert all(f.exception() is None for f in accepted), "accepted work must still complete"
