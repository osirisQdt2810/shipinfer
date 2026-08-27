"""Two processes on one GPU each load their *share* of a model's instances — on a real device.

The offline tests pin the arithmetic against a stubbed driver. This is the same claim made
where it matters: a server started with ``devices.shared_by`` / ``devices.share_rank`` on a
real CUDA device builds the divided instance count, on that device, and the two ranks together
carry exactly the configured count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shipinfer.core.settings import DeviceSettings, ServerSettings
from shipinfer.engine import InferenceServer
from shipinfer.runtime.platform import device_count

_CONFIG = """
name: echo
platform: mock
max_batch_size: 8
inputs:
  - {name: x, data_type: FP32, dims: [4]}
outputs:
  - {name: y, data_type: FP32, dims: [4]}
instance_groups:
  - {kind: KIND_GPU, count: 3}
dynamic_batching:
  enabled: false
parameters:
  latency_ms: 0.1
"""


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "model_repository"
    (root / "echo" / "1").mkdir(parents=True)
    (root / "echo" / "config.yaml").write_text(_CONFIG.lstrip())
    return root


def _instances_for_rank(tmp_path: Path, rank: int):
    # GPUs 2-5 are the ones this box lets the tests use; fall back to 0 on a smaller machine.
    gpu = 2 if device_count() > 2 else 0
    settings = ServerSettings(
        model_repository=_repository(tmp_path),
        devices=DeviceSettings(visible_gpus=[gpu], shared_by=[2], share_rank=[rank]),
    )
    server = InferenceServer(settings).start()
    try:
        return gpu, [instance.device for instance in server.model("echo").instances]
    finally:
        server.stop()


@pytest.mark.gpu
class TestAShardLoadsItsShareOnARealDevice:
    def test_rank_zero_gets_two_of_three(self, tmp_path) -> None:
        gpu, devices = _instances_for_rank(tmp_path, rank=0)

        assert len(devices) == 2
        assert all(d.is_cuda and d.index == gpu for d in devices)

    def test_rank_one_gets_the_remaining_one(self, tmp_path) -> None:
        gpu, devices = _instances_for_rank(tmp_path, rank=1)

        assert len(devices) == 1
        assert devices[0].is_cuda and devices[0].index == gpu

    def test_the_two_ranks_together_carry_the_configured_count(self, tmp_path) -> None:
        _, first = _instances_for_rank(tmp_path / "a", rank=0)
        _, second = _instances_for_rank(tmp_path / "b", rank=1)

        assert len(first) + len(second) == 3, "floor division would have lost one"
