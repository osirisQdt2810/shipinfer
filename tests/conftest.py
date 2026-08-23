"""Shared fixtures.

Two rules the whole suite depends on:

1. **The default run needs no GPU.** ``pytest`` selects ``-m "not gpu"``, and everything
   outside that marker must pass on a laptop with no NVIDIA driver. The scheduler's
   fairness and balancing guarantees are the most valuable things here to test, and a test
   that needs sixteen GPUs gets written once and then never run again.
2. **No shared global state between tests.** Each test builds its own server, its own
   metrics registry and its own repository, so a counter from one cannot leak into another.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.metrics import ServerMetrics
from shipinfer.core.request import InferenceRequest, RequestContext, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.scheduling.work import WorkItem

DATA = Path(__file__).parent / "data"


# -- markers ------------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items) -> None:
    """Gate the GPU tiers on a container, then skip them when there is no device.

    Two separate concerns, and the order matters.

    **The container gate comes first.** The rule that measurements run in a container was
    enforced by a hook reading the *text* of a shell command, and review showed a deny-list
    over text cannot be made sound: ``( pytest tests/ )``, ``eval "pytest tests/"``,
    ``coverage run -m pytest`` and six other ordinary spellings all walked through it. This
    check is inside the process that would do the work, so no spelling avoids it — and it
    has to run before the skip logic below, which returns early on a host that *has* GPUs
    and would therefore let the whole device tier run there.

    Only the device tiers are gated. The offline tier must keep running anywhere: that is
    ADR-001, it is what CI does on a plain runner, and it is the promise that makes the pure
    layers verifiable without a driver.

    **Then the skip.** ``-m gpu`` on a GPU-less host should report "skipped", not "failed":
    the tests are fine, the hardware is absent, and conflating the two makes a red CI run
    meaningless.
    """
    from shipinfer.runtime import containment
    from shipinfer.runtime.platform import device_count

    device_markers = {"gpu", "multigpu"}
    selected = [i for i in items if device_markers & set(i.keywords)]
    if selected:
        try:
            containment.require_container(f"{len(selected)} GPU-tier test(s)")
        except RuntimeError as exc:
            # `pytest.exit` rather than letting the raise escape: an exception out of a
            # collection hook is reported as INTERNALERROR with a traceback, which reads
            # like a broken suite rather than a refused run. The operator needs the reason,
            # not our stack.
            pytest.exit(str(exc), returncode=pytest.ExitCode.USAGE_ERROR)

    count = device_count()
    if count >= 2:
        return
    no_gpu = pytest.mark.skip(reason="needs a CUDA device")
    no_multi = pytest.mark.skip(reason="needs at least 2 CUDA devices")
    for item in items:
        if "multigpu" in item.keywords and count < 2:
            item.add_marker(no_multi)
        elif "gpu" in item.keywords and count < 1:
            item.add_marker(no_gpu)


# -- repositories ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def demo_repository_path() -> Path:
    """The repository shipped with the project, used as a realistic fixture."""
    return Path(__file__).resolve().parents[1] / "model_repository"


@pytest.fixture()
def tmp_repository(tmp_path: Path) -> Iterator[Path]:
    """A minimal two-model repository, writable by the test."""
    root = tmp_path / "model_repository"
    (root / "echo" / "1").mkdir(parents=True)
    (root / "echo" / "config.yaml").write_text("""
name: echo
platform: mock
max_batch_size: 8
inputs:
  - {name: x, data_type: FP32, dims: [4]}
outputs:
  - {name: y, data_type: FP32, dims: [4]}
instance_groups:
  - {kind: KIND_CPU, count: 2}
dynamic_batching:
  enabled: true
  max_queue_delay_us: 2000
  preferred_batch_sizes: [2, 4, 8]
parameters:
  latency_ms: 0.5
""".lstrip())
    (root / "slow" / "1").mkdir(parents=True)
    (root / "slow" / "config.yaml").write_text("""
name: slow
platform: mock
max_batch_size: 4
inputs:
  - {name: x, data_type: FP32, dims: [2]}
outputs:
  - {name: y, data_type: FP32, dims: [2]}
instance_groups:
  - {kind: KIND_CPU, count: 1}
dynamic_batching:
  enabled: false
parameters:
  latency_ms: 5.0
""".lstrip())
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def settings(tmp_repository: Path) -> ServerSettings:
    return ServerSettings(model_repository=tmp_repository)


@pytest.fixture()
def metrics() -> ServerMetrics:
    return ServerMetrics()


# -- request helpers ---------------------------------------------------------------------


@pytest.fixture()
def make_request():
    """Build an ``InferenceRequest`` with sensible defaults."""

    def _make(
        model: str = "echo",
        *,
        camera: str = "cam0",
        frame: int = 0,
        width: int = 4,
        rows: int = 1,
        **kwargs,
    ) -> InferenceRequest:
        return InferenceRequest(
            model_name=model,
            inputs={"x": Tensor.from_numpy(np.zeros((rows, width), dtype=np.float32))},
            context=RequestContext(camera_id=camera, frame_id=frame),
            **kwargs,
        )

    return _make


@pytest.fixture()
def make_item(make_request):
    """Build a ``WorkItem`` around a request."""

    def _make(**kwargs) -> WorkItem:
        request = make_request(**kwargs)
        return WorkItem(request, ResponseFuture(request))

    return _make
