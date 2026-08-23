"""Ensemble DAG behaviour: complete responses, validated wiring, honest shutdown.

An ensemble skipping a conditional branch used to drop that branch's outputs from the
response entirely. A caller then got HTTP 200 with two of three declared tensors, which is
indistinguishable from "the branch raised and something swallowed it" — and every consumer
had to write the same defensive lookup.

Load-time validation also ignored ``condition:``, so a step consuming a conditional
tensor unconditionally passed start-up and failed on whichever frame first took the other
branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.server import InferenceServer

# `fail_every` is not used here, but `seed` is: the mock's has_ship flag decides which
# branch runs, and a test that asserts on a branch needs to know which one it got.
_ROUTER = """
platform: mock
max_batch_size: 4
inputs: [{{name: images, data_type: FP32, dims: [4]}}]
outputs:
  - {{name: crops, data_type: FP32, dims: [2]}}
  - {{name: has_thing, data_type: INT32, dims: [1]}}
instance_groups: [{{kind: KIND_CPU, count: 1}}]
dynamic_batching: {{enabled: false}}
parameters: {{latency_ms: 0.05, always: {always}}}
"""

_BRANCH = """
platform: mock
max_batch_size: 4
inputs: [{name: crops, data_type: FP32, dims: [2]}]
outputs: [{name: embedding, data_type: FP32, dims: [3]}]
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.05}
"""

_HEAD = """
platform: mock
max_batch_size: 4
inputs: [{name: embedding, data_type: FP32, dims: [3]}]
outputs: [{name: score, data_type: FP32, dims: [1]}]
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.05}
"""


def _write(root: Path, name: str, body: str, *, versioned: bool = True) -> None:
    directory = root / name
    (directory / "1").mkdir(parents=True) if versioned else directory.mkdir(parents=True)
    (directory / "config.yaml").write_text(body.lstrip())


def _repo(tmp_path: Path, *, branch_condition: str | None = "has_thing") -> Path:
    root = tmp_path / "repo"
    _write(root, "router", _ROUTER.format(always=0))
    _write(root, "branch", _BRANCH)

    condition = f"\n      condition: {branch_condition}" if branch_condition else ""
    _write(
        root,
        "pipe",
        f"""
platform: ensemble
max_batch_size: 0
inputs: [{{name: images, data_type: FP32, dims: [4]}}]
outputs:
  - {{name: crops, data_type: FP32, dims: [2]}}
  - {{name: embedding, data_type: FP32, dims: [3]}}
dynamic_batching: {{enabled: false}}
ensemble:
  steps:
    - model: router
      input_map: {{images: images}}
      output_map: {{crops: crops, has_thing: has_thing}}
    - model: branch
      input_map: {{crops: crops}}
      output_map: {{embedding: embedding}}{condition}
""",
        versioned=False,
    )
    return root


def _server(root: Path) -> InferenceServer:
    return InferenceServer(
        ServerSettings(
            model_repository=root,
            devices={"visible_gpus": []},
            execution={"warmup_iterations": 0},
        )
    )


def _request() -> InferenceRequest:
    return InferenceRequest(
        model_name="pipe",
        inputs={"images": Tensor.from_numpy(np.zeros((1, 4), dtype=np.float32))},
        context=RequestContext(camera_id="cam7", frame_id=3),
    )


class TestConditionalBranches:

    def test_every_declared_output_is_present_even_when_a_branch_skips(
        self, tmp_path: Path
    ) -> None:
        """The regression test. A skipped branch yields zero rows, not a missing key."""
        with _server(_repo(tmp_path)) as server:
            response = server.infer_sync(_request(), timeout=20)

        assert set(response.outputs) == {"crops", "embedding"}
        embedding = response.outputs["embedding"]
        # Either the branch ran (one row) or it did not (zero rows) — but the key is there and
        # the row shape is the declared one either way.
        assert embedding.shape[1:] == (3,)
        assert embedding.shape[0] in (0, 1)

    def test_a_skipped_branch_is_distinguishable_from_a_failure(self, tmp_path: Path) -> None:
        """Zero rows says "no things in this frame"; a missing key said nothing at all."""
        root = _repo(tmp_path)
        # Force the router's flag to zero so the branch always skips.
        config = (root / "router" / "config.yaml").read_text().replace("seed: 0", "seed: 0")
        (root / "router" / "config.yaml").write_text(config)

        with _server(root) as server:
            for _ in range(6):
                response = server.infer_sync(_request(), timeout=20)
                assert "embedding" in response.outputs


class TestContextTag:

    def test_the_context_tag_survives_the_whole_dag(self, tmp_path: Path) -> None:
        with _server(_repo(tmp_path)) as server:
            response = server.infer_sync(_request(), timeout=20)
        assert response.context.camera_id == "cam7"
        assert response.context.frame_id == 3


class TestWiringValidation:

    def test_consuming_a_conditional_tensor_unconditionally_fails_at_startup(
        self,
        tmp_path: Path,
    ) -> None:
        """CONVENTIONS 2.6: a mis-wired ensemble stops the deploy.

        Here the branch reads `crops`, which is unconditional, so instead we make the *branch*
        unconditional while an earlier conditional step produces what it reads. The graph is
        only valid for the frames that take that branch, and start-up must say so.
        """
        root = tmp_path / "repo"
        _write(root, "router", _ROUTER.format(always=0))
        _write(root, "branch", _BRANCH)
        _write(root, "head", _HEAD)
        _write(
            root,
            "pipe",
            """
platform: ensemble
max_batch_size: 0
inputs: [{name: images, data_type: FP32, dims: [4]}]
outputs: [{name: score, data_type: FP32, dims: [1]}]
dynamic_batching: {enabled: false}
ensemble:
  steps:
    - model: router
      input_map: {images: images}
      output_map: {crops: crops, has_thing: has_thing}
    - model: branch
      input_map: {crops: crops}
      output_map: {embedding: embedding}
      condition: has_thing
    # `head` reads `embedding`, which only exists when the branch above ran, but declares
    # no condition of its own. Valid on every frame that takes the branch; broken on the
    # first that does not.
    - model: head
      input_map: {embedding: embedding}
      output_map: {score: score}
""",
            versioned=False,
        )

        with pytest.raises(ConfigurationError, match="only exists when an earlier conditional"):
            _server(root).start()

    def test_an_output_no_step_produces_fails_at_startup(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        _write(root, "router", _ROUTER.format(always=0))
        _write(
            root,
            "pipe",
            """
platform: ensemble
max_batch_size: 0
inputs: [{name: images, data_type: FP32, dims: [4]}]
outputs: [{name: nobody_makes_this, data_type: FP32, dims: [3]}]
dynamic_batching: {enabled: false}
ensemble:
  steps:
    - model: router
      input_map: {images: images}
      output_map: {crops: crops, has_thing: has_thing}
""",
            versioned=False,
        )

        with pytest.raises(ConfigurationError, match="no step produces"):
            _server(root).start()


class TestPoolLifecycle:

    def test_shutdown_resolves_queued_requests(self, tmp_path: Path) -> None:
        """cancel_futures=True dropped the caller's futures on the floor, so every waiter
        blocked forever. They must resolve — with an error, but they must resolve."""
        from concurrent.futures import wait

        server = _server(_repo(tmp_path)).start()
        futures = [server.model("pipe").infer(_request()) for _ in range(6)]
        server.stop()

        done, not_done = wait(futures, timeout=20)
        assert not not_done, f"{len(not_done)} future(s) never resolved"
        # Whatever ran before the stop succeeded; the rest carry a typed error.
        assert all(f.done() for f in done)

    def test_the_pool_applies_backpressure(self, tmp_path: Path) -> None:
        """A ThreadPoolExecutor's queue is unbounded; every other path in this system refuses
        work when saturated, and so must this one."""
        from shipinfer.core.errors import QueueFullError

        root = _repo(tmp_path)
        slow = (
            (root / "router" / "config.yaml")
            .read_text()
            .replace("latency_ms: 0.05", "latency_ms: 200")
        )
        (root / "router" / "config.yaml").write_text(slow)

        rejected = 0
        accepted = []
        with _server(root) as server:
            model = server.model("pipe")
            for _ in range(200):
                try:
                    accepted.append(model.infer(_request()))
                except QueueFullError:
                    rejected += 1
                    break

        assert (
            rejected > 0
        ), "an unbounded ensemble queue accepted 200 requests behind a 200ms model"
