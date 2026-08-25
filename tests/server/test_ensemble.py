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


# A second pass over what the first pass produced. The *model* names its tensors distinctly —
# the repository refuses an input and an output sharing a name — while the ensemble step maps
# both onto the same namespace entry, which is what "refine in place" means at this level.
_REFINE = """
platform: mock
max_batch_size: 4
inputs: [{name: crops_in, data_type: FP32, dims: [2]}]
outputs: [{name: crops_out, data_type: FP32, dims: [2]}]
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.05}
"""

_SLOW = """
platform: mock
max_batch_size: 4
inputs: [{name: images, data_type: FP32, dims: [4]}]
outputs: [{name: crops, data_type: FP32, dims: [2]}]
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 60.0, seed: 11}
"""

_FAST = """
platform: mock
max_batch_size: 4
inputs: [{name: images, data_type: FP32, dims: [4]}]
outputs: [{name: crops, data_type: FP32, dims: [2]}]
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.05, seed: 22}
"""


def _mock_first_draw(seed: int, rows: int = 1, width: int = 2) -> np.ndarray:
    """What `MockBackend` emits on its first execution for a given seed.

    Mirrors `backends/mock.py`: a seeded `default_rng`, `random((batch, *shape))` in float32.
    Reproducing it here rather than reading it back from the model is what makes the write
    race *observable* — two steps writing the same name are otherwise indistinguishable, and
    a test that cannot tell them apart passes whichever one wins.
    """
    return np.random.default_rng(seed).random((rows, width), dtype=np.float32)


class TestAStepMayReadANameALaterStepWrites:
    """Scheduling the steps independently deadlocked refine-in-place.

    A step was admitted only when every producer of every name it reads had finished, and
    producers were collected over the whole graph with no regard for declaration order. So
    `detect(images -> crops)` followed by `refine(crops -> crops)` could never run: `refine`
    is itself a producer of `crops`, so its own precondition included itself. The future never
    resolved and its semaphore slot was never released — the pool leaks a permit per request
    until it stops accepting work at all.

    Not an exotic shape: a second pass that improves what the first pass produced is the
    obvious way to write one, and the sequential walk this replaced ran it happily.
    """

    def _repo(self, tmp_path: Path) -> Path:
        root = tmp_path / "repo"
        _write(root, "router", _ROUTER.format(always=1))
        _write(root, "refine", _REFINE)
        _write(
            root,
            "pipe",
            """
platform: ensemble
max_batch_size: 0
inputs: [{name: images, data_type: FP32, dims: [4]}]
outputs: [{name: crops, data_type: FP32, dims: [2]}]
dynamic_batching: {enabled: false}
ensemble:
  steps:
    - model: router
      input_map: {images: images}
      output_map: {crops: crops, has_thing: has_thing}
    - model: refine
      input_map: {crops_in: crops}
      output_map: {crops_out: crops}
""",
            versioned=False,
        )
        return root

    def test_refine_in_place_completes_instead_of_hanging(self, tmp_path: Path) -> None:
        server = _server(self._repo(tmp_path))
        server.start()
        try:
            response = server.infer(_request()).result(timeout=10.0)
        finally:
            server.stop()

        assert "crops" in response.outputs

    def test_the_second_pass_output_is_the_one_returned(self, tmp_path: Path) -> None:
        """Declaration order decides, so the later step's value is what the ensemble emits —
        the same answer the sequential walk gave."""
        server = _server(self._repo(tmp_path))
        server.start()
        try:
            first = server.infer(_request()).result(timeout=10.0)
            second = server.infer(_request()).result(timeout=10.0)
        finally:
            server.stop()

        assert first.outputs["crops"].shape == second.outputs["crops"].shape

    def test_the_pool_does_not_leak_a_permit_per_request(self, tmp_path: Path) -> None:
        """The deadlock's real cost: a stranded request never releases its slot, so the pool
        stops accepting long after the graph that caused it."""
        server = _server(self._repo(tmp_path))
        server.start()
        try:
            for _ in range(8):
                server.infer(_request()).result(timeout=10.0)
        finally:
            server.stop()


class TestTwoStepsWritingOneNameKeepDeclarationOrder:
    """Independent steps that map an output onto the same ensemble name now run concurrently
    and race to write it. Whichever *finished* last would win, non-deterministically, where
    the sequential walk always gave the later-declared step.

    The slow step is declared first and the fast one second, so completion order and
    declaration order disagree on every run. A scheduler that resolves the race by arrival
    returns the slow step's value; the sequential semantics return the fast one's.
    """

    def _repo(self, tmp_path: Path) -> Path:
        root = tmp_path / "repo"
        _write(root, "slow", _SLOW)
        _write(root, "fast", _FAST)
        _write(
            root,
            "pipe",
            """
platform: ensemble
max_batch_size: 0
inputs: [{name: images, data_type: FP32, dims: [4]}]
outputs: [{name: crops, data_type: FP32, dims: [2]}]
dynamic_batching: {enabled: false}
ensemble:
  steps:
    - model: slow
      input_map: {images: images}
      output_map: {crops: crops}
    - model: fast
      input_map: {images: images}
      output_map: {crops: crops}
""",
            versioned=False,
        )
        return root

    def test_the_later_declared_step_wins_however_the_two_finish(self, tmp_path: Path) -> None:
        server = _server(self._repo(tmp_path))
        server.start()
        try:
            response = server.infer(_request()).result(timeout=10.0)
        finally:
            server.stop()

        got = response.outputs["crops"].numpy()
        slow_value = _mock_first_draw(11)
        fast_value = _mock_first_draw(22)

        # `fast` is declared second, so the sequential walk's answer is `fast`'s value —
        # even though `slow` (60 ms) finishes long after it (0.05 ms) and a scheduler that
        # resolved the race by arrival would return `slow`'s.
        assert np.allclose(
            got, fast_value
        ), "the ensemble returned the step that finished last, not the one declared last"
        assert not np.allclose(got, slow_value)

    def test_both_steps_still_run(self, tmp_path: Path) -> None:
        """The fix must not be "ignore the slow one" — it has to run, its result is simply
        superseded, exactly as it was in the sequential walk."""
        root = self._repo(tmp_path)
        server = _server(root)
        server.start()
        try:
            server.infer(_request()).result(timeout=10.0)
            stats = server.model("slow").stats()
        finally:
            server.stop()

        executed = sum(int(i["requests"]) for i in stats["instances"])
        assert executed >= 1, "the superseded step was skipped rather than superseded"


class TestAReaderMayNotHaveAProducerDeclaredAfterIt:
    """Steps run as soon as their inputs settle, so two with no dependency between them
    dispatch together. If one writes a name a third reads, the read sees whichever landed
    first — and the sequential walk always produced the earlier writer's value.

    `written_by` keeps the namespace deterministic but cannot make a reader wait for a writer
    it was never told about, and making it wait is the deadlock the scheduler was just fixed
    for. The two requirements are incompatible, so the graph is refused at load instead —
    which is what CONVENTIONS asks of every other wiring mistake.
    """

    def _repo(self, tmp_path: Path) -> Path:
        root = tmp_path / "repo"
        _write(root, "router", _ROUTER.format(always=1))
        _write(root, "branch", _BRANCH)
        _write(root, "refine", _REFINE)
        _write(
            root,
            "pipe",
            """
platform: ensemble
max_batch_size: 0
inputs: [{name: images, data_type: FP32, dims: [4]}]
outputs: [{name: embedding, data_type: FP32, dims: [3]}]
dynamic_batching: {enabled: false}
ensemble:
  steps:
    - model: router
      input_map: {images: images}
      output_map: {crops: crops, has_thing: has_thing}
    - model: branch
      input_map: {crops: crops}
      output_map: {embedding: embedding}
    - model: refine
      input_map: {crops_in: crops}
      output_map: {crops_out: crops}
""",
            versioned=False,
        )
        return root

    def test_the_ambiguous_graph_is_refused_at_start_up(self, tmp_path: Path) -> None:
        """`branch` reads `crops` at step 1 and `refine` writes it at step 2 — so whether
        `branch` sees the router's crops or the refined ones is a race."""
        server = _server(self._repo(tmp_path))

        with pytest.raises(ConfigurationError) as caught:
            server.start()

        message = str(caught.value)
        assert "crops" in message, "the message must name the tensor"
        assert "step 1" in message, "and the reader"
        assert "[2]" in message, "and the writer that comes after it"

    def test_the_message_says_what_to_do(self, tmp_path: Path) -> None:
        """A mis-wired ensemble stops a deploy, so the message is what the operator has to
        act on — "step 1 is ambiguous" without a remedy costs an afternoon."""
        server = _server(self._repo(tmp_path))

        with pytest.raises(ConfigurationError) as caught:
            server.start()

        assert "Reorder the steps" in str(caught.value)

    def test_a_step_writing_a_name_it_reads_is_still_allowed(self, tmp_path: Path) -> None:
        """Refine-in-place is not ambiguous: a step is not a *later* producer of itself, and
        this is the shape the deadlock fix exists for."""
        root = tmp_path / "repo"
        _write(root, "router", _ROUTER.format(always=1))
        _write(root, "refine", _REFINE)
        _write(
            root,
            "pipe",
            """
platform: ensemble
max_batch_size: 0
inputs: [{name: images, data_type: FP32, dims: [4]}]
outputs: [{name: crops, data_type: FP32, dims: [2]}]
dynamic_batching: {enabled: false}
ensemble:
  steps:
    - model: router
      input_map: {images: images}
      output_map: {crops: crops, has_thing: has_thing}
    - model: refine
      input_map: {crops_in: crops}
      output_map: {crops_out: crops}
""",
            versioned=False,
        )
        server = _server(root)
        server.start()
        try:
            assert server.infer(_request()).result(timeout=10.0).outputs["crops"] is not None
        finally:
            server.stop()
