"""The fixture that replaced the mock backend, tested through the path the server runs.

``latency_ms:`` is a promise several tests rest their whole argument on -- the rate limiter
only proves anything if executions genuinely overlap, and the write-race test only proves
anything if a slow model really finishes after a fast one. A fixture that is *nearly* free
turns those into tests that pass against a deleted limiter.

So the assertions here are made after ``torch.jit.optimize_for_inference``, which is what
``TorchScriptBackend._do_initialize`` applies. Timed on the machine running the suite and
compared against the fixture's own calibration, never against a hard-coded millisecond
count, so a slow CI box cannot make this flaky.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from tests.support.models import build_model, iterations_for, unit_cost_ms


def _through_the_backend(
    module: torch.jit.ScriptModule, tmp_path: Path
) -> torch.jit.ScriptModule:
    """Save, load and optimise exactly as ``TorchScriptBackend`` does."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.pt"
    module.save(str(path))
    return torch.jit.optimize_for_inference(torch.jit.load(str(path)).eval())


def _milliseconds(module: torch.jit.ScriptModule, runs: int = 5) -> float:
    sample = torch.zeros(1, 4)
    for _ in range(3):
        module(sample)
    start = time.perf_counter()
    for _ in range(runs):
        module(sample)
    return (time.perf_counter() - start) * 1000.0 / runs


class TestTheWorkSurvivesTheBackendsOwnOptimiser:
    """The defect this file exists for: the cost was optimised away and nothing noticed."""

    def test_a_slow_model_is_still_slow_after_freezing(self, tmp_path: Path) -> None:
        """It was not. `optimize_for_inference` freezes the module, so work that started
        from a buffer was constant-folded and a result multiplied by zero was dead code: a
        `latency_ms: 60` model executed in 0.05 ms, and every test that needed real overlap
        passed vacuously.
        """
        unit = unit_cost_ms()
        slow = build_model([4], [4], iterations_for(20.0, unit))
        idle = build_model([4], [4], 0)

        slow_ms = _milliseconds(_through_the_backend(slow, tmp_path / "slow"))
        idle_ms = _milliseconds(_through_the_backend(idle, tmp_path / "idle"))

        assert slow_ms > 20 * idle_ms, "the declared work must survive the optimiser"
        assert slow_ms > 2.0, f"20 ms of declared work took {slow_ms:.3f} ms"

    def test_the_cost_is_linear_in_the_declared_work(self, tmp_path: Path) -> None:
        """Ten times the iterations, near enough ten times the cost.

        Not pedantry: the spin used to *contract* toward zero, every entry went subnormal
        within a hundred iterations, and subnormal matmul is an order of magnitude slower --
        so cost per iteration grew with the count and no calibration could be right at both
        ends. The matrix is orthogonal now, which is what keeps every iteration one price.
        """
        unit = unit_cost_ms()
        few = _through_the_backend(build_model([4], [4], 200), tmp_path / "few")
        many = _through_the_backend(build_model([4], [4], 2000), tmp_path / "many")

        ratio = _milliseconds(many) / max(_milliseconds(few), 1e-6)

        assert 5.0 < ratio < 20.0, f"10x the work cost {ratio:.1f}x the time"
        assert unit > 0

    @pytest.mark.parametrize("target_ms", [1.0, 5.0])
    def test_a_declared_latency_lands_near_what_it_asked_for(
        self, target_ms: float, tmp_path: Path
    ) -> None:
        """Within 3x, which is loose on purpose: this is a shared machine and the point is
        that the number means something, not that it is a benchmark."""
        module = build_model([4], [4], iterations_for(target_ms, unit_cost_ms()))

        measured = _milliseconds(_through_the_backend(module, tmp_path / f"m{target_ms}"))

        assert (
            target_ms / 3 < measured < target_ms * 3
        ), f"asked {target_ms}, got {measured:.3f}"


class TestTheFixtureMatchesTheShapeItsConfigDeclares:
    """A landmine under shared infrastructure: the output side had no shape at all.

    `_shapes()` carried the *input* dims into the trace from the start; outputs went through
    `math.prod`, so a config declaring `dims: [300, 6]` got a module returning `(N, 1800)`.
    The first real request would hit `Tensor.validate_against` and be refused — with an error
    indistinguishable from `disagrees_with_its_config`'s deliberate one, which is what makes
    it worth a test rather than a fix alone.
    """

    def _repository(self, root: Path, dims: str) -> Path:
        (root / "det" / "1").mkdir(parents=True)
        (root / "det" / "config.yaml").write_text(
            "platform: pytorch\n"
            "max_batch_size: 8\n"
            "inputs: [{name: images, data_type: FP32, dims: [3, 64, 64]}]\n"
            f"outputs: [{{name: output0, data_type: FP32, dims: {dims}}}]\n"
            "instance_groups: [{kind: KIND_CPU, count: 1}]\n"
            "dynamic_batching: {enabled: false}\n"
        )
        return root

    def test_a_multi_dimensional_output_keeps_its_declared_dims(self, tmp_path: Path) -> None:
        """The detector's own shape, which `tests/cli/test_run_engine.py` already declares."""
        from tests.support.models import materialise

        root = self._repository(tmp_path / "repo", "[300, 6]")
        materialise(root)

        module = torch.jit.load(str(root / "det" / "1" / "model.pt"))
        output = module(torch.zeros(1, 3, 64, 64))

        assert tuple(output.shape) == (1, 300, 6), "flattened to (N, 1800) before this"

    def test_a_one_dimensional_output_is_unchanged(self, tmp_path: Path) -> None:
        """The common case must not grow a spurious axis."""
        from tests.support.models import materialise

        root = self._repository(tmp_path / "repo", "[16]")
        materialise(root)

        module = torch.jit.load(str(root / "det" / "1" / "model.pt"))

        assert tuple(module(torch.zeros(1, 3, 64, 64)).shape) == (1, 16)


class TestTheCostHoldsOnAWorkerThreadToo:
    """Every model instance runs on its own thread, so main-thread timing proves nothing.

    TorchScript profiles a loop the first time a *thread* executes it, and that overhead is
    per iteration. With a fine-grained unit of work a `latency_ms: 200` fixture needed ~8000
    iterations and took **over a minute** on an instance's worker thread while taking 200 ms
    on the main one — every request through it timed out, and the offline tier hung rather
    than failed. This is the shape of that regression.
    """

    def test_a_slow_model_costs_the_same_on_a_fresh_thread(self, tmp_path: Path) -> None:
        import threading

        module = _through_the_backend(
            build_model([2], [2], iterations_for(50.0, unit_cost_ms())), tmp_path / "slow"
        )
        sample = torch.zeros(1, 2)
        module(sample)

        elapsed: dict[str, float] = {}

        def on_worker() -> None:
            start = time.perf_counter()
            module(sample)
            elapsed["first"] = (time.perf_counter() - start) * 1000.0

        worker = threading.Thread(target=on_worker)
        worker.start()
        worker.join(timeout=30.0)

        assert not worker.is_alive(), "a fresh thread's first execution never finished"
        assert elapsed["first"] < 5000, (
            f"50 ms of declared work took {elapsed['first']:.0f} ms on a worker thread; the "
            "unit of work is too fine and TorchScript's per-thread profiling dominates"
        )
