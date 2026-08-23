"""A deterministic, hardware-free backend."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from shipinfer.backends.base import BackendContext, ModelBackend
from shipinfer.backends.registry import BACKENDS
from shipinfer.core.errors import InferenceError
from shipinfer.core.types import Tensor

__all__ = ["MockBackend"]


@BACKENDS.register("mock", "fake", description="Deterministic fake — tests and benchmarks")
class MockBackend(ModelBackend):
    """Produces shaped, reproducible outputs without touching a GPU.

    This is not a testing afterthought, it is load-bearing. The scheduler, the fairness
    guarantees, the spillover behaviour and the batching windows are the parts of this
    system most worth verifying, and all of them are observable with a backend that merely
    *takes time*. Verifying them against a real TensorRT engine would mean the tests need
    sixteen GPUs, which means they would be written once and then never run.

    ``parameters`` in the model config drive it:

    * ``latency_ms`` — how long one execution sleeps, whatever the batch size;
    * ``per_item_latency_ms`` — additional time per row, so batching shows a real win;
    * ``fail_every`` — raise on every Nth execution, to exercise the error path;
    * ``seed`` — makes the outputs reproducible across runs.
    """

    platform = "mock"
    requires_gpu = False

    def __init__(self, context: BackendContext) -> None:
        super().__init__(context)
        params = context.config.parameters
        self._latency_s = float(params.get("latency_ms", 0.0)) / 1000.0
        self._per_item_s = float(params.get("per_item_latency_ms", 0.0)) / 1000.0
        self._fail_every = int(params.get("fail_every", 0))
        self._rng = np.random.default_rng(int(params.get("seed", 0)))
        self._executions = 0

    def _do_initialize(self) -> None:
        return None

    def execute(self, inputs: dict[str, Tensor], batch_size: int) -> dict[str, Tensor]:
        self._executions += 1
        if self._fail_every and self._executions % self._fail_every == 0:
            raise InferenceError(
                f"mock backend failing execution #{self._executions} by config"
            )

        delay = self._latency_s + self._per_item_s * batch_size
        if delay > 0:
            # A real sleep, not a busy loop: the point is to model a *blocked worker*, and
            # a spin would make the scheduler look better than it is by keeping a core hot.
            time.sleep(delay)

        outputs: dict[str, Tensor] = {}
        for spec in self.output_specs:
            shape = tuple(max(dim, 1) for dim in spec.shape)
            if spec.dtype.numpy_dtype.kind == "f":
                data = self._rng.random((batch_size, *shape), dtype=np.float32)
                data = data.astype(spec.dtype.numpy_dtype, copy=False)
            else:
                data = self._rng.integers(
                    0, 8, size=(batch_size, *shape), dtype=np.int64
                ).astype(spec.dtype.numpy_dtype, copy=False)
            outputs[spec.name] = Tensor.from_numpy(data)
        return outputs

    def stats(self) -> dict[str, Any]:
        return {"executions": self._executions}
