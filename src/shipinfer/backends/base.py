"""The model-backend contract — the seam every inference runtime plugs into.

A backend is given a batch of already-assembled input tensors and returns a batch of
output tensors. It does **not** decide what to batch, where to run, or when: that is the
scheduler's job. Keeping the contract that narrow is what lets TensorRT, ONNX Runtime,
torch and a deterministic mock be genuinely interchangeable, and it is why adding a
runtime is a new file plus a registration rather than an edit to the server.

The contract is deliberately synchronous. Asynchrony in this system lives one level up, in
the instance's worker thread and its stream pool: a backend that returned futures would
duplicate that machinery and make ordering guarantees impossible to reason about.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar

from shipinfer.core.logging import get_logger
from shipinfer.core.settings import ExecutionSettings
from shipinfer.core.types import Device, Tensor, TensorSpec
from shipinfer.repository import ModelArtifact, ModelConfig
from shipinfer.runtime.graphs import GraphCache
from shipinfer.runtime.memory import MemoryPool
from shipinfer.runtime.stream import StreamPool

__all__ = ["BackendContext", "ModelBackend"]

_LOG = get_logger("backends")


@dataclass(slots=True)
class BackendContext:
    """Everything a backend instance needs, assembled by the server.

    Passing one object rather than eight positional arguments is not cosmetic: it means a
    new capability (a second allocator, a profiler handle) reaches every backend without
    changing a single constructor signature.
    """

    artifact: ModelArtifact
    device: Device
    memory: MemoryPool
    execution: ExecutionSettings
    #: ``None`` on CPU, where there is nothing to overlap.
    streams: StreamPool | None = None
    #: ``None`` when graph capture is off or unsupported.
    graphs: GraphCache | None = None
    #: Free-form, backend-specific extras (a shared TRT logger, a torch dtype override).
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> ModelConfig:
        return self.artifact.config

    @property
    def instance_name(self) -> str:
        return f"{self.artifact.name}:{self.artifact.version}@{self.device}"


class ModelBackend(abc.ABC):
    """One loaded copy of one model on one device."""

    #: The ``platform`` value in ``config.yaml`` that selects this backend.
    platform: ClassVar[str] = "abstract"
    #: Refuse to construct on a CPU device. Set False for backends with a CPU path.
    requires_gpu: ClassVar[bool] = False

    def __init__(self, context: BackendContext) -> None:
        self._context = context
        self._initialized = False
        if self.requires_gpu and not context.device.is_cuda:
            raise ValueError(
                f"backend {self.platform!r} requires a CUDA device, got {context.device}"
            )

    # -- introspection -------------------------------------------------------------------

    @property
    def context(self) -> BackendContext:
        return self._context

    @property
    def device(self) -> Device:
        return self._context.device

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        """Declared inputs.

        The default is the config's view. A backend that can read the truth from its
        artefact (TensorRT can) should override and return *that*, so a config that has
        drifted from the engine fails at load time instead of producing garbage.
        """
        return self._context.config.input_specs

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        return self._context.config.output_specs

    @property
    def max_batch_size(self) -> int:
        return self._context.config.effective_max_batch_size

    # -- lifecycle -----------------------------------------------------------------------

    def initialize(self) -> None:
        """Load the model. Idempotent."""
        if self._initialized:
            return
        self._do_initialize()
        self._initialized = True
        _LOG.info("loaded %s on %s", self._context.instance_name, self.device)

    def finalize(self) -> None:
        """Release everything. Idempotent, and must not raise during shutdown."""
        if not self._initialized:
            return
        try:
            self._do_finalize()
        except Exception:
            _LOG.exception("error finalising %s", self._context.instance_name)
        finally:
            self._initialized = False

    @abc.abstractmethod
    def _do_initialize(self) -> None: ...

    def _do_finalize(self) -> None:
        """Override when there is something to release."""

    # -- execution -----------------------------------------------------------------------

    @abc.abstractmethod
    def execute(self, inputs: dict[str, Tensor], batch_size: int) -> dict[str, Tensor]:
        """Run one batch.

        Args:
            inputs: batch-major tensors, already validated against :attr:`input_specs`.
            batch_size: rows in the batch — passed explicitly because a padded batch's
                tensor shape and its *useful* row count can differ.

        Returns:
            Batch-major outputs with exactly ``batch_size`` rows.

        Raises:
            InferenceError: on any execution failure. The instance catches it, fails the
                batch's futures individually, and stays alive: one bad batch must not take
                a GPU worker down.
        """

    def warmup(self, iterations: int) -> None:
        """Run throwaway batches so the first real request does not pay for lazy init.

        Skipping this makes the first p99 after every deploy meaningless: cuBLAS
        autotuning, lazy CUDA module loading and TensorRT's first-call allocations all
        land on whichever unlucky request arrives first.

        The default builds zero-filled inputs from :attr:`input_specs`; a backend whose
        model rejects zeros should override.
        """
        if iterations <= 0:
            return
        try:
            batch = self._warmup_batch()
        except Exception as exc:
            _LOG.debug("cannot build a warm-up batch for %s: %s", self.platform, exc)
            return
        for _ in range(iterations):
            self.execute(batch, batch_size=1)
        _LOG.debug("warmed up %s with %d iteration(s)", self._context.instance_name, iterations)

    def _warmup_batch(self) -> dict[str, Tensor]:
        import numpy as np

        batch: dict[str, Tensor] = {}
        for spec in self.input_specs:
            shape = tuple(max(dim, 1) for dim in spec.shape)
            batch[spec.name] = Tensor.from_numpy(
                np.zeros((1, *shape), dtype=spec.dtype.numpy_dtype)
            )
        return batch

    def stats(self) -> dict[str, Any]:
        """Backend-specific counters, merged into the instance's stats."""
        return {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self._context.instance_name}>"
