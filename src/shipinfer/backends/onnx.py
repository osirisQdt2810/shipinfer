"""ONNX Runtime backend — the portable fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from shipinfer.backends.base import BackendContext, ModelBackend
from shipinfer.core.errors import BackendUnavailableError, InferenceError
from shipinfer.core.logging import get_logger
from shipinfer.core.types import DataType, Tensor, TensorSpec

__all__ = ["OnnxRuntimeBackend"]

_LOG = get_logger("backends.onnx")

#: ONNX type strings -> ours. Only the ones a perception model actually emits.
_ONNX_TYPES = {
    "tensor(float)": DataType.FP32,
    "tensor(float16)": DataType.FP16,
    "tensor(int64)": DataType.INT64,
    "tensor(int32)": DataType.INT32,
    "tensor(int8)": DataType.INT8,
    "tensor(uint8)": DataType.UINT8,
    "tensor(bool)": DataType.BOOL,
}


class OnnxRuntimeBackend(ModelBackend):
    """Runs an ``.onnx`` graph, preferring the CUDA execution provider.

    Portability, not peak speed: bring-up, numeric comparison against TensorRT, and
    hosts where building an engine is impractical. Deliberately not wired to CUDA
    graphs — ONNX Runtime owns its allocations and stream, so the stable-address
    precondition for capture is not ours to guarantee.
    """

    platform = "onnxruntime"
    requires_gpu = False

    def __init__(self, context: BackendContext) -> None:
        super().__init__(context)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendUnavailableError(
                'ONNX Runtime is not installed. Install it with: pip install "shipinfer[onnx]"'
            ) from exc
        self._ort: Any = ort
        self._session: Any = None
        self._output_names: tuple[str, ...] = ()

    def _do_initialize(self) -> None:
        params = self.context.config.parameters
        path = self.context.artifact.file(str(params.get("model_file", "model.onnx")))

        options = self._ort.SessionOptions()
        options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # One intra-op thread per session: the server already runs one worker per instance
        # and several instances per GPU, so letting ORT also fan out oversubscribes the box
        # and makes tail latency worse, not better.
        options.intra_op_num_threads = int(params.get("intra_op_threads", 1))
        options.inter_op_num_threads = int(params.get("inter_op_threads", 1))

        providers: list[Any] = []
        if self.device.is_cuda:
            providers.append(("CUDAExecutionProvider", {"device_id": self.device.index}))
        providers.append("CPUExecutionProvider")

        self._session = self._ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        active = self._session.get_providers()
        if self.device.is_cuda and "CUDAExecutionProvider" not in active:
            _LOG.warning(
                "%s asked for CUDA but ONNX Runtime fell back to %s",
                self.context.instance_name,
                active,
            )
        self._output_names = tuple(o.name for o in self._session.get_outputs())

    def _do_finalize(self) -> None:
        self._session = None

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        if self._session is None:
            return super().input_specs
        return tuple(_to_spec(i) for i in self._session.get_inputs())

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        if self._session is None:
            return super().output_specs
        return tuple(_to_spec(o) for o in self._session.get_outputs())

    def execute(self, inputs: dict[str, Tensor], batch_size: int) -> dict[str, Tensor]:
        if self._session is None:
            raise InferenceError(f"{self.context.instance_name} is not initialised")
        feed = {name: tensor.numpy() for name, tensor in inputs.items()}
        try:
            results = self._session.run(list(self._output_names), feed)
        except Exception as exc:
            raise InferenceError(f"{self.context.instance_name}: {exc}") from exc
        return {
            name: Tensor.from_numpy(np.ascontiguousarray(array))
            for name, array in zip(self._output_names, results, strict=True)
        }


def _to_spec(node: Any) -> TensorSpec:
    dtype = _ONNX_TYPES.get(node.type)
    if dtype is None:
        raise InferenceError(f"unsupported ONNX tensor type {node.type!r} for {node.name!r}")
    # ORT reports symbolic dims as strings; -1 is our word for the same thing.
    dims = tuple(d if isinstance(d, int) and d > 0 else -1 for d in node.shape[1:])
    return TensorSpec(name=node.name, dtype=dtype, shape=dims)
