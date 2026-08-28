"""TorchScript backend — for prototyping and CPU parity."""

from __future__ import annotations

from typing import Any

import numpy as np

from shipinfer.backends.base import BackendContext, ModelBackend
from shipinfer.core.errors import BackendUnavailableError, InferenceError
from shipinfer.core.types import Tensor

__all__ = ["TorchScriptBackend"]


class TorchScriptBackend(ModelBackend):
    """Runs a scripted/traced ``torch.jit`` module.

    The right backend while a model is still moving: no export step, no engine build, and
    the same weights the training code used. Also the reference the TensorRT numerics get
    compared against when an engine's output looks wrong.

    Inference runs under ``inference_mode`` and, on CUDA, in half precision when the config
    asks — both are the difference between "torch as a convenience" and "torch as something
    you could briefly ship".
    """

    platform = "pytorch"
    requires_gpu = False

    def __init__(self, context: BackendContext) -> None:
        super().__init__(context)
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError(
                'PyTorch is not installed. Install it with: pip install "shipinfer[torch]"'
            ) from exc
        self._torch: Any = torch
        self._module: Any = None
        self._torch_device = f"cuda:{self.device.index}" if self.device.is_cuda else "cpu"
        self._half = bool(context.config.parameters.get("fp16", False)) and self.device.is_cuda

    def _do_initialize(self) -> None:
        params = self.context.config.parameters
        path = self.context.artifact.file(str(params.get("model_file", "model.pt")))
        module = self._torch.jit.load(str(path), map_location=self._torch_device)
        module.eval()
        if self._half:
            module.half()
        # Fuse and specialise on the first few calls rather than every call.
        self._module = (
            self._torch.jit.optimize_for_inference(module)
            if hasattr(self._torch.jit, "optimize_for_inference")
            else module
        )

    def _do_finalize(self) -> None:
        self._module = None
        if self.device.is_cuda:
            self._torch.cuda.empty_cache()

    def execute(self, inputs: dict[str, Tensor], batch_size: int) -> dict[str, Tensor]:
        if self._module is None:
            raise InferenceError(f"{self.context.instance_name} is not initialised")

        torch = self._torch
        ordered = [spec.name for spec in self.input_specs]
        args = []
        for name in ordered:
            array = inputs[name].numpy()
            tensor = torch.from_numpy(array).to(self._torch_device, non_blocking=True)
            args.append(tensor.half() if self._half and tensor.is_floating_point() else tensor)

        try:
            with torch.inference_mode():
                result = self._module(*args)
        except Exception as exc:
            raise InferenceError(f"{self.context.instance_name}: {exc}") from exc

        values = result if isinstance(result, (tuple, list)) else (result,)
        specs = self.output_specs
        if len(values) != len(specs):
            raise InferenceError(
                f"{self.context.instance_name}: module returned {len(values)} tensor(s) "
                f"but the config declares {len(specs)}"
            )
        return {
            spec.name: Tensor.from_numpy(
                np.ascontiguousarray(
                    value.detach().to("cpu").to(_torch_dtype(torch, spec)).numpy()
                )
            )
            for spec, value in zip(specs, values, strict=True)
        }


def _torch_dtype(torch: Any, spec: Any) -> Any:
    """Cast back to the declared dtype so an fp16 module still honours an fp32 contract."""
    return {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "FP64": torch.float64,
        "INT32": torch.int32,
        "INT64": torch.int64,
        "INT8": torch.int8,
        "UINT8": torch.uint8,
        "BOOL": torch.bool,
    }[spec.dtype.value]
