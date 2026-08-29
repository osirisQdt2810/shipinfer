"""Registry of model backends.

The heavy backends are registered **lazily**: importing this module must not import
TensorRT, torch or onnxruntime. ``shipinfer repo ls`` should list a TensorRT model on a
laptop that has never had CUDA installed, and a server that loads only ONNX models should
not pay a two-second torch import for the privilege.
"""

from __future__ import annotations

from shipinfer.backends.base import BackendContext, ModelBackend
from shipinfer.core.registry import Registry

__all__ = ["BACKENDS", "build_backend"]

BACKENDS: Registry[ModelBackend] = Registry("backend", ModelBackend)

# Lazy registrations: every backend here costs an import of a heavy runtime, so none is
# paid for until a config names it.
BACKENDS.register_lazy(
    "tensorrt",
    "shipinfer.backends.tensorrt:TensorRTBackend",
    "trt",
    description="NVIDIA TensorRT engines (.plan) — the production path",
)
BACKENDS.register_lazy(
    "onnxruntime",
    "shipinfer.backends.onnx:OnnxRuntimeBackend",
    "onnx",
    description="ONNX Runtime with the CUDA execution provider",
)
BACKENDS.register_lazy(
    "pytorch",
    "shipinfer.backends.torch_backend:TorchScriptBackend",
    "torch",
    "torchscript",
    description="TorchScript modules, CPU or CUDA",
)


def build_backend(platform: str, context: BackendContext) -> ModelBackend:
    """Instantiate the backend named by a model's ``platform``."""
    return BACKENDS.create(platform, context)
