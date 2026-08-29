"""Model backends — one execution runtime per module, selected through :data:`BACKENDS`.

Importing this package registers the light backends eagerly and the heavy ones lazily, so
``import shipinfer`` never pulls in TensorRT, torch or onnxruntime. Adding a runtime is a
new module plus one registration; nothing else in the tree changes.
"""

from shipinfer.backends.base import BackendContext, ModelBackend
from shipinfer.backends.registry import BACKENDS, build_backend

__all__ = [
    "BACKENDS",
    "BackendContext",
    "ModelBackend",
    "build_backend",
]
