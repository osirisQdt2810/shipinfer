"""The TensorRT backend, split by concern.

* :mod:`~shipinfer.backends.tensorrt.logger` — ``trt.ILogger`` -> this project's logging
* :mod:`~shipinfer.backends.tensorrt.engine` — deserialise and introspect the real I/O
* :mod:`~shipinfer.backends.tensorrt.bindings` — persistent device + pinned buffers
* :mod:`~shipinfer.backends.tensorrt.backend` — the :class:`ModelBackend` itself

Importing this package imports TensorRT, which is why the registry registers it lazily.
"""

from shipinfer.backends.tensorrt.backend import TensorRTBackend
from shipinfer.backends.tensorrt.bindings import Binding, BindingSet
from shipinfer.backends.tensorrt.engine import EngineIO, LoadedEngine, load_engine

__all__ = [
    "Binding",
    "BindingSet",
    "EngineIO",
    "LoadedEngine",
    "TensorRTBackend",
    "load_engine",
]
