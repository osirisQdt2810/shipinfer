"""Loading a serialised TensorRT engine and reading its true I/O contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shipinfer.core.errors import BackendLoadError
from shipinfer.core.logging import get_logger
from shipinfer.core.types import DYNAMIC, DataType, TensorSpec

__all__ = ["EngineIO", "LoadedEngine", "load_engine", "trt_dtype_to_datatype"]

_LOG = get_logger("backends.tensorrt.engine")

#: TensorRT's ``DataType`` names -> ours. Keyed on the name so this table survives the
#: enum reshuffles between TensorRT 8 and 10.
_TRT_DTYPE_NAMES = {
    "FLOAT": DataType.FP32,
    "HALF": DataType.FP16,
    "INT8": DataType.INT8,
    "INT32": DataType.INT32,
    "INT64": DataType.INT64,
    "BOOL": DataType.BOOL,
    "UINT8": DataType.UINT8,
}


def trt_dtype_to_datatype(trt_dtype: Any) -> DataType:
    name = str(trt_dtype).rsplit(".", 1)[-1].upper()
    try:
        return _TRT_DTYPE_NAMES[name]
    except KeyError:
        raise BackendLoadError(f"unsupported TensorRT dtype {name!r}") from None


@dataclass(frozen=True, slots=True)
class EngineIO:
    """One engine tensor as the engine itself describes it."""

    name: str
    dtype: DataType
    #: Full shape *including* the batch dimension, with -1 for dynamic extents.
    shape: tuple[int, ...]
    is_input: bool

    def to_spec(self, strip_batch: bool) -> TensorSpec:
        """The server-facing spec.

        ``strip_batch`` reflects ``max_batch_size > 0`` in the config: the server owns dim
        0 in that case, so it must not appear in the declared shape.
        """
        shape = self.shape[1:] if strip_batch and self.shape else self.shape
        return TensorSpec(name=self.name, dtype=self.dtype, shape=tuple(shape))


@dataclass(slots=True)
class LoadedEngine:
    """A deserialised engine plus its introspected I/O."""

    engine: Any
    io: tuple[EngineIO, ...]
    path: Path

    @property
    def inputs(self) -> tuple[EngineIO, ...]:
        return tuple(t for t in self.io if t.is_input)

    @property
    def outputs(self) -> tuple[EngineIO, ...]:
        return tuple(t for t in self.io if not t.is_input)

    @property
    def has_dynamic_shapes(self) -> bool:
        return any(DYNAMIC in t.shape for t in self.io)


def load_engine(trt: Any, logger: Any, path: Path) -> LoadedEngine:
    """Deserialise ``path`` and read back what the engine actually expects.

    Reading the contract from the engine rather than trusting ``config.yaml`` is the whole
    point of this function. A config that has drifted from its engine — a re-export with a
    renamed output, a changed input resolution — otherwise produces silently wrong results
    or a CUDA error deep inside ``execute_async``. Here it is a load-time failure with both
    shapes printed.

    Raises:
        BackendLoadError: if the file is missing, unreadable, or was built for a different
            TensorRT version or GPU architecture.
    """
    if not path.is_file():
        raise BackendLoadError(f"TensorRT engine not found: {path}")

    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise BackendLoadError(f"cannot read TensorRT engine {path}: {exc}") from exc

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise BackendLoadError(
            f"TensorRT refused to deserialise {path}. An engine is specific to the "
            "TensorRT version and GPU architecture it was built on — rebuild it on this host."
        )

    io = tuple(_introspect(trt, engine))
    _LOG.debug(
        "engine %s: %d input(s) %s, %d output(s) %s",
        path.name,
        len([t for t in io if t.is_input]),
        [t.name for t in io if t.is_input],
        len([t for t in io if not t.is_input]),
        [t.name for t in io if not t.is_input],
    )
    return LoadedEngine(engine=engine, io=io, path=path)


def _introspect(trt: Any, engine: Any) -> list[EngineIO]:
    """Read the I/O table across the TensorRT 8.5 API break.

    TensorRT 8.5 replaced index-addressed bindings with named tensors, and 10.0 removed the
    old API entirely. Supporting both is three lines here and saves every deployment from
    being pinned to one TensorRT minor version.
    """
    if hasattr(engine, "num_io_tensors"):  # TensorRT >= 8.5
        out: list[EngineIO] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            out.append(
                EngineIO(
                    name=name,
                    dtype=trt_dtype_to_datatype(engine.get_tensor_dtype(name)),
                    shape=tuple(int(d) for d in engine.get_tensor_shape(name)),
                    is_input=engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                )
            )
        return out

    return [  # TensorRT < 8.5
        EngineIO(
            name=engine.get_binding_name(i),
            dtype=trt_dtype_to_datatype(engine.get_binding_dtype(i)),
            shape=tuple(int(d) for d in engine.get_binding_shape(i)),
            is_input=engine.binding_is_input(i),
        )
        for i in range(engine.num_bindings)
    ]
