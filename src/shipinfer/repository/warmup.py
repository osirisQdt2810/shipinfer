"""Turning ``model_warmup`` in a ``config.yaml`` into tensors a backend can execute.

Triton's ``model_warmup`` exists because *how often* you warm up is the less interesting
half of the question. A fixed iteration count of zeros loads the CUDA modules and pays
TensorRT's first-call allocations, and then the first real frame still finds an unwarmed
path: a detector's NMS never sorts when the image is blank, a TensorRT engine picks tactics
per shape, and a fused preprocessing kernel takes a different branch on a frame with ships
in it. The sample decides which kernels get warmed; the count only decides how many times.

This lives in ``repository`` rather than in ``backends`` because it is a *reading of the
config*: numpy and the config are all it needs, so it is testable with no accelerator and no
backend at all. The backend receives finished tensors and never learns where they came from,
which is the contract in ``backends/base.py`` (ADR-001).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.types import DataType, Tensor
from shipinfer.repository.model_config import ModelConfig, WarmupInput, WarmupSample

__all__ = ["WarmupBatch", "build_warmup_batches"]

#: Fixed seed for ``random_data``. A warm-up that differs run to run makes two deployments
#: of the same engine incomparable, and the point of random data here is the *code path* the
#: values take, not the values.
_SEED = 20260824


@dataclass(frozen=True, slots=True)
class WarmupBatch:
    """One materialised warm-up sample, ready to hand to ``ModelBackend.execute``.

    Built once per instance at load time, so it is allowed to allocate — this is the only
    place in the serving path where that is true, and it is true because it happens before
    the instance reports ready.
    """

    name: str
    inputs: dict[str, Tensor]
    batch_size: int
    count: int


def build_warmup_batches(config: ModelConfig, version_dir: Path) -> tuple[WarmupBatch, ...]:
    """Materialise every ``model_warmup`` sample declared by ``config``.

    Args:
        config: the model's config. ``model_warmup`` empty yields an empty tuple, which is
            the caller's signal to fall back to the implicit zero-filled warm-up.
        version_dir: the model's version directory — where ``input_data_file`` is resolved
            from, matching Triton, where warm-up data files live beside the artefact.

    Returns:
        One :class:`WarmupBatch` per declared sample, in config order.

    Raises:
        ConfigurationError: for a data file that is missing, unreadable, or the wrong size
            for the shape it claims to fill. Loudly, at load time: a warm-up that silently
            did not happen is worse than no warm-up, because the deployment then believes
            its first p99 is representative.
    """
    if not config.model_warmup:
        return ()

    declared = {io.name: io for io in config.inputs}
    rng = np.random.default_rng(_SEED)
    batches: list[WarmupBatch] = []
    for sample in config.model_warmup:
        inputs = {
            name: _tensor_for(
                config=config,
                sample=sample,
                tensor_name=name,
                warmup_input=source,
                dtype=declared[name].data_type,
                dims=source.dims or declared[name].dims,
                version_dir=version_dir,
                rng=rng,
            )
            for name, source in sample.inputs.items()
        }
        batches.append(
            WarmupBatch(
                name=sample.name,
                inputs=inputs,
                batch_size=sample.batch_size,
                count=sample.count,
            )
        )
    return tuple(batches)


def _tensor_for(
    *,
    config: ModelConfig,
    sample: WarmupSample,
    tensor_name: str,
    warmup_input: WarmupInput,
    dtype: DataType,
    dims: list[int],
    version_dir: Path,
    rng: np.random.Generator,
) -> Tensor:
    """One warm-up tensor. The three sources are mutually exclusive by validation."""
    shape = (sample.batch_size, *dims)
    where = f"model {config.name!r} warm-up sample {sample.name!r}, input {tensor_name!r}"

    if warmup_input.zero_data:
        return Tensor.from_numpy(np.zeros(shape, dtype=dtype.numpy_dtype))
    if warmup_input.random_data:
        return Tensor.from_numpy(_random(rng, shape, dtype))
    assert warmup_input.input_data_file is not None  # guaranteed by WarmupInput's validator
    return Tensor.from_numpy(
        _from_file(version_dir / warmup_input.input_data_file, shape, dtype, where)
    )


def _random(rng: np.random.Generator, shape: tuple[int, ...], dtype: DataType) -> np.ndarray:
    """Uniform noise of the right dtype.

    Floats get [0, 1) and integers a small non-negative range, because a warm-up value that
    overflows its dtype would fail the load for a reason that has nothing to do with the
    model. The range is not meant to be realistic — realism is what ``input_data_file`` is
    for; this is for waking a data-dependent kernel that zeros leave asleep.
    """
    numpy_dtype = dtype.numpy_dtype
    if numpy_dtype.kind == "f":
        return rng.random(shape, dtype=np.float32).astype(numpy_dtype, copy=False)
    if numpy_dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(numpy_dtype, copy=False)
    return rng.integers(0, 8, size=shape).astype(numpy_dtype, copy=False)


def _from_file(path: Path, shape: tuple[int, ...], dtype: DataType, where: str) -> np.ndarray:
    """A raw little-endian dump of exactly ``shape`` elements, as Triton reads it.

    The size check is the whole reason this is not two lines: ``frombuffer`` on a file with
    one extra byte raises a ``ValueError`` naming buffer lengths, which tells an operator
    nothing about which model, which sample, or what was expected.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError(f"{where}: cannot read warm-up data {path}: {exc}") from exc

    expected = int(np.prod(shape)) * dtype.itemsize
    if len(raw) != expected:
        raise ConfigurationError(
            f"{where}: {path} holds {len(raw)} bytes but {shape} of {dtype.value} "
            f"needs {expected}"
        )
    return np.frombuffer(raw, dtype=dtype.numpy_dtype).reshape(shape).copy()
