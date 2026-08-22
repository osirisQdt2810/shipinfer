"""Wire schemas for the HTTP API — KServe v2 (a.k.a. the Triton HTTP protocol).

Speaking a protocol somebody else already specified is worth more than a nicer one of our
own: existing Triton clients, load generators and dashboards work against this server
unchanged, and a team migrating between the two does not rewrite its callers.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shipinfer.core.errors import ValidationError
from shipinfer.core.types import DataType, Tensor

__all__ = [
    "ErrorBody",
    "InferInputTensor",
    "InferOutputTensor",
    "InferenceRequestBody",
    "InferenceResponseBody",
    "ModelMetadata",
    "ServerMetadata",
    "tensor_from_wire",
    "tensor_to_wire",
]


class InferInputTensor(BaseModel):
    """One request tensor, JSON-encoded."""

    model_config = ConfigDict(extra="forbid")

    name: str
    shape: list[int]
    datatype: DataType
    data: list[Any]


class InferOutputTensor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    shape: list[int]
    datatype: DataType
    data: list[Any]


class InferenceRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    inputs: list[InferInputTensor]
    outputs: list[dict[str, Any]] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class InferenceResponseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    model_name: str
    model_version: str
    outputs: list[InferOutputTensor]
    parameters: dict[str, Any] = Field(default_factory=dict)


class TensorMetadata(BaseModel):
    name: str
    datatype: DataType
    shape: list[int]


class ModelMetadata(BaseModel):
    name: str
    versions: list[str]
    platform: str
    inputs: list[TensorMetadata]
    outputs: list[TensorMetadata]


class ServerMetadata(BaseModel):
    name: Literal["shipinfer"] = "shipinfer"
    version: str
    extensions: list[str]


class ErrorBody(BaseModel):
    error: str


def tensor_from_wire(spec: InferInputTensor) -> Tensor:
    """Decode a JSON tensor.

    JSON is a poor transport for a 6 MB frame and this function is not the fast path — it
    exists so ``curl`` and a browser work. Anything throughput-sensitive should use the
    in-process API or the binary/shared-memory path; the encoding cost here is real and
    unavoidable, and pretending otherwise would be the worse choice.
    """
    try:
        array = np.asarray(spec.data, dtype=spec.datatype.numpy_dtype).reshape(spec.shape)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"input {spec.name!r}: {exc}") from exc
    return Tensor.from_numpy(np.ascontiguousarray(array))


def tensor_to_wire(name: str, tensor: Tensor) -> InferOutputTensor:
    array = tensor.numpy()
    return InferOutputTensor(
        name=name,
        shape=list(array.shape),
        datatype=tensor.dtype,
        data=array.reshape(-1).tolist(),
    )
