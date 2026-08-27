"""Wire schemas for the HTTP API — KServe v2 (a.k.a. the Triton HTTP protocol).

Speaking a protocol somebody else already specified is worth more than a nicer one of our
own: existing Triton clients, load generators and dashboards work against this server
unchanged, and a team migrating between the two does not rewrite its callers.
"""

from __future__ import annotations

from collections.abc import Mapping
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


class RequestTag(BaseModel):
    """The provenance tag, carried in KServe's request-level ``parameters`` object.

    In ``parameters`` rather than as new top-level fields, so the server stays wire
    compatible with a stock Triton client — an unknown top-level key would be rejected by
    one, while ``parameters`` is exactly the extension point the protocol provides.

    Both fields are optional, and a request without them is legitimate. But it is worth
    knowing what it costs: every untagged request shares the fairness key ``"-"``, so they
    all queue in one lane. That is the honest default (the server cannot invent a camera
    id), not a recommendation — a multi-camera client that omits the tag gets FIFO and
    the starvation ADR-005 exists to prevent.
    """

    model_config = ConfigDict(extra="ignore")

    camera_id: str = ""
    frame_id: int = -1

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> RequestTag:
        """Read the tag out of a request's ``parameters``.

        Raises:
            ValidationError: for a frame_id that is not an integer. Coercing it to the
                default would silently merge that client's frames into the untagged lane,
                which is precisely the bug this tag exists to make impossible.
        """
        try:
            return cls(
                camera_id=str(parameters.get("camera_id", "") or ""),
                frame_id=int(parameters.get("frame_id", -1)),
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"parameters.frame_id must be an integer, got "
                f"{parameters.get('frame_id')!r}"
            ) from exc


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
