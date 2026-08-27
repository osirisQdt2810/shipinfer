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
    "DrainResult",
    "ErrorBody",
    "InferInputTensor",
    "InferOutputTensor",
    "InferenceRequestBody",
    "InferenceResponseBody",
    "ModelMetadata",
    "ServerMetadata",
    "StreamInfo",
    "StreamList",
    "StreamRemoved",
    "StreamRequest",
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


# -- the stream control plane (arch.md section 2) ------------------------------------------
#
# Not KServe. `POST /streams` is this project's own surface -- cameras and videos enter the
# deployment here, and the tensor side-door above is for a caller who already has pixels.
# They share a package because they share a process and an error mapping, and nothing else.


class StreamRequest(BaseModel):
    """One camera, as a client asks for it: ``{"url": "rtsp://..."}`` and little else.

    Deliberately :class:`~shipinfer.launch.control.CameraSpec`'s three fields and not
    :class:`~shipinfer.core.settings.ingest.CameraConfig`'s twenty. Codec, transport, decode
    size and priority are *deployment* settings that the shard resolves from its own tree
    (CONVENTIONS 2.6); what only the caller knows is which video it wants read.

    ``extra="forbid"`` is the load-bearing line. A client that posts ``{"uri": ...}`` or
    ``{"fps": 30, "priority": "high"}`` against a server that silently drops what it does not
    recognise gets a 201 and a camera reading nothing, or a camera at the wrong rate -- and
    finds out from a dashboard rather than from the response. A 422 naming the field is the
    cheaper failure.
    """

    model_config = ConfigDict(extra="forbid")

    #: Optional. Empty means "name it for me": the server mints the next free ``cam-<n>``
    #: with :func:`~shipinfer.launch.control.mint_camera_id`, the same helper
    #: ``shipinfer run --inputs`` uses, so ids minted by the two paths cannot collide.
    camera_id: str = ""
    url: str
    #: Target frame rate; ``0.0`` means "whatever the source delivers".
    fps: float = 0.0


class StreamInfo(BaseModel):
    """One camera as the server sees it, assembled from a runner's health report.

    Every field is optional-shaped on purpose, because two runners answer this question with
    different amounts of knowledge and neither is lying: an in-process runner knows the
    camera's ingest ``state`` and has no shard to report beyond its own, while a launcher
    knows the ``shard`` and reads the state back out of that shard's report.
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    #: The source, when the runner's health carries one. Neither runner does today -- a
    #: camera's URL is its shard's configuration, not the launcher's -- so this reads ``""``
    #: for a camera this process did not just place. It is here rather than tracked in the
    #: router because a router that remembered urls would answer from its own memory about
    #: cameras added over gRPC, and be confidently wrong after a shard restart.
    url: str = ""
    #: Which shard holds it. ``None`` only when the runner reports no shard at all.
    shard: int | None = None
    #: Placed but not yet accepted -- the launcher's reservation window
    #: (``runners/fleet.py``). A pending camera is not being read yet, and reporting it as
    #: running is how an operator concludes a dark camera is fine.
    pending: bool = False
    #: The ingest state as the shard reports it (``connecting``, ``streaming``,
    #: ``exhausted``, ...), or ``""`` when nothing has said yet.
    state: str = ""


class StreamList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    streams: list[StreamInfo]


class StreamRemoved(BaseModel):
    """The answer to ``DELETE /streams/{id}``: the placement is gone, was the thread?

    ``clean=False`` is a **body signal and never a 5xx**: the camera *has* been removed and
    the caller must not retry, but a decoder thread outlived its deadline and still holds
    buffers. A 500 would say the removal failed, and a control plane that retried it would
    get a 404 and conclude something worse.
    """

    model_config = ConfigDict(extra="forbid")

    clean: bool


class DrainResult(BaseModel):
    """The answer to ``POST /streams/drain``: how many camera threads were abandoned.

    ``0`` is the clean drain. Non-zero is a lifetime signal rather than a statistic -- one
    deadline is charged to the whole camera set, so a thread still unfinished at it is
    genuinely stuck and still references this process's buffers.
    """

    model_config = ConfigDict(extra="forbid")

    abandoned: int


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
