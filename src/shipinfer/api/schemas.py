"""Wire schemas for the HTTP API — KServe v2 (a.k.a. the Triton HTTP protocol).

Speaking a protocol somebody else already specified is worth more than a nicer one of our
own: existing Triton clients, load generators and dashboards work against this server
unchanged, and a team migrating between the two does not rewrite its callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shipinfer.core.errors import ValidationError
from shipinfer.core.request import Priority
from shipinfer.core.settings.ingest import usable_camera_id
from shipinfer.core.types import DataType, Tensor

__all__ = [
    "BAND_NAMES",
    "BandName",
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


#: The bands a client may name, lower-cased, in the order the enum declares them. Derived from
#: :class:`~shipinfer.core.request.Priority` rather than written out, so a band added there is
#: offered here without a second edit -- and cannot be offered here without existing.
BAND_NAMES: Final[tuple[str, ...]] = tuple(band.name.lower() for band in Priority)

#: The wire type of :attr:`StreamRequest.priority`: the band *names*, never the numbers.
#:
#: A ``Literal`` rather than ``Priority`` itself, because the annotation is what the schema is
#: generated from and ``Priority`` is an ``IntEnum``: FastAPI published
#: ``{"enum": [0, 1, 2, 3], "type": "integer"}`` on ``/openapi.json`` while the validator
#: refused every one of those integers, so a generated client did exactly what the document
#: told it to and got a 422. The names are the only thing that was ever accepted, and now they
#: are the only thing advertised.
#:
#: Subscripted with the derived tuple and not with four spelled-out strings, so the document
#: and the enum cannot drift apart; ``Literal[t]`` for a tuple ``t`` is the same type as
#: ``Literal[*t]`` at runtime, which static tooling cannot see -- hence the ignore.
#:
#: Matched case-insensitively, by :meth:`StreamRequest._band_name_is_case_insensitive`
#: normalising the input first; what is *published* is the lower-cased spelling, because a
#: document has to offer one.
BandName = Literal[BAND_NAMES]  # type: ignore[valid-type]


class StreamRequest(BaseModel):
    """One camera, as a client asks for it: ``{"url": "rtsp://..."}`` and little else.

    Deliberately :class:`~shipinfer.launch.control.CameraSpec`'s fields and not
    :class:`~shipinfer.core.settings.ingest.CameraConfig`'s twenty. Codec, transport and
    decode size are *deployment* settings that the shard resolves from its own tree
    (CONVENTIONS 2.6); what only the caller knows is which video it wants read, and — since a
    fleet shard has no camera table to resolve a band from — which lane it wants it read in.

    ``extra="forbid"`` is the load-bearing line. A client that posts ``{"uri": ...}`` or
    ``{"framerate": 30}`` against a server that silently drops what it does not recognise
    gets a 201 and a camera reading nothing, or a camera at the wrong rate -- and finds out
    from a dashboard rather than from the response. A 422 naming the field is the cheaper
    failure.

    **The field constraints make the same argument about values, and they are not cosmetic.**
    The next thing to inspect a ``url`` or an ``fps`` is
    :class:`~shipinfer.core.settings.ingest.CameraConfig`, a layer below the router, and its
    refusal is a *pydantic* ``ValidationError`` -- a ``ValueError``, and not a
    :class:`~shipinfer.core.errors.ShipInferError`, so the handler's typed mapping does not
    see it. Unconstrained here, ``{"url": ""}`` from a deploy script whose variable did not
    expand answered **500** in process and, over gRPC, a refusal from every shard -> a
    ``NoShardAvailableError`` -> a **retryable 503** for a request that can never succeed.
    Declared here, FastAPI answers 422 naming the field before the handler is entered, so
    both runners give the caller the same terminal answer. The constraints are not a mirror
    of ``CameraConfig``'s (``core/settings/ingest.py``) but the *same rule applied earlier*:
    non-empty ``uri``, ``fps >= 0``, ``priority`` by name, and -- through
    :func:`~shipinfer.core.settings.ingest.usable_camera_id`, which both models call -- a
    ``camera_id`` with no whitespace in it. A mirror would have drifted, and it did:
    ``camera_id`` was the field this argument was written for and the field it was first
    missing.
    """

    model_config = ConfigDict(extra="forbid")

    #: Optional. Empty means "name it for me": the server mints the next free ``cam-<n>``
    #: with :func:`~shipinfer.launch.control.mint_camera_id`, the same helper
    #: ``shipinfer run --inputs`` uses, so ids minted by the two paths cannot collide.
    camera_id: str = ""
    #: ``rtsp://...`` for a camera, or a file path for a replayed video. Required, and never
    #: blank: a camera with no source is a decoder thread that fails on its first open.
    url: str = Field(min_length=1)
    #: Target frame rate; ``0.0`` means "whatever the source delivers". Never negative --
    #: a rate below zero has no meaning downstream and ``CameraConfig`` refuses it anyway,
    #: one layer too late to be a 422.
    fps: float = Field(default=0.0, ge=0.0)
    #: ``replay`` sources only: restart the file at EOF. Exposed because it is the one field
    #: of :class:`~shipinfer.launch.control.CameraSpec` that decides whether *this* camera
    #: ever ends, and it is not a deployment default -- ``shipinfer run --inputs`` has
    #: ``--no-loop`` for exactly this, and without it here a client that posts a finite video
    #: file gets it replayed forever with no way to ask otherwise. Ignored by a live source,
    #: which has no end to reach.
    loop: bool = True
    #: Optional scheduler lane, **by name**: ``tracking_critical``, ``high``, ``normal`` or
    #: ``background`` (:data:`BAND_NAMES`). ``None`` -- the default -- leaves the band to the
    #: deployment, which is what a caller who has no opinion should send.
    #:
    #: Names and never numbers, which is why the type is :data:`BandName` and not
    #: :class:`~shipinfer.core.request.Priority`. ``Priority`` is an ``IntEnum``, so accepting
    #: it on the wire would also accept ``{"priority": 0}`` --
    #: :attr:`~shipinfer.core.request.Priority.TRACKING_CRITICAL`, one keystroke away from a
    #: caller who meant "unset" and got the highest lane on the deployment. A name cannot be
    #: misread that way, and it is what the operator wrote in ``ingest.cameras`` anyway.
    #: An unknown name is a 422 naming the field, which FastAPI answers before the handler
    #: runs -- the same answer an unknown key gets, for the same reason.
    priority: BandName | None = None

    @field_validator("camera_id")
    @classmethod
    def _camera_id_is_usable(cls, value: str) -> str:
        """``""`` means "name it for me"; anything else must be an id a camera can carry.

        The empty string is the one value this validator lets past, because it is not an id
        at all -- it is the request to mint one, and :func:`shipinfer.api.streams._mint`
        answers it with a name that is legal by construction. Everything else is checked by
        :func:`~shipinfer.core.settings.ingest.usable_camera_id`, the same function
        ``CameraConfig`` uses, so the door and the record cannot disagree.

        Without this, ``{"camera_id": "quay 1"}`` reached the runner and was refused a layer
        down -- a 400 in process, and on a fleet a **retryable 503** for a request that can
        never succeed, because the launcher sees only that every shard said no. That is the
        exact 400-vs-503 conflation ``url`` and ``fps`` were constrained to remove, on the
        third field, which is why the rule is now shared code rather than a copy.
        """
        return value if not value else usable_camera_id(value)

    @field_validator("url")
    @classmethod
    def _url_is_not_blank(cls, value: str) -> str:
        """``min_length`` catches ``""``; this catches ``"   "``, which is the same mistake.

        A shell that renders an unset variable into a JSON body produces whitespace as
        readily as it produces nothing, and a whitespace-only url is a source no decoder can
        open. ``CameraConfig.uri`` rejects both with this same message a layer down; saying
        it here is what makes it a 422 instead of a 500.
        """
        if not value.strip():
            raise ValueError("url must not be empty")
        return value

    @field_validator("priority", mode="before")
    @classmethod
    def _band_name_is_case_insensitive(cls, value: object) -> object:
        """``TRACKING_CRITICAL`` and ``tracking_critical`` name the same lane.

        The band is a member *name*, and that name is written upper-case everywhere it comes
        from -- :class:`~shipinfer.core.request.Priority` in Python, ``Priority.BACKGROUND``
        in a gRPC client's generated enum -- while this schema publishes it lower-cased. A
        client that sent the spelling its own stub gave it would otherwise get a 422 for
        naming the right lane, which is a document disagreeing with itself rather than a
        caller making a mistake. ``api/streams.py`` already looks the member up with
        ``Priority[name.upper()]``, so the case was never what it was checking.

        Only ``str`` is touched, and only its case. Everything the field refused before it
        still refuses: ``{"priority": 0}`` is an ``int`` and passes through untouched to be
        rejected by the :data:`BandName` literal (the falsy-zero trap this field exists for,
        ADR-005), ``"tracking-critical"`` is not a member name in any case, and ``""`` is not
        one either. This normalises a spelling; it does not widen the vocabulary.
        """
        return value.lower() if isinstance(value, str) else value


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
    #: The shard holding this camera has **exited**, so nothing is reading it and nothing
    #: will re-place it (ADR-018). Distinct from a missing entry, which would say the camera
    #: was never placed, and from an unreachable shard, which is alive and may answer again:
    #: this one is terminal until a caller removes the camera and adds it back. The view lags
    #: a death by up to the launcher's supervision poll.
    lost: bool = False

    # `loop` is deliberately absent, for the reason `url` is optional-shaped: no runner's
    # health report carries it, so a field here would answer `true` for every camera --
    # including one posted with `loop: false` -- and be confidently wrong about the one
    # thing that decides whether a video ever ends. `StreamRequest.loop` is where it is
    # asked for; `state: "exhausted"` is how a listing shows a non-looping file finished.


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
