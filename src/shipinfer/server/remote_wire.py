"""What crosses a shared ring: a request in one direction, a response in the other.

Pure functions over bytes, so the format is pinned offline and both ends of a ring agree by
construction — the same module encodes and decodes. No pickle: a slot is a fixed-size region
that another process reads while this one is still running, and the format has to be
checkable (versioned, bounded, no code execution) rather than convenient.

Layout of a request slot::

    RequestHead (struct) | name/shape table, one TensorHead per input | raw tensor bytes …

Layout of a response slot::

    ResponseHead (struct) | one TensorHead per output | raw tensor bytes …

Every tensor here is a **host** tensor — the Python plane hands its models
``Tensor.from_numpy(rows)`` (`pipeline/graph/objects.py`, `detect.py`), so a remote submit is
a memcpy into the slot and the owner's backend stages host → device exactly as it does for a
local request. The tag rides in the head (ADR-002: it survives every path), and so do priority,
deadline and the timings that have been taken so far, so the owner's queue treats the request
as it would a local one and the response's timings are whole.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from shipinfer.core.errors import RingProtocolError, ValidationError
from shipinfer.core.request import InferenceRequest, InferenceResponse, Priority, RequestContext
from shipinfer.core.request.timings import Timings
from shipinfer.core.types import DataType, Device, Tensor

__all__ = [
    "WIRE_VERSION",
    "RemoteFailureError",
    "decode_request",
    "decode_response",
    "encode_failure",
    "encode_request",
    "encode_response",
    "encoded_request_size",
    "encoded_response_size",
    "peek_request_id",
]

WIRE_VERSION = 1
_MAGIC_REQUEST = 0x5245_5131  # "REQ1"
_MAGIC_RESPONSE = 0x5245_5331  # "RES1"
_MAX_NAME = 64
_MAX_DIMS = 8

#: magic u32 | version u32 | request_id u64 | model_version i64 | priority i32 | tensors u32 |
#: deadline_ns i64 | received_ns i64 | queued_ns i64 | frame_id i64 | captured_ns i64 |
#: captured_unix_ns i64 | camera_id 64s | model_name 64s | trace_id 64s
_REQUEST_HEAD = struct.Struct("<IIQqiIqqqqqq64s64s64s")
#: magic u32 | version u32 | request_id u64 | model_version i64 | tensors u32 | status u32 |
#: received_ns i64 | queued_ns i64 | batched_ns i64 | compute_start_ns i64 | compute_end_ns i64 |
#: completed_ns i64 | model_name 64s | executed_on 16s | message 192s
#: `status` 0 is a result; anything else is a failure whose `message` the submitter raises, so an
#: owner that could not run the request says so instead of leaving a future to time out.
_RESPONSE_HEAD = struct.Struct("<IIQqIIqqqqqq64s16s192s")
STATUS_OK = 0
STATUS_FAILED = 1
#: name 64s | dtype u8 | ndim u8 | pad u16 | nbytes u64 | dims 8 x u32
_TENSOR_HEAD = struct.Struct("<64sBBHQ8I")

_DTYPE_CODES: dict[DataType, int] = {dtype: i for i, dtype in enumerate(DataType)}
_DTYPES_BY_CODE: dict[int, DataType] = {i: dtype for dtype, i in _DTYPE_CODES.items()}


class RemoteFailureError(Exception):
    """The owner ran into an error and sent it back instead of a result. The message is the
    owner's error, class and text; the submitter turns it into a typed failure for the caller.
    """


@dataclass(frozen=True, slots=True)
class _TensorHead:
    name: str
    dtype: DataType
    shape: tuple[int, ...]
    nbytes: int


def _fixed(text: str, width: int, what: str) -> bytes:
    raw = text.encode()
    if len(raw) >= width:
        raise ValidationError(f"{what} {text!r} does not fit in {width - 1} bytes")
    return raw


def _unfixed(raw: bytes) -> str:
    return raw.rstrip(b"\0").decode(errors="replace")


def _host(tensor: Tensor, name: str) -> np.ndarray:
    """The tensor's bytes on the host — copying them there when they live on a device.

    Only host bytes cross a ring; a device-resident tensor is D2H'd here, on the caller's
    thread, which is the submitter's (or the owner's) side of the copy ADR-002 routes through
    the host. Synchronous by design: the bytes must be complete before the slot is published.
    """
    if tensor.host is not None:
        return np.ascontiguousarray(tensor.host)
    return _device_to_host(tensor, name)


def _device_to_host(tensor: Tensor, name: str) -> np.ndarray:  # noqa: ARG001 - name aids errors
    from shipinfer.runtime.tensor import to_torch

    view = to_torch(tensor)
    return np.ascontiguousarray(view.detach().to("cpu", non_blocking=False).numpy())


def _tensor_heads(tensors: Mapping[str, Tensor]) -> list[tuple[_TensorHead, np.ndarray]]:
    heads: list[tuple[_TensorHead, np.ndarray]] = []
    for name in sorted(tensors):
        array = _host(tensors[name], name)
        if array.ndim > _MAX_DIMS:
            raise ValidationError(
                f"tensor {name!r} has {array.ndim} dims; the wire carries at most {_MAX_DIMS}"
            )
        heads.append(
            (
                _TensorHead(
                    name, DataType.from_numpy(array.dtype), tuple(array.shape), array.nbytes
                ),
                array,
            )
        )
    return heads


def _pack_tensor_head(head: _TensorHead) -> bytes:
    dims = list(head.shape) + [0] * (_MAX_DIMS - len(head.shape))
    return _TENSOR_HEAD.pack(
        _fixed(head.name, _MAX_NAME, "tensor name"),
        _DTYPE_CODES[head.dtype],
        len(head.shape),
        0,
        head.nbytes,
        *dims,
    )


def _unpack_tensor_head(view: memoryview, offset: int) -> _TensorHead:
    fields = _TENSOR_HEAD.unpack_from(view, offset)
    name, code, ndim = _unfixed(fields[0]), fields[1], fields[2]
    if code not in _DTYPES_BY_CODE:
        raise RingProtocolError(f"tensor {name!r}: unknown dtype code {code}")
    # fields: name, dtype code, ndim, pad, nbytes, dims[8]
    return _TensorHead(name, _DTYPES_BY_CODE[code], tuple(fields[5 : 5 + ndim]), fields[4])


def _write_tensors(
    view: memoryview, offset: int, heads: list[tuple[_TensorHead, np.ndarray]]
) -> int:
    for head, _ in heads:
        view[offset : offset + _TENSOR_HEAD.size] = _pack_tensor_head(head)
        offset += _TENSOR_HEAD.size
    for head, array in heads:
        # One memcpy straight into the slot: `.tobytes()` here materialised a third full
        # copy of the payload on the dispatch path (6.29 MB per embedder batch).
        view[offset : offset + head.nbytes] = array.reshape(-1).view(np.uint8)
        offset += head.nbytes
    return offset


def _read_tensors(
    view: memoryview, offset: int, count: int, *, copy: bool
) -> dict[str, Tensor]:
    heads = []
    for _ in range(count):
        heads.append(_unpack_tensor_head(view, offset))
        offset += _TENSOR_HEAD.size
    out: dict[str, Tensor] = {}
    for head in heads:
        if offset + head.nbytes > len(view):
            raise RingProtocolError(
                f"tensor {head.name!r} runs past the slot ({offset + head.nbytes} > {len(view)})"
            )
        raw = np.frombuffer(
            view[offset : offset + head.nbytes], dtype=head.dtype.numpy_dtype
        ).reshape(head.shape)
        out[head.name] = Tensor.from_numpy(raw.copy() if copy else raw)
        offset += head.nbytes
    return out


def peek_request_id(view: memoryview) -> int:
    """The request id of whatever head is in ``view`` — request and response heads agree on
    where it sits, so the reader can find the pending future before decoding the rest."""
    if len(view) < 16:
        raise RingProtocolError(f"slot of {len(view)} bytes holds no head")
    return struct.unpack_from("<Q", view, 8)[0]


# -- requests -----------------------------------------------------------------------------


def encoded_request_size(request: InferenceRequest) -> int:
    """Bytes a slot needs for ``request`` — what sizes a ring's slots from a model's config."""
    heads = _tensor_heads(request.inputs)
    return _REQUEST_HEAD.size + len(heads) * _TENSOR_HEAD.size + sum(h.nbytes for h, _ in heads)


def encode_request(request: InferenceRequest, view: memoryview) -> int:
    """Write ``request`` into ``view``. Returns the bytes used.

    Raises:
        ValueError: an input is device-resident, or the request does not fit the slot — the
            caller sized the ring from the model's config, so a miss here is a config bug and
            is said so before a byte is written.
    """
    heads = _tensor_heads(request.inputs)
    needed = (
        _REQUEST_HEAD.size + len(heads) * _TENSOR_HEAD.size + sum(h.nbytes for h, _ in heads)
    )
    if needed > len(view):
        raise ValidationError(
            f"request {request.request_id} needs {needed} bytes but the slot holds {len(view)}; "
            f"size the ring's slots from the model's max batch"
        )
    ctx = request.context
    t = request.timings
    _REQUEST_HEAD.pack_into(
        view,
        0,
        _MAGIC_REQUEST,
        WIRE_VERSION,
        request.request_id,
        -1 if request.model_version is None else request.model_version,
        int(request.priority),
        len(heads),
        request.deadline_ns,
        t.received_ns,
        t.queued_ns,
        ctx.frame_id,
        ctx.captured_ns,
        ctx.captured_unix_ns,
        _fixed(ctx.camera_id, _MAX_NAME, "camera_id"),
        _fixed(request.model_name, _MAX_NAME, "model_name"),
        _fixed(ctx.trace_id, _MAX_NAME, "trace_id"),
    )
    return _write_tensors(view, _REQUEST_HEAD.size, heads)


def decode_request(view: memoryview, *, copy: bool = False) -> InferenceRequest:
    """Read a request back. With ``copy=False`` the inputs *view* the slot: the slot must stay
    claimed until the request completes (the owner releases it in the completion callback)."""
    if len(view) < _REQUEST_HEAD.size:
        raise RingProtocolError(f"slot of {len(view)} bytes holds no request head")
    f = _REQUEST_HEAD.unpack_from(view, 0)
    if f[0] != _MAGIC_REQUEST or f[1] != WIRE_VERSION:
        raise RingProtocolError(
            f"not a version-{WIRE_VERSION} request (magic {f[0]:#x}, version {f[1]})"
        )
    inputs = _read_tensors(view, _REQUEST_HEAD.size, f[5], copy=copy)
    request = InferenceRequest(
        model_name=_unfixed(f[13]),
        inputs=inputs,
        request_id=f[2],
        model_version=None if f[3] < 0 else f[3],
        context=RequestContext(
            camera_id=_unfixed(f[12]),
            frame_id=f[9],
            captured_ns=f[10],
            captured_unix_ns=f[11],
            trace_id=_unfixed(f[14]),
        ),
        priority=Priority(f[4]),
        deadline_ns=f[6],
    )
    request.timings.received_ns = f[7]
    request.timings.queued_ns = f[8]
    return request


# -- responses ----------------------------------------------------------------------------


def encoded_response_size(response: InferenceResponse) -> int:
    heads = _tensor_heads(response.outputs)
    return (
        _RESPONSE_HEAD.size + len(heads) * _TENSOR_HEAD.size + sum(h.nbytes for h, _ in heads)
    )


def encode_response(response: InferenceResponse, view: memoryview) -> int:
    heads = _tensor_heads(response.outputs)
    needed = (
        _RESPONSE_HEAD.size + len(heads) * _TENSOR_HEAD.size + sum(h.nbytes for h, _ in heads)
    )
    if needed > len(view):
        raise ValidationError(
            f"response {response.request_id} needs {needed} bytes but the slot holds {len(view)}"
        )
    t = response.timings
    _RESPONSE_HEAD.pack_into(
        view,
        0,
        _MAGIC_RESPONSE,
        WIRE_VERSION,
        response.request_id,
        response.model_version,
        len(heads),
        STATUS_OK,
        t.received_ns,
        t.queued_ns,
        t.batched_ns,
        t.compute_start_ns,
        t.compute_end_ns,
        t.completed_ns,
        _fixed(response.model_name, _MAX_NAME, "model_name"),
        _fixed(str(response.executed_on), 16, "executed_on"),
        b"",
    )
    return _write_tensors(view, _RESPONSE_HEAD.size, heads)


def encode_failure(
    request_id: int, model_name: str, error: BaseException, view: memoryview
) -> int:
    """Write a failure for ``request_id``: the owner could not run it, and says why.

    The message is the error's class and text, cut to fit; the submitter raises it as a
    `ServerStateError` so the caller's future fails the way a local failure would rather than
    waiting out the stage timeout.
    """
    if len(view) < _RESPONSE_HEAD.size:
        raise ValidationError(
            f"a failure needs {_RESPONSE_HEAD.size} bytes but the slot holds {len(view)}"
        )
    message = f"{type(error).__name__}: {error}".encode(errors="replace")[:191]
    _RESPONSE_HEAD.pack_into(
        view,
        0,
        _MAGIC_RESPONSE,
        WIRE_VERSION,
        request_id,
        -1,
        0,
        STATUS_FAILED,
        0,
        0,
        0,
        0,
        0,
        0,
        _fixed(model_name, _MAX_NAME, "model_name"),
        b"cpu",
        message,
    )
    return _RESPONSE_HEAD.size


def decode_response(
    view: memoryview, context: RequestContext, *, copy: bool = True
) -> InferenceResponse:
    """Read a response back for the request whose ``context`` the submitter kept.

    ``copy=True`` by default: the submitter releases the result slot as soon as it has the
    response, and the outputs are small (embeddings, boxes) next to the inputs they answer.
    """
    if len(view) < _RESPONSE_HEAD.size:
        raise RingProtocolError(f"slot of {len(view)} bytes holds no response head")
    f = _RESPONSE_HEAD.unpack_from(view, 0)
    if f[0] != _MAGIC_RESPONSE or f[1] != WIRE_VERSION:
        raise RingProtocolError(
            f"not a version-{WIRE_VERSION} response (magic {f[0]:#x}, version {f[1]})"
        )
    if f[5] != STATUS_OK:
        raise RemoteFailureError(
            _unfixed(f[14]) or "the owner failed the request without a message"
        )
    outputs = _read_tensors(view, _RESPONSE_HEAD.size, f[4], copy=copy)
    timings = Timings(
        received_ns=f[6],
        queued_ns=f[7],
        batched_ns=f[8],
        compute_start_ns=f[9],
        compute_end_ns=f[10],
        completed_ns=f[11],
    )
    return InferenceResponse(
        request_id=f[2],
        model_name=_unfixed(f[12]),
        model_version=f[3],
        outputs=outputs,
        context=context,
        timings=timings,
        executed_on=Device.parse(_unfixed(f[13])),
    )
