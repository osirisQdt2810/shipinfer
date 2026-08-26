"""The ring's wire format: a request and a response survive the round trip, byte for byte."""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.errors import RingProtocolError
from shipinfer.core.request import InferenceRequest, InferenceResponse, Priority, RequestContext
from shipinfer.core.request.timings import Timings
from shipinfer.core.types import DataType, Device, Tensor
from shipinfer.server import remote_wire as wire


def _request(**overrides) -> InferenceRequest:
    rows = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    fields = {
        "model_name": "person_embedder",
        "inputs": {"images": Tensor.from_numpy(rows)},
        "context": RequestContext(
            camera_id="quay-7",
            frame_id=1234,
            captured_ns=55,
            captured_unix_ns=66,
            trace_id="t-1",
        ),
        "priority": getattr(Priority, "TRACKING_CRITICAL", Priority.NORMAL),
        "deadline_ns": 999_999,
        "model_version": 3,
    }
    fields.update(overrides)
    request = InferenceRequest(**fields)
    request.timings.received_ns = 100
    request.timings.queued_ns = 200
    return request


class TestARequestSurvivesTheRoundTrip:
    def test_every_field_and_every_byte(self) -> None:
        request = _request()
        slot = memoryview(bytearray(wire.encoded_request_size(request)))
        used = wire.encode_request(request, slot)
        assert used == len(slot)

        back = wire.decode_request(slot, copy=True)
        assert back.request_id == request.request_id
        assert back.model_name == "person_embedder" and back.model_version == 3
        assert back.priority is request.priority and back.deadline_ns == 999_999
        assert (back.context.camera_id, back.context.frame_id) == ("quay-7", 1234)
        assert (
            back.context.captured_ns,
            back.context.captured_unix_ns,
            back.context.trace_id,
        ) == (55, 66, "t-1")
        assert (back.timings.received_ns, back.timings.queued_ns) == (100, 200)
        assert back.inputs.keys() == {"images"}
        np.testing.assert_array_equal(
            back.inputs["images"].numpy(), request.inputs["images"].numpy()
        )
        assert back.inputs["images"].dtype is DataType.FP32 and back.inputs["images"].shape == (
            2,
            3,
            4,
            4,
        )

    def test_a_view_shares_the_slots_memory_and_a_copy_does_not(self) -> None:
        request = _request()
        slot = memoryview(bytearray(wire.encoded_request_size(request)))
        wire.encode_request(request, slot)
        viewed = wire.decode_request(slot, copy=False).inputs["images"].numpy()
        copied = wire.decode_request(slot, copy=True).inputs["images"].numpy()
        # poke the slot: the view sees it, the copy does not
        slot[-4:] = np.float32(-1.0).tobytes()
        assert viewed.reshape(-1)[-1] == -1.0
        assert copied.reshape(-1)[-1] != -1.0

    def test_several_inputs_of_different_dtypes_in_name_order(self) -> None:
        request = _request(
            inputs={
                "z_mask": Tensor.from_numpy(np.ones((2, 5), dtype=np.uint8)),
                "a_boxes": Tensor.from_numpy(np.arange(8, dtype=np.int32).reshape(2, 4)),
            }
        )
        slot = memoryview(bytearray(wire.encoded_request_size(request)))
        wire.encode_request(request, slot)
        back = wire.decode_request(slot)
        assert list(back.inputs) == ["a_boxes", "z_mask"]
        assert back.inputs["a_boxes"].dtype is DataType.INT32
        np.testing.assert_array_equal(
            back.inputs["z_mask"].numpy(), np.ones((2, 5), dtype=np.uint8)
        )

    def test_a_missing_model_version_stays_missing(self) -> None:
        request = _request(model_version=None)
        slot = memoryview(bytearray(wire.encoded_request_size(request)))
        wire.encode_request(request, slot)
        assert wire.decode_request(slot).model_version is None


class TestWhatIsRefused:
    def test_a_slot_too_small_is_refused_before_a_byte_is_written(self) -> None:
        request = _request()
        slot = memoryview(bytearray(wire.encoded_request_size(request) - 1))
        with pytest.raises(ValueError, match="size the ring's slots"):
            wire.encode_request(request, slot)
        assert bytes(slot[:4]) == b"\0\0\0\0", "nothing written"

    def test_a_device_resident_input_is_refused(self) -> None:
        import types

        handle = types.SimpleNamespace(ptr=1, nbytes=64, device=Device.cuda(0))
        request = _request(
            inputs={"images": Tensor(dtype=DataType.FP32, shape=(16,), handle=handle)}
        )
        with pytest.raises(ValueError, match="device-resident"):
            wire.encoded_request_size(request)

    def test_garbage_is_not_a_request(self) -> None:
        with pytest.raises(RingProtocolError, match="not a version"):
            wire.decode_request(memoryview(bytearray(4096)))

    def test_a_truncated_slot_is_a_protocol_error_not_a_crash(self) -> None:
        request = _request()
        full = bytearray(wire.encoded_request_size(request))
        wire.encode_request(request, memoryview(full))
        truncated = memoryview(full[: wire._REQUEST_HEAD.size + wire._TENSOR_HEAD.size + 16])
        with pytest.raises(RingProtocolError, match="runs past the slot"):
            wire.decode_request(truncated)

    def test_a_name_that_does_not_fit_is_refused(self) -> None:
        request = _request(context=RequestContext(camera_id="c" * 70, frame_id=1))
        with pytest.raises(ValueError, match="camera_id"):
            wire.encode_request(
                request, memoryview(bytearray(wire.encoded_request_size(request)))
            )


class TestAResponseSurvivesTheRoundTrip:
    def test_outputs_timings_and_where_it_ran(self) -> None:
        context = RequestContext(camera_id="quay-7", frame_id=1234)
        response = InferenceResponse(
            request_id=42,
            model_name="person_embedder",
            model_version=3,
            outputs={
                "embedding": Tensor.from_numpy(
                    np.linspace(0, 1, 2 * 512, dtype=np.float32).reshape(2, 512)
                )
            },
            context=context,
            timings=Timings(
                received_ns=1,
                queued_ns=2,
                batched_ns=3,
                compute_start_ns=4,
                compute_end_ns=5,
                completed_ns=6,
            ),
            executed_on=Device.cuda(3),
        )
        slot = memoryview(bytearray(wire.encoded_response_size(response)))
        assert wire.encode_response(response, slot) == len(slot)

        back = wire.decode_response(slot, context)
        assert (
            back.request_id == 42
            and back.model_name == "person_embedder"
            and back.model_version == 3
        )
        assert back.executed_on == Device.cuda(
            3
        ), "the proof that spillover happened rides back"
        assert (
            back.context is context
        ), "the submitter keeps the context; the wire carries the id"
        assert (back.timings.received_ns, back.timings.completed_ns) == (1, 6)
        np.testing.assert_array_equal(
            back.outputs["embedding"].numpy(), response.outputs["embedding"].numpy()
        )

    def test_a_request_head_is_not_a_response(self) -> None:
        request = _request()
        slot = memoryview(bytearray(max(wire.encoded_request_size(request), 4096)))
        wire.encode_request(request, slot)
        with pytest.raises(RingProtocolError, match="not a version"):
            wire.decode_response(slot, request.context)

    def test_a_failure_crosses_back_as_the_owners_error(self) -> None:
        context = RequestContext(camera_id="quay-7", frame_id=9)
        slot = memoryview(bytearray(4096))
        used = wire.encode_failure(
            42, "person_embedder", ValueError("engine refused the batch"), slot
        )
        assert used == wire._RESPONSE_HEAD.size
        with pytest.raises(
            wire.RemoteFailureError, match="ValueError: engine refused the batch"
        ):
            wire.decode_response(slot, context)
