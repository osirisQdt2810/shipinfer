"""Tensor and spec semantics."""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.types import (
    DYNAMIC,
    DataType,
    Device,
    MemoryKind,
    Tensor,
    TensorSpec,
    stack_tensors,
    validate_against,
)


class TestDevice:
    def test_parse(self) -> None:
        assert Device.parse("cpu") == Device.cpu()
        assert Device.parse("cuda") == Device.cuda(0)
        assert Device.parse("cuda:3") == Device.cuda(3)

    def test_str_round_trips(self) -> None:
        for text in ("cpu", "cuda:0", "cuda:7"):
            assert str(Device.parse(text)) == text

    def test_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="unknown device"):
            Device.parse("tpu:0")

    def test_is_hashable_and_ordered(self) -> None:
        devices = {Device.cuda(1), Device.cuda(0), Device.cpu()}
        assert len(devices) == 3
        assert sorted(devices)[0] == Device.cpu()


class TestDataType:
    def test_round_trips_through_numpy(self) -> None:
        for dtype in DataType:
            assert DataType.from_numpy(dtype.numpy_dtype) is dtype

    def test_rejects_an_untransportable_dtype(self) -> None:
        with pytest.raises(ValueError, match="unsupported numpy dtype"):
            DataType.from_numpy(np.dtype(object))


class TestTensorSpec:
    def test_dynamic_dims_match_anything(self) -> None:
        spec = TensorSpec("x", DataType.FP32, (DYNAMIC, 4))
        assert spec.is_dynamic
        assert spec.matches((7, 4))
        assert not spec.matches((7, 5))

    def test_rank_must_match(self) -> None:
        assert not TensorSpec("x", DataType.FP32, (4,)).matches((1, 4))

    def test_nbytes(self) -> None:
        assert TensorSpec("x", DataType.FP32, (3, 4)).nbytes(2) == 2 * 3 * 4 * 4

    def test_nbytes_refuses_to_guess_a_dynamic_extent(self) -> None:
        with pytest.raises(ValueError, match="dynamic shape"):
            TensorSpec("x", DataType.FP32, (DYNAMIC,)).nbytes(1)

    def test_negative_dims_other_than_dynamic_are_refused(self) -> None:
        with pytest.raises(ValueError, match="invalid dimension"):
            TensorSpec("x", DataType.FP32, (-4,))


class TestTensor:
    def test_from_numpy_shares_a_contiguous_buffer(self) -> None:
        array = np.zeros((2, 3), dtype=np.float32)
        assert Tensor.from_numpy(array).numpy() is array

    def test_from_numpy_makes_a_non_contiguous_array_contiguous(self) -> None:
        """Every downstream copy assumes one flat span; discovering otherwise inside a
        CUDA memcpy is a much worse place to find out."""
        array = np.zeros((4, 6), dtype=np.float32)[:, ::2]
        tensor = Tensor.from_numpy(array)
        assert tensor.numpy().flags["C_CONTIGUOUS"]

    def test_requires_exactly_one_backing_store(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Tensor(dtype=DataType.FP32, shape=(1,))

    def test_slice_batch_is_a_view(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(6, 2)
        sliced = Tensor.from_numpy(source).slice_batch(2, 4)
        assert np.shares_memory(sliced.numpy(), source)
        np.testing.assert_array_equal(sliced.numpy(), source[2:4])

    def test_device_tensors_refuse_a_silent_readback(self) -> None:
        """A hidden synchronising D2H at an arbitrary point is how a pipeline loses overlap."""

        class FakeHandle:
            ptr = 0x1000
            nbytes = 16
            kind = MemoryKind.DEVICE
            device = Device.cuda(0)

        tensor = Tensor.from_handle(FakeHandle(), DataType.FP32, (1, 4))
        assert tensor.is_device_resident
        with pytest.raises(RuntimeError, match="not host-visible"):
            tensor.numpy()


class TestStackTensors:
    def test_stacks_along_the_batch_axis(self) -> None:
        parts = [Tensor.from_numpy(np.full((2, 3), i, dtype=np.float32)) for i in range(3)]
        stacked = stack_tensors(parts)
        assert stacked.shape == (6, 3)

    def test_single_element_is_returned_unchanged(self) -> None:
        only = Tensor.from_numpy(np.zeros((1, 3), dtype=np.float32))
        assert stack_tensors([only]) is only

    def test_dtype_mismatch(self) -> None:
        with pytest.raises(ValueError, match="dtype mismatch"):
            stack_tensors(
                [
                    Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32)),
                    Tensor.from_numpy(np.zeros((1, 2), dtype=np.int32)),
                ]
            )

    def test_row_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="row shape mismatch"):
            stack_tensors(
                [
                    Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32)),
                    Tensor.from_numpy(np.zeros((1, 3), dtype=np.float32)),
                ]
            )


class TestValidateAgainst:
    SPECS = (
        TensorSpec("x", DataType.FP32, (4,)),
        TensorSpec("opt", DataType.FP32, (2,), optional=True),
    )

    def test_accepts_a_valid_map(self) -> None:
        validate_against(
            {"x": Tensor.from_numpy(np.zeros((3, 4), dtype=np.float32))},
            self.SPECS,
            what="input",
        )

    def test_missing_required(self) -> None:
        with pytest.raises(ValueError, match="missing required"):
            validate_against({}, self.SPECS, what="input")

    def test_unexpected_name_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="known:"):
            validate_against(
                {
                    "x": Tensor.from_numpy(np.zeros((1, 4), dtype=np.float32)),
                    "z": Tensor.from_numpy(np.zeros((1, 4), dtype=np.float32)),
                },
                self.SPECS,
                what="input",
            )

    def test_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            validate_against(
                {"x": Tensor.from_numpy(np.zeros((1, 4), dtype=np.int32))},
                self.SPECS,
                what="input",
            )
