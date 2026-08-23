"""Do the three image-op implementations agree?

The fast one is only trustworthy if it computes the same thing as the readable one. These
tests are what make it defensible to run fused CUDA kernels in production and a numpy
reference in CI.

Note the *deliberate* asymmetry in tolerance: torch and the native kernel both do bilinear
sampling with half-pixel centres and must agree to floating-point noise, while the numpy
reference samples nearest-neighbour and is only expected to agree structurally. Pretending
nearest and bilinear should match to 1e-6 would mean weakening the check that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.runtime.ops import IMAGE_OPS, NormalizeParams, NumpyImageOps, get_image_ops
from shipinfer.runtime.platform import is_available

PARAMS = NormalizeParams(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0), swap_rb=True)


@pytest.fixture()
def frames() -> list[np.ndarray]:
    """A ragged batch, because 50 cameras do not agree on resolution."""
    rng = np.random.default_rng(1234)
    return [
        rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
        rng.integers(0, 255, (300, 300, 3), dtype=np.uint8),
    ]


class TestNumpyReference:
    """Properties the reference must have, which then pin the others."""

    def test_output_shape_and_range(self, frames) -> None:
        result = NumpyImageOps().letterbox_batch(frames, (256, 256), PARAMS)
        assert result.tensor.shape == (3, 3, 256, 256)
        assert result.tensor.dtype == np.float32
        assert result.tensor.min() >= 0.0 and result.tensor.max() <= 1.0

    def test_aspect_ratio_is_preserved(self, frames) -> None:
        result = NumpyImageOps().letterbox_batch(frames, (256, 256), PARAMS)
        # 1280x720 into 256x256 -> scale 0.2, 144 rows used, 56 padded each side.
        assert result.scales[1] == pytest.approx(256 / 1280)
        assert result.pads[1].tolist() == [0.0, 56.0]

    def test_square_input_needs_no_padding(self, frames) -> None:
        result = NumpyImageOps().letterbox_batch(frames, (256, 256), PARAMS)
        assert result.pads[2].tolist() == [0.0, 0.0]

    def test_padding_uses_the_fill_value(self) -> None:
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = NumpyImageOps().letterbox_batch([image], (64, 64), PARAMS, pad_value=114)
        # Top rows are pad: 114/255 after normalisation.
        assert result.tensor[0, 0, 0, 0] == pytest.approx(114 / 255, abs=1e-6)

    def test_channel_swap(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 0] = 255  # blue in BGR
        swapped = NumpyImageOps().letterbox_batch([image], (8, 8), PARAMS).tensor
        assert swapped[0, 2].max() == pytest.approx(1.0)  # lands in channel 2 as RGB
        assert swapped[0, 0].max() == pytest.approx(0.0)

        kept = (
            NumpyImageOps()
            .letterbox_batch([image], (8, 8), NormalizeParams(std=(255.0,) * 3, swap_rb=False))
            .tensor
        )
        assert kept[0, 0].max() == pytest.approx(1.0)


class TestNumpyCrops:
    """The crop path, on the host — so a bug here fails CI without needing a GPU.

    It did once: clipping boxes with ``np.clip(..., out=fancy_index)`` silently refused to
    cast and raised only for some dtypes, and the GPU tier was the only thing that caught
    it. These tests move that coverage down a tier.
    """

    FRAME = np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (48, 1, 3))

    def test_shape_and_count(self) -> None:
        boxes = np.array([[0, 0, 32, 24], [32, 24, 64, 48]], dtype=np.float32)
        crops = NumpyImageOps().crop_batch(self.FRAME, boxes, (16, 16), PARAMS)
        assert crops.shape == (2, 3, 16, 16)
        assert crops.dtype == np.float32

    def test_boxes_are_clipped_to_the_frame(self) -> None:
        boxes = np.array([[-50, -50, 200, 200]], dtype=np.float32)
        crops = NumpyImageOps().crop_batch(self.FRAME, boxes, (8, 8), PARAMS)
        assert np.isfinite(crops).all()

    def test_float_boxes_do_not_raise(self) -> None:
        boxes = np.array([[1.5, 2.5, 30.25, 20.75]], dtype=np.float32)
        assert NumpyImageOps().crop_batch(self.FRAME, boxes, (8, 8), PARAMS).shape[0] == 1

    def test_degenerate_box_is_black(self) -> None:
        boxes = np.array([[10, 10, 10, 10]], dtype=np.float32)
        crops = NumpyImageOps().crop_batch(self.FRAME, boxes, (8, 8), PARAMS)
        assert np.abs(crops).max() == pytest.approx(0.0)

    def test_no_boxes_returns_an_empty_batch(self) -> None:
        empty = np.empty((0, 4), dtype=np.float32)
        assert NumpyImageOps().crop_batch(self.FRAME, empty, (8, 8), PARAMS).shape == (
            0,
            3,
            8,
            8,
        )


class TestNms:
    """NMS must agree exactly — it returns indices, so "close" is meaningless."""

    BOXES = np.array(
        [
            [10, 10, 100, 100],
            [15, 15, 105, 105],  # heavy overlap with the first
            [300, 300, 400, 400],
            [305, 305, 405, 405],  # heavy overlap with the third
            [500, 500, 520, 520],  # isolated
        ],
        dtype=np.float32,
    )
    SCORES = np.array([0.90, 0.80, 0.95, 0.70, 0.60], dtype=np.float32)

    def test_reference_suppresses_overlaps(self) -> None:
        kept = NumpyImageOps().nms(self.BOXES, self.SCORES, 0.5, 0.1, 10)
        assert kept.tolist() == [2, 0, 4]  # score order, overlaps removed

    def test_score_threshold(self) -> None:
        kept = NumpyImageOps().nms(self.BOXES, self.SCORES, 0.5, 0.85, 10)
        assert kept.tolist() == [2, 0]

    def test_max_output(self) -> None:
        kept = NumpyImageOps().nms(self.BOXES, self.SCORES, 0.5, 0.0, 1)
        assert kept.tolist() == [2]

    def test_empty_input(self) -> None:
        empty = np.empty((0, 4), dtype=np.float32)
        assert NumpyImageOps().nms(empty, np.empty(0, np.float32), 0.5, 0.1, 10).size == 0

    @pytest.mark.gpu
    def test_every_implementation_agrees(self) -> None:
        reference = NumpyImageOps().nms(self.BOXES, self.SCORES, 0.5, 0.1, 10)
        for name in _gpu_implementations():
            ops = IMAGE_OPS.create(name, device_index=0)
            assert ops.nms(self.BOXES, self.SCORES, 0.5, 0.1, 10).tolist() == reference.tolist()


@pytest.mark.gpu
class TestGpuParity:
    def test_geometry_matches_the_reference(self, frames) -> None:
        """Scales and pads are exact integers/ratios — they must match bit for bit, because
        post-processing inverts the letterbox with these numbers."""
        reference = NumpyImageOps().letterbox_batch(frames, (256, 256), PARAMS)
        for name in _gpu_implementations():
            result = IMAGE_OPS.create(name, device_index=0).letterbox_batch(
                frames, (256, 256), PARAMS
            )
            np.testing.assert_array_equal(result.scales, reference.scales)
            np.testing.assert_array_equal(result.pads, reference.pads)

    def test_gpu_implementations_agree_with_each_other(self, frames) -> None:
        """torch and the fused kernel both do bilinear with half-pixel centres, so they are
        expected to agree to floating-point noise — not merely to be 'close enough'."""
        import torch

        names = _gpu_implementations()
        if len(names) < 2:
            pytest.skip("needs both the torch and native implementations")

        out = torch.empty((len(frames), 3, 256, 256), dtype=torch.float32, device="cuda:0")
        results = {}
        for name in names:
            out.zero_()
            IMAGE_OPS.create(name, device_index=0).letterbox_to_device(frames, out, PARAMS)
            results[name] = out.cpu().numpy().copy()
            assert results[name].any(), f"{name} wrote nothing"

        first, second = names[0], names[1]
        np.testing.assert_allclose(results[first], results[second], atol=1e-5)

    def test_structurally_close_to_the_numpy_reference(self, frames) -> None:
        """Nearest vs bilinear on random noise differs pixel to pixel, but the *padding*
        must land in exactly the same place — that is the part a bug would break."""
        reference = NumpyImageOps().letterbox_batch(frames, (256, 256), PARAMS)
        result = get_image_ops(device_index=0).letterbox_batch(frames, (256, 256), PARAMS)
        assert result.tensor.shape == reference.tensor.shape
        # Row 1 is 1280x720 -> 56 rows of padding top and bottom, identical in both.
        np.testing.assert_allclose(
            result.tensor[1, :, :50, :], reference.tensor[1, :, :50, :], atol=1e-5
        )

    def test_crop_batch_agrees(self) -> None:
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 200, 200], [300, 100, 500, 400]], dtype=np.float32)

        reference = NumpyImageOps().crop_batch(frame, boxes, (64, 64), PARAMS)
        for name in _gpu_implementations():
            result = IMAGE_OPS.create(name, device_index=0).crop_batch(
                frame, boxes, (64, 64), PARAMS
            )
            assert result.shape == reference.shape == (2, 3, 64, 64)
            # Means over a large crop are robust to the interpolation difference; a wrong
            # box, a wrong channel order or a wrong normalisation would move them a lot.
            np.testing.assert_allclose(
                result.mean(axis=(2, 3)), reference.mean(axis=(2, 3)), atol=0.02
            )

    def test_degenerate_box_yields_a_black_crop(self) -> None:
        """Detectors emit them; killing a batch over one is the wrong trade."""
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        boxes = np.array([[50, 50, 50, 50]], dtype=np.float32)
        for name in _gpu_implementations():
            result = IMAGE_OPS.create(name, device_index=0).crop_batch(
                frame, boxes, (16, 16), PARAMS
            )
            assert np.abs(result).max() == pytest.approx(0.0, abs=1e-6)


def _gpu_implementations() -> list[str]:
    """Registered ops that can run on this host."""
    from shipinfer.runtime.native import is_native_available

    if not is_available():
        return []
    names = ["torch"]
    if is_native_available():
        names.append("native")
    return names
