"""The batched crop gather must compute exactly what the per-box loop computed.

``TorchImageOps.crop_batch`` used to run one ``F.interpolate`` per box: at 15 people a frame
and 1000 frames a second that is 15 000 launches a second to produce 15 000 small crops, and
the cost grew with the crowd — the one shape of work this server exists to keep flat. The
gather that replaced it is only worth having if it is the *same function*, so the loop is
frozen at the bottom of this module and every case below is checked against it.

These run on CPU torch, in the offline tier, deliberately. The parts most likely to be wrong
— half-pixel centres, the far neighbour clamped inside the patch, a black row for a
degenerate box, the order of the channel swap and the normalisation — are all arithmetic and
none of them need a device. :class:`TestCropBatchMatchesTheLoopOnCuda` repeats the table on a
GPU for the parts that do: the index upload, the gather itself and the copy back.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.runtime.ops.base import NormalizeParams
from shipinfer.runtime.ops.torch_ops import TorchImageOps, _bilinear_axis

torch = pytest.importorskip("torch")


# -- fixtures for the eye ----------------------------------------------------------------


def _gradient_frame(height: int, width: int) -> np.ndarray:
    """A frame where the pixel value *is* the coordinate, so a failure is readable.

    Noise would be as sensitive, but a diff on noise says only that the wrong pixel was
    fetched; here it says which one.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack(
        [xs % 256, ys % 256, (xs * 7 + ys * 13) % 256],
        axis=-1,
        dtype=np.uint8,
        casting="unsafe",
    )


def _random_boxes(count: int, shape: tuple[int, ...], seed: int = 7) -> np.ndarray:
    """Float boxes from a detector's point of view: unrounded, and some off the frame."""
    rng = np.random.default_rng(seed)
    height, width = shape[:2]
    xs = rng.uniform(-10.0, width + 10.0, size=(count, 2))
    ys = rng.uniform(-10.0, height + 10.0, size=(count, 2))
    return np.stack(
        [xs.min(axis=1), ys.min(axis=1), xs.max(axis=1), ys.max(axis=1)], axis=1
    ).astype(np.float32)


FRAME = _gradient_frame(48, 64)
NOISE = np.random.default_rng(20260826).integers(0, 256, (97, 131, 3), dtype=np.uint8)
PIXEL = np.full((1, 1, 3), 200, dtype=np.uint8)

#: ``(image, boxes, dst_size)`` per named case. Every one of these came from a way the
#: geometry can go wrong, not from a desire for coverage.
CASES = [
    pytest.param(FRAME, np.array([[8, 6, 40, 30]], np.float32), (16, 16), id="plain-box"),
    pytest.param(FRAME, np.array([[-30, -20, 200, 400]], np.float32), (12, 20), id="clipped"),
    pytest.param(
        FRAME,
        np.array([[0, 0, 16, 16], [10, 10, 10, 10], [4, 4, 20, 30], [7, 9, 7, 40]], np.float32),
        (8, 8),
        id="degenerate-among-valid",
    ),
    pytest.param(FRAME, np.array([[40, 30, 8, 6]], np.float32), (8, 8), id="reversed"),
    pytest.param(FRAME, np.array([[5, 5, 6, 6]], np.float32), (6, 6), id="one-pixel-box"),
    pytest.param(FRAME, np.array([[3, 4, 5, 7]], np.float32), (16, 32), id="upsampled"),
    pytest.param(NOISE, np.array([[0, 0, 130, 96]], np.float32), (4, 5), id="downsampled-hard"),
    pytest.param(FRAME, np.array([[2, 3, 33, 41]], np.float32), (1, 1), id="one-pixel-output"),
    pytest.param(FRAME, np.array([[1.5, 2.5, 30.25, 20.75]], np.float32), (9, 11), id="float"),
    pytest.param(FRAME, np.array([[60, 44, 64, 48]], np.float32), (7, 7), id="far-edge"),
    pytest.param(NOISE, _random_boxes(20, NOISE.shape), (14, 10), id="twenty-random"),
    pytest.param(PIXEL, np.array([[0, 0, 1, 1]], np.float32), (4, 4), id="one-pixel-frame"),
]

IMAGENET = NormalizeParams(mean=(103.5, 116.3, 123.7), std=(57.4, 57.1, 58.4))
PARAM_SETS = [
    pytest.param(NormalizeParams(swap_rb=True), id="unit-rgb"),
    pytest.param(NormalizeParams(swap_rb=False), id="unit-bgr"),
    pytest.param(IMAGENET, id="imagenet-rgb"),
    pytest.param(
        NormalizeParams(mean=IMAGENET.mean, std=IMAGENET.std, swap_rb=False), id="imagenet-bgr"
    ),
]


def _tolerance(params: NormalizeParams) -> float:
    """Absolute tolerance in normalised units, from a budget of 0.025 of a grey level.

    Both implementations combine the same four corner pixels; they differ only in how. ATen
    writes ``h0 * (w0 * p00 + w1 * p01) + h1 * (...)`` where this nests two ``lerp``s, and
    the source index is a multiply-then-subtract here where a fused multiply-add is
    permitted there — which can also move a sample exactly on an integer boundary to the
    neighbouring index, whose weight is then ~1. All of those are a couple of ULPs of 255,
    under a thousandth of a grey level; the budget is twenty times that and still orders of
    magnitude below any real defect. A half-pixel shift, an off-by-one clamp, a
    frame-clamped far neighbour or a swapped channel all move pixels by whole grey levels.
    """
    return 0.025 / min(params.std)


def _degenerate_rows(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """The crops the contract requires to be black, stated independently of the code.

    Deliberately re-derived from the boxes rather than read off the implementation: a test
    that asks the code which rows it decided to blacken cannot catch it deciding wrongly.
    """
    height, width = image.shape[:2]
    x1 = np.clip(boxes[:, 0], 0, width - 1).astype(np.int64)
    x2 = np.clip(boxes[:, 2], 0, width - 1).astype(np.int64)
    y1 = np.clip(boxes[:, 1], 0, height - 1).astype(np.int64)
    y2 = np.clip(boxes[:, 3], 0, height - 1).astype(np.int64)
    return np.nonzero((x2 <= x1) | (y2 <= y1))[0]


def _assert_matches_the_loop(
    result: np.ndarray, image: np.ndarray, boxes: np.ndarray, dst_size, params, device
) -> None:
    expected = _reference_crop_batch(image, boxes, dst_size, params, device=device)
    assert result.shape == expected.shape == (len(boxes), 3, *dst_size)
    assert result.dtype == expected.dtype == np.float32
    # The gather works in a channels-first layout and transposes at the end; a caller that
    # hands this straight to a backend gets a scrambled batch if the transpose stays a view.
    assert result.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(result, expected, rtol=0, atol=_tolerance(params))

    # A black crop is `(0 - mean) / std`, which involves no interpolation at all — so it is
    # exact, and a tolerance would hide a row that was merely nearly untouched.
    rows = _degenerate_rows(image, boxes)
    np.testing.assert_array_equal(result[rows], expected[rows])


# -- the sample tables -------------------------------------------------------------------


class TestSampleTables:
    """``_bilinear_axis`` is where every geometry decision is made; pin it on its own.

    Pure numpy, so these fail on any machine, and they say what the tables mean rather than
    what they contain.
    """

    def test_an_exact_size_axis_is_the_identity(self) -> None:
        origin = np.array([5, 0, 17], dtype=np.int64)
        extent = np.array([8, 8, 8], dtype=np.int64)
        i0, i1, w1 = _bilinear_axis(origin, extent, 8)
        np.testing.assert_array_equal(i0, origin[:, None] + np.arange(8))
        np.testing.assert_array_equal(i1, i0 + np.array([1] * 7 + [0]))
        # Exactly zero, not nearly: this is what makes an exact-size crop a bit-exact copy.
        assert (w1 == 0.0).all()

    @pytest.mark.parametrize("extent", [1, 2, 3, 5, 8, 13, 64])
    @pytest.mark.parametrize("dst", [1, 2, 3, 7, 16])
    def test_every_index_stays_inside_the_patch(self, extent: int, dst: int) -> None:
        origin = np.array([11], dtype=np.int64)
        i0, i1, w1 = _bilinear_axis(origin, np.array([extent], np.int64), dst)
        assert i0.dtype == i1.dtype == np.int64 and w1.dtype == np.float32
        assert (i0 >= origin[0]).all() and (i0 <= origin[0] + extent - 1).all()
        assert (i1 >= origin[0]).all() and (i1 <= origin[0] + extent - 1).all()
        assert set(np.unique(i1 - i0).tolist()) <= {0, 1}
        assert (w1 >= 0.0).all() and (w1 <= 1.0).all()
        assert (np.diff(i0, axis=1) >= 0).all(), "sampling must walk forward"

    def test_the_far_neighbour_is_clamped_to_the_patch_not_the_frame(self) -> None:
        """The counter-example that rules out ``grid_sample``.

        A 2-pixel box upsampled to 4 columns: the last column samples past the box, and the
        contract holds the box's own edge. ``grid_sample`` would blend in ``p[origin + 2]``,
        which belongs to whatever is standing next to this person.
        """
        i0, i1, _w1 = _bilinear_axis(np.array([30], np.int64), np.array([2], np.int64), 4)
        np.testing.assert_array_equal(i0, [[30, 30, 30, 31]])
        np.testing.assert_array_equal(i1, [[31, 31, 31, 31]])

    def test_an_empty_extent_collapses_onto_the_origin(self) -> None:
        """Degenerate boxes are blacked out afterwards, but the gather still runs — and an
        out-of-range index on CUDA is a device-side assert that poisons the context."""
        origin = np.array([4, 9], dtype=np.int64)
        for extent in (0, -6):
            i0, i1, _w1 = _bilinear_axis(origin, np.full(2, extent, np.int64), 5)
            assert (i0 == origin[:, None]).all() and (i1 == origin[:, None]).all()


# -- the crop itself ---------------------------------------------------------------------


class TestCropBatchMatchesTheLoop:
    @pytest.mark.parametrize(("image", "boxes", "dst_size"), CASES)
    @pytest.mark.parametrize("params", PARAM_SETS)
    def test_case(self, image, boxes, dst_size, params) -> None:
        ops = TorchImageOps()
        result = ops.crop_batch(image, boxes, dst_size, params)
        _assert_matches_the_loop(result, image, boxes, dst_size, params, ops._device)

    def test_an_exact_size_crop_is_a_bit_exact_copy(self) -> None:
        """When the box is already the destination size the scale is exactly 1.0, every
        source index is an integer and every weight is exactly zero — so the crop is the raw
        patch, normalised, and not merely the raw patch to within a tolerance. Any drift
        here means the half-pixel arithmetic is off."""
        ops = TorchImageOps()
        params = NormalizeParams(mean=(1.0, 2.0, 3.0), std=(4.0, 5.0, 6.0), swap_rb=True)
        boxes = np.array([[10, 6, 26, 18]], dtype=np.float32)  # 16 wide, 12 high
        result = ops.crop_batch(FRAME, boxes, (12, 16), params)

        patch = FRAME[6:18, 10:26, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        mean = np.asarray(params.mean, np.float32)[None, :, None, None]
        std = np.asarray(params.std, np.float32)[None, :, None, None]
        np.testing.assert_array_equal(result, (patch - mean) / std)
        np.testing.assert_array_equal(
            result, _reference_crop_batch(FRAME, boxes, (12, 16), params, device=ops._device)
        )

    def test_no_boxes_returns_an_empty_batch(self) -> None:
        result = TorchImageOps().crop_batch(
            FRAME, np.empty((0, 4), np.float32), (8, 8), NormalizeParams()
        )
        assert result.shape == (0, 3, 8, 8)
        assert result.dtype == np.float32

    def test_a_non_bilinear_ops_object_refuses_to_crop(self) -> None:
        """The tables encode bilinear sampling. Cropping bilinearly for an operator who
        configured nearest would show up as a slightly worse embedding, never as an error."""
        ops = TorchImageOps(interpolation="nearest")
        with pytest.raises(ConfigurationError, match="bilinear"):
            ops.crop_batch(
                FRAME, np.array([[1, 1, 9, 9]], np.float32), (4, 4), NormalizeParams()
            )


class TestChunking:
    """A batched gather is not O(1) in memory the way the loop was, so the pass is bounded."""

    def test_the_ranges_cover_every_crop_exactly_once(self) -> None:
        ops = TorchImageOps()
        chunks = list(ops._crop_chunks(37, 8, 8))
        assert chunks[0][0] == 0 and chunks[-1][1] == 37
        assert all(lo < hi for lo, hi in chunks)
        assert [hi for _lo, hi in chunks[:-1]] == [lo for lo, _hi in chunks[1:]]

    def test_one_crop_larger_than_the_budget_still_gets_a_range(self) -> None:
        ops = TorchImageOps()
        assert list(ops._crop_chunks(2, 4096, 4096)) == [(0, 1), (1, 2)]

    def test_a_batch_larger_than_one_chunk_matches_the_loop(self, monkeypatch) -> None:
        params = IMAGENET
        boxes = _random_boxes(9, NOISE.shape, seed=11)
        ops = TorchImageOps()
        monkeypatch.setattr(TorchImageOps, "_CROP_CHUNK_ELEMENTS", 3 * 12 * 10 * 2)
        assert len(list(ops._crop_chunks(len(boxes), 12, 10))) >= 3, "the case must chunk"

        result = ops.crop_batch(NOISE, boxes, (12, 10), params)
        _assert_matches_the_loop(result, NOISE, boxes, (12, 10), params, ops._device)


class TestTheConstantTablesAreBuiltOnce:
    """Every crop used to upload its own mean, std and channel flip. On the embedder's hot
    path that is a synchronous copy per call for values fixed at load time."""

    def test_equal_params_share_one_pair(self) -> None:
        ops = TorchImageOps()
        first = ops._normalization(IMAGENET, ops._device)
        again = ops._normalization(
            NormalizeParams(mean=IMAGENET.mean, std=IMAGENET.std, swap_rb=False), ops._device
        )
        assert first[0] is again[0] and first[1] is again[1]

    def test_different_values_get_their_own_pair(self) -> None:
        ops = TorchImageOps()
        mean, std = ops._normalization(IMAGENET, ops._device)
        other_mean, other_std = ops._normalization(NormalizeParams(), ops._device)
        assert other_mean is not mean and other_std is not std
        assert other_mean.flatten().tolist() == [0.0, 0.0, 0.0]

    def test_the_channel_order_is_the_swap(self) -> None:
        ops = TorchImageOps()
        assert ops._channel_order(True, ops._device).tolist() == [2, 1, 0]
        assert ops._channel_order(False, ops._device).tolist() == [0, 1, 2]
        assert ops._channel_order(True, ops._device) is ops._channel_order(True, ops._device)

    def test_letterbox_shares_the_cache_with_the_crop_path(self) -> None:
        ops = TorchImageOps()
        ops.letterbox_batch([FRAME], (16, 16), IMAGENET)
        mean, std = ops._normalization(IMAGENET, ops._device)
        assert len(ops._norm_cache) == 1, "letterbox built its own pair"
        assert mean.flatten().tolist() == pytest.approx(list(IMAGENET.mean))
        assert std.flatten().tolist() == pytest.approx(list(IMAGENET.std))

    def test_a_caller_that_never_repeats_itself_cannot_grow_the_cache(self) -> None:
        ops = TorchImageOps()
        for index in range(TorchImageOps._CACHE_LIMIT * 2 + 1):
            ops._normalization(NormalizeParams(mean=(float(index),) * 3), ops._device)
        assert 0 < len(ops._norm_cache) <= TorchImageOps._CACHE_LIMIT
        # and it still works after the drop
        mean, _std = ops._normalization(IMAGENET, ops._device)
        assert mean.flatten().tolist() == pytest.approx(list(IMAGENET.mean))


class _CountingMode(torch.overrides.TorchFunctionMode):
    """Counts torch API calls, which is the closest an offline test gets to counting
    launches: on CPU there are none, and the property under test is about how many times the
    implementation asks torch to do something, not about the device it asks."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return func(*args, **(kwargs or {}))


class TestTheWorkDoesNotScaleWithTheBoxCount:
    """The whole point. A crowded camera must not cost more torch calls than an empty one —
    that is what the per-box loop got wrong, and a benchmark on an uncrowded frame would
    never have shown it."""

    def test_two_boxes_and_thirty_two_cost_the_same_calls(self) -> None:
        params = NormalizeParams()
        boxes = np.stack(
            [
                np.array([2 + i % 5, 3 + i % 7, 40 + i % 5, 44 + i % 7], np.float32)
                for i in range(32)
            ]
        )
        ops = TorchImageOps()
        ops.crop_batch(FRAME, boxes[:2], (8, 8), params)  # fill the constant tables

        counts = []
        for count in (2, 32):
            with _CountingMode() as mode:
                ops.crop_batch(FRAME, boxes[:count], (8, 8), params)
            counts.append(mode.calls)

        assert counts[0] == counts[1], f"16x the boxes cost {counts[1] - counts[0]} more calls"
        assert counts[1] < 60, f"a single crop pass should be a handful of ops, not {counts[1]}"


@pytest.mark.gpu
class TestCropBatchMatchesTheLoopOnCuda:
    """The same table on a device: the index upload, the gather and the copy back are the
    parts CPU torch does not exercise."""

    @pytest.mark.parametrize(("image", "boxes", "dst_size"), CASES)
    @pytest.mark.parametrize("params", [IMAGENET, NormalizeParams(swap_rb=False)])
    def test_case(self, image, boxes, dst_size, params) -> None:
        ops = TorchImageOps(device_index=0)
        assert ops.on_device, "the gpu tier must not quietly measure the CPU path"
        result = None
        try:
            result = ops.crop_batch(image, boxes, dst_size, params)
            _assert_matches_the_loop(result, image, boxes, dst_size, params, ops._device)
        finally:
            del result, ops
            torch.cuda.empty_cache()


# -- the loop this module replaced -------------------------------------------------------


def _reference_crop_batch(
    image: np.ndarray,
    boxes: np.ndarray,
    dst_size: tuple[int, int],
    params: NormalizeParams,
    *,
    device,
) -> np.ndarray:
    """A frozen copy of ``TorchImageOps.crop_batch`` at 0b18100 — the per-box loop the
    batched gather replaces.

    Spelled out here rather than imported or reconstructed from the class, so it cannot
    follow the implementation: a reference that shares code with the thing it checks only
    proves the code equals itself. It is deliberately unchanged from the original, down to
    the ``mode`` argument that used to come from ``self._interpolation``.
    """
    dst_h, dst_w = dst_size
    if boxes.size == 0:
        return np.empty((0, 3, dst_h, dst_w), dtype=np.float32)

    src_h, src_w = image.shape[:2]
    frame = (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device, non_blocking=True)
        .permute(2, 0, 1)
        .float()
    )
    clipped = np.empty_like(boxes, dtype=np.int64)
    clipped[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_w - 1)
    clipped[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_h - 1)

    out = torch.zeros((boxes.shape[0], 3, dst_h, dst_w), dtype=torch.float32, device=device)
    for i, (x1, y1, x2, y2) in enumerate(clipped):
        if x2 <= x1 or y2 <= y1:
            continue  # degenerate box -> zeros, never an exception
        patch = frame[:, y1:y2, x1:x2].unsqueeze(0)
        out[i] = torch.nn.functional.interpolate(
            patch, size=(dst_h, dst_w), mode="bilinear", align_corners=False
        )[0]

    if params.swap_rb:
        out = out.flip(1)
    mean = torch.tensor(params.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(params.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    out.sub_(mean).div_(std)
    return out.cpu().numpy()
