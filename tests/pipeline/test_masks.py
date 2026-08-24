"""The mask arithmetic, asserted on the number it produces.

`InstanceMaskArea` is the only place in the pipeline that turns a segmentation engine's two
outputs into a mask, and the suite's only assertion about it was `mask_area_px is not None`.
Meanwhile the conftest fixture deliberately builds a closed-form case — all weight on
prototype plane 0, saturated so sigmoid ~ 1, every cell inside the instance — and no test
ever asserted the answer that implies.

Each of these is green if the arithmetic is inverted:

- flip ``cut = log(t / (1 - t))`` and every area becomes ``total - area``;
- use ``planes.shape[1]`` (the coefficient count) instead of ``[2]`` (the cell count) for the
  scale and every area is off by 32x;
- drop the score gate and a crop the segmenter found nothing in reports the largest area of
  the batch, which is what the docstring's measured 0.011/0.033 scores are about.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.errors import InferenceError
from shipinfer.pipeline.graph.masks import InstanceMaskArea

#: Prototype planes per row, as the shipped engine emits.
COEFFICIENTS = 32
#: A small prototype extent, so the expected areas are easy to state exactly.
PROTO_HW = (4, 4)
CROP_HW = (16, 16)


def outputs(
    *,
    rows: int = 1,
    score: float = 0.9,
    weight: float = 10.0,
    plane_value: float | None = None,
    coefficients: int = COEFFICIENTS,
    proto_hw: tuple[int, int] = PROTO_HW,
) -> dict[str, np.ndarray]:
    """One segmentation batch: `(N, 1, 6+M)` detections and `(N, M, h, w)` prototypes.

    All the weight goes on plane 0, so the mask logits are exactly ``weight * plane_value``
    and the expected area is closed-form.
    """
    detections = np.zeros((rows, 1, 6 + coefficients), dtype=np.float32)
    detections[:, 0, 4] = score
    detections[:, 0, 5] = 8.0
    detections[:, 0, 6] = 1.0
    planes = np.zeros((rows, coefficients, *proto_hw), dtype=np.float32)
    planes[:, 0] = weight if plane_value is None else plane_value
    return {"output0": detections, "output1": planes}


def area(**kwargs) -> np.ndarray:
    combine = InstanceMaskArea(crop_hw=CROP_HW, score_threshold=0.25)
    return combine(outputs(**kwargs))["mask_area_px"]


class TestTheAreaIsInCropPixels:
    def test_an_all_foreground_mask_is_the_whole_crop(self) -> None:
        """Every prototype cell inside the instance, scaled from cells to crop pixels. The
        cell scale is `crop_h*crop_w / (h*w)`; using the coefficient count instead is a 32x
        error that no shape check would catch."""
        assert area()[0, 0] == pytest.approx(CROP_HW[0] * CROP_HW[1])

    def test_an_all_background_mask_is_zero(self) -> None:
        """Inverting the logit threshold turns this into the whole crop."""
        assert area(weight=-10.0)[0, 0] == pytest.approx(0.0)

    def test_half_the_plane_inside_is_half_the_crop(
        self,
    ) -> None:
        combine = InstanceMaskArea(crop_hw=CROP_HW, score_threshold=0.25)
        batch = outputs()
        batch["output1"][0, 0, :2, :] = -10.0  # top half outside the instance

        result = combine(batch)["mask_area_px"]

        assert result[0, 0] == pytest.approx(CROP_HW[0] * CROP_HW[1] / 2)

    def test_the_scale_follows_the_crop_size(self) -> None:
        """Same mask, a bigger crop: four times the pixels."""
        small = InstanceMaskArea(crop_hw=(16, 16))(outputs())["mask_area_px"][0, 0]
        large = InstanceMaskArea(crop_hw=(32, 32))(outputs())["mask_area_px"][0, 0]

        assert large == pytest.approx(small * 4)


class TestTheScoreGate:
    def test_a_row_below_the_threshold_reports_no_area(self) -> None:
        """Measured on the shipped engine, two distant vessels scored 0.011 and 0.033 with
        best rows whose masks covered the *whole plane*. Taking the argmax unconditionally
        makes a crop the segmenter found nothing in report the largest area of the batch."""
        assert area(score=0.01)[0, 0] == pytest.approx(0.0)

    def test_a_row_at_the_threshold_is_kept(self) -> None:
        assert area(score=0.25)[0, 0] == pytest.approx(CROP_HW[0] * CROP_HW[1])

    def test_each_row_is_gated_independently(self) -> None:
        combine = InstanceMaskArea(crop_hw=CROP_HW, score_threshold=0.25)
        batch = outputs(rows=2)
        batch["output0"][1, 0, 4] = 0.01  # the second crop found nothing

        result = combine(batch)["mask_area_px"]

        assert result[0, 0] == pytest.approx(CROP_HW[0] * CROP_HW[1])
        assert result[1, 0] == pytest.approx(0.0)


class TestTheShapeContract:
    def test_one_area_per_crop(self) -> None:
        assert area(rows=5).shape == (5, 1)

    def test_a_missing_output_is_named(self) -> None:
        combine = InstanceMaskArea(crop_hw=CROP_HW)
        with pytest.raises(InferenceError, match="output1"):
            combine({"output0": outputs()["output0"]})

    def test_a_coefficient_count_that_disagrees_is_refused(self) -> None:
        """A truncated basis builds a plausible mask from the wrong planes, so the
        disagreement is named rather than silently sliced."""
        combine = InstanceMaskArea(crop_hw=CROP_HW)
        batch = outputs()
        batch["output1"] = batch["output1"][:, :16]

        with pytest.raises(InferenceError, match="coefficient"):
            combine(batch)

    def test_outputs_disagreeing_on_batch_size_are_refused(self) -> None:
        combine = InstanceMaskArea(crop_hw=CROP_HW)
        batch = outputs(rows=3)
        batch["output1"] = batch["output1"][:2]

        with pytest.raises(InferenceError, match="batch size"):
            combine(batch)

    def test_a_detection_output_of_the_wrong_rank_is_refused(self) -> None:
        combine = InstanceMaskArea(crop_hw=CROP_HW)
        batch = outputs()
        batch["output0"] = batch["output0"][0]

        with pytest.raises(InferenceError, match="6\\+coeffs"):
            combine(batch)


class TestConstruction:
    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.5, 1.5])
    def test_an_impossible_mask_threshold_is_refused(self, threshold: float) -> None:
        """`log(t/(1-t))` is undefined at both ends, and a silently clamped threshold would
        make every area either zero or the whole crop."""
        with pytest.raises(ValueError, match="mask_threshold"):
            InstanceMaskArea(crop_hw=CROP_HW, mask_threshold=threshold)
