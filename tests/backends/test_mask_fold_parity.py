"""Does the device fold compute what the readable one does?

The same rule `tests/runtime/test_ops_parity.py` exists for, one layer along: the fold that
runs on the engine's own device tensors is only defensible because a numpy reference agrees
with it, and that reference is what the offline tier and the cross-plane golden check.

Runs on CPU torch, deliberately. The arithmetic is the claim; the device it runs on is not,
and a parity test that needed a driver would be a parity test nobody runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.backends.tensorrt.fold import MaskAreaFold
from shipinfer.core.errors import InferenceError
from shipinfer.topology.elements.masks import InstanceMaskArea

torch = pytest.importorskip("torch")

CROP = (640, 640)
COEFFS = 8
PLANE = (40, 40)


def scene(
    count: int, rows: int = 5, *, seed: int = 7, scores: list[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One batch of ``(N, R, 6 + M)`` detections and its ``(N, M, h, w)`` prototype bank."""
    rng = np.random.default_rng(seed)
    detections = rng.normal(size=(count, rows, _PREFIX + COEFFS)).astype(np.float32)
    if scores is None:
        detections[:, :, 4] = rng.uniform(0.0, 1.0, size=(count, rows))
    else:
        detections[:, :, 4] = np.asarray(scores, dtype=np.float32).reshape(count, rows)
    protos = rng.normal(size=(count, COEFFS, *PLANE)).astype(np.float32)
    return detections, protos


_PREFIX = 6


def folds(**overrides: object) -> tuple[InstanceMaskArea, MaskAreaFold]:
    """The two implementations, configured from ONE set of numbers.

    Built as a pair rather than separately: two constructors with the same literals typed
    twice is how a parity test ends up comparing two different configurations and passing.
    """
    settings: dict[str, object] = {
        "crop_hw": CROP,
        "detections": "output0",
        "prototypes": "output1",
        "name": "mask_area_px",
        "score_threshold": 0.25,
        "mask_threshold": 0.5,
    }
    settings.update(overrides)
    return InstanceMaskArea(**settings), MaskAreaFold(**settings)  # type: ignore[arg-type]


class TestTheTwoImplementationsAgree:
    @pytest.mark.parametrize("count", [1, 3, 8])
    def test_the_areas_are_the_same_numbers(self, count: int) -> None:
        """Every batch size the engine can be called at, including one."""
        detections, protos = scene(count)
        reference, device = folds()

        expected = reference({"output0": detections, "output1": protos})["mask_area_px"]
        got = device(torch.from_numpy(detections), torch.from_numpy(protos), count)

        np.testing.assert_allclose(got.numpy(), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("mask_threshold", [0.1, 0.5, 0.9])
    def test_they_agree_at_every_mask_cut(self, mask_threshold: float) -> None:
        """The cut is the knob that decides the answer, so one value proves little.

        A fold that ignored `mask_threshold` entirely would pass a single-threshold test —
        the same vacuity `test_fold_wiring` was fixed for on the C++ side.
        """
        detections, protos = scene(4)
        reference, device = folds(mask_threshold=mask_threshold)

        expected = reference({"output0": detections, "output1": protos})["mask_area_px"]
        got = device(torch.from_numpy(detections), torch.from_numpy(protos), 4)

        np.testing.assert_allclose(got.numpy(), expected, rtol=0, atol=0)
        assert float(expected.sum()) > 0.0, "a cut that admits nothing proves nothing"

    def test_a_crop_the_engine_found_nothing_in_reports_no_area(self) -> None:
        """`score_threshold`, and it has to be the same side of the comparison on both.

        Off by one comparison and a crop with no instance reports the area of the whole
        plane, which downstream reads as a very large ship.
        """
        detections, protos = scene(2, rows=2, scores=[0.1, 0.2, 0.9, 0.3])
        reference, device = folds()

        expected = reference({"output0": detections, "output1": protos})["mask_area_px"]
        got = device(torch.from_numpy(detections), torch.from_numpy(protos), 2)

        assert float(expected[0, 0]) == 0.0, "no row cleared the floor"
        assert float(expected[1, 0]) > 0.0, "and this one did"
        np.testing.assert_allclose(got.numpy(), expected, rtol=0, atol=0)

    def test_only_the_rows_of_this_batch_are_read(self) -> None:
        """The device tensors are sized for the engine's MAXIMUM and the tail is last
        batch's data, so a fold that read all of them would answer for rows nobody asked
        about — and `ModelInstance`'s scatter would hand one request another's area."""
        detections, protos = scene(8)
        reference, device = folds()

        expected = reference({"output0": detections[:3], "output1": protos[:3]})
        got = device(torch.from_numpy(detections), torch.from_numpy(protos), 3)

        assert got.shape == (3, 1), "one row per crop of THIS batch"
        np.testing.assert_allclose(got.numpy(), expected["mask_area_px"], rtol=0, atol=0)


class TestTheRefusalsAreTheSame:
    def test_a_truncated_basis_is_refused_rather_than_sliced(self) -> None:
        """Both halves refuse it, because a mask built from the wrong planes is plausible.

        The numpy twin owns the shape and name checks; this one keeps the coefficient check
        because it is the only one that can still be wrong once the tensors are in hand.
        """
        detections, protos = scene(2)
        _, device = folds()
        short = torch.from_numpy(protos[:, : COEFFS - 2])

        with pytest.raises(InferenceError, match="truncated basis"):
            device(torch.from_numpy(detections), short, 2)
