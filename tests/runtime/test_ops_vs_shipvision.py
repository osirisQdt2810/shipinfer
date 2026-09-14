"""The premise `V124a-PHASE3` rests on: do our ops and shipvision's already agree?

That item thins `runtime/ops` onto `shipvision.imgproc`, which is only safe where the two
already compute the same thing. Measured rather than assumed, and pinned here so the premise
cannot rot silently between now and the refactor -- a submodule bump that moves either side
should fail HERE, with a message saying what it costs, rather than as a surprise mid-move.

It also pins the one place they are MEANT to differ. `numpy_ops.py` samples nearest so the
CUDA kernel has a bit-exact reference (its own docstring says so, and `test_ops_parity.py`'s
header states the same asymmetry); shipvision's numpy oracle is bilinear. That difference is
a decision, so the test asserts it is still there AND that it is the ONLY one.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.runtime.ops.base import NormalizeParams
from shipinfer.runtime.ops.numpy_ops import NumpyImageOps
from shipinfer.topology import bridge as bridge_module

needs_shipvision = pytest.mark.skipif(
    not bridge_module.shipvision_available(),
    reason="shipvision is not importable; the submodule is not checked out",
)
torch = pytest.importorskip("torch")

FRAMES_HW = ((480, 640), (300, 300), (720, 1280))


def frames() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in FRAMES_HW]


def theirs(kind: str):
    import shipvision.imgproc as imgproc

    return {"numpy": imgproc.NumpyImageOps, "torch": imgproc.TorchImageOps}[kind]()


def as_array(value) -> np.ndarray:
    value = value.tensor if hasattr(value, "tensor") else value
    return np.asarray(value.cpu() if hasattr(value, "cpu") else value)


@needs_shipvision
class TestTheTorchOpsAlreadyAgree:
    """The half that CAN move: torch is a hard dependency (ADR-003), so delegating it adds
    nothing to the floor -- and these numbers are why it is a move rather than a rewrite."""

    def test_letterbox_is_bit_exact(self) -> None:
        from shipinfer.runtime.ops.torch_ops import TorchImageOps

        mine = as_array(
            TorchImageOps().letterbox_batch(frames(), (640, 640), NormalizeParams())
        )
        other = as_array(theirs("torch").letterbox(frames(), (640, 640))[0])

        assert mine.shape == other.shape
        assert np.array_equal(mine, other), (
            f"the torch letterboxes have drifted (max |d| {np.abs(mine - other).max():.3g}). "
            "V124a-PHASE3 plans to delete ours in favour of theirs; that is only free while "
            "this is exact"
        )

    @pytest.mark.parametrize(
        ("label", "box"),
        [("no scaling", (10.0, 10.0, 138.0, 138.0)), ("downscale", (10.0, 10.0, 266.0, 266.0))],
    )
    def test_crop_agrees_when_it_is_not_upscaling(
        self, label: str, box: tuple[float, ...]
    ) -> None:
        """Not bit-exact and not expected to be: same maths, different op order in float32."""
        delta = self.crop_delta(box, (128, 128))

        assert delta < 1e-3, f"{label}: torch crops differ by {delta:.3g}, past float32 noise"

    # doc: long the one place the move is not free, and why a tolerance would have hidden it
    @pytest.mark.parametrize(
        ("scale", "box"),
        [
            (1.07, (10.0, 10.0, 130.0, 130.0)),
            (2.0, (10.0, 10.0, 74.0, 74.0)),
            (4.0, (10.0, 10.0, 42.0, 42.0)),
        ],
    )
    def test_crop_DIVERGES_once_it_upscales(self, scale: float, box: tuple[float, ...]) -> None:
        """AND THIS IS A FINDING, not a tolerance to widen. Measured 14 Sep: the two torch
        crops agree to float32 noise while downscaling or copying, and separate as soon as
        they sample denser than the source -- 0.05 at 1.07x, 0.30 at 2x, 0.43 at 4x.

        It is a production path: a person box far from the camera is smaller than the
        embedder's input and is upscaled into it, so `V124a-PHASE3` swapping our crop for
        shipvision's would change embedder inputs for exactly the small detections identity
        is hardest on. Pinned as a difference so the move prices it instead of assuming it.
        """
        delta = self.crop_delta(box, (128, 128))

        assert delta > 1e-3, (
            f"the torch crops now agree at {scale}x upscale (max |d| {delta:.3g}). If that is "
            "deliberate, V124a-PHASE3 just got cheaper and this test should say so instead"
        )

    def crop_delta(self, box: tuple[float, ...], target: tuple[int, int]) -> float:
        from shipinfer.runtime.ops.torch_ops import TorchImageOps

        image = frames()[0]
        boxes = np.array([box], dtype=np.float32)
        mine = as_array(TorchImageOps().crop_batch(image, boxes, target, NormalizeParams()))
        other = as_array(theirs("torch").crop_batch(image, boxes, target))
        return float(np.abs(mine.astype(np.float64) - other.astype(np.float64)).max())


@needs_shipvision
class TestTheNumpyOpsDifferOnlyByTheResample:
    """The half that must NOT move, and the evidence for why it is a decision not an accident.

    `numpy_ops.py` is the dependency-free floor (`ops/__init__.py`: "native if built, else
    torch, else numpy") and CI runs that tier with no submodule, so delegating it would put
    shipvision under ADR-001. It is also the nearest-neighbour reference the CUDA kernel is
    matched against, which is a different job from shipvision's bilinear oracle.
    """

    def test_they_agree_exactly_when_nothing_is_resampled(self) -> None:
        """So the difference is the sampler and nothing else -- not padding, not the
        normalisation, not the channel swap, not the geometry."""
        exact = [np.asarray(frames()[0][:480, :480])]
        mine = as_array(NumpyImageOps().letterbox_batch(exact, (480, 480), NormalizeParams()))
        other = as_array(theirs("numpy").letterbox(exact, (480, 480))[0])

        assert np.array_equal(mine, other), "even with no resize the numpy paths now differ"

    def test_and_differ_once_anything_is(self) -> None:
        """The guard on the decision. If this ever passes, either our sampler became bilinear
        -- and the kernel lost its bit-exact reference -- or theirs became nearest."""
        mine = as_array(
            NumpyImageOps().letterbox_batch(frames(), (640, 640), NormalizeParams())
        )
        other = as_array(theirs("numpy").letterbox(frames(), (640, 640))[0])

        assert not np.array_equal(mine, other), (
            "our numpy oracle now matches shipvision's. Ours is nearest ON PURPOSE, so the "
            "CUDA kernel has an unambiguous reference; if that changed, `test_ops_parity.py`'s "
            "stated asymmetry and the comment in `numpy_ops.py::_resize_nearest` are both stale"
        )


@needs_shipvision
class TestSuppressAgrees:
    def test_nms_picks_the_same_boxes(self) -> None:
        """100 random fleets, because one hand-made case agrees by luck as often as by rule."""
        import shipvision.imgproc as imgproc

        rng = np.random.default_rng(7)
        for trial in range(100):
            count = int(rng.integers(1, 80))
            corner = rng.random((count, 2)).astype(np.float32) * 100
            extent = rng.random((count, 2)).astype(np.float32) * 40 + 1
            boxes = np.concatenate([corner, corner + extent], axis=1).astype(np.float32)
            scores = rng.random(count).astype(np.float32)
            iou, floor = float(rng.choice([0.3, 0.5, 0.7])), float(rng.choice([0.0, 0.25]))

            mine = np.asarray(NumpyImageOps().nms(boxes, scores, iou, floor, count)).ravel()
            other, _ = imgproc.suppress(
                boxes, scores, iou_threshold=iou, score_threshold=floor, max_output=count
            )

            assert np.array_equal(
                mine, np.asarray(other).ravel()
            ), f"trial {trial}: nms disagrees at iou={iou} score={floor}"
