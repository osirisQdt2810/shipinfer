"""One number per object, from a segmentation engine's two outputs.

A YOLO segmentation head does not emit masks. It emits **detections** — 300 slots of
``[x1, y1, x2, y2, score, class]`` followed by 32 mask coefficients — and a bank of 32
**prototype** planes at a quarter of the input resolution. A mask is the coefficients'
linear combination of those planes, through a sigmoid. Anything that wants a mask has to do
that arithmetic, and this module is the only place in the pipeline that does it.

**Why an area and not a mask.** Reassembly holds a stage's output until the frame is
complete, so a stage that stores pixels turns a 1024-frame bound into tens of gigabytes: one
160x160 float plane per object is 100 KB, and fifteen objects a frame across a full buffer is
1.5 GB of pixels nobody reads. The mask is not publishable either — the architecture document
is explicit that this bus carries metadata while frames stay in shared memory — so it is
reduced at the seam where it is produced rather than at the seam where it would be dropped.

**Why the score threshold is not optional.** Measured on the shipped ``yolo26n-seg`` engine
over the four ship crops of ``ship_2K/ship1.jpg``: the two near vessels score 0.688 and
0.195, and the two distant ones — a few hundred pixels stretched to 640x640 — score 0.011 and
0.033, with best rows whose masks cover *the whole plane*. Take the argmax unconditionally
and a crop the segmenter found nothing in reports the largest area of the four. That number
would be plausible, wrong and silent, which is the class of failure this project refuses.

**What is claimed.** That the arithmetic is the engine's own and the area is in the crop's
own pixels. Not that the number measures a vessel: the crop handed to the segmenter is a
detection box stretched to a square, which is out of distribution for a whole-image
segmentation model, and the scores above say so. Saying which of the two a number supports is
the difference between evidence and decoration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from shipinfer.core.errors import InferenceError

__all__ = ["InstanceMaskArea"]

#: Columns before the mask coefficients in one detection row: ``x1, y1, x2, y2, score, class``.
_PREFIX = 6
#: Column of the confidence within that prefix.
_SCORE = 4


@dataclass(frozen=True, slots=True)
class InstanceMaskArea:
    """Fold a segmentation engine's outputs into one foreground area per crop.

    Shaped as an :class:`~shipinfer.pipeline.graph.objects.ObjectStage` ``combine`` function:
    every one of the engine's outputs in, the quantities the stage forwards out. A
    per-output reducer cannot express this, because the answer is two outputs multiplied
    together.

    Args:
        crop_hw: the crop size the engine was fed, so a count of mask cells becomes an area
            in the crop's own pixels instead of in prototype cells (a factor of 16 at
            640x640 against 160x160 prototypes).
        detections: the engine output holding ``(N, R, 6 + M)`` rows.
        prototypes: the engine output holding ``(N, M, h, w)`` planes.
        name: the key the stage's ``outputs`` map reads. Not a model output name — this
            quantity is one the model does not have a name for.
        score_threshold: a row below this is not an instance, and its area is 0.
        mask_threshold: a mask *probability* at or above this is inside the instance.
            Applied to the logits through :func:`math.log`, which is the same comparison
            without a sigmoid over 25 600 cells per object.
    """

    crop_hw: tuple[int, int]
    detections: str = "output0"
    prototypes: str = "output1"
    name: str = "mask_area_px"
    score_threshold: float = 0.25
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_threshold < 1.0:
            raise ValueError(f"mask_threshold must be in (0, 1), got {self.mask_threshold}")

    def __call__(self, outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """``(N, 1)`` float32 areas, one row per crop, in crop pixels.

        Raises:
            InferenceError: an output is missing, or the coefficient count disagrees with
                the prototype bank. Both mean the engine is not the one this stage was
                configured for, and a truncated basis produces a plausible mask from the
                wrong planes — so it is named here rather than silently sliced.
        """
        rows = self._require(outputs, self.detections)
        protos = self._require(outputs, self.prototypes)
        if rows.ndim != 3 or rows.shape[2] <= _PREFIX:
            raise InferenceError(
                f"segmentation output {self.detections!r} must be (N, rows, 6+coeffs), got "
                f"{rows.shape}"
            )
        if protos.ndim != 4:
            raise InferenceError(
                f"segmentation output {self.prototypes!r} must be (N, coeffs, h, w), got "
                f"{protos.shape}"
            )
        count, channels = protos.shape[0], protos.shape[1]
        coefficients = rows.shape[2] - _PREFIX
        if coefficients != channels:
            raise InferenceError(
                f"segmentation engine emits {coefficients} mask coefficient(s) per row but "
                f"{channels} prototype plane(s); one of the two outputs is not the one this "
                f"stage was configured for, and combining them would build a mask from a "
                f"truncated basis"
            )
        if rows.shape[0] != count:
            raise InferenceError(
                f"segmentation outputs disagree on batch size: {self.detections!r} has "
                f"{rows.shape[0]} row(s), {self.prototypes!r} has {count}"
            )

        # The strongest row of each crop. The crop *is* one object, so its instance is the
        # engine's best answer for that crop, not the union of everything it saw in it.
        best = np.argmax(rows[:, :, _SCORE], axis=1)
        chosen = rows[np.arange(count), best]
        found = chosen[:, _SCORE] >= self.score_threshold

        # One einsum rather than a loop over objects: each crop has its own prototype bank,
        # so this is (N, M) x (N, M, h*w) -> (N, h*w). At batch 8 and 160x160 that is 6.5 M
        # multiply-accumulates, microseconds of numpy, and it happens once per segmenter call.
        planes = protos.reshape(count, channels, -1)
        logits = np.einsum("nm,nmk->nk", chosen[:, _PREFIX:], planes)
        cut = math.log(self.mask_threshold / (1.0 - self.mask_threshold))
        cells = np.count_nonzero(logits >= cut, axis=1)

        crop_h, crop_w = self.crop_hw
        cell_px = (crop_h * crop_w) / float(planes.shape[2])
        areas = np.where(found, cells * cell_px, 0.0)
        return {self.name: areas.astype(np.float32).reshape(-1, 1)}

    @staticmethod
    def _require(outputs: Mapping[str, np.ndarray], name: str) -> np.ndarray:
        array = outputs.get(name)
        if array is None:
            raise InferenceError(
                f"segmentation output {name!r} is missing (got: {sorted(outputs)}); a "
                f"detection-only engine has one output and a segmentation engine has two"
            )
        return np.asarray(array)
