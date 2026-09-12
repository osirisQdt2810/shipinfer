"""The mask fold as an engine attachment — the Python twin of ``tensorrt/fold.{h,cpp}``.

Thin on purpose, and the same division the C++ side draws: every refusal belongs to the pure
half (:class:`~shipinfer.topology.elements.masks.InstanceMaskArea` validates the engine's
shapes and names), and what is left here is the one thing that cannot be pure — arithmetic on
the tensors TensorRT just wrote into, on the instance's own stream, while it still owns the
batch.

Why it exists at all: a YOLO-seg engine emits a ``(N, 32, 160, 160)`` prototype bank that
nothing above reads once an area has been computed from it — 3.1 MB a row copied home to be
reduced to one float. The C++ plane stopped paying that in #232/#239; this is V88's other half.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shipinfer.core.errors import InferenceError

__all__ = ["MaskAreaFold"]

#: Columns before the mask coefficients in a YOLO-seg detection row: x, y, w, h, score, class.
_PREFIX = 6
#: The column holding the row's score, which decides which row is the crop's instance.
_SCORE = 4


# doc: long the six numbers the chain hands this fold, and what each of them decides
@dataclass(frozen=True, slots=True)
class MaskAreaFold:
    """One area per crop, computed on the device from two of the engine's outputs.

    Constructed from the same numbers :class:`InstanceMaskArea` carries, because the chain is
    the single source of truth for them — the resolved plan hands the C++ composition root a
    ``MaskAreaSpec`` and this is the same spec arriving by the same route.

    Args:
        crop_hw: the crop size the engine was fed, so a count of mask cells becomes an area in
            the crop's own pixels rather than in prototype cells.
        detections: the engine output holding ``(N, R, 6 + M)`` rows.
        prototypes: the engine output holding ``(N, M, h, w)`` planes. This is the one that
            stops being copied home.
        name: the name the folded output is advertised under. Not a model output name — the
            engine has no name for this quantity.
        score_threshold: a row below this is not an instance, and its area is 0.
        mask_threshold: a mask *probability* at or above this is inside the instance. Compared
            as a logit, which is the same test without a sigmoid over 25 600 cells an object.
    """

    crop_hw: tuple[int, int]
    detections: str
    prototypes: str
    name: str
    score_threshold: float
    mask_threshold: float

    @property
    def logit_cut(self) -> float:
        """``mask_threshold`` as a logit, so no sigmoid runs over the planes."""
        return math.log(self.mask_threshold / (1.0 - self.mask_threshold))

    def __call__(self, rows: Any, planes: Any, count: int) -> Any:
        """``(count, 1)`` float32 areas on the device the inputs are on.

        ``rows`` is the detections tensor and ``planes`` the prototype bank; ``count`` is how
        many rows of this batch to read, because both are sized for the engine's maximum and
        the tail is last batch's data.

        Raises:
            InferenceError: the outputs disagree on the coefficient count. Sliced instead, a
                truncated basis builds a plausible mask from the wrong planes.
        """
        import torch

        used_rows = rows[:count]
        used_planes = planes[:count]
        channels = used_planes.shape[1]
        coefficients = used_rows.shape[2] - _PREFIX
        if coefficients != channels:
            raise InferenceError(
                f"segmentation engine emits {coefficients} mask coefficient(s) per row but "
                f"{channels} prototype plane(s); one of the two outputs is not the one this "
                f"fold was configured for, and combining them would build a mask from a "
                f"truncated basis"
            )

        # THE STRONGEST ROW OF EACH CROP, because the crop *is* one object: its instance is
        # the engine's best answer for that crop and not the union of everything in it. Same
        # choice as the numpy twin, and the parity test is what holds them to it.
        best = torch.argmax(used_rows[:, :, _SCORE], dim=1)
        index = torch.arange(count, device=used_rows.device)
        chosen = used_rows[index, best]
        found = chosen[:, _SCORE] >= self.score_threshold

        flat = used_planes.reshape(count, channels, -1)
        logits = torch.einsum("nm,nmk->nk", chosen[:, _PREFIX:], flat)
        cells = (logits >= self.logit_cut).sum(dim=1)

        crop_h, crop_w = self.crop_hw
        cell_px = (crop_h * crop_w) / float(flat.shape[2])
        areas = torch.where(found, cells.to(torch.float32) * cell_px, 0.0)
        return areas.reshape(-1, 1)
