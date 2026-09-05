"""A segmentation engine's two outputs, so both planes' mask fold runs on the same numbers.

The record seam compares `build_records` on rows a scenario STATES, which for `mask_area_px`
means it states an already-reduced `(N, 1)`. That is exactly what made
`CSRC-SEGMENT-FOLD-MISSING` invisible: the C++ plane had no fold at all, published
`output0[crop][0][0]` -- a box coordinate -- and the record gate could not see it. So this
seam compares the FOLD, upstream of the records: a scenario states what a YOLO-seg engine
answered and each plane reduces it to one area per crop.

**Why the numbers here are all far from the cut.** The comparison is a byte compare of the
areas, and an area is a COUNT of cells whose logit clears `log(m / (1 - m))`. numpy sums a
dot product its own way and the C++ loop sums left to right, so a logit sitting near the cut
could round either side and change a count. Every value below is +-1 or +-2 against a cut of
0, which no summation order can move across it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shipinfer.core.errors import ConfigurationError

__all__ = ["MaskScenario", "arrays_of", "load_mask_scenario"]

#: Columns before the mask coefficients in one detection row, and the score's column.
PREFIX, SCORE = 6, 4


@dataclass(frozen=True, slots=True)
class MaskScenario:
    """One segmentation response: the two shapes, the two cuts, and the values."""

    name: str
    crop: tuple[int, int] = (16, 8)
    #: ``(candidates, stride)`` per crop, where ``stride`` is ``PREFIX + coefficients``.
    rows_shape: tuple[int, int] = (2, 8)
    #: ``(coefficients, height, width)`` per crop.
    protos_shape: tuple[int, int, int] = (2, 4, 2)
    score_threshold: float = 0.25
    mask_threshold: float = 0.5
    #: ``{(crop, candidate): (score, coefficients...)}`` -- a row the scenario states.
    detections: dict[tuple[int, int], tuple[float, ...]] = field(default_factory=dict)
    #: ``{(crop, coefficient): plane values}``, row-major.
    planes: dict[tuple[int, int], tuple[float, ...]] = field(default_factory=dict)
    crops: int = 0


def arrays_of(scenario: MaskScenario) -> dict[str, np.ndarray]:
    """The scenario as the two arrays the engine would have answered with.

    Every unstated row scores 0 with zero coefficients, and every unstated plane is zero: a
    scenario says what it is about and the rest is the engine's own silence.
    """
    candidates, stride = scenario.rows_shape
    coefficients, height, width = scenario.protos_shape
    # The two shapes are NOT cross-checked here, deliberately. A scenario states what the
    # engine answered, including a pair that disagrees about the coefficient count -- and the
    # FOLD is what has to refuse that, on both planes. Refusing it in the loader would move
    # the assertion out of the code under test and into the harness.
    rows = np.zeros((scenario.crops, candidates, stride), dtype=np.float32)
    for (crop, candidate), values in scenario.detections.items():
        rows[crop, candidate, SCORE] = values[0]
        rows[crop, candidate, PREFIX:] = values[1:]
    protos = np.zeros((scenario.crops, coefficients, height, width), dtype=np.float32)
    for (crop, coefficient), values in scenario.planes.items():
        protos[crop, coefficient] = np.array(values, dtype=np.float32).reshape(height, width)
    return {"output0": rows, "output1": protos}


def load_mask_scenario(path: Path) -> MaskScenario:
    """Parse one scenario, naming the line of any refusal."""
    values: dict[str, object] = {}
    detections: dict[tuple[int, int], tuple[float, ...]] = {}
    planes: dict[tuple[int, int], tuple[float, ...]] = {}
    crops = 0
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{path}:{number}"
        directive, *words = line.split()
        if directive == "scenario":
            values["name"] = words[0]
        elif directive == "crop":
            values["crop"] = (_int(words[0], where), _int(words[1], where))
        elif directive == "rows":
            values["rows_shape"] = (_int(words[0], where), _int(words[1], where))
        elif directive == "protos":
            values["protos_shape"] = tuple(_int(word, where) for word in words[:3])
        elif directive in ("score_threshold", "mask_threshold"):
            values[directive] = _number(words[0], where)
        elif directive == "det":
            if len(words) < 3:
                raise ConfigurationError(f"{where}: expected `det <crop> <cand> <score> ...`")
            crop, candidate = _int(words[0], where), _int(words[1], where)
            if (crop, candidate) in detections:
                raise ConfigurationError(f"{where}: a second `det {crop} {candidate}`")
            detections[crop, candidate] = tuple(_number(w, where) for w in words[2:])
            crops = max(crops, crop + 1)
        elif directive == "plane":
            if len(words) < 3:
                raise ConfigurationError(f"{where}: expected `plane <crop> <coeff> <v>...`")
            crop, coefficient = _int(words[0], where), _int(words[1], where)
            if (crop, coefficient) in planes:
                raise ConfigurationError(f"{where}: a second `plane {crop} {coefficient}`")
            planes[crop, coefficient] = tuple(_number(w, where) for w in words[2:])
            crops = max(crops, crop + 1)
        elif directive == "crops":
            crops = max(crops, _int(words[0], where))
        else:
            raise ConfigurationError(f"{where}: unknown directive {directive!r}")

    if "name" not in values:
        raise ConfigurationError(f"{path}: no `scenario <name>` line")
    return MaskScenario(
        detections=detections,
        planes=planes,
        crops=crops,
        **values,  # type: ignore[arg-type]
    )


def _int(word: str, where: str) -> int:
    try:
        return int(word)
    except ValueError:
        raise ConfigurationError(f"{where}: {word!r} is not an integer") from None


def _number(word: str, where: str) -> float:
    try:
        return float(word)
    except ValueError:
        raise ConfigurationError(f"{where}: {word!r} is not a number") from None
