"""The mask fold on this plane, so the C++ gate has a golden to be held to.

`InstanceMaskArea` directly and not through `PoolSegment`: the element adds cropping, a model
pool and a scatter, and none of that is what the two planes have to agree about. What they
have to agree about is the arithmetic -- argmax row, score floor, the coefficients against the
prototype bank, the threshold count, cells to crop pixels.
"""

from __future__ import annotations

from pathlib import Path

from shipinfer.topology.elements.masks import InstanceMaskArea

from .mask_scenario import MaskScenario, arrays_of, load_mask_scenario

SCENARIOS = Path(__file__).resolve().parent / "scenarios" / "masks"
GOLDEN = Path(__file__).resolve().parent / "golden" / "masks"

__all__ = ["GOLDEN", "SCENARIOS", "load", "render_masks"]


def render_masks(scenario: MaskScenario) -> str:
    """One area per crop, one per line, in the order the crops went in.

    A plain list and not JSON: the value is one number per crop, and the seam beside this one
    (`golden/number_spellings.tsv`) already proves the two planes spell a float identically.
    """
    fold = InstanceMaskArea(
        crop_hw=scenario.crop,
        score_threshold=scenario.score_threshold,
        mask_threshold=scenario.mask_threshold,
    )
    areas = fold(arrays_of(scenario))[fold.name]
    return "".join(f"{float(area[0])!r}\n" for area in areas)


def load(name: str) -> MaskScenario:
    """A scenario by name under ``scenarios/masks/``, or by path."""
    named = Path(name)
    return load_mask_scenario(named if named.suffix == ".scn" else SCENARIOS / f"{name}.scn")
