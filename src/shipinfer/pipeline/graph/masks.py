"""The mask-area fold, re-exported from where the element that needs it lives.

It moved to `topology/elements/masks.py` with P6-SEGMENT-CROP: `PoolSegment` folds a
segmentation engine's two outputs into one area per crop, and `topology/` may not import
`pipeline/`. This module keeps the name `pipeline/graph/graph.py` and the tests already use,
so there is one definition and the import goes the one way the layering allows.
"""

from __future__ import annotations

from shipinfer.topology.elements.masks import InstanceMaskArea

__all__ = ["InstanceMaskArea"]
