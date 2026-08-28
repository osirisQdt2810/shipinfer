"""The one conversion an event's fields need: a model row as JSON-ready floats.

It lives beside the event rather than beside either producer because **both** producers need
it and they sit on opposite sides of the accelerator seam: ``pipeline/graph/state.py``
builds records from a frame's stage outputs, ``topology/elements/output.py`` builds them
from a chain item's metadata, and the DeepStream probe reads the same values out of NvDs
metadata. Three call sites, one helper — the alternative is three spellings that drift, and
the spelling matters here more than it looks (see :func:`as_embedding`).

``numpy`` is the only import, and it is legal in ``core`` (ADR-001 draws the line at torch,
not at arrays). Nothing here touches a device.
"""

from __future__ import annotations

import numpy as np

__all__ = ["as_embedding"]


def as_embedding(row: np.ndarray) -> tuple[float, ...]:
    """One detection row's vector as the schema's plain-float list, or ``None``.

    The single tolist seam (B1): every publisher of ``*_feature_vec`` converts here, so a
    dtype or NaN rule changes in one place. ``None`` means "this row was never embedded" —
    distinct from an empty vector, which would read as an embedder that returned nothing.
    """
    return tuple(np.asarray(row).reshape(-1).tolist())
