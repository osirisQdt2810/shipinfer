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
    """A model row as a tuple of floats — a **copy**, so the source batch can be freed.

    ``tolist()``, not ``float(v) for v in ...``. The generator was a per-element Python loop
    on the emission path: ``person_embedder`` emits 2048 floats and the documented load is
    ~15 000 objects/s, so ~30 M ``float()`` calls a second, paid even with the ``null`` sink.
    ``tolist()`` does the identical conversion — it yields Python floats, not ``np.float32``
    — in one C call, measured 6x faster on both 512-d and 2048-d rows. ADR-003 and the
    ponytail principle: numpy already does this well.

    And no ``.astype(float)`` first: ``tolist()`` on the float32 view already yields Python
    floats and already copies — the astype was a second, redundant full float64
    materialisation, ~240 MB/s of pure allocation at the design load (#32 round 7).

    Public, and in ``core``, because it has three callers on both sides of the seam: the
    graph's emission path, the DeepStream probe (the two-planes rule — the same tensors read
    out of NvDs metadata must convert the same way) and the ``output`` element. ONE helper,
    not copies that drift; the generator has already been written twice.
    """
    return tuple(np.asarray(row).reshape(-1).tolist())
