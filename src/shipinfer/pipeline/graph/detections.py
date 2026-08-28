"""Re-export: the detection value types now live in ``topology.elements.detections``.

They moved because ``track`` consumes them. The chain's ``detect`` element decodes into
``meta["detections"]`` and the tracker reads that key, so the decode has to sit in
``topology`` -- and it can, unchanged, because it imports nothing but numpy and
``core.errors``. What is left here is one line, so the counting-simulation pipeline
(``pipeline/graph/detect.py``, ``state.py``, ``graph.py``, ``deepstream/``) keeps working
against the same objects during the coexistence arch.md section 9 describes.

**One definition, two names.** A shim rather than a copy, because ``isinstance`` and ``is``
comparisons cross this boundary: ``pipeline`` builds a ``Detections`` and a ``topology``
element will be handed it. Two classes with the same fields would satisfy neither.
"""

from __future__ import annotations

from shipinfer.topology.elements.detections import (
    UNKNOWN_LABEL,
    DecodeParams,
    Detection,
    Detections,
    Normalization,
    decode_detections,
)

__all__ = [
    "UNKNOWN_LABEL",
    "DecodeParams",
    "Detection",
    "Detections",
    "Normalization",
    "decode_detections",
]
