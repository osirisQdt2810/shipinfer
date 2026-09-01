#!/usr/bin/env python3
"""Compose crowd frames for the T3b measurement: an N x N mosaic of real person photos.

The entry point lives here rather than under ``benchmarks/`` because
``scripts/hooks/require_container.py`` refuses ``python -m benchmarks.*`` wholesale --
correctly for the bench runners, and wrongly for this one, which reads JPEGs with PIL,
writes JPEGs, touches no device and produces no measurement. Documenting a command the
project's own hook denies teaches the reader to reach for ``SHIPINFER_ALLOW_HOST_RUN``,
which is how the container rule was lost the first time; the same reasoning moved P6's
golden emitter to ``scripts/emit_parity_golden.py``.

    python scripts/compose_crowd_frames.py --src benchmarks/baseline/data/person \\
        --out .artifacts/person_crowd --grid 2 --frames 10
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness.crowd import main

if __name__ == "__main__":
    raise SystemExit(main())
