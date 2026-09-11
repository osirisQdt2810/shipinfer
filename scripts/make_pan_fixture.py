#!/usr/bin/env python3
"""Write a panning frame sequence from one large photo — the bench's tracking fixture.

The entry point lives here rather than under ``benchmarks/`` for the reason
``compose_crowd_frames.py`` gives: ``scripts/hooks/require_container.py`` refuses
``python -m benchmarks.*`` wholesale, correctly for the runners and wrongly for this, which
reads a JPEG with PIL, writes JPEGs, touches no device and measures nothing.

    python scripts/make_pan_fixture.py --src benchmarks/baseline/data/person_4K \\
        --out .artifacts/person_pan --frames 400
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness.pan import main

if __name__ == "__main__":
    raise SystemExit(main())
