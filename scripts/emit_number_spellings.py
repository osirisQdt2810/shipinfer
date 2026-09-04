#!/usr/bin/env python3
"""Emit the double spellings both planes must agree on, byte for byte.

`json_number` in `csrc/shipinfer/core/events/json.h` reproduces Python's `repr` rule --
scientific only when the decimal exponent is < -4 or >= 16 -- because a perception event is a
wire format and the parity gate is a byte compare. That rule used to be pinned by a dozen
values hand-written into the C++ gate, each checked against CPython once, by hand.

This is that check, committed: CPython emits the table, the C++ gate reads the same file, and
`tests/test_number_spellings.py` holds this emitter to it. So a change to either writer names
the plane that moved, the way the chain-plan seam does.

    python scripts/emit_number_spellings.py --out benchmarks/parity/golden/number_spellings.tsv
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "benchmarks" / "parity" / "golden" / "number_spellings.tsv"

#: The boundaries of the rule itself, both signs, plus the ends of the double range where the
#: FIXED form does not fit in a 64-byte buffer at all -- which is how the C++ writer decides.
BOUNDARIES = (
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    0.25,
    2.675,
    0.1,
    0.3,
    1e-4,
    -1e-4,
    9.999e-5,
    1e-5,
    -1e-5,
    0.00012345,
    1e15,
    -1e15,
    9999999999999998.0,
    999999999999999.9,
    1e16,
    -1e16,
    1.5e16,
    1e17,
    5e-324,
    -5e-324,
    1.7976931348623157e308,
    -1.7976931348623157e308,
    1e300,
    1e-300,
    123456789.123456789,
    123456789012345.0,
)

#: The ranges an event's numbers actually come from: a score, a pixel, an embedding
#: component, a mask area, a small threshold. Seeded, so the file is reproducible.
RANGES = ((0.0, 1.0), (0.0, 4000.0), (-1.0, 1.0), (0.0, 1e6), (1e-8, 1e-3))


def spellings() -> dict[str, str]:
    """``{literal: repr}``, keyed by a round-tripping literal so C++ can read it back."""
    values = list(BOUNDARIES)
    rng = random.Random(20260904)
    for _ in range(60):
        values += [rng.uniform(low, high) for low, high in RANGES]
    # Every tenth decade, with a one-ulp nudge either side: the interesting cases are at the
    # rule's boundaries, and 600-odd values keep this file reviewable.
    for exponent in range(-300, 301, 10):
        power = 10.0**exponent
        values += [power, -power, math.nextafter(power, 0.0), math.nextafter(power, math.inf)]
    return {f"{value:.17g}": repr(value) for value in values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=GOLDEN)
    args = parser.parse_args(argv)
    table = spellings()
    header = [
        "# <literal>\\t<repr>: every double spelling both planes must agree on.",
        "# Emitted by `scripts/emit_number_spellings.py`; read by BOTH planes' gates.",
        f"# {len(table)} values: the boundaries of Python's repr rule, a seeded sample from",
        "# the ranges an event carries, and every tenth decade with a one-ulp nudge either side.",
    ]
    body = [f"{literal}\t{spelling}" for literal, spelling in sorted(table.items())]
    args.out.write_text("\n".join(header + body) + "\n", encoding="ascii")
    print(f"wrote {args.out} ({len(body)} spellings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
