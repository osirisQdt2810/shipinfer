"""The double spellings both planes must agree on, held to CPython's own `repr`.

`json_number` in `csrc/shipinfer/core/events/json.h` reproduces Python's rule -- scientific
only when the decimal exponent is < -4 or >= 16 -- because a perception event is a wire format
and the parity gate is a byte compare. That rule was pinned by a dozen values hand-written
into the C++ gate, each checked against CPython once, by hand, in a review.

The table is a shared artefact now: `scripts/emit_number_spellings.py` writes it, the C++ gate
reads the same file, and this holds the emitter to the committed bytes. So a change to either
writer names the plane that moved -- the same shape as the chain-plan seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.emit_number_spellings import GOLDEN, spellings

ROOT = Path(__file__).resolve().parents[1]


def _committed() -> dict[str, str]:
    rows = {}
    for line in GOLDEN.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        literal, spelling = line.split("\t")
        rows[literal] = spelling
    return rows


class TestTheCommittedTableIsWhatCPythonSays:
    def test_the_emitter_reproduces_it_exactly(self) -> None:
        assert spellings() == _committed(), (
            "the committed table is not what this interpreter emits. If the change IS the "
            "decision, re-emit with `python scripts/emit_number_spellings.py` and say so"
        )

    def test_every_spelling_is_this_interpreters_repr(self) -> None:
        """Not just self-consistent: each row is `repr(float(literal))`, on this CPython."""
        wrong = {
            literal: (spelling, repr(float(literal)))
            for literal, spelling in _committed().items()
            if spelling != repr(float(literal))
        }

        assert not wrong, f"rows that are not `repr` of their own literal: {wrong}"

    def test_the_literal_round_trips(self) -> None:
        """`%.17g` is what makes the file readable by `strtod` on the other side."""
        for literal, spelling in _committed().items():
            assert float(literal) == float(spelling), literal


class TestTheTableCoversTheRuleAndNotOnlyEasyValues:
    """A table of 0.0 and 1.0 would pass every writer ever written."""

    def test_it_carries_both_sides_of_both_boundaries(self) -> None:
        committed = _committed()
        by_value = {
            float(literal): spelling
            for literal in committed
            for spelling in [committed[literal]]
        }

        assert by_value[1e15] == "1000000000000000.0", "just inside the fixed range"
        assert by_value[1e16] == "1e+16", "just outside it"
        assert by_value[1e-4] == "0.0001", "just inside the small end"
        assert by_value[1e-5] == "1e-05", "just outside it"

    def test_it_carries_the_ends_of_the_double_range(self) -> None:
        """Where the FIXED form does not fit a 64-byte buffer -- which is how the C++ writer
        decides, so a table that stopped at 1e300 would leave that branch unproven."""
        committed = {float(literal) for literal in _committed()}

        assert 5e-324 in committed and 1.7976931348623157e308 in committed

    def test_it_is_several_hundred_rows_and_not_a_dozen(self) -> None:
        assert len(_committed()) >= 500

    @pytest.mark.parametrize("value", [0.0, -0.0])
    def test_both_zeroes_are_present_and_distinguished(self, value: float) -> None:
        assert _committed()[f"{value:.17g}"] == repr(value)
