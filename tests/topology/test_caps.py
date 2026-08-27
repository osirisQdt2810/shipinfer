"""Caps: the two-word type system every chain edge is checked against.

The property worth pinning here is the *refusal*, not the parse: a wildcard must never
bridge ``gpu`` and ``cpu``. That single rule is what makes a chain that would silently
download 1000 frames a second to host memory fail at load instead of at 3 a.m.

Scope, because these tests cannot see it and the module they test cannot enforce it:
everything here is about **one pair** of caps. The chain-level guarantee — that a wildcard
element in the middle cannot launder a device frame into a host-only sink — needs cap
propagation across the whole graph, and lives in ``tests/topology/test_chain.py``.
"""

from __future__ import annotations

import pytest

from shipinfer.core.errors import CapsSyntaxError
from shipinfer.topology.caps import Caps, negotiate, parse_caps


class TestParsing:
    @pytest.mark.parametrize("text", ["nv12@gpu", "bgr@cpu", "tensor@gpu", "meta@cpu", "*@*"])
    def test_documented_caps_round_trip(self, text: str) -> None:
        """Every cap arch.md §8 names, plus the sink wildcard, survives parse -> str."""
        assert str(Caps.parse(text)) == text

    def test_case_and_whitespace_are_normalised(self) -> None:
        assert Caps.parse("  NV12 @ GPU ") == Caps("nv12", "gpu")

    @pytest.mark.parametrize(
        "text",
        [
            "nv12@vram",  # location is a closed vocabulary; this is the tempting typo
            "nv12",  # no location: the whole point of the type is that it is stated
            "@gpu",  # no format
            "nv12@",  # empty location
            "nv12@gpu@cpu",  # two locations
            "12nv@gpu",  # a format token starts with a letter
        ],
    )
    def test_a_misspelled_cap_is_refused(self, text: str) -> None:
        with pytest.raises(CapsSyntaxError):
            Caps.parse(text)

    def test_a_cap_built_in_code_is_validated_too(self) -> None:
        """The constructor checks, not only ``parse`` — otherwise the check is skippable."""
        with pytest.raises(CapsSyntaxError):
            Caps("nv12", "vram")

    def test_parse_caps_keeps_declaration_order(self) -> None:
        """Order is preference order, so this never sorts."""
        assert parse_caps(["bgr@cpu", "nv12@gpu"]) == (Caps("bgr", "cpu"), Caps("nv12", "gpu"))


class TestMatching:
    def test_identical_caps_match_and_different_formats_do_not(self) -> None:
        assert Caps.parse("nv12@gpu").matches(Caps.parse("nv12@gpu"))
        assert not Caps.parse("nv12@gpu").matches(Caps.parse("bgr@gpu"))

    @pytest.mark.parametrize("other", ["nv12@gpu", "bgr@cpu", "meta@cpu"])
    def test_a_full_wildcard_takes_anything(self, other: str) -> None:
        assert Caps.parse("*@*").matches(Caps.parse(other))

    def test_each_half_wildcards_independently(self) -> None:
        assert Caps.parse("*@gpu").matches(Caps.parse("nv12@gpu"))
        assert not Caps.parse("*@gpu").matches(Caps.parse("nv12@cpu"))
        assert Caps.parse("nv12@*").matches(Caps.parse("nv12@cpu"))
        assert not Caps.parse("nv12@*").matches(Caps.parse("bgr@cpu"))

    @pytest.mark.parametrize("accepted", ["nv12@cpu", "bgr@cpu"])
    def test_no_pair_bridges_the_two_memories(self, accepted: str) -> None:
        """A device cap never matches a host cap. Half of arch.md §8's rule.

        Only half, and the docstring says so because the other half is easy to assume from
        here. Matching is decided by the **pair**, so nothing in this module stops a chain
        from evading §8 with an intermediate element that wildcards both of its sides: each
        of its two edges would pass this test. That is a chain-level property, pinned in
        ``tests/topology/test_chain.py`` by
        ``test_a_wildcard_passthrough_cannot_launder_a_gpu_frame_to_a_cpu_sink``.
        """
        assert not Caps.parse("nv12@gpu").matches(Caps.parse(accepted))
        assert negotiate([Caps.parse("nv12@gpu")], [Caps.parse(accepted)]) is None


class TestNegotiation:
    def test_the_producers_first_workable_cap_wins(self) -> None:
        """Declaration order is preference order, producer first.

        A detector that lists ``nv12@gpu, bgr@cpu`` is saying "device-resident if you can,
        host memory if you cannot", and nobody has to configure a preference for that.
        """
        produced = parse_caps(["nv12@gpu", "bgr@cpu"])
        accepted = parse_caps(["bgr@cpu", "nv12@gpu"])

        assert negotiate(produced, accepted) == Caps("nv12", "gpu")

    def test_negotiation_resolves_wildcards_to_the_concrete_side(self) -> None:
        assert negotiate(parse_caps(["nv12@gpu"]), parse_caps(["*@*"])) == Caps("nv12", "gpu")
        assert negotiate(parse_caps(["*@gpu"]), parse_caps(["nv12@*"])) == Caps("nv12", "gpu")

    def test_a_cap_neither_side_pinned_stays_a_wildcard(self) -> None:
        """Honest rather than invented: nothing has said what will flow on that edge."""
        assert negotiate(parse_caps(["*@*"]), parse_caps(["*@*"])) == Caps("*", "*")

    def test_an_empty_side_negotiates_nothing(self) -> None:
        """A sink has no output caps, so an element placed after one has no edge to take."""
        assert negotiate((), parse_caps(["nv12@gpu"])) is None
        assert negotiate(parse_caps(["nv12@gpu"]), ()) is None

    def test_resolving_two_caps_that_do_not_match_is_refused(self) -> None:
        """Returning one of them arbitrarily would put a wrong cap on a validated edge."""
        with pytest.raises(CapsSyntaxError):
            Caps.parse("nv12@gpu").resolve(Caps.parse("bgr@cpu"))
