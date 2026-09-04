"""Two slots that could fill one event field for one detection, refused at LOAD.

Both planes' `build_records` already refuse this per frame, and so do `PoolEmbed._scatter`
and `ChainWalk.inbound`: there is no answer to which of two vectors is an object's. Refusing
it at load is strictly better than refusing it per frame -- the per-frame refusal is a total
outage for the chain, plus an unrate-limited `_LOG.exception` per frame on the Python plane,
which is ~1000 log lines a second at the design load.

What makes this checkable at all is `Element.selects_rows` and `declared_classes()`: the
loader asks the element rather than carrying a list of implementation names, which is the
reason both are declarations.
"""

from __future__ import annotations

import pytest
import yaml

from shipinfer.core.errors import ChainStructureError
from shipinfer.topology import ChainSpec, Topology

#: The head every chain below shares: a detector whose table names both labels.
HEAD = """
name: overlap
elements:
  decode: {impl: replay}
  detect:
    impl: pool
    model: ship_detector
    params: {decode: {class_labels: {0: person, 8: ship}}}
"""


def load(*elements: str) -> Topology:
    """`HEAD` plus one line per extra element, plus an output wired to all of them.

    The `after:` on the output is not decoration: without it the default rule ("the element
    declared before me") orphans every branch but the last, and `_check_structure` refuses
    the chain before the check under test is reached.
    """
    slots = [line.split(":", 1)[0].strip() for line in elements]
    body = (
        HEAD
        + "\n".join(f"  {line}" for line in elements)
        + f"\n  output: {{impl: jsonlines, after: {slots}}}\n"
    )
    return Topology.from_spec(ChainSpec.model_validate(yaml.safe_load(body)))


class TestTwoSlotsCoveringOneRow:
    def test_two_embedders_with_no_selection_are_refused(self) -> None:
        """Both cover every row, so every detection is contested."""
        with pytest.raises(ChainStructureError, match="both fill the event's 'embedding'"):
            load(
                "a: {impl: pool, kind: embed, model: ship_embedder, after: detect}",
                "b: {impl: pool, kind: embed, model: person_embedder, after: detect}",
            )

    def test_one_declared_and_one_not_is_refused(self) -> None:
        """The asymmetric case, which is the one a resolved plan makes reachable: an
        unselected slot covers every row, including the other's."""
        with pytest.raises(ChainStructureError, match="declares no `classes:`"):
            load(
                "a: {impl: pool, kind: embed, model: ship_embedder, after: detect,"
                " params: {classes: [ship]}}",
                "b: {impl: pool, kind: embed, model: person_embedder, after: detect}",
            )

    def test_overlapping_selections_are_refused_and_the_overlap_is_named(self) -> None:
        with pytest.raises(ChainStructureError, match=r"the class\(es\) \['ship'\]"):
            load(
                "a: {impl: pool, kind: embed, model: ship_embedder, after: detect,"
                " params: {classes: [ship]}}",
                "b: {impl: pool, kind: embed, model: person_embedder, after: detect,"
                " params: {classes: [ship, person]}}",
            )

    def test_disjoint_selections_load(self) -> None:
        """The production shape: one embedder per class, which is what the fix looks like."""
        chain = load(
            "a: {impl: pool, kind: embed, model: ship_embedder, after: detect,"
            " params: {classes: [ship]}}",
            "b: {impl: pool, kind: embed, model: person_embedder, after: detect,"
            " params: {classes: [person]}}",
        )

        assert [node.name for node in chain.nodes][:4] == ["decode", "detect", "a", "b"]

    def test_two_slots_of_DIFFERENT_field_kinds_do_not_collide(self) -> None:
        """A segmenter fills `mask_area_px` and an embedder `embedding`, so covering the same
        row is the ordinary case -- the check is per field, not per row."""
        chain = load(
            "seg: {impl: pool, kind: segment, model: ship_segmenter, after: detect,"
            " params: {classes: [ship]}}",
            "emb: {impl: pool, kind: embed, model: ship_embedder, after: seg,"
            " params: {classes: [ship]}}",
        )

        assert len(chain) == 5

    def test_a_slot_that_selects_NOTHING_cannot_contest_a_row(self) -> None:
        """`classes: []` is "select nothing", explicitly not `None` -- so it cannot collide
        with a slot that selects everything, however degenerate the pair is."""
        chain = load(
            "a: {impl: pool, kind: embed, model: ship_embedder, after: detect,"
            " params: {classes: []}}",
            "b: {impl: pool, kind: embed, model: person_embedder, after: detect}",
        )

        assert chain.node("a").element.declared_classes() == ()
        assert chain.node("b").element.declared_classes() is None

    def test_a_frame_guard_does_not_excuse_an_overlap(self) -> None:
        """Deliberate: `when:` is a runtime fact about a frame and this is a statement about
        the file, so two slots that could never run together are still refused."""
        with pytest.raises(ChainStructureError, match="both fill the event's 'embedding'"):
            load(
                "a: {impl: pool, kind: embed, model: ship_embedder, after: detect,"
                " params: {classes: [ship]}, when: fps == 20}",
                "b: {impl: pool, kind: embed, model: person_embedder, after: detect,"
                " params: {classes: [ship]}, when: fps == 10}",
            )

    def test_an_element_that_does_not_select_rows_is_not_considered(self) -> None:
        """`selects_rows` is the question, not the kind: a frame-level element cannot contest
        a row, and a mock chain of them must still load -- `test_inprocess.py`'s chain is two
        `runner-embed` doubles with no `classes:` at all. The check asks the element rather
        than carrying a list of implementation names, which is why that flag exists.

        `PoolSegment` is the deliberate gap: it parses `classes:` (the resolved plan needed
        it) while its Python half is still whole-frame, so it declares `selects_rows = False`
        and two overlapping segment slots are caught per frame instead. `P6-SEGMENT-CROP` is
        the item that closes it, and this check covers it for free that day.
        """
        chain = load(
            "seg_a: {impl: pool, kind: segment, model: ship_segmenter, after: detect}",
            "seg_b: {impl: pool, kind: segment, model: ship_segmenter, after: detect}",
        )

        assert not chain.node("seg_a").element.selects_rows, "the documented gap"
        assert len(chain) == 5, "so the chain loads, and the per-frame refusal is the guard"
