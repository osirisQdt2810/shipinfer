"""The fan-in merge rule, in isolation.

Two branches split at ``detect`` and rejoin at ``track`` (arch.md §1), and "whichever
finished last wins" is not an answer when the two carry different metadata: one branch has
the ship's ``identities``, the other the person's ``vectors``, and the tracker needs both.
:meth:`~shipinfer.runners.inprocess.InprocessRunner._inbound` is the rule, and it is tested
here directly — the walk that calls it is tested end to end in ``test_inprocess.py``, but a
merge is worth pinning where the inputs can be stated by hand.

Everything here runs with the runner **stopped**: the donor is topology data resolved by the
loader and stored on the node (``ElementNode.donor``, pinned in
``tests/topology/test_chain.py::TestTheDonorAtAFanIn``), so the merge is answerable without a
thread or an open element. This file asks only what the runner does with it.
"""

from __future__ import annotations

import textwrap

import pytest

from shipinfer.core.request import RequestContext
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.topology import Caps, ChainItem, ChainSpec, Topology

#: A branch that splits and rejoins, with the two inbound edges carrying **different** caps:
#: ``detect`` hands ``join`` a frame (``nv12@gpu``) and ``tap`` hands it metadata
#: (``meta@cpu``). ``after: [tap, detect]`` puts the metadata predecessor *first* in
#: declaration order on purpose — the donor rule is about the negotiated cap, not about
#: arrival or declaration order, and with the two aligned the test would not know which one
#: it proved.
FAN_IN = """
name: fan_in
elements:
  decode: {impl: mock}
  detect: {impl: mock, model: ship_detector}
  tap:    {impl: mock, kind: track, after: detect}
  join:   {impl: mock, kind: track, after: [tap, detect]}
  output: {impl: mock}
"""


def load(text: str) -> Topology:
    return Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(text)))


def item(caps: str = "nv12@gpu", payload: object = None, **meta: object) -> ChainItem:
    """One item, tagged like every other item in the system."""
    return ChainItem(
        RequestContext(camera_id="cam-1", frame_id=7),
        Caps.parse(caps),
        payload=payload,
        meta=dict(meta),
    )


@pytest.fixture()
def runner() -> InprocessRunner:
    """A stopped runner over the fan-in chain: enough for the merge, no threads."""
    return InprocessRunner(load(FAN_IN))


class TestTheFanInMerge:
    def test_metadata_from_both_branches_arrives(self, runner: InprocessRunner) -> None:
        """The property the tracker depends on: it sees both branches' results."""
        node = runner.topology.node("join")

        merged = runner._inbound(
            node,
            {"tap": item(caps="meta@cpu", tracks=[1]), "detect": item(boxes=[(0, 0, 1, 1)])},
        )

        assert merged is not None
        assert merged.meta == {"tracks": [1], "boxes": [(0, 0, 1, 1)]}

    def test_a_collision_is_resolved_by_declaration_order(
        self, runner: InprocessRunner
    ) -> None:
        """First writer wins, in ``node.inputs`` order — not last, and never at random.

        Two branches that both wrote ``class`` disagree about the frame, which is a chain-file
        problem. Whichever way it resolves, it must resolve the *same* way on every frame and
        every run: a merge whose answer depended on thread timing would make one camera's
        events flip between deploys with nothing in the chain file changed.
        """
        node = runner.topology.node("join")

        merged = runner._inbound(
            node,
            {
                "tap": item(caps="meta@cpu", **{"class": "person"}),
                "detect": item(**{"class": "ship"}),
            },
        )

        assert merged is not None
        assert node.inputs == ("tap", "detect"), "the fixture's declaration order"
        assert merged.meta["class"] == "person", "the first declared predecessor wins"

    def test_the_payload_comes_from_the_predecessor_with_the_preferred_cap(
        self, runner: InprocessRunner
    ) -> None:
        """Half a frame handle plus half a metadata dict is not a payload.

        ``join``'s donor is ``detect`` (the loader's answer — it accepts ``nv12@gpu`` before
        ``meta@cpu``), and this is the runner honouring it: the payload and the caps come
        from that one predecessor even though the metadata branch is declared first.
        """
        node = runner.topology.node("join")
        frame = item(payload="frame:cam-1:7")
        meta_only = item(caps="meta@cpu", payload=None, tracks=[1])

        merged = runner._inbound(node, {"tap": meta_only, "detect": frame})

        assert merged is not None
        assert merged.payload == "frame:cam-1:7"
        assert merged.caps == Caps.parse("nv12@gpu")

    def test_the_tag_survives_the_merge(self, runner: InprocessRunner) -> None:
        """ADR-002: the ``(camera_id, frame_id)`` tag rides untouched, merge or no merge."""
        node = runner.topology.node("join")
        frame = item(payload="frame")

        merged = runner._inbound(node, {"tap": item(caps="meta@cpu"), "detect": frame})

        assert merged is not None
        assert merged.context is frame.context, "the tag is carried, never rebuilt"


class TestWhatTheMergeDoesNotDo:
    def test_a_single_predecessor_is_passed_through_untouched(
        self, runner: InprocessRunner
    ) -> None:
        """A straight line is the common case and must allocate nothing.

        Identity, not equality: at 1000 frames per second through nine elements, a defensive
        copy per hand-over is nine thousand copies a second for no reader's benefit.
        """
        node = runner.topology.node("tap")
        only = item(boxes=[(0, 0, 1, 1)])

        assert runner._inbound(node, {"detect": only}) is only

    def test_a_predecessor_that_consumed_its_item_contributes_nothing(
        self, runner: InprocessRunner
    ) -> None:
        """``None`` from an element means *consumed*, so its successors receive nothing from it."""
        node = runner.topology.node("join")
        frame = item(payload="frame")

        merged = runner._inbound(node, {"tap": None, "detect": frame})

        assert merged is frame

    def test_no_contributor_at_all_means_this_element_does_not_run(
        self, runner: InprocessRunner
    ) -> None:
        """Not a failure: it is what a branch that ended in a sink looks like."""
        node = runner.topology.node("join")

        assert runner._inbound(node, {"tap": None, "detect": None}) is None
        assert runner._inbound(node, {}) is None


class TestASkippedPredecessor:
    """``when:`` is skip-and-continue, so a skipped element hands *its* inbound item on."""

    def test_it_passes_its_own_inbound_item_to_the_successors(
        self, runner: InprocessRunner
    ) -> None:
        """The walk stores the item a skipped element was given under that element's name.

        This is the contract ``ElementNode.admits`` documents: the item is not dropped and the
        walk does not stop, so the merge sees the *predecessor's* input rather than a gap. A
        gap would silently strip the branch's earlier metadata — a person's boxes would never
        reach the tracker because the segmenter did not run on them.
        """
        node = runner.topology.node("join")
        # `tap` was skipped, so what stands in for its output is what *it* was handed.
        skipped_inbound = item(boxes=[(0, 0, 1, 1)], **{"class": "person"})

        merged = runner._inbound(
            node, {"tap": skipped_inbound, "detect": item(payload="frame", tracks=[1])}
        )

        assert merged is not None
        assert merged.meta["boxes"] == [(0, 0, 1, 1)]
        assert merged.meta["class"] == "person"
        assert merged.meta["tracks"] == [1]
