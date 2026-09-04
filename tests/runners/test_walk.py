"""The fan-in merge rule, in isolation.

Two branches split at ``detect`` and rejoin at ``track`` (arch.md §1), and "whichever
finished last wins" is not an answer when the two carry different metadata: one branch has
the ship's ``identities``, the other the person's ``vectors``, and the tracker needs both.
:meth:`~shipinfer.runners.walk.ChainWalk.inbound` is the rule, and it is tested
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

from shipinfer.core.errors import InferenceError
from shipinfer.core.request import RequestContext
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.topology import Caps, ChainItem, ChainSpec, RowIndexed, Topology

#: A branch that splits and rejoins, with the two inbound edges carrying **different** caps:
#: ``detect`` hands ``join`` the frame (``bgr@cpu``) and ``tap`` hands it metadata
#: (``meta@cpu``). ``after: [detect, tap]`` puts the *frame* predecessor first in declaration
#: order on purpose, because ``ShipvisionTrack`` lists ``meta@cpu`` first in its ``accepts``
#: and the loader nominates the donor in that order: declaration says ``detect``, preference
#: says ``tap``, and with the two aligned the test would not know which one it proved.
FAN_IN = """
name: fan_in
elements:
  decode: {impl: replay}
  detect: {impl: pool, model: ship_detector}
  tap:    {impl: shipvision, kind: track, after: detect}
  join:   {impl: shipvision, kind: track, after: [detect, tap]}
  output: {impl: none}
"""

#: The two-embedder rejoin: two
#: embedders forked off ``detect`` (``embed_person`` carries ``after: detect`` precisely so it
#: does *not* follow ``embed_ship``) and rejoined at ``track``. Each covers its own classes
#: and files a partial ``meta["vectors"]``, so this is the wiring on which "the metadata is
#: the union" has to mean the union of two *scatter-backs*. It is the shape C8b gives
#: ``topology/ship_person.yaml`` and the shape ``tests/topology/test_pool_embed_crops.py``'s
#: ``TWO_BRANCHES`` declares with the real ``pool`` elements; the shipped file itself still
#: guards both embedders with ``when: class == ...`` against a field nothing sets, so on it
#: neither of them runs.
TWO_EMBEDDERS = """
name: two_embedders
elements:
  decode:       {impl: replay}
  detect:       {impl: pool, model: ship_detector}
  # Disjoint `classes:`, which the loader now requires of two slots filling one event field
  # (`_check_one_filler_per_row`): two batches covering one detection is refused per frame by
  # both planes' record builders, so a chain that guarantees it is refused at load. The
  # fixture's subject is the MERGE of two scatter-backs, which is unchanged -- these tests
  # hand `inbound` its mappings directly.
  embed_ship:   {impl: pool, kind: embed, model: ship_embedder,   after: detect,
                 params: {classes: [ship]}}
  embed_person: {impl: pool, kind: embed, model: person_embedder, after: detect,
                 params: {classes: [person]}}
  track:        {impl: shipvision, kind: track, after: [embed_ship, embed_person]}
  output:       {impl: none}
"""

#: The same rejoin with a *third* branch, for the refusal's re-scan. Three contributors is
#: where "name the two the merge is holding" and "name the branches that claimed the row"
#: stop being the same answer: branches one and two union cleanly and branch three contests a
#: row of the union, so the merge is holding a value neither of the two culprits alone filed.
THREE_EMBEDDERS = """
name: three_embedders
elements:
  decode:        {impl: replay}
  # A third label on the detector, so the third branch can declare a disjoint selection.
  detect:        {impl: pool, model: ship_detector,
                  params: {decode: {class_labels: {0: person, 3: vehicle, 8: ship}}}}
  embed_ship:    {impl: pool, kind: embed, model: ship_embedder,    after: detect,
                  params: {classes: [ship]}}
  embed_person:  {impl: pool, kind: embed, model: person_embedder,  after: detect,
                  params: {classes: [person]}}
  embed_vehicle: {impl: pool, kind: embed, model: vehicle_embedder, after: detect,
                  params: {classes: [vehicle]}}
  track:         {impl: shipvision, kind: track, after: [embed_ship, embed_person, embed_vehicle]}
  output:        {impl: none}
"""

#: Two ``segment`` slots on the same rejoining branches. ``PoolSegment`` keeps
#: ``_PoolElement._finish``'s default and files the model's raw ``response.outputs`` --
#: ``{output name: Tensor}`` -- under ``meta["masks"]``, which is a mapping and is *not* a
#: scatter-back. This chain is the one that says so.
TWO_SEGMENTERS = """
name: two_segmenters
elements:
  decode:     {impl: replay}
  detect:     {impl: pool, model: ship_detector}
  seg_ship:   {impl: pool, kind: segment, model: ship_segmenter,   after: detect}
  seg_person: {impl: pool, kind: segment, model: person_segmenter, after: detect}
  track:      {impl: shipvision, kind: track, after: [seg_ship, seg_person]}
  output:     {impl: none}
"""


def load(text: str) -> Topology:
    return Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(text)))


def item(caps: str = "bgr@cpu", payload: object = None, **meta: object) -> ChainItem:
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


@pytest.fixture()
def rejoin() -> InprocessRunner:
    """A stopped runner over the two-embedder chain, for the scatter-back merge."""
    return InprocessRunner(load(TWO_EMBEDDERS))


@pytest.fixture()
def three_way() -> InprocessRunner:
    """A stopped runner over the three-embedder chain, for the refusal's re-scan."""
    return InprocessRunner(load(THREE_EMBEDDERS))


@pytest.fixture()
def segmenters() -> InprocessRunner:
    """A stopped runner over the two-segmenter chain, for the mappings that are *not* rows."""
    return InprocessRunner(load(TWO_SEGMENTERS))


class TestTheFanInMerge:
    def test_metadata_from_both_branches_arrives(self, runner: InprocessRunner) -> None:
        """The property the tracker depends on: it sees both branches' results."""
        node = runner.topology.node("join")

        merged = runner._walker.inbound(
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

        merged = runner._walker.inbound(
            node,
            {
                "tap": item(caps="meta@cpu", **{"class": "person"}),
                "detect": item(**{"class": "ship"}),
            },
        )

        assert merged is not None
        assert node.inputs == ("detect", "tap"), "the fixture's declaration order"
        assert merged.meta["class"] == "ship", "the first declared predecessor wins"

    def test_the_payload_comes_from_the_predecessor_with_the_preferred_cap(
        self, runner: InprocessRunner
    ) -> None:
        """Half a frame handle plus half a metadata dict is not a payload.

        ``join``'s donor is ``tap`` (the loader's answer — ``ShipvisionTrack`` accepts
        ``meta@cpu`` before ``bgr@cpu``), and this is the runner honouring it: the payload and
        the caps come from that one predecessor even though ``detect`` is declared first.
        """
        node = runner.topology.node("join")
        frame = item(payload="frame:cam-1:7")
        meta_only = item(caps="meta@cpu", payload="tracks:cam-1:7", tracks=[1])

        merged = runner._walker.inbound(node, {"detect": frame, "tap": meta_only})

        assert merged is not None
        assert merged.payload == "tracks:cam-1:7"
        assert merged.caps == Caps.parse("meta@cpu")

    def test_the_tag_survives_the_merge(self, runner: InprocessRunner) -> None:
        """ADR-002: the ``(camera_id, frame_id)`` tag rides untouched, merge or no merge."""
        node = runner.topology.node("join")
        donated = item(caps="meta@cpu", payload="tracks")

        merged = runner._walker.inbound(node, {"detect": item(payload="frame"), "tap": donated})

        assert merged is not None
        assert merged.context is donated.context, "the tag is carried, never rebuilt"


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

        assert runner._walker.inbound(node, {"detect": only}) is only

    def test_a_predecessor_that_consumed_its_item_contributes_nothing(
        self, runner: InprocessRunner
    ) -> None:
        """``None`` from an element means *consumed*, so its successors receive nothing from it."""
        node = runner.topology.node("join")
        tracks = item(caps="meta@cpu", payload="tracks")

        merged = runner._walker.inbound(node, {"detect": None, "tap": tracks})

        assert merged is tracks

    def test_no_contributor_at_all_means_this_element_does_not_run(
        self, runner: InprocessRunner
    ) -> None:
        """Not a failure: it is what a branch that ended in a sink looks like."""
        node = runner.topology.node("join")

        assert runner._walker.inbound(node, {"tap": None, "detect": None}) is None
        assert runner._walker.inbound(node, {}) is None


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
        # `detect` was skipped, so what stands in for its output is what *it* was handed.
        skipped_inbound = item(boxes=[(0, 0, 1, 1)], **{"class": "person"})

        merged = runner._walker.inbound(
            node,
            {"detect": skipped_inbound, "tap": item(caps="meta@cpu", tracks=[1])},
        )

        assert merged is not None
        assert merged.meta["boxes"] == [(0, 0, 1, 1)]
        assert merged.meta["class"] == "person"
        assert merged.meta["tracks"] == [1]


class TestTwoBranchesFilingTheSameMapping:
    """The two-embedder rejoin, and the half of the frame it used to lose.

    Two embedders in parallel off ``detect``, rejoining at ``track``, each filing a *partial*
    ``meta["vectors"]`` — a :class:`~shipinfer.topology.base.RowIndexed`, ``{detection row:
    vector}`` over its own classes only. A merge that took one branch's mapping wholesale
    would hand the tracker half the frame's appearance vectors: at the sizing in
    ``CLAUDE.md`` that is ~15 000 person crops a second cropped, embedded on a GPU and then
    dropped, every person arriving with ``embedding=None``, and no exception, no counter and
    no log line anywhere — both elements' per-frame counters say they ran, and they did. So a
    key both branches wrote as a *declared scatter-back* is merged, not chosen between.

    Every mapping in this class is a ``RowIndexed`` on purpose. That type is the declaration
    the merge reads; a plain ``dict`` here would test a rule the runner no longer has, and
    :class:`TestTwoBranchesFilingAModelsRawOutputs` is what pins the other side of it.
    """

    def test_the_tracker_receives_both_branches_rows(self, rejoin: InprocessRunner) -> None:
        """The property: the union of the two coverages, keyed by detection row."""
        node = rejoin.topology.node("track")
        ships = item(vectors=RowIndexed({0: "ship-0", 2: "ship-2"}))
        people = item(vectors=RowIndexed({1: "person-1", 3: "person-3"}))

        merged = rejoin._walker.inbound(node, {"embed_ship": ships, "embed_person": people})

        assert merged is not None
        assert merged.meta["vectors"] == {
            0: "ship-0",
            1: "person-1",
            2: "ship-2",
            3: "person-3",
        }

    def test_neither_branch_s_mapping_is_mutated(self, rejoin: InprocessRunner) -> None:
        """The union is a new dict.

        The branches' items are what the walk stored under their names, and an element may
        still be holding one; merging into the first contributor's dict in place would edit
        an object the merge does not own — the kind of aliasing that shows up as one camera's
        vectors appearing on another's frame.
        """
        node = rejoin.topology.node("track")
        ships = item(vectors=RowIndexed({0: "ship-0"}))
        people = item(vectors=RowIndexed({1: "person-1"}))

        merged = rejoin._walker.inbound(node, {"embed_ship": ships, "embed_person": people})

        assert merged is not None
        assert ships.meta["vectors"] == {0: "ship-0"}
        assert people.meta["vectors"] == {1: "person-1"}
        assert merged.meta["vectors"] is not ships.meta["vectors"]
        assert isinstance(
            merged.meta["vectors"], RowIndexed
        ), "the union stays declared, or a third branch would not union into it"

    def test_a_mapping_written_before_the_fork_is_not_a_collision(
        self, rejoin: InprocessRunner
    ) -> None:
        """A diamond, which is the ordinary case and must not be refused.

        A scatter-back filed *before* the fork — one embedder ahead of the split — rides down
        both branches as the same object, because ``derive`` copies the dict and not the
        values, so every entry in it collides with itself. Those are not two branches
        disagreeing.
        """
        node = rejoin.topology.node("track")
        before_the_fork = RowIndexed({0: "row-0", 1: "row-1"})

        merged = rejoin._walker.inbound(
            node,
            {
                "embed_ship": item(shared=before_the_fork, vectors=RowIndexed({0: "ship-0"})),
                "embed_person": item(
                    shared=before_the_fork, vectors=RowIndexed({1: "person-1"})
                ),
            },
        )

        assert merged is not None
        assert merged.meta["shared"] == before_the_fork
        assert merged.meta["vectors"] == {0: "ship-0", 1: "person-1"}

    def test_a_mapping_written_before_the_fork_is_not_copied(
        self, rejoin: InprocessRunner
    ) -> None:
        """The identity skip is a *fast path*, and this is what pins it.

        It is not what makes the diamond work — deleting it leaves the answer identical,
        because a mapping merged into itself meets every entry with the same object and no
        refusal is reachable. What it saves is the O(rows) copy, on a key that is already the
        answer, at every fan-in of every frame. Without this assertion the branch has no test
        and one refactor turns a skipped copy into a made one, invisibly.

        The pre-fork value is a ``RowIndexed`` deliberately: since the union is gated on that
        type, it is the *only* kind of value the fast path can save a copy of, so a plain dict
        here would leave the branch as untested as deleting the assertion.
        """
        node = rejoin.topology.node("track")
        before_the_fork = RowIndexed({0: "row-0", 1: "row-1"})

        merged = rejoin._walker.inbound(
            node,
            {
                "embed_ship": item(shared=before_the_fork, vectors=RowIndexed({0: "ship-0"})),
                "embed_person": item(
                    shared=before_the_fork, vectors=RowIndexed({1: "person-1"})
                ),
            },
        )

        assert merged is not None
        assert merged.meta["shared"] is before_the_fork, "handed through, not rebuilt"

    def test_two_branches_claiming_one_row_is_a_typed_refusal(
        self, rejoin: InprocessRunner
    ) -> None:
        """Refused, and the message names the chain file's two slots.

        Two elements covering the same detection means both cropped it and both paid for it,
        which is what ``classes:`` exists to prevent — and there is no answer to "which of
        these two vectors is this object's". Silently keeping one would attach an appearance
        vector chosen by declaration order.
        """
        node = rejoin.topology.node("track")

        with pytest.raises(InferenceError) as raised:
            rejoin._walker.inbound(
                node,
                {
                    "embed_ship": item(vectors=RowIndexed({1: "ship-1"})),
                    "embed_person": item(vectors=RowIndexed({1: "person-1"})),
                },
            )

        message = str(raised.value)
        assert "'embed_ship'" in message and "'embed_person'" in message
        assert "'vectors'" in message and "[1]" in message
        assert (
            "`params: classes:`" in message
        ), "the fix is a chain-file knob, and only slots that have one can reach this"

    def test_a_three_way_rejoin_names_only_the_branches_that_claimed_the_row(
        self, three_way: InprocessRunner
    ) -> None:
        """The refusal names the two slots at fault, and not the innocent third.

        This is what :class:`~shipinfer.runners.walk.ChainWalk`'s collision re-scan
        exists for, and with two branches it is untestable because the two the merge is
        holding are always the two at fault. With three it is not: ``embed_ship`` covers row 0
        and ``embed_person`` covers row 1, they union cleanly, and then ``embed_vehicle``
        contests row 1. The merge is now holding the *union* of the first two on one side —
        an object neither of them filed — so "name the two being merged" would send the
        operator to ``embed_ship``, whose ``classes:`` has nothing to do with the overlap.

        An operator reading this message goes and edits two ``params: classes:`` lists. Naming
        the wrong one costs a wrong edit and leaves the real overlap in place.
        """
        node = three_way.topology.node("track")

        with pytest.raises(InferenceError) as raised:
            three_way._walker.inbound(
                node,
                {
                    "embed_ship": item(vectors=RowIndexed({0: "ship-0"})),
                    "embed_person": item(vectors=RowIndexed({1: "person-1"})),
                    "embed_vehicle": item(vectors=RowIndexed({1: "vehicle-1"})),
                },
            )

        message = str(raised.value)
        assert "'embed_person'" in message and "'embed_vehicle'" in message
        assert "'embed_ship'" not in message, "it covered row 0 and is not at fault"
        assert "[1]" in message

    def test_a_non_mapping_collision_still_resolves_first_writer_wins(
        self, rejoin: InprocessRunner
    ) -> None:
        """The old rule, kept: two branches disagreeing about a scalar is a chain-file problem
        and not a merge problem, and there is no union to take. It resolves the same way on
        every frame and every run because it resolves by ``node.inputs`` order.

        A ``RowIndexed`` meeting anything else is that same disagreement — about what the key
        *is* — so it resolves the same way rather than raising, which would turn a
        mis-declared chain into a per-frame exception storm instead of a stable, inspectable
        wrong answer.
        """
        node = rejoin.topology.node("track")

        scalars = rejoin._walker.inbound(
            node,
            {
                "embed_ship": item(**{"class": "ship"}),
                "embed_person": item(**{"class": "person"}),
            },
        )
        mixed = rejoin._walker.inbound(
            node,
            {
                "embed_ship": item(vectors=RowIndexed({0: "ship-0"})),
                "embed_person": item(vectors="not a scatter-back"),
            },
        )

        assert scalars is not None and mixed is not None
        assert node.inputs == ("embed_ship", "embed_person"), "the fixture's declaration order"
        assert scalars.meta["class"] == "ship"
        assert mixed.meta["vectors"] == {0: "ship-0"}


class TestTwoBranchesFilingAModelsRawOutputs:
    """A mapping that is not a scatter-back keeps the rule it had before the union existed.

    ``PoolSegment`` and ``PoolRecognize`` keep ``_PoolElement._finish``'s default, which files
    the model's raw ``response.outputs`` — ``{output name: Tensor}`` — under ``meta["masks"]``
    and ``meta["identities"]``. Those are mappings and they are emphatically *not* keyed by
    detection row, so the union must not touch them. Sniffing ``isinstance(..., Mapping)``
    cannot tell the two apart, which is why the writer declares itself with
    :class:`~shipinfer.topology.base.RowIndexed` and only that type unions.

    Both tests below describe the *same* chain — two ``segment`` slots on rejoining branches —
    and differ only in whether the two engines happen to name their output the same. Under a
    ``Mapping`` gate that coincidence decides between an exception on every frame and a
    fabricated composite; under the declared gate it decides nothing, which is the property.
    """

    def test_engines_naming_their_output_the_same_are_not_refused(
        self, segmenters: InprocessRunner
    ) -> None:
        """First-writer-wins, exactly as it resolved before the union rule existed.

        Both segmenters emit an output called ``masks``, so a ``Mapping``-gated union reads
        the *output name* as a detection row, finds it claimed twice, and fails every frame of
        the chain — with a message pointing at a ``params: classes:`` the ``segment`` family
        does not have. There is nothing for an operator to fix and nothing wrong with the
        chain.
        """
        node = segmenters.topology.node("track")

        merged = segmenters._walker.inbound(
            node,
            {
                "seg_ship": item(masks={"masks": "ship-masks"}),
                "seg_person": item(masks={"masks": "person-masks"}),
            },
        )

        assert merged is not None
        assert node.inputs == ("seg_ship", "seg_person"), "the fixture's declaration order"
        assert merged.meta["masks"] == {"masks": "ship-masks"}

    def test_engines_naming_their_outputs_differently_are_not_unioned(
        self, segmenters: InprocessRunner
    ) -> None:
        """No composite dict is invented, either.

        The other half of the same failure: with different output names there is no collision,
        so a ``Mapping``-gated union quietly hands ``track`` a ``{'ship_masks': ...,
        'person_masks': ...}`` that neither engine produced and no consumer was written
        against. First-writer-wins is a stable, inspectable answer; a fabricated one is not.
        """
        node = segmenters.topology.node("track")

        merged = segmenters._walker.inbound(
            node,
            {
                "seg_ship": item(masks={"ship_masks": "T1"}),
                "seg_person": item(masks={"person_masks": "T2"}),
            },
        )

        assert merged is not None
        assert merged.meta["masks"] == {"ship_masks": "T1"}, "not a union of the two"
