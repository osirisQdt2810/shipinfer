"""The in-process runner, end to end over the production chain's wiring with mock elements.

This is the file that says the runner works: nine elements, two branches that split at
``detect`` and rejoin at ``track``, six frames from three cameras, and the assertion that
every one of them reaches the sink carrying the tag it arrived with. It runs offline, with no
GPU and no driver, because every element in it is a mock and
:class:`~shipinfer.topology.base.Element` constructors are required to be hardware-free — the
same property that makes ``tests/topology/test_chain.py`` runnable anywhere.

**On the duplicated fixture.** ``MOCK_CHAIN`` below is ``tests/topology/test_chain.py``'s
chain, adapted: the ``tests/`` directories are not packages (pytest's default import mode
gives two same-named modules in non-package directories one module name), so there is nothing
to import it from without inventing a shared test package for one string. It is adapted, not
copied verbatim — ``detect`` and ``recognize`` carry substitution markers so a test can swap
in an element that blocks, fails or refuses to open, which is what the edge cases here need.
``tests/topology/test_chain.py::TestTheProductionChainFile`` is what keeps *both* honest
against ``topology/ship_person.yaml``.

**Why the assertions read metadata rather than the mocks' counters.** ``_Mock.processes`` is a
plain int and its docstring says, in as many words, not to reach for it from a multi-threaded
runner test: with four workers a lost update would make this suite flaky. What is emitted is
the property anyway — "the segmenter did not run on a person" is a fact about the person's
event, not about a counter.
"""

from __future__ import annotations

import textwrap
import threading
import time
from collections.abc import Iterator
from concurrent.futures import wait
from typing import Any, ClassVar

import pytest

from shipinfer.core.errors import (
    ConfigurationError,
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    ServerStateError,
)
from shipinfer.core.request import InferenceRequest, RequestContext, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.scheduling.queues import FairPriorityQueue
from shipinfer.scheduling.work import WorkItem
from shipinfer.topology import (
    Caps,
    ChainItem,
    ChainSpec,
    ElementContext,
    ElementKind,
    Topology,
)
from shipinfer.topology.elements.mock import MockDetect, MockOutput
from shipinfer.topology.registry import registry_for

#: ``topology/ship_person.yaml``'s wiring with every implementation replaced by its mock: the
#: same nine slots, the same ``when:`` conditions repeated on every element of the ship
#: branch, and the same two ``after:`` lines that spell out the fork and the rejoin. A
#: **branching** chain, which is the only kind worth running a fan-in merge against.
MOCK_CHAIN = """
name: mock_ship_person
elements:
  decode:       {impl: mock}
  detect:       {impl: __DETECT__, model: ship_detector, params: __PARAMS__}
  segment:      {impl: mock, model: ship_segmenter, when: class == ship}
  embed_ship:   {impl: mock, model: ship_embedder, when: class == ship, after: segment}
  embed_person: {impl: mock, model: person_embedder, when: class == person, after: detect}
  recognize:    {impl: __RECOGNIZE__, model: ship_recognizer, when: class == ship, after: embed_ship}
  track:        {impl: mock, per: camera, after: [recognize, embed_person]}
  mtmc:         {impl: mock, scope: global}
  output:       {impl: mock}
"""

#: A **straight line** with one conditional element in the middle, which is the shape the
#: branch conditions are actually deployed in and the one where skip-and-continue is the only
#: thing keeping the sink fed: ``segment`` runs on ships, and a person still has to reach the
#: output through the gap it leaves. ``MOCK_CHAIN`` cannot show this — every one of its
#: conditional elements has an unconditional sibling on the other branch, so the item reaches
#: the tracker either way and a runner that dropped a skipped item would still look correct.
LINEAR_CHAIN = """
name: linear
elements:
  decode:  {impl: mock}
  detect:  {impl: mock, model: ship_detector, params: {class: __CLASS__}}
  segment: {impl: mock, model: ship_segmenter, when: class == ship}
  output:  {impl: mock}
"""

#: A fan-in whose two inbound edges carry **different** negotiated caps: ``detect`` hands
#: ``track`` the frame (``nv12@gpu``) and ``track_a`` hands it metadata (``meta@cpu``). The
#: loader nominates ``detect`` as ``track``'s donor, because ``track`` accepts ``nv12@gpu``
#: first — so when ``detect`` consumes an item there is a contributor left, but not one that
#: donates under the cap the loader negotiated for the donor's edge. ``MOCK_CHAIN`` cannot
#: show this: both of *its* fan-in edges are ``nv12@gpu``, so any contributor would do.
FAN_IN_CHAIN = """
name: donor_gap
elements:
  decode:  {impl: mock}
  detect:  {impl: runner-drops, model: ship_detector, params: {drop_camera: __DROP__}}
  track_a: {impl: mock, after: decode}
  track:   {impl: mock, after: [detect, track_a]}
  output:  {impl: mock}
"""

#: The slots in the order the loader resolves them, which is the order elements are opened in
#: and the reverse of the order they are closed in. ``embed_person`` before ``embed_ship``
#: because Kahn's algorithm keeps declaration order *among the elements that are ready*, and
#: ``embed_person`` becomes ready at ``detect`` while ``embed_ship`` waits for ``segment`` —
#: the resolved order, which is what ``Topology.describe()`` prints, not the file's order.
ORDER = (
    "decode",
    "detect",
    "segment",
    "embed_person",
    "embed_ship",
    "recognize",
    "track",
    "mtmc",
    "output",
)

TAGS = (("cam-1", 1), ("cam-1", 2), ("cam-2", 1), ("cam-2", 2), ("cam-3", 1), ("cam-3", 2))


# -- test-only elements -------------------------------------------------------------------
#
# Registered at module scope under names no chain file uses, following
# `tests/topology/test_chain.py`, which registers its own probes into the real registries. A
# runner has three failure modes worth driving from a real chain -- an element that blocks, one
# that raises mid-frame, one that refuses to open -- and none of them can be provoked from
# outside the chain.


@registry_for(ElementKind.DETECT).register("runner-gate")
class GateDetect(MockDetect):
    """A detector that parks the worker until a test releases it.

    Exists to make backpressure *deterministic*: with the only worker held here, the queue's
    depth is whatever the test put in it, so "the lane is full" is an assertion rather than a
    race.
    """

    def __init__(self, name: str, params: Any = None, *, model: str | None = None) -> None:
        super().__init__(name, params, model=model)
        self.entered = threading.Event()
        self.release = threading.Event()

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        self.entered.set()
        self.release.wait(10.0)
        return super()._do_process(item)


@registry_for(ElementKind.DETECT).register("runner-drops")
class DroppingDetect(MockDetect):
    """A detector that *consumes* the item for one class instead of handing it on.

    An element returning ``None`` is ordinary — a filter, a sink — and it is what makes a
    fan-in lose its nominated donor mid-run. Params-driven so the same chain can be loaded
    with the donor producing and with it dropping, which is the only way to show the merge
    rule refuses one and not the other.
    """

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        if item.context.camera_id == str(self.params.get("drop_camera", "")):
            return None
        return super()._do_process(item)


@registry_for(ElementKind.DETECT).register("runner-boom")
class BoomDetect(MockDetect):
    """A detector that fails on one nominated frame and works on every other."""

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        if item.context.frame_id == int(self.params.get("boom_frame", -1)):
            raise RuntimeError("the detector fell over")
        return super()._do_process(item)


@registry_for(ElementKind.RECOGNIZE).register("runner-unopenable")
class UnopenableRecognize(MockDetect):
    """An element that cannot acquire what it needs — a missing model, a dead camera."""

    kind: ClassVar[ElementKind] = ElementKind.RECOGNIZE

    def _do_open(self, context: ElementContext) -> None:
        raise ServerStateError("this element cannot open")


class PausedQueue(FairPriorityQueue):
    """The configured fair queue, handing out nothing until the test says ``ready``.

    Needed for one property and worth the six lines: the walk re-checks expiry, and only an
    item that was *fresh when its batch was drained* and stale by the time the worker reached
    it can reach that check. That needs two items in **one** batch — and with
    ``max_delay_us`` at zero the queue hands over whatever is present the instant a worker
    asks, so "both items were in the same batch" would otherwise be a race against the
    worker's poll loop rather than a fact.
    """

    def __init__(self, name: str, capacity: int, **options: Any) -> None:
        super().__init__(name, capacity, **options)
        self.ready = threading.Event()

    def get_batch(self, window: Any, *, poll_s: float = 0.05) -> Any:
        self.ready.wait(10.0)
        return super().get_batch(window, poll_s=poll_s)


# -- helpers ------------------------------------------------------------------------------


def load(
    *, detect: str = "mock", recognize: str = "mock", params: str = "{class: ship}"
) -> Topology:
    """The mock chain, with the two swappable slots filled in."""
    text = (
        textwrap.dedent(MOCK_CHAIN)
        .replace("__DETECT__", detect)
        .replace("__RECOGNIZE__", recognize)
        .replace("__PARAMS__", params)
    )
    return Topology.from_spec(ChainSpec.from_yaml(text))


def load_linear(*, klass: str) -> Topology:
    """The straight-line chain, with the class its detector stamps on every item."""
    text = textwrap.dedent(LINEAR_CHAIN).replace("__CLASS__", klass)
    return Topology.from_spec(ChainSpec.from_yaml(text))


def load_fan_in(*, drop_camera: str) -> Topology:
    """The two-cap fan-in, with the camera whose items the donor consumes."""
    text = textwrap.dedent(FAN_IN_CHAIN).replace("__DROP__", drop_camera)
    return Topology.from_spec(ChainSpec.from_yaml(text))


def settings(**pipeline: Any) -> ServerSettings:
    """Deployment settings with the pipeline knobs a runner reads."""
    return ServerSettings(pipeline=pipeline)


def item(camera: str, frame: int, *, captured_ns: int = 0) -> ChainItem:
    """One item as a producer would submit it: a tag and nothing else yet."""
    return ChainItem(
        RequestContext(camera_id=camera, frame_id=frame, captured_ns=captured_ns),
        Caps.parse("nv12@gpu"),
    )


def sink(chain: Topology) -> MockOutput:
    element = chain.node("output").element
    assert isinstance(element, MockOutput)
    return element


def emitted_tags(chain: Topology) -> set[tuple[str, int]]:
    return {emitted.key for emitted in sink(chain).emitted}


def submit_all(runner: InprocessRunner, chain: Topology) -> list[ResponseFuture]:
    """Submit the six items and wait for every walk to finish.

    Waiting on the futures rather than sleeping: the future completing *is* the runner saying
    the walk is over, and a sleep would either be flaky or slow.
    """
    futures = [runner.submit(item(camera, frame)) for camera, frame in TAGS]
    _, not_done = wait(futures, timeout=15.0)
    assert not not_done, f"{len(not_done)} item(s) never finished walking"
    return futures


@pytest.fixture()
def running() -> Iterator[Any]:
    """``start(chain, **options)`` -> a started runner, stopped however the test ends."""
    runners: list[InprocessRunner] = []

    def _start(
        chain: Topology, config: ServerSettings | None = None, **options: Any
    ) -> InprocessRunner:
        runner = InprocessRunner(chain, config, **options)
        runners.append(runner)
        return runner.start()  # type: ignore[return-value]

    yield _start
    for runner in runners:
        runner.stop(timeout_s=5.0)


# -- the end-to-end walk ------------------------------------------------------------------


class TestTheChainRuns:
    @pytest.mark.parametrize("workers", [1, 4])
    def test_every_item_reaches_the_sink_carrying_its_own_tag(self, running, workers) -> None:
        """Six frames, three cameras, nine elements — and the tag rides untouched (ADR-002).

        The identity check is the strong form: the emitted item's context must be the *same
        object* the producer submitted, not an equal one. Every element derives its successor
        from its input, so an implementation that rebuilt the tag — the mistake that makes a
        reassembled frame mix two cameras — fails here rather than in production.
        """
        chain = load()
        runner = running(chain, settings(workers=workers))

        futures = submit_all(runner, chain)

        assert [future.exception() for future in futures] == [None] * 6
        assert emitted_tags(chain) == set(TAGS)
        submitted = {id(future.context) for future in futures}
        assert {id(e.context) for e in sink(chain).emitted} == submitted

    def test_the_walk_gathers_every_element_s_contribution(self, running) -> None:
        """What arrives at the sink is the whole chain's work, not the last element's."""
        chain = load()
        runner = running(chain, settings(workers=2))

        submit_all(runner, chain)

        for emitted in sink(chain).emitted:
            assert set(emitted.meta) >= {
                "frame_id",
                "boxes",
                "class",
                "masks",
                "vectors",
                "identities",
                "tracks",
                "global_ids",
            }, emitted.meta

    def test_the_fan_in_carries_both_branches_into_the_tracker(self, running) -> None:
        """``track`` rejoins the ship and person branches (arch.md §1).

        The ship branch contributes ``identities`` through ``recognize`` and ``vectors``
        through ``embed_ship``; the skipped ``embed_person`` contributes its own inbound item,
        which is what carries ``boxes`` around the branch it did not run on. All of it has to
        be in one item by the time the tracker sees it, or the tracker is tracking half a
        frame.
        """
        chain = load()
        runner = running(chain, settings(workers=1))

        submit_all(runner, chain)

        emitted = sink(chain).emitted[0]
        assert emitted.meta["identities"] == ["ship-1"]
        assert emitted.meta["vectors"] == [[0.0, 1.0]]
        assert emitted.meta["boxes"] == [(0, 0, 10, 10)]

    @pytest.mark.parametrize(
        ("klass", "present", "absent"),
        [("ship", "identities", None), ("person", None, "identities")],
    )
    def test_a_conditional_element_runs_only_for_its_own_class(
        self, running, klass, present, absent
    ) -> None:
        """``when:`` guards the element it is written on, and the item continues regardless.

        With ``class == person`` the whole ship branch — segment, embed_ship, recognize — is
        skipped, and the person still has to reach the tracker and the sink. That is
        skip-and-continue; a runner that dropped the item instead would lose every person in
        the deployment.
        """
        chain = load(params=f"{{class: {klass}}}")
        runner = running(chain, settings(workers=2))

        submit_all(runner, chain)

        assert emitted_tags(chain) == set(TAGS), "every item is emitted whatever its class"
        for emitted in sink(chain).emitted:
            assert emitted.meta["class"] == klass
            assert ("masks" in emitted.meta) == (klass == "ship")
            if present is not None:
                assert present in emitted.meta
            if absent is not None:
                assert absent not in emitted.meta

    def test_a_skipped_element_hands_its_own_item_to_the_next_one(self, running) -> None:
        """Skip-and-continue on a straight line, where a gap would lose the frame outright.

        ``decode -> detect -> segment{when: class == ship} -> output`` with a **person**: the
        segmenter must not run, and the person must still be emitted. This is the assertion
        that fails if the walk stops storing a skipped element's inbound item under that
        element's name — the successor then has no contributor at all, does not run, and every
        person in the deployment disappears between the detector and the sink while the runner
        reports a clean walk.
        """
        chain = load_linear(klass="person")
        segment = chain.node("segment").element
        runner = running(chain, settings(workers=1))

        future = runner.submit(item("cam-1", 1))

        assert future.exception(timeout=10.0) is None
        assert emitted_tags(chain) == {("cam-1", 1)}
        assert segment.processes == 0, "the segmenter does not run on a person"
        emitted = sink(chain).emitted[0]
        assert emitted.meta["class"] == "person"
        assert emitted.meta["boxes"], "the detector's work survives the gap"
        assert "masks" not in emitted.meta

    def test_the_same_chain_runs_the_conditional_element_for_its_own_class(
        self, running
    ) -> None:
        """The other half, so the test above cannot pass by the segmenter never running."""
        chain = load_linear(klass="ship")
        segment = chain.node("segment").element
        runner = running(chain, settings(workers=1))

        assert runner.submit(item("cam-1", 1)).exception(timeout=10.0) is None
        assert segment.processes == 1
        assert sink(chain).emitted[0].meta["masks"]


# -- lifecycle ----------------------------------------------------------------------------


class TestTheLifecycle:
    def test_submitting_before_start_is_a_typed_refusal(self) -> None:
        """Not an implicit start: opening a chain on a producer's thread is how a CUDA
        context ends up on the wrong thread (ADR-002)."""
        runner = InprocessRunner(load())

        with pytest.raises(ServerStateError, match="before start"):
            runner.submit(item("cam-1", 1))

    def test_submitting_after_stop_is_the_same_refusal(self) -> None:
        runner = InprocessRunner(load(), settings(workers=1)).start()
        runner.stop(timeout_s=5.0)

        with pytest.raises(ServerStateError):
            runner.submit(item("cam-1", 1))

    def test_starting_twice_opens_the_chain_once(self) -> None:
        chain = load()
        runner = InprocessRunner(chain, settings(workers=1))
        try:
            runner.start()
            runner.start()
        finally:
            runner.stop(timeout_s=5.0)

        assert [chain.node(slot).element.opens for slot in ORDER] == [1] * len(ORDER)
        assert runner.workers == 1

    def test_stopping_twice_closes_the_chain_once(self) -> None:
        """A shutdown path and a failed-start path both call stop; neither may double-close."""
        chain = load()
        runner = InprocessRunner(chain, settings(workers=2)).start()

        runner.stop(timeout_s=5.0)
        runner.stop(timeout_s=5.0)

        assert not runner.is_running
        assert [chain.node(slot).element.closes for slot in ORDER] == [1] * len(ORDER)
        assert all(not chain.node(slot).element.is_open for slot in ORDER)

    def test_stopping_before_starting_is_a_no_op(self) -> None:
        chain = load()

        InprocessRunner(chain).stop()

        assert [chain.node(slot).element.closes for slot in ORDER] == [0] * len(ORDER)

    def test_a_runner_can_be_restarted(self) -> None:
        """A supervisor restarts a shard; the second cycle must serve traffic, not silence.

        The queue is closed by ``stop`` — that is how queued items are failed with a typed
        error rather than dropped — so a restart mints a fresh one. Without that, every
        submission after a restart would be refused with ``RequestCancelledError`` and the
        shard would look alive and serve nothing.
        """
        chain = load()
        runner = InprocessRunner(chain, settings(workers=1))
        for _ in range(2):
            runner.start()
            futures = submit_all(runner, chain)
            assert [future.exception() for future in futures] == [None] * 6
            runner.stop(timeout_s=5.0)

        assert [chain.node(slot).element.opens for slot in ORDER] == [2] * len(ORDER)
        assert len(sink(chain).emitted) == 12

    def test_an_element_that_cannot_open_unwinds_the_ones_before_it(self) -> None:
        """A partial start leaves nothing acquired — the ADR-002 rule about shared boxes.

        ``recognize`` is the sixth element in topological order, so the five before it were
        opened and must be closed again before the failure is re-raised. An element that
        stayed open would hold a decoder thread, a socket or a CUDA context on a box somebody
        else is using, and the operator's next act after reading the error is to fix the
        config and start again.
        """
        chain = load(recognize="runner-unopenable")
        runner = InprocessRunner(chain, settings(workers=1))

        with pytest.raises(ServerStateError, match="cannot open"):
            runner.start()

        assert not runner.is_running
        before = ORDER[: ORDER.index("recognize")]
        after = ORDER[ORDER.index("recognize") + 1 :]
        assert [chain.node(slot).element.opens for slot in before] == [1] * len(before)
        assert [chain.node(slot).element.closes for slot in before] == [1] * len(before)
        assert [chain.node(slot).element.opens for slot in after] == [0] * len(after)
        assert not any(chain.node(slot).element.is_open for slot in ORDER)

    def test_a_failed_start_can_be_followed_by_a_successful_one(self) -> None:
        """The unwind must not poison the queue: the operator fixes the fault and restarts."""
        chain = load(recognize="runner-unopenable")
        runner = InprocessRunner(chain, settings(workers=1))
        with pytest.raises(ServerStateError):
            runner.start()

        good = load()
        second = InprocessRunner(good, settings(workers=1))
        try:
            second.start()
            assert [future.exception() for future in submit_all(second, good)] == [None] * 6
        finally:
            second.stop(timeout_s=5.0)

    def test_the_context_manager_starts_and_stops(self) -> None:
        chain = load()

        with InprocessRunner(chain, settings(workers=1)) as runner:
            assert runner.is_running
            runner.submit(item("cam-1", 1)).result(timeout=10.0)

        assert not runner.is_running
        assert all(not chain.node(slot).element.is_open for slot in ORDER)

    def test_an_element_is_told_the_placement_the_runner_was_given(self) -> None:
        """An element is *told* which shard and device it is on; it never chooses (§1)."""
        chain = load()
        runner = InprocessRunner(chain, settings(workers=1), shard_id=3, models=object())
        try:
            runner.start()
            context = chain.node("detect").element.context
            assert context is not None
            assert context.shard_id == 3
            assert context.device is None, "no accelerator on the offline tier"
            assert context.models is runner.models
        finally:
            runner.stop(timeout_s=5.0)

    def test_the_chain_is_closed_in_the_reverse_of_the_order_it_opened(self) -> None:
        """A sink that fails to flush must not leave the decoder upstream of it open.

        Reverse order is the whole reason ``_do_stop`` iterates ``reversed(nodes)``, and
        without this test the loop could iterate forwards and every other assertion in the
        file would still pass — the counters only say *that* each element closed. Recording
        the order on the instances rather than in the mock class keeps it a property of this
        test: a shared list in ``topology/elements/mock.py`` would be module state that two
        tests running in one process could interleave.
        """
        chain = load()
        closed: list[str] = []
        for node in chain.nodes:
            element = node.element

            def record(_original=element._do_close, _name=node.name) -> None:
                _original()
                closed.append(_name)

            element._do_close = record  # type: ignore[method-assign]

        runner = InprocessRunner(chain, settings(workers=1)).start()
        runner.stop(timeout_s=5.0)

        assert [node.name for node in chain.nodes] == list(ORDER), "the resolved order"
        assert closed == list(reversed(ORDER))

    def test_a_runner_with_no_workers_is_refused_at_construction(self) -> None:
        """It accepts items and walks none of them, which looks exactly like a hung chain."""
        with pytest.raises(ConfigurationError, match="workers must be >= 1"):
            InprocessRunner(load(), workers=0)

    def test_an_injected_queue_that_is_already_closed_is_refused_at_start(self) -> None:
        """A closed queue fails every submission: the shard looks alive and serves nothing.

        Only reachable for an *injected* queue: one this runner built is rebuilt on a restart.
        Replacing the caller's object instead would silently discard the capacity and overflow
        policy they chose, which is the more expensive mistake.
        """
        queue = FairPriorityQueue("spent", 4)
        queue.close()
        runner = InprocessRunner(load(), settings(workers=1), queue=queue)

        with pytest.raises(ServerStateError, match="closed queue"):
            runner.start()

        assert not runner.is_running
        assert all(not chain_node.element.is_open for chain_node in runner.topology)


# -- refusals and failures ----------------------------------------------------------------


class TestBackpressureAndFailure:
    def test_a_full_lane_refuses_the_newest_item(self, running) -> None:
        """The producer is told "no" so it can drop its own frame (ADR-005, arch.md §5②).

        The single worker is parked inside the gate element, so the queue holds exactly what
        this test put there: one item in a one-slot queue, and the next submission is refused
        with the depth and capacity an operator can act on.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = running(chain, settings(workers=1, queue_capacity=1))

        first = runner.submit(item("cam-1", 1))
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        second = runner.submit(item("cam-1", 2))

        with pytest.raises(QueueFullError) as caught:
            runner.submit(item("cam-1", 3))

        assert caught.value.capacity == 1
        # The number worth paging on is *which* camera flooded, not the shard total (ADR-005).
        assert runner.metrics.items_dropped.value(camera="cam-1") == 1
        assert runner.stats()["items"]["dropped"] == 1
        gate.release.set()
        _, not_done = wait([first, second], timeout=15.0)
        assert not not_done
        assert emitted_tags(chain) == {("cam-1", 1), ("cam-1", 2)}

    def test_an_element_failure_costs_one_item_and_not_the_worker(self, running) -> None:
        """A worker that died would stop serving every camera on the shard.

        The failed item's future carries a typed failure naming the element and the tag — the
        frame does not vanish — and the other five walk to the sink on the same worker.
        """
        chain = load(detect="runner-boom", params="{class: ship, boom_frame: 2}")
        runner = running(chain, settings(workers=1))

        futures = submit_all(runner, chain)

        failed = [f for f, tag in zip(futures, TAGS, strict=True) if tag[1] == 2]
        survived = [f for f, tag in zip(futures, TAGS, strict=True) if tag[1] != 2]
        assert len(failed) == 3
        for future in failed:
            with pytest.raises(InferenceError, match="detect"):
                future.result(timeout=5.0)
        assert [future.exception() for future in survived] == [None] * 3
        assert emitted_tags(chain) == {("cam-1", 1), ("cam-2", 1), ("cam-3", 1)}
        assert runner.is_running
        assert runner.stats()["items"] == {
            "accepted": 6,
            "walked": 3,
            "failed": 3,
            "expired": 0,
            "dropped": 0,
        }

    def test_an_item_past_its_deadline_never_reaches_the_chain(self, running) -> None:
        """Spending a GPU on a frame that is already too late to act on is pure waste.

        The deadline is measured from *capture*, as the ingest sink measures it: the gap
        between capture and dequeue is exactly the queue latency a frame deadline exists to
        catch.
        """
        chain = load()
        runner = running(
            chain,
            ServerSettings(pipeline={"workers": 1}, ingest={"frame_deadline_ms": 1}),
        )

        stale = runner.submit(item("cam-1", 1, captured_ns=time.monotonic_ns() - 1_000_000_000))
        fresh = runner.submit(item("cam-2", 1, captured_ns=time.monotonic_ns()))

        with pytest.raises(RequestCancelledError):
            stale.result(timeout=10.0)
        assert fresh.exception(timeout=10.0) is None
        assert emitted_tags(chain) == {("cam-2", 1)}

    def test_shutting_down_fails_what_is_still_queued_rather_than_dropping_it(self) -> None:
        """A shutdown that silently discards queued frames is not an orderly shutdown."""
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = InprocessRunner(chain, settings(workers=1, queue_capacity=8)).start()
        held = runner.submit(item("cam-1", 1))
        assert gate.entered.wait(10.0)
        queued = [runner.submit(item("cam-1", frame)) for frame in (2, 3)]

        gate.release.set()
        runner.stop(timeout_s=5.0)

        assert held.exception(timeout=5.0) is None
        for future in queued:
            assert isinstance(future.exception(timeout=5.0), RequestCancelledError)

    def test_the_walk_re_checks_the_deadline_it_may_have_passed_while_waiting(
        self, running
    ) -> None:
        """The queue drops what expired on the way *out*; this catches what expired after.

        Two items are drained in one wakeup batch, both fresh at that instant. The worker
        parks in the gate with the first one, and by the time it reaches the second the
        deadline has passed — so the second must be failed without a model ever seeing it.
        Spending a GPU on a frame that is already too late to act on is pure waste, and this
        is the only check that can catch this one: the queue is done with the item.

        The two capture clocks differ by design. A 500 ms budget and a second item captured
        400 ms ago means the gap the gate opens (250 ms) expires *that* one and leaves the
        held one with budget to spare — so this test still says "the item behind the gate
        expired" and not "everything in the batch expired", which is what makes it discriminate
        from the re-check in front of each model element.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        queue = PausedQueue("held", 8)
        runner = running(
            chain,
            ServerSettings(
                pipeline={"workers": 1, "frames_per_wakeup": 2},
                ingest={"frame_deadline_ms": 500},
            ),
            queue=queue,
        )
        now = time.monotonic_ns()
        held = runner.submit(item("cam-1", 1, captured_ns=now))
        late = runner.submit(item("cam-1", 2, captured_ns=now - 400_000_000))

        queue.ready.set()
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        # Sleeping past the deadline is what the gate makes deterministic: the drain has
        # already happened (both items were fresh then) and the second item's expiry check
        # has not.
        time.sleep(0.25)
        gate.release.set()

        assert held.exception(timeout=10.0) is None
        error = late.exception(timeout=10.0)
        assert isinstance(error, RequestCancelledError)
        assert "before the walk" in str(error), "the walk's check, not the queue's"
        assert runner.stats()["items"]["expired"] == 1
        assert runner.metrics.items_expired.value(camera="cam-1") == 1
        assert emitted_tags(chain) == {("cam-1", 1)}

    def test_stopping_fails_the_item_an_abandoned_worker_was_still_walking(self) -> None:
        """The one item per worker that is neither in the queue nor at the sink.

        ``stop`` closing the queue resolves what is still *queued*. The item a worker is
        already inside the chain with is not in the queue any more, and a worker still stuck
        at the join deadline will never resolve it — so without this the producer holding that
        future waits forever, which is exactly the frame that vanishes with no typed outcome
        that ADR-005 and ``base.py``'s ``submit`` contract exist to prevent.

        The gate makes "stuck" deterministic: the worker is parked inside ``detect`` and the
        deadline is 0.2 s, so ``stop`` must return having failed the item rather than having
        waited for it.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = InprocessRunner(chain, settings(workers=1, queue_capacity=8)).start()
        held = runner.submit(item("cam-1", 1))
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        workers = list(runner._threads)

        started = time.monotonic()
        runner.stop(timeout_s=0.2)
        elapsed = time.monotonic() - started

        assert held.done(), "resolved by stop() itself, not by the worker that is stuck in it"
        assert elapsed < 5.0, "stop waited for the gate instead of failing the item"
        error = held.exception(timeout=1.0)
        assert isinstance(error, RequestCancelledError)
        assert "the runner stopped" in str(error)
        assert runner.stats()["items"]["failed"] == 1
        assert runner.metrics.items_failed.value(camera="cam-1") == 1
        # Release the abandoned worker and let it exit, so the test leaves no thread behind.
        # It resumes into a closed chain and logs one refusal per element it reaches; that is
        # the documented consequence of abandoning it, and its future is already resolved.
        gate.release.set()
        for worker in workers:
            worker.join(10.0)
            assert not worker.is_alive()

    def test_stopping_fails_every_item_of_an_abandoned_worker_s_wake_up_batch(self) -> None:
        """A worker does not hold one item off the queue — it holds a whole wake-up batch.

        With ``frames_per_wakeup: 4`` one drain hands the worker four items, and the three
        behind the one it is wedged inside are no longer in the queue either. Closing the queue
        cannot resolve them, and the worker never will. A registry with one slot per worker
        failed only the first, and three producers were left holding futures nobody owned —
        exactly the frame that vanishes with no typed outcome that ADR-005 and ``base.py``'s
        ``submit`` contract exist to prevent.

        ``PausedQueue`` is what makes "one batch" a fact rather than a race: the worker is held
        out of the queue until all four items are in it, so the drain returns four and not one
        followed by three.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        queue = PausedQueue("held", 8)
        runner = InprocessRunner(
            chain,
            settings(workers=1, frames_per_wakeup=4),
            queue=queue,
        ).start()
        futures = [runner.submit(item("cam-1", frame)) for frame in (1, 2, 3, 4)]
        workers = list(runner._threads)

        queue.ready.set()
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        started = time.monotonic()
        runner.stop(timeout_s=0.2)
        elapsed = time.monotonic() - started

        assert elapsed < 5.0, "stop waited for the gate instead of failing the batch"
        assert [future.done() for future in futures] == [
            True
        ] * 4, "all four, not just the first"
        for future in futures:
            error = future.exception(timeout=1.0)
            assert isinstance(error, RequestCancelledError), error
            assert "the runner stopped" in str(error)
        assert runner.stats()["items"]["failed"] == 4
        assert runner.metrics.items_failed.value(camera="cam-1") == 4
        # Release the abandoned worker and let it exit, so the test leaves no thread behind.
        gate.release.set()
        for worker in workers:
            worker.join(10.0)
            assert not worker.is_alive()

    def test_the_walk_re_checks_the_deadline_in_front_of_every_model_element(
        self, running
    ) -> None:
        """One check at the top of a nine-element walk is a check for the first element only.

        Every model element submits to the pool and *sleeps* on the answer, so a chain can
        spend several stage timeouts between the top of the walk and the segmenter. Here the
        gate holds the item inside ``detect`` well past its 100 ms budget: it was fresh when
        the walk began and when the detector ran, and it must be failed at ``segment`` — the
        next element that would submit — rather than walked to the sink having consumed the
        whole chain's GPU on a frame nobody can act on any more.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        segment = chain.node("segment").element
        runner = running(
            chain,
            ServerSettings(pipeline={"workers": 1}, ingest={"frame_deadline_ms": 100}),
        )

        late = runner.submit(item("cam-1", 1, captured_ns=time.monotonic_ns()))
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        time.sleep(0.25)
        gate.release.set()

        error = late.exception(timeout=10.0)
        assert isinstance(error, RequestCancelledError)
        assert "before element 'segment'" in str(error), "the element it had reached"
        assert segment.processes == 0, "no model saw a frame that was already too late"
        assert emitted_tags(chain) == set(), "failed, not walked to the sink"
        assert runner.stats()["items"]["expired"] == 1
        assert runner.metrics.items_expired.value(camera="cam-1") == 1

    def test_a_fan_in_refuses_to_donate_under_a_cap_nobody_negotiated(self, running) -> None:
        """The donor consumed its item, and the branch left over carries a different cap.

        ``track`` takes the frame from ``detect`` (``nv12@gpu``) and metadata from ``track_a``
        (``meta@cpu``), and the loader nominated ``detect`` as its donor. With ``detect``
        consuming this camera's items the only contribution left is the ``meta@cpu`` one — and
        handing *that* payload on as the donation would label an item with a cap the loader
        never negotiated for the donor's edge, which is the relabelling the ``*@*`` fix on the
        ``pool`` element was about. So the item fails, typed, naming both sides.
        """
        chain = load_fan_in(drop_camera="cam-1")
        runner = running(chain, settings(workers=1))

        dropped = runner.submit(item("cam-1", 1))

        error = dropped.exception(timeout=10.0)
        assert isinstance(error, InferenceError)
        assert "track" in str(error) and "detect" in str(error), "both sides are named"
        assert "nv12@gpu" in str(error), "and the cap that was negotiated"
        assert emitted_tags(chain) == set(), "nothing travelled under an unnegotiated cap"
        assert runner.stats()["items"]["failed"] == 1
        assert runner.metrics.items_failed.value(camera="cam-1") == 1

    def test_the_same_fan_in_merges_normally_when_its_donor_produces(self, running) -> None:
        """The other half, so the refusal above cannot pass by the chain never working."""
        chain = load_fan_in(drop_camera="cam-9")
        runner = running(chain, settings(workers=1))

        assert runner.submit(item("cam-1", 1)).exception(timeout=10.0) is None
        emitted = sink(chain).emitted[0]
        assert emitted.meta["boxes"], "the donor's branch is in the merge"
        assert emitted.meta["tracks"], "and so is the other one"

    def test_a_work_item_that_did_not_come_through_submit_is_refused_by_name(self) -> None:
        """A typed refusal on a worker thread, not an ``AttributeError`` from inside one.

        The queue's element type is ``WorkItem``, so anything put into an injected queue
        directly — a test, a caller reusing the lane — reaches ``_walk`` carrying no chain
        item. It has a future, so it gets a typed failure like every other item.
        """
        chain = load()
        queue = FairPriorityQueue("foreign", 4)
        runner = InprocessRunner(chain, settings(workers=1), queue=queue).start()
        try:
            request = InferenceRequest(model_name="chain", inputs={}, context=RequestContext())
            work = WorkItem(request, ResponseFuture(request))
            queue.put(work)

            error = work.future.exception(timeout=10.0)
        finally:
            runner.stop(timeout_s=5.0)

        assert isinstance(error, InferenceError)
        assert "carries no chain item" in str(error)
        assert "submit()" in str(error), "the message says how items are meant to arrive"


class TestWhatTheElementsAreTold:
    def test_the_settings_the_elements_cannot_read_are_resolved_onto_the_context(self) -> None:
        """``topology`` is pure, so a knob reaches an element only if the runner carries it.

        ``pipeline.stage_timeout_ms`` and ``ingest.input_name`` are the two the ``pool``
        elements need and cannot look up: an element that imported the settings tree would be
        choosing its own configuration, which is what
        :class:`~shipinfer.topology.base.ElementContext` being frozen and runner-built exists
        to prevent. Before this was carried, lowering ``stage_timeout_ms`` to 500 ms changed
        nothing and every ``pool`` element still waited the module default of five seconds.
        """
        runner = InprocessRunner(
            load(),
            ServerSettings(
                pipeline={"workers": 1, "stage_timeout_ms": 500},
                ingest={"input_name": "pixels"},
            ),
            shard_id=3,
        )

        context = runner.element_context()

        assert context.stage_timeout_s == 0.5, "milliseconds in the settings, seconds here"
        assert context.input_name == "pixels"
        assert context.shard_id == 3, "and the fields that were already carried still are"

    def test_the_default_settings_resolve_to_the_defaults_the_elements_mirror(self) -> None:
        """The two literals in ``topology/elements/pool.py`` are the fallback, not a fiction."""
        context = InprocessRunner(load()).element_context()

        assert context.stage_timeout_s == 5.0
        assert context.input_name == "images"


# -- observability ------------------------------------------------------------------------


class TestHealthAndStats:
    def test_health_answers_before_the_runner_starts(self) -> None:
        """The first question asked of a runner that will not start is what state it is in."""
        health = InprocessRunner(load(), settings(workers=2)).health()

        assert health["runner"] == "inprocess"
        assert health["state"] == "stopped"
        assert health["topology"] == "mock_ship_person"
        assert health["elements"] == dict.fromkeys(ORDER, False)
        assert health["workers"] == {"wanted": 2, "alive": 0}

    def test_health_reports_the_open_chain_and_the_live_workers(self, running) -> None:
        chain = load()
        runner = running(chain, settings(workers=2))

        health = runner.health()

        assert health["state"] == "running"
        assert health["elements"] == dict.fromkeys(ORDER, True)
        assert health["workers"] == {"wanted": 2, "alive": 2}
        assert health["queue"]["depth"] == 0, "nothing submitted yet"

    def test_every_counter_is_attributed_to_the_camera_that_earned_it(self, running) -> None:
        """The fleet total is never the number that matters — *which* camera is (ADR-005).

        The previous generation could report "1000 entries in the buffer" and never "camera 7
        lost 40% of its frames", and four integers behind a lock reproduce exactly that blind
        spot. Two frames from each of three cameras, and each camera's own count is readable.
        """
        chain = load()
        runner = running(chain, settings(workers=2))

        submit_all(runner, chain)

        metrics = runner.metrics
        for camera in ("cam-1", "cam-2", "cam-3"):
            assert metrics.items_accepted.value(camera=camera) == 2
            assert metrics.items_walked.value(camera=camera) == 2
            assert metrics.items_failed.value(camera=camera) == 0
        assert metrics.items_walked.value(camera="cam-9") == 0, "a camera that sent nothing"
        assert runner.stats()["items"]["walked"] == 6, "and the roll-up still agrees"

    def test_stats_counts_what_was_accepted_and_what_was_walked(self, running) -> None:
        chain = load()
        runner = running(chain, settings(workers=2))

        submit_all(runner, chain)

        stats = runner.stats()
        assert stats["items"] == {
            "accepted": 6,
            "walked": 6,
            "failed": 0,
            "expired": 0,
            "dropped": 0,
        }
        assert stats["queue"]["accepted"] == 6
        assert stats["workers"] == 2
