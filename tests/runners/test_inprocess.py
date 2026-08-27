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
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    ServerStateError,
)
from shipinfer.core.request import RequestContext, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.runners.inprocess import InprocessRunner
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

#: The slots in the order the loader resolves them, which is the order elements are opened in
#: and the reverse of the order they are closed in.
ORDER = (
    "decode",
    "detect",
    "segment",
    "embed_ship",
    "embed_person",
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

    def test_stats_counts_what_was_accepted_and_what_was_walked(self, running) -> None:
        chain = load()
        runner = running(chain, settings(workers=2))

        submit_all(runner, chain)

        stats = runner.stats()
        assert stats["items"] == {"accepted": 6, "walked": 6, "failed": 0, "expired": 0}
        assert stats["queue"]["accepted"] == 6
        assert stats["workers"] == 2
