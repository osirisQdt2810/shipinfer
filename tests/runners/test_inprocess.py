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
    RequestTimeoutError,
    RingClosedError,
    ServerStateError,
    ValidationError,
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
from shipinfer.topology.elements.mock import (
    MockDetect,
    MockOutput,
    MockRecognize,
    MockSegment,
)
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
  segment:      {impl: __SEGMENT__, model: ship_segmenter, when: class == ship}
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

#: A gated detector in front of a ``recognize`` slot the test chooses, so the question "is
#: this element charged an expiry check?" is asked of exactly one element. In ``MOCK_CHAIN``
#: the answer would always come from ``segment``, which is the next element that needs a model
#: after ``detect`` and would fail the item before ``recognize`` was reached.
EXPIRY_CHAIN = """
name: expiry
elements:
  decode:    {impl: mock}
  detect:    {impl: runner-gate, model: ship_detector}
  recognize: {impl: __RECOGNIZE__, after: detect__MODEL__}
  output:    {impl: mock}
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


class _Pooled:
    """Mixin: a mock that declares it resolves a repository model, without owning one.

    The stand-in for a ``pool`` element, and it has to be a *double* rather than a flag on the
    shipped mocks. :attr:`~shipinfer.topology.base.Element.needs_model` is what tells the
    process building a runner to construct an ``InferenceServer`` (``cli/commands/run.py``),
    so a ``MockDetect`` that answered ``True`` would make a chain of mocks load the whole
    model repository to run elements that invent a box.

    What it buys here is the runner's *other* reader of the same declaration: the expiry
    re-check in the walk, which used to ask ``node.kind in MODEL_KINDS``. That question and
    this one differ for every mock in the file, which is why the gate needs an element that
    says yes and there is none to hand.

    :attr:`~shipinfer.topology.base.Element.requires_model_name` is deliberately left
    ``False``: that is the *loader's* declaration, and setting it would make every chain in
    this file carry a ``model:`` for an element that resolves nothing. The split is what lets
    a double be one without being the other.
    """

    needs_model: ClassVar[bool] = True


@registry_for(ElementKind.SEGMENT).register("runner-pooled")
class PooledSegment(_Pooled, MockSegment):
    """A segmenter that would submit to the pool and sleep on the answer."""


@registry_for(ElementKind.RECOGNIZE).register("runner-pooled")
class PooledRecognize(_Pooled, MockRecognize):
    """The same, one kind over, so the gate can be asked about one element at a time."""


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

    def rearm(self) -> threading.Event:
        """Fresh latches for the next start cycle; returns the *previous* release event.

        Replacing the objects rather than clearing them, which is the whole point: a worker
        abandoned in the previous cycle is already inside ``release.wait()`` on the old event,
        so clearing would leave the two cycles sharing one latch and releasing the new
        worker's item would release the old worker's too. Returning the old one is how a test
        drives exactly one of the two.
        """
        previous = self.release
        self.entered = threading.Event()
        self.release = threading.Event()
        return previous


@registry_for(ElementKind.DETECT).register("runner-typed")
class TypedFailureDetect(MockDetect):
    """A detector that raises one nominated typed error, exactly as a ``pool`` element does.

    ``pool.py`` promises a ``QueueFullError`` from a saturated model queue reaches the
    submitter untouched; the mock chain runs offline with no model pool, so this is what makes
    that promise testable in the offline tier. Params-driven so one chain covers every member
    of the family — including a plain ``ValueError``, which is the case that *must* still be
    wrapped.
    """

    #: What each ``params.raises`` name costs the runner. Constructed here rather than in the
    #: element so the test can read the expected type off the same table the element raises
    #: from, instead of the two drifting apart.
    ERRORS: ClassVar[dict[str, Any]] = {
        "queue_full": lambda: QueueFullError("model:ship_detector", 64, 64),
        # A `QueueFullError` *subclass* that is not backpressure: phase D makes it one so the
        # dispatcher's spill loop retries another candidate, and the runner must still not
        # charge a dead peer to `items_backpressure`.
        "ring_closed": lambda: RingClosedError("shard-1", "ring:ship_detector"),
        "timeout": lambda: RequestTimeoutError("model 'ship_detector' did not answer"),
        "validation": lambda: ValidationError("the payload is not a tensor"),
        "foreign": lambda: ValueError("something nobody typed"),
    }

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        which = str(self.params.get("raises", ""))
        if which:
            raise self.ERRORS[which]()
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


class PausingRunner(InprocessRunner):
    """A runner whose *own* admission queue is a :class:`PausedQueue`, one per start cycle.

    Injecting the paused queue is the shorter spelling and it forbids the restart one test
    below needs: ``_do_start`` refuses an *injected* queue that a previous ``stop`` closed,
    deliberately, because replacing the caller's object would discard the capacity and
    overflow policy they chose. Overriding the hook the runner already builds its own queue
    through leaves that refusal exactly where it is, and still makes "one drain, one batch of
    three" a fact rather than a race against the worker's poll loop.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        #: Every queue this runner has built, newest last -- one per start cycle, so a test
        #: can release cycle one's drain without touching cycle two's.
        self.queues: list[PausedQueue] = []
        super().__init__(*args, **kwargs)

    def _build_queue(self) -> PausedQueue:
        queue = PausedQueue(f"held-{len(self.queues)}", self._settings.pipeline.queue_capacity)
        self.queues.append(queue)
        return queue


# -- helpers ------------------------------------------------------------------------------


def load(
    *,
    detect: str = "mock",
    segment: str = "mock",
    recognize: str = "mock",
    params: str = "{class: ship}",
) -> Topology:
    """The mock chain, with the three swappable slots filled in."""
    text = (
        textwrap.dedent(MOCK_CHAIN)
        .replace("__DETECT__", detect)
        .replace("__SEGMENT__", segment)
        .replace("__RECOGNIZE__", recognize)
        .replace("__PARAMS__", params)
    )
    return Topology.from_spec(ChainSpec.from_yaml(text))


def load_expiry(*, recognize: str, model: str = "") -> Topology:
    """The gated chain, with the element whose expiry gate is under test."""
    text = (
        textwrap.dedent(EXPIRY_CHAIN)
        .replace("__RECOGNIZE__", recognize)
        .replace("__MODEL__", f", model: {model}" if model else "")
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


#: The right-hand side of the ledger identity `InprocessRunner._do_stats` documents. Every
#: accepted item is in exactly one of these, or still in flight; the identity used to be false
#: because the three the *queue* resolves on its own had typed futures and no counter.
#:
#: ``backpressure`` and not ``dropped``: the two used to be one counter, and mixing an
#: admission refusal (never ``accepted``) with a model queue refusing mid-walk (``accepted``,
#: and inside the chain) is what forced the identity to carry a "minus the queue's rejected"
#: correction term. ``dropped`` is deliberately absent from this tuple.
OUTCOMES = (
    "walked",
    "failed",
    "expired",
    "timed_out",
    "backpressure",
    "queue_closed",
    "queue_evicted",
    "queue_expired",
    "in_flight",
)


def accounted(stats: dict[str, Any]) -> int:
    """How many accepted items ``stats()`` can name an outcome for."""
    return sum(stats["items"][key] for key in OUTCOMES)


def settled(runner: InprocessRunner, timeout_s: float = 5.0) -> dict[str, Any]:
    """``runner.stats()`` once the workers' slots have caught up with their futures.

    ``items["in_flight"]`` is a gauge, and the runner's docstring says so: a worker resolves
    an item's future at the end of its walk and narrows its slot on the way out of the loop
    body, so a ``stats()`` taken the instant a future completes can still count that item as
    in flight. Polling for zero rather than sleeping keeps the exact-dictionary assertions
    below both meaningful and non-flaky.
    """
    deadline = time.monotonic() + timeout_s
    stats = runner.stats()
    while stats["items"]["in_flight"] and time.monotonic() < deadline:
        time.sleep(0.005)
        stats = runner.stats()
    return stats


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
        assert settled(runner)["items"] == {
            "accepted": 6,
            "walked": 3,
            "failed": 3,
            "expired": 0,
            "timed_out": 0,
            "dropped": 0,
            "backpressure": 0,
            "queue_closed": 0,
            "queue_evicted": 0,
            "queue_expired": 0,
            "in_flight": 0,
        }

    @pytest.mark.parametrize(
        ("raises", "expected", "counter"),
        [
            ("queue_full", QueueFullError, "items_backpressure"),
            ("ring_closed", RingClosedError, "items_failed"),
            ("timeout", RequestTimeoutError, "items_timed_out"),
            ("validation", ValidationError, "items_failed"),
        ],
    )
    def test_an_element_s_own_typed_failure_reaches_the_submitter_as_itself(
        self, running, raises, expected, counter
    ) -> None:
        """Backpressure, a stage timeout and a bad payload are three events, not one.

        The walk used to wrap *every* element exception in ``InferenceError``, which flattened
        the whole :class:`~shipinfer.core.errors.base.ShipInferError` family into one type: a
        submitter could not tell "the detector's queue is full, back off" from "the detector
        never answered, it is saturated" from "I handed it the wrong payload, fix the chain",
        and ``topology/elements/pool.py`` promised in as many words that a ``QueueFullError``
        is "propagated untouched". It was not. Each one is also charged to its own counter, so
        an overloaded shard does not read as a shard full of bugs.

        ``ring_closed`` is the row that says inheritance is not the rule. ``RingClosedError``
        is a :class:`QueueFullError` *subclass* — deliberately, so phase D's dispatcher spill
        loop treats a closed ring as a refusal and tries the next candidate — but the operator
        response is the opposite of backpressure's: a peer is gone, and no amount of shedding
        load brings it back. Counting it as ``backpressure`` would put a dead process on the
        "shed load or add capacity" graph, which is the confusion this whole split removes.
        """
        chain = load(detect="runner-typed", params=f"{{class: ship, raises: {raises}}}")
        runner = running(chain, settings(workers=1))

        error = runner.submit(item("cam-1", 1)).exception(timeout=10.0)

        assert type(error) is expected, f"wrapped into {type(error).__name__}"
        assert getattr(runner.metrics, counter).value(camera="cam-1") == 1
        for other in (
            "items_dropped",
            "items_backpressure",
            "items_timed_out",
            "items_failed",
        ):
            if other != counter:
                assert getattr(runner.metrics, other).value(camera="cam-1") == 0
        assert emitted_tags(chain) == set(), "the walk stopped at the element that raised"

    def test_backpressure_from_an_element_keeps_the_depth_and_capacity_it_carried(
        self, running
    ) -> None:
        """The numbers are the whole reason ``QueueFullError`` is not a bare exception.

        "We dropped a frame" is not actionable; "the queue that refused holds 64 and had 64 in
        it" is. Re-wrapping the error kept the *text* in the message and threw the fields away,
        so the one consumer that matters — a producer deciding whether to shed or to retry —
        had to parse a string.
        """
        chain = load(detect="runner-typed", params="{class: ship, raises: queue_full}")
        runner = running(chain, settings(workers=1))

        error = runner.submit(item("cam-1", 1)).exception(timeout=10.0)

        assert isinstance(error, QueueFullError)
        assert (error.queue_name, error.depth, error.capacity) == (
            "model:ship_detector",
            64,
            64,
        )

    def test_an_exception_that_is_not_ours_is_still_wrapped_and_still_names_the_element(
        self, running
    ) -> None:
        """The other half: a bare ``ValueError`` says neither which element nor which frame.

        Without this, "preserve typed errors" could be implemented as "preserve everything",
        and the message an operator gets for an ordinary bug would lose the two facts that
        make it diagnosable.
        """
        chain = load(detect="runner-typed", params="{class: ship, raises: foreign}")
        runner = running(chain, settings(workers=1))

        error = runner.submit(item("cam-1", 7)).exception(timeout=10.0)

        assert type(error) is InferenceError
        assert "detect" in str(error) and "cam-1" in str(error) and "7" in str(error)
        assert "something nobody typed" in str(error), "the original is not lost either"
        assert runner.metrics.items_failed.value(camera="cam-1") == 1

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

    def test_an_abandoned_worker_walks_no_further_than_the_item_it_was_wedged_in(
        self,
    ) -> None:
        """The stop signal has to be read per *item*, not per wake-up batch.

        A worker holds a whole ``frames_per_wakeup`` batch, and ``stop`` fails every item of it
        from the in-flight slot precisely because the worker never will. The loop then read the
        signal only at the top of the outer ``while``, so a worker released after the deadline
        finished the item it was wedged inside -- documented and unavoidable -- and then went
        on to walk the ``frames_per_wakeup - 1`` behind it. Their futures had already been
        resolved at the stop, so the walk delivered nothing to anybody; what it did deliver was
        ghost events into the sink, through a chain the restart had just re-opened, attributed
        to a cycle that had been stopped. Downstream sees frames from a shard that is not
        running.

        Three items in one drain, the worker parked in the first, the restart in between: the
        sink must gain exactly the one item the worker was already inside, and not the two
        behind it.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = PausingRunner(
            chain, settings(workers=1, frames_per_wakeup=3, queue_capacity=8)
        )

        runner.start()
        futures = [runner.submit(item("cam-1", frame)) for frame in (1, 2, 3)]
        stale_worker = runner._threads[0]
        runner.queues[0].ready.set()
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        runner.stop(timeout_s=0.2)

        for future in futures:
            assert isinstance(future.exception(timeout=1.0), RequestCancelledError)
        emitted_at_stop = len(sink(chain).emitted)

        # The restart is what makes the ghost events *possible*: it re-opens every element, so
        # a stale worker's walk is no longer refused by a closed chain. `rearm` hands back
        # cycle one's latch so releasing it frees exactly the abandoned worker.
        stale_release = gate.rearm()
        runner.start()
        fresh_worker = runner._threads[0]
        runner.queues[1].ready.set()

        stale_release.set()
        # Cycle two's latch too, so a regression is a *ghost event* rather than a worker
        # parked in the gate for ten seconds. Nothing has been submitted into cycle two, so
        # the only thread this frees is the stale one, on the two items it must not walk.
        gate.release.set()
        stale_worker.join(10.0)
        stale_exited = not stale_worker.is_alive()

        assert emitted_tags(chain) == {("cam-1", 1)}, "the rest of the batch was walked"
        assert (
            len(sink(chain).emitted) == emitted_at_stop + 1
        ), "the item it was wedged inside, and nothing behind it"
        assert stale_exited, "an abandoned worker must exit at the next item, not rejoin"
        runner.stop(timeout_s=5.0)
        fresh_worker.join(10.0)
        assert not fresh_worker.is_alive()

    def test_a_stopped_runner_reports_nothing_in_flight_once_its_stale_worker_wakes(
        self,
    ) -> None:
        """``in_flight`` used to come back up after a stop and never come down again.

        ``stop`` drains the in-flight slots and fails every item in them, so the gauge reads
        zero the instant it returns. But the abandoned worker still holds the *list*, and its
        ``finally`` republishes the remainder of its wake-up batch the moment whatever wedged
        it lets go -- into the same object ``stats()`` was reading. From then on a stopped
        runner reported two items in flight, forever, for two futures it had already resolved:
        an operator draining a shard before a deploy watches exactly that number.

        The fix is that the stopped cycle's list is replaced and not merely emptied. The stale
        thread goes on writing into the object it was handed, and nothing reads it.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        queue = PausedQueue("held", 8)
        runner = InprocessRunner(
            chain, settings(workers=1, frames_per_wakeup=3), queue=queue
        ).start()
        futures = [runner.submit(item("cam-1", frame)) for frame in (1, 2, 3)]
        workers = list(runner._threads)

        queue.ready.set()
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        runner.stop(timeout_s=0.2)

        assert runner.stats()["items"]["in_flight"] == 0, "the drain emptied the slots"
        for future in futures:
            assert isinstance(future.exception(timeout=1.0), RequestCancelledError)

        # Let the stale worker out and *wait for it to be done*, so the republish this test is
        # about has certainly happened before the gauge is read again.
        gate.release.set()
        for worker in workers:
            worker.join(10.0)
            assert not worker.is_alive()

        assert (
            runner.stats()["items"]["in_flight"] == 0
        ), "a stopped runner reported work in flight that it had already failed"

    def test_a_worker_abandoned_before_a_restart_cannot_clear_the_new_worker_s_slot(
        self,
    ) -> None:
        """abandon -> restart -> abandon used to lose the second cycle's futures entirely.

        Each worker publishes what it still owes an answer for into a slot indexed by its own
        number, and ``stop`` drains those slots so an abandoned worker does not take its items'
        futures with it. The slot list was **rebound** on every start while a worker read it
        off ``self`` — so an abandoned worker from cycle one, finishing its walk after cycle
        two had begun, wrote its "nothing left" into cycle *two's* list at the same index. The
        new worker's batch was erased by a thread the runner had stopped tracking, and the next
        shutdown found an empty slot: the second item's future never resolved, which is exactly
        the frame that vanishes with no typed outcome that ADR-005 exists to prevent.

        Two independent latches are what make the sequence deterministic — ``rearm`` hands back
        the old one, so the first cycle's worker can be released while the second's stays
        parked.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = InprocessRunner(chain, settings(workers=1, queue_capacity=8))

        runner.start()
        first = runner.submit(item("cam-1", 1))
        stale_worker = runner._threads[0]
        assert gate.entered.wait(10.0), "the first cycle's worker never reached the gate"
        runner.stop(timeout_s=0.2)
        assert isinstance(first.exception(timeout=1.0), RequestCancelledError)

        stale_release = gate.rearm()
        runner.start()
        fresh_worker = runner._threads[0]
        second = runner.submit(item("cam-2", 1))
        assert gate.entered.wait(10.0), "the second cycle's worker never reached the gate"

        # Let cycle one's worker out. It walks its item to the sink -- the future is already
        # resolved, so nothing is delivered twice -- and then runs the `finally` whose store
        # used to land in cycle two's list, at cycle two's worker's index.
        stale_release.set()
        deadline = time.monotonic() + 10.0
        while ("cam-1", 1) not in emitted_tags(chain) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ("cam-1", 1) in emitted_tags(chain), "the abandoned worker never finished"
        # The clobber is the two stores after `_finish`, so give the stale thread its shot at
        # them before asking the question. With the fix this changes nothing.
        time.sleep(0.1)

        runner.stop(timeout_s=0.2)

        assert second.done(), "cycle two's item lost its owner to cycle one's worker"
        error = second.exception(timeout=1.0)
        assert isinstance(error, RequestCancelledError)
        assert "the runner stopped" in str(error)
        # Release cycle two's worker and let both threads exit, so the test leaves none behind.
        gate.release.set()
        for worker in (stale_worker, fresh_worker):
            worker.join(10.0)
            assert not worker.is_alive()

    def test_a_worker_abandoned_before_a_restart_cannot_consume_the_new_cycle_s_queue(
        self,
    ) -> None:
        """A stale worker that rejoins loses futures that ``stop()`` had already promised.

        A worker abandoned at a stop deadline is not gone: it is parked inside
        ``element.process()``. Everything it read off ``self`` when it woke therefore belonged
        to whichever cycle was current *then* — and cycle two rebuilt all of it. Clearing
        ``self._stopping`` made its loop condition true again, and ``self._queue`` had been
        rebuilt, so it called ``get_batch`` on the **new** cycle's queue, took real work off
        it, and published that work into cycle **one's** slot list, which no shutdown drains.
        The result is the exact failure ADR-005 and ``base.py``'s ``submit`` contract exist to
        prevent: ``stop()`` returned, ``is_running`` was ``False``, ``in_flight`` read 0, and a
        producer was still holding a future nobody would ever resolve.

        Two items, because the loss is a *race* between the stale worker and the live one over
        the queue — with one item either thread might win it, and with two the stale thread
        cannot avoid taking one if it is still a consumer at all. Both futures must be typed
        by the second shutdown, whichever worker held which.

        The sink is the second half of the same property. Cycle one's stop closed every
        element; a stale worker that goes back to the queue walks cycle two's items through
        elements a stopped runner owns, and emits them. Its own item — the one it was parked
        on — does reach the sink, and that is documented and unavoidable; nothing *after* that
        may.

        Determinism comes from the latches, not from sleeping: ``rearm`` hands back cycle
        one's release event so exactly one of the two workers is freed, and the freed worker is
        then **joined** — with the fix it finds an event set forever and a queue closed
        forever and exits, so by the time the two new items are submitted the only consumer
        left is cycle two's. That join is asserted last rather than first, so a regression
        reports the lost future it caused instead of the thread that caused it.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = InprocessRunner(chain, settings(workers=1, queue_capacity=8))

        runner.start()
        first = runner.submit(item("cam-1", 1))
        stale_worker = runner._threads[0]
        assert gate.entered.wait(10.0), "the first cycle's worker never reached the gate"
        runner.stop(timeout_s=0.2)
        assert isinstance(first.exception(timeout=1.0), RequestCancelledError)

        stale_release = gate.rearm()
        runner.start()
        fresh_worker = runner._threads[0]

        # Let cycle one's worker out. It finishes the item it was parked on -- the future is
        # already resolved, so nothing is delivered twice -- and then takes its next loop turn,
        # which is the turn this test is about.
        stale_release.set()
        deadline = time.monotonic() + 10.0
        while ("cam-1", 1) not in emitted_tags(chain) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ("cam-1", 1) in emitted_tags(chain), "the abandoned worker never finished"
        stale_worker.join(10.0)
        stale_exited = not stale_worker.is_alive()
        emitted_by_cycle_one = len(sink(chain).emitted)

        second = runner.submit(item("cam-2", 1))
        third = runner.submit(item("cam-3", 1))
        assert gate.entered.wait(10.0), "the second cycle's worker never reached the gate"
        runner.stop(timeout_s=0.2)

        for future, tag in ((second, "cam-2"), (third, "cam-3")):
            assert future.done(), f"{tag}'s future was lost to cycle one's worker"
            assert isinstance(future.exception(timeout=1.0), RequestCancelledError)
        assert (
            len(sink(chain).emitted) == emitted_by_cycle_one
        ), "a stopped cycle's worker emitted through elements the runner had closed"
        assert stale_exited, "an abandoned worker must exit on its next turn, not rejoin"
        # Release cycle two's worker and let both threads exit, so the test leaves none behind.
        gate.release.set()
        for worker in (stale_worker, fresh_worker):
            worker.join(10.0)
            assert not worker.is_alive()

    def test_the_walk_re_checks_the_deadline_in_front_of_every_model_element(
        self, running
    ) -> None:
        """One check at the top of a nine-element walk is a check for the first element only.

        Every element that resolves a repository model submits to the pool and *sleeps* on the
        answer, so a chain can spend several stage timeouts between the top of the walk and the
        segmenter. Here the gate holds the item inside ``detect`` well past its 100 ms budget:
        it was fresh when the walk began and when the detector ran, and it must be failed at
        ``segment`` — the next element that would submit — rather than walked to the sink
        having consumed the whole chain's GPU on a frame nobody can act on any more.

        ``segment`` is a :class:`PooledSegment` rather than the plain mock because the gate
        asks the *element* whether it resolves a model, and no shipped mock does. It used to
        ask the kind, which every ``segment`` slot answers the same way — including a mock
        that never waits for anything.
        """
        chain = load(detect="runner-gate", segment="runner-pooled")
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

    def test_an_element_that_resolves_no_model_is_not_charged_an_expiry_check(
        self, running
    ) -> None:
        """The gate is the implementation's business, not the kind's — the walk half of C2.

        ``recognize`` is a model *kind*, and the gate used to read ``node.kind in
        MODEL_KINDS``, so an element that submits to nothing — a gallery query in phase C, a
        mock here — was refused an already-late item exactly as a network would be. That is
        not a saving: the frame has crossed the whole chain and the only cost left is the
        microseconds this element takes, so dropping it there throws away work already paid
        for and emits nothing.

        Same gate, same 100 ms budget and the same 250 ms overshoot as the test below; the
        difference is only which implementation fills the slot.
        """
        chain = load_expiry(recognize="mock")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = running(
            chain,
            ServerSettings(pipeline={"workers": 1}, ingest={"frame_deadline_ms": 100}),
        )

        late = runner.submit(item("cam-1", 1, captured_ns=time.monotonic_ns()))
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        time.sleep(0.25)
        gate.release.set()

        assert late.exception(timeout=10.0) is None, "a local element cost it its frame"
        assert emitted_tags(chain) == {("cam-1", 1)}, "walked to the sink, not dropped"
        assert runner.stats()["items"]["expired"] == 0

    def test_the_same_slot_filled_by_a_model_element_is_still_charged_one(
        self, running
    ) -> None:
        """The half that must not have moved, one element over from the test above.

        :class:`PooledRecognize` on the same slot stands in for the pool implementation: it
        declares that it resolves a repository model, so it would submit and sleep, and a
        frame nobody can act on any more must not be given another GPU. Same chain, same
        gate, same overshoot — one word of the element's declaration is the whole difference,
        which is what makes the pair evidence rather than two tests.
        """
        chain = load_expiry(recognize="runner-pooled", model="ship_recognizer")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
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
        assert "before element 'recognize'" in str(error)
        assert emitted_tags(chain) == set()
        assert runner.stats()["items"]["expired"] == 1

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

    def test_the_elements_record_on_the_runners_own_registry(self) -> None:
        """Not a fresh one, and the identity is the assertion.

        One exporter has to carry both halves of a dropped frame — the runner counted it
        against ``shipinfer_runner_items_dropped_total{camera}`` and an element that counted
        its own refusal onto a private registry would publish a number nothing scrapes. The
        same argument ``_ingest`` makes when it hands ``IngestMetrics`` this registry.
        """
        runner = InprocessRunner(load())

        context = runner.element_context()

        assert context.metrics is runner.metrics.registry

    def test_the_worker_count_is_the_one_this_runner_will_really_use(self) -> None:
        """The constructor's override, not ``pipeline.workers`` — they can disagree.

        The element that needs this number needs the true one: a barrier that waits for other
        cameras' frames must leave a worker free to deliver them, and one told "four" on a
        runner started with one would park the only thread there is and then close its instant
        by timeout, forever. Asserted with the two deliberately different, which is the only
        way to tell a carried number from a re-read setting.
        """
        runner = InprocessRunner(load(), ServerSettings(pipeline={"workers": 4}), workers=1)

        context = runner.element_context()

        assert runner.workers == 1, "the override won, as it always did"
        assert context.workers == 1, "and it is what the elements are told"

    def test_the_image_ops_are_not_resolved_yet_and_say_so(self) -> None:
        """``None`` and not a host-side default. An element that needs ops and finds none
        raises, which is a deploy that stops; a numpy fallback nobody asked for is a chain
        that runs at a fiftieth of the speed and reports nothing."""
        assert InprocessRunner(load()).element_context().ops is None


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

        stats = settled(runner)
        assert stats["items"] == {
            "accepted": 6,
            "walked": 6,
            "failed": 0,
            "expired": 0,
            "timed_out": 0,
            "dropped": 0,
            "backpressure": 0,
            "queue_closed": 0,
            "queue_evicted": 0,
            "queue_expired": 0,
            "in_flight": 0,
        }
        assert stats["queue"]["accepted"] == 6
        assert stats["workers"] == 2

    def test_the_ledger_names_an_outcome_for_every_item_that_was_accepted(self) -> None:
        """``accepted`` used to outrun the sum of the outcomes, and the gap looked like work.

        Three items on one held worker: one taken off the queue and inside the chain, two still
        queued. ``stop`` fails the first from its in-flight slot and the queue's ``close``
        fails the other two — and *that* outcome had a typed future and no counter at all, so
        an operator reading ``stats()`` saw ``accepted: 3`` against one outcome and could not
        tell two lost frames from two slow ones. The same gap swallows what ``drop_expired``
        drops at the drain and what ``DROP_OLDEST`` evicts.

        The identity is checked twice on purpose: once while the items are alive, where
        ``in_flight`` is the whole of it, and once after the shutdown has resolved them.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = InprocessRunner(chain, settings(workers=1, queue_capacity=8)).start()
        held = runner.submit(item("cam-1", 1))
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        queued = [runner.submit(item("cam-2", frame)) for frame in (1, 2)]
        workers = list(runner._threads)

        live = runner.stats()
        assert live["items"]["accepted"] == 3
        assert live["items"]["in_flight"] == 3, "one inside the chain, two still in the lane"
        assert accounted(live) == 3

        runner.stop(timeout_s=0.2)

        assert isinstance(held.exception(timeout=1.0), RequestCancelledError)
        for future in queued:
            assert isinstance(future.exception(timeout=1.0), RequestCancelledError)
        stats = runner.stats()
        assert stats["items"]["failed"] == 1, "the one the abandoned worker was walking"
        assert stats["items"]["queue_closed"] == 2, "and the two the queue itself failed"
        assert stats["items"]["in_flight"] == 0
        assert accounted(stats) == stats["items"]["accepted"] == 3
        # The camera is the number that matters, not the shard total (ADR-005).
        assert runner.metrics.items_queue_closed.value(camera="cam-2") == 2
        assert runner.metrics.items_queue_closed.value(camera="cam-1") == 0
        gate.release.set()
        for worker in workers:
            worker.join(10.0)
            assert not worker.is_alive()

    def test_a_model_queue_refusing_mid_walk_is_backpressure_not_a_dropped_submission(
        self, running
    ) -> None:
        """The two backpressure populations are not the same items, so they are not one term.

        A ``pool`` element's model queue refusing a request mid-walk loses an item that was
        **accepted** -- it came off the lane, a worker owns it, and ``stats()`` owes it an
        outcome. A submission the runner's own lane refuses at the door was never accepted and
        owes nothing. One counter for both made the ledger identity carry a correction term
        ("minus the queue's ``rejected``"), and a ledger that needs a correction term is a
        ledger nobody checks.
        """
        chain = load(detect="runner-typed", params="{class: ship, raises: queue_full}")
        runner = running(chain, settings(workers=1))

        error = runner.submit(item("cam-1", 1)).exception(timeout=10.0)

        assert isinstance(error, QueueFullError)
        stats = settled(runner)
        assert stats["items"]["backpressure"] == 1
        assert stats["items"]["dropped"] == 0, "the door refused nobody"
        assert runner.metrics.items_backpressure.value(camera="cam-1") == 1
        assert runner.metrics.items_dropped.value(camera="cam-1") == 0
        assert accounted(stats) == stats["items"]["accepted"] == 1

    def test_a_refused_submission_is_not_a_term_in_the_ledger(self) -> None:
        """The other half, and the one that used to make the identity false.

        Three submissions into a one-slot lane with the only worker held: two are accepted and
        the third is refused at the door. While the refusal was counted on the same
        ``dropped`` term the ledger summed, the right-hand side read three against an
        ``accepted`` of two -- an over-count by exactly ``queue["rejected"]``, which the
        docstring had to spell out as a caveat instead of the numbers simply adding up.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = InprocessRunner(chain, settings(workers=1, queue_capacity=1)).start()
        workers = list(runner._threads)
        try:
            first = runner.submit(item("cam-1", 1))
            assert gate.entered.wait(10.0), "the worker never reached the gate"
            second = runner.submit(item("cam-1", 2))
            with pytest.raises(QueueFullError):
                runner.submit(item("cam-1", 3))

            live = runner.stats()
            assert live["items"]["accepted"] == 2, "the third never got in"
            assert live["items"]["dropped"] == 1, "and it is still counted, per camera"
            assert live["items"]["backpressure"] == 0, "refused at the door, not mid-walk"
            assert accounted(live) == live["items"]["accepted"] == 2
        finally:
            gate.release.set()
            _, not_done = wait([first, second], timeout=15.0)
            assert not not_done
            runner.stop(timeout_s=5.0)
        for worker in workers:
            worker.join(10.0)
            assert not worker.is_alive()

    def test_the_ledger_counts_the_items_a_closed_injected_queue_failed(self) -> None:
        """The counter belongs to whoever performs the close, and here that is the runner.

        ``_do_start`` refuses to *replace* an injected queue, because doing so would throw away
        the capacity and overflow policy the caller chose — so it is fair to ask whether the
        runner should count what closing that queue fails. It should, and does: a queue the
        runner closes is a queue whose typed futures the runner can enumerate, injected or not,
        and the ``accepted``-outran-the-outcomes gap is the same gap either way.

        The half that stays with the caller is the close the *caller* performs. There is no
        callback and no drained list on that path, and the queue's own ``stats()`` does not
        separate "failed by close" from the rest, so the runner would have to guess -- and a
        guessed counter reads exactly like a real one. ``_close_queue`` documents that split.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        injected = FairPriorityQueue("injected", 8)
        runner = InprocessRunner(chain, settings(workers=1), queue=injected).start()
        held = runner.submit(item("cam-1", 1))
        assert gate.entered.wait(10.0), "the worker never reached the gate"
        queued = runner.submit(item("cam-2", 1))
        workers = list(runner._threads)

        runner.stop(timeout_s=0.2)

        assert isinstance(queued.exception(timeout=1.0), RequestCancelledError)
        assert isinstance(held.exception(timeout=1.0), RequestCancelledError)
        assert runner.stats()["items"]["queue_closed"] == 1
        assert runner.metrics.items_queue_closed.value(camera="cam-2") == 1
        assert (
            runner.metrics.items_queue_closed.value(camera="cam-1") == 0
        ), "the held item was failed from its in-flight slot, not by the close"
        gate.release.set()
        for worker in workers:
            worker.join(10.0)
            assert not worker.is_alive()

    def test_the_ledger_counts_what_drop_oldest_sacrificed_to_make_room(self) -> None:
        """An eviction is a lost frame with a typed future, and it had no counter either.

        ``DROP_OLDEST`` is the policy whose silent version is the bug this whole system was
        rebuilt to remove, so "the lane evicted one of camera 1's frames" has to be readable
        somewhere. It is the queue that evicts, not the runner, so the runner's own counters
        could never see it — ``stats()["items"]`` now carries the queue's number next to its
        own.
        """
        chain = load(detect="runner-gate")
        gate = chain.node("detect").element
        assert isinstance(gate, GateDetect)
        runner = InprocessRunner(
            chain,
            settings(workers=1, queue_capacity=1, overflow_policy="drop_oldest"),
        ).start()
        workers = list(runner._threads)
        try:
            held = runner.submit(item("cam-1", 1))
            assert gate.entered.wait(10.0), "the worker never reached the gate"
            sacrificed = runner.submit(item("cam-1", 2))
            runner.submit(item("cam-1", 3))

            assert isinstance(sacrificed.exception(timeout=5.0), QueueFullError)
            stats = runner.stats()
            assert stats["items"]["accepted"] == 3, "all three were admitted"
            assert stats["items"]["dropped"] == 0, "nothing was refused at the door"
            assert stats["items"]["queue_evicted"] == 1
            assert accounted(stats) == 3
        finally:
            gate.release.set()
            runner.stop(timeout_s=5.0)
        assert held.exception(timeout=5.0) is None
        for worker in workers:
            worker.join(10.0)
