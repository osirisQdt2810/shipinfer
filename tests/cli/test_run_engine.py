"""``shipinfer run`` builds the dependencies its chain needs, and only those.

The defect this file pins: ``run`` built a runner with no ``models=``, so
:class:`~shipinfer.topology.base.ElementContext` carried ``models=None`` and the first
``pool`` element refused at ``open()`` — ``--runner inprocess`` could not execute any real
topology, only a chain of mocks. The fleet path had always been right: a shard builds an
:class:`~shipinfer.engine.InferenceServer`, starts it, and passes it in (``cli/shard.py``).

What is asserted here is the *order*, because that is where the remaining mistakes live: the
pool has to be up before the chain opens against it, and it has to outlive the workers that
are still submitting to it. So the double records rather than mocks, and the tests compare a
whole sequence instead of counting calls.

Two dependencies are gated the same way and asserted the same way: the model pool
(``models=``) and, since C3, image pre-processing (``ops=``). A ``pool`` detector letterboxes
its frame and cannot resolve an implementation itself — ``topology`` may not import
``runtime`` — so this command resolves one, and only for a chain that declares
:attr:`~shipinfer.topology.base.Element.needs_image_ops`. A chain of mocks must build neither.

Everything is offline. The engine is a recording double in every test but the last, which
starts a real ``InferenceServer`` over the mock backend and a detector-shaped repository — the
one that proves a ``pool`` element actually opens, rather than that a keyword was forwarded. The
ops it is handed there resolve to ``NumpyImageOps``, because ``get_image_ops`` degrades to it
on a host with no accelerator; that is the whole reason this tier can run the real element.
"""

from __future__ import annotations

import importlib
import inspect
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

from shipinfer.cli.commands.run import (
    dependency_is_needed,
    image_ops_are_needed,
    model_pool_is_needed,
    run,
)
from shipinfer.core.errors import ConfigurationError, ServerStateError
from shipinfer.core.request import ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.runners import RUNNERS
from shipinfer.runners.base import Runner
from shipinfer.topology import ChainItem, ChainSpec, ElementKind, Topology
from shipinfer.topology.base import ElementContext
from shipinfer.topology.elements.mock import MockDetect
from shipinfer.topology.registry import registry_for

#: The module object, not the ``run`` function ``shipinfer.cli.commands`` re-exports under the
#: same name -- ``monkeypatch.setattr("shipinfer.cli.commands.run._wait", ...)`` resolves the
#: dotted path by attribute and lands on the function.
run_module = importlib.import_module("shipinfer.cli.commands.run")

#: A chain with a `pool` element: the shape of every real topology, and the one that used to
#: fail at `open()`. `detector` is the model :func:`detector_repository` declares, and it is a
#: *detector-shaped* one: one input `images[3x8x8]:FP32` and one output `output0[300x6]:FP32`.
#:
#: That shape is load-bearing, not decoration. A `pool` detector submits a letterboxed
#: `(1, 3, H, W)` frame, so it refuses at `open()` both an input the artefact does not declare
#: and one no letterbox fits -- and the shared `tmp_repository` fixture's `echo` declares
#: `x[4]`, which is the second of those. Naming it here (with a `decode.dst_size` to get past
#: the extent resolution) made this file's real-engine test assert that a chain which cannot
#: run had opened successfully. `input:` is named anyway, because `ingest.input_name` is where
#: an element that does not say gets it and the default is not this model's; `decode.dst_size`
#: is not, because the model declares its extent and that is the path a real deployment takes.
POOL_CHAIN = textwrap.dedent("""
    name: pool_chain
    elements:
      decode: {impl: mock}
      detect: {impl: pool, model: detector, params: {input: images}}
      output: {impl: mock}
    """)

#: The same chain with the mock detector: a model *kind*, naming a model, needing no pool.
#: Asking `node.kind in MODEL_KINDS` instead of the element would load a repository for this.
MOCK_CHAIN = POOL_CHAIN.replace("{impl: pool", "{impl: mock").replace(
    "name: pool_chain", "name: mock_chain"
)

#: The same chain again with an `nvinfer`-shaped detector (:class:`DetectElsewhere` below):
#: it *names* `model: echo` and runs it outside this process. The chain the split exists for
#: -- asking `requires_model_name` here would build an `InferenceServer` for it.
ELSEWHERE_CHAIN = POOL_CHAIN.replace("{impl: pool", "{impl: run-engine-elsewhere").replace(
    "name: pool_chain", "name: elsewhere_chain"
)

#: Its sibling (:class:`DetectHere`), identical but for one attribute: it resolves that same
#: name against this process's pool, so this chain does need one.
HERE_CHAIN = POOL_CHAIN.replace("{impl: pool", "{impl: run-engine-here").replace(
    "name: pool_chain", "name: here_chain"
)

#: A chain that needs image ops and *no* model pool (:class:`DetectWithOpsAndNoPool`) -- the
#: shape of the first `crop` element, and the only combination the shipped set cannot make.
OPS_ONLY_CHAIN = POOL_CHAIN.replace(
    "{impl: pool, model: detector, params: {input: images}}", "{impl: run-engine-ops-only}"
).replace("name: pool_chain", "name: ops_only_chain")

#: What happened, in order, across the engine double and the runner double. One list rather
#: than a counter per object because the failure this file exists to prevent is an *ordering*
#: one -- a pool started after the chain opened against it, or stopped while a worker was
#: still submitting -- and neither shows up in a call count.
EVENTS: list[str] = []


class RecordingEngine:
    """Stands in for :class:`~shipinfer.engine.InferenceServer`, recording its lifecycle.

    A double rather than the real thing for the ordering tests: an ``InferenceServer`` scans a
    repository and starts worker threads, and none of that is what is under test here. It
    satisfies ``ModelResolver`` structurally (one ``get``), which is all a runner is promised.
    """

    instances: ClassVar[list[RecordingEngine]] = []
    #: What ``start()`` raises, or ``None``. A class attribute so a test can arm it before
    #: ``run`` constructs the instance it will fail on.
    fail_to_start: ClassVar[Exception | None] = None

    def __init__(self, settings: ServerSettings | None = None) -> None:
        self.settings = settings
        self.started = False
        self.stopped = False
        RecordingEngine.instances.append(self)

    def start(self) -> RecordingEngine:
        EVENTS.append("engine.start")
        if RecordingEngine.fail_to_start is not None:
            raise RecordingEngine.fail_to_start
        self.started = True
        return self

    def stop(self) -> None:
        EVENTS.append("engine.stop")
        self.stopped = True

    def get(self, name: str) -> str:  # pragma: no cover - no element opens in these tests
        return name


class FourGpuEngine(RecordingEngine):
    """A ``RecordingEngine`` that reports four GPUs, so the ops spread has something to spread.

    Two attributes deep on purpose: ``get_thread_local_image_ops`` reads ``visible_gpus`` and
    ``has_accelerator`` off the engine's ``DeviceManager`` and nothing else, which is why the
    command can ask a real engine for its device list without building a second one. Reporting
    ``has_accelerator`` as ``False`` while naming four devices is not a contradiction here: it
    is what keeps this test offline, because the truthy branch calls ``bind_current_thread``.
    """

    class _Devices:
        visible_gpus: ClassVar[tuple[int, ...]] = (0, 1, 2, 3)
        has_accelerator: ClassVar[bool] = False

    def __init__(self, settings: ServerSettings | None = None) -> None:
        super().__init__(settings)
        self.devices = FourGpuEngine._Devices()
        self.memory = None


class RefusingEngine:
    """An ``InferenceServer`` that fails the test if anybody builds one.

    The only honest way to assert "no engine was built": a spy that records a construction
    still lets the run continue, so a regression would have to be caught by an assertion at
    the end, and the run would meanwhile have loaded a repository that was never meant to be
    read. This fails on the line that made the mistake.
    """

    def __init__(self, settings: ServerSettings | None = None) -> None:
        raise AssertionError("an engine was built for a run that needs none")


@registry_for(ElementKind.DETECT).register("run-engine-elsewhere")
class DetectElsewhere(MockDetect):
    """A detect that names a ``model:`` and resolves it somewhere that is not this process.

    The shape of the ``nvinfer`` element the deepstream runner will register: the graph
    compiler hands the artefact's name to GStreamer, so an operator must write it in the chain
    file (``requires_model_name``) and nothing here ever asks a model pool for it
    (``needs_model``). Those are the two questions, and this is the only element shape at which
    they disagree -- every implementation the package ships answers both the same way, so
    without a double the reader in :func:`~shipinfer.cli.commands.run.model_pool_is_needed`
    can be pointed at either attribute and the suite stays green.

    Registered here rather than in ``topology/elements/``, and under a name prefixed with this
    file's, for the reason ``tests/topology/test_model_requirement.py`` gives: the registries
    are process-wide, so a double that lives beside the test that uses it cannot make some
    other file's result depend on collection order.
    """

    requires_model_name: ClassVar[bool] = True
    needs_model: ClassVar[bool] = False


@registry_for(ElementKind.DETECT).register("run-engine-here")
class DetectHere(DetectElsewhere):
    """The same element with the second answer flipped: it *does* run its model here.

    One attribute apart from :class:`DetectElsewhere`, which is what makes the pair a
    measurement rather than an assertion: a predicate hard-wired to ``False`` would pass its
    test and fail this one.
    """

    requires_model_name: ClassVar[bool] = True
    needs_model: ClassVar[bool] = True

    def _do_open(self, context: ElementContext) -> None:
        """Refuse a context with no pool, which is what declaring ``needs_model`` promises.

        Not reached by these tests -- the recording runner never opens its chain -- but a
        double that claimed to need a pool and then ran happily without one would be a double
        of nothing.
        """
        if context.models is None:
            raise ConfigurationError(f"{self.name!r} was opened with no model pool")


@registry_for(ElementKind.DETECT).register("run-engine-ops-only")
class DetectWithOpsAndNoPool(MockDetect):
    """An element that pre-processes and runs no model in this process.

    The one combination no shipped element makes today, and the one the comment in
    ``run.py`` promises is handled: ``needs_image_ops`` without ``needs_model``. Phase C's
    ``crop`` element is it -- it letterboxes or crops for something downstream and asks this
    process's pool for nothing -- so the wiring it will land on is pinned here before it
    exists rather than after it is found to be wrong.
    """

    needs_image_ops: ClassVar[bool] = True
    needs_model: ClassVar[bool] = False


@RUNNERS.register("recording-pool")
class RecordingRunner(Runner):
    """A runner that opens its chain here — so a caller must hand it a model pool.

    Registered process-wide, which is the bargain ``tests/cli/test_run_inputs.py`` already
    strikes for ``still``: ``run()`` resolves the name through ``RUNNERS``, and
    ``needs_model_pool`` is read off the *class* before anything is built, so a double has to
    be in the registry to be reachable at all.
    """

    name: ClassVar[str] = "recording-pool"
    manages_cameras: ClassVar[bool] = True
    needs_model_pool: ClassVar[bool] = True

    instances: ClassVar[list[RecordingRunner]] = []
    #: What ``_do_start``/``_do_stop`` raise, or ``None``. Class attributes for
    #: :class:`RecordingEngine`'s reason -- ``run()`` builds the instance, so a test can only
    #: arm the failure before it exists.
    fail_to_start: ClassVar[BaseException | None] = None
    fail_to_stop: ClassVar[BaseException | None] = None

    def __init__(
        self, topology: Topology, settings: ServerSettings | None = None, **kwargs: Any
    ):
        super().__init__(topology, settings, **kwargs)
        RecordingRunner.instances.append(self)

    def _do_start(self) -> None:
        EVENTS.append("runner.start")
        if RecordingRunner.fail_to_start is not None:
            raise RecordingRunner.fail_to_start

    def _do_stop(self, timeout_s: float) -> None:
        EVENTS.append("runner.stop")
        if RecordingRunner.fail_to_stop is not None:
            raise RecordingRunner.fail_to_stop

    def _do_submit(self, item: ChainItem) -> ResponseFuture:  # pragma: no cover - unused
        raise NotImplementedError


@RUNNERS.register("recording-pool-cameraless")
class CameralessPoolRunner(RecordingRunner):
    """A runner that opens its chain here and owns no ingest plane.

    The two ``ClassVar``s are independent, and this is the combination that makes the order of
    the two questions observable: a pool would be built for it, and the cameras it was given
    have to be refused *before* that happens. No production runner is shaped this way yet --
    ``inprocess`` is the only ``needs_model_pool`` runner and it manages cameras too -- which
    is why the case exists as a double rather than as a name from the registry.
    """

    name: ClassVar[str] = "recording-pool-cameraless"
    manages_cameras: ClassVar[bool] = False
    needs_model_pool: ClassVar[bool] = True


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    """Clear the recorders, and let the command past the container gate.

    The gate is real and stays real (``runtime/containment.py``); what these tests exercise is
    the wiring one layer above it, and every other ``run()`` test that gets past ``--dry-run``
    stubs it the same way.
    """
    EVENTS.clear()
    RecordingEngine.instances.clear()
    RecordingEngine.fail_to_start = None
    RecordingRunner.instances.clear()
    RecordingRunner.fail_to_start = None
    RecordingRunner.fail_to_stop = None
    monkeypatch.setattr("shipinfer.runtime.containment.require_container", lambda *a, **k: None)
    monkeypatch.setattr(run_module, "_wait", lambda built, **kwargs: EVENTS.append("wait"))
    yield
    RecordingEngine.fail_to_start = None
    RecordingRunner.fail_to_start = None
    RecordingRunner.fail_to_stop = None


def write_chain(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "chain.yaml"
    path.write_text(text)
    return path


@pytest.fixture()
def pool_chain_file(tmp_path: Path) -> Path:
    return write_chain(tmp_path, POOL_CHAIN)


@pytest.fixture()
def detector_repository(tmp_path: Path) -> Path:
    """A one-model repository whose model can actually receive a letterboxed frame.

    Local to this file rather than a third model in the shared ``tmp_repository`` fixture: that
    one is a *two-model* repository by contract (``tests/engine/test_server.py`` asserts
    ``server.models() == ["echo", "slow"]`` and a three-instance health report), and its
    ``echo`` declares ``x[4]`` because the engine tests submit ``(1, 4)`` tensors to it. Both
    are right for what they are; neither is a detector.

    ``platform: mock`` on a ``KIND_CPU`` group, so :class:`~shipinfer.engine.InferenceServer`
    starts it with no accelerator. The dims are the artefact's statement of its own geometry --
    ``PoolDetect`` resolves its letterbox target from them, which is what a real deployment
    does -- and ``8x8`` only to keep the fixture small.
    """
    root = tmp_path / "detector_repository"
    (root / "detector" / "1").mkdir(parents=True)
    (root / "detector" / "config.yaml").write_text(textwrap.dedent("""
        name: detector
        platform: mock
        max_batch_size: 4
        inputs:
          - {name: images, data_type: FP32, dims: [3, 8, 8]}
        outputs:
          - {name: output0, data_type: FP32, dims: [300, 6]}
        instance_groups:
          - {kind: KIND_CPU, count: 1}
        """).lstrip())
    return root


@pytest.fixture()
def mock_chain_file(tmp_path: Path) -> Path:
    return write_chain(tmp_path, MOCK_CHAIN)


@pytest.fixture()
def ops_only_chain_file(tmp_path: Path) -> Path:
    return write_chain(tmp_path, OPS_ONLY_CHAIN)


def topology_of(text: str) -> Topology:
    return Topology.from_spec(ChainSpec.from_yaml(text))


class TestWhoNeedsAPool:
    """The two declarations the decision is made from, asked directly.

    Both are class attributes rather than a check against ``"inprocess"`` or ``"pool"``, so
    the next runner and the next element answer for themselves (CONVENTIONS 2.3).
    """

    def test_a_chain_with_a_pool_element_on_a_runner_that_opens_it_here(self) -> None:
        assert model_pool_is_needed("inprocess", topology_of(POOL_CHAIN)) is True

    def test_a_chain_of_mocks_needs_none_although_its_kinds_name_models(self) -> None:
        """`detect` is a model kind and names `model: echo`; `impl: mock` still runs none."""
        assert model_pool_is_needed("inprocess", topology_of(MOCK_CHAIN)) is False

    def test_the_fleet_needs_none_because_each_shard_builds_its_own(self) -> None:
        assert model_pool_is_needed("fleet", topology_of(POOL_CHAIN)) is False

    def test_an_unknown_runner_is_refused_by_the_registry_that_would_build_it(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown runner"):
            model_pool_is_needed("nope", topology_of(POOL_CHAIN))


class TestAChainThatNamesAModelItRunsElsewhere:
    """The half of the split ``model_pool_is_needed`` is the only reader of.

    Every element the package ships answers ``requires_model_name`` and ``needs_model``
    identically, so the two attributes are indistinguishable from inside the shipped set: the
    predicate below reads ``needs_model``, and swapping it to ``requires_model_name`` left the
    whole of ``tests/cli``, ``tests/topology`` and ``tests/runners`` green. :class:`DetectElsewhere`
    is the chain that tells them apart, and it is not hypothetical -- it is the deepstream
    chain phase C is being built towards, on which the wrong attribute means a launcher that
    loads the model repository onto every visible device to serve a chain that never submits
    to it.
    """

    def test_it_loads_because_it_names_the_model_the_chain_file_must_carry(self) -> None:
        """First the loader's half, so the second assertion is about a chain that exists."""
        chain = topology_of(ELSEWHERE_CHAIN)

        element = chain.node("detect").element
        assert element.requires_model_name is True
        assert (
            element.model == "detector"
        ), "the name an operator writes is carried, not dropped"

    def test_it_needs_no_pool_although_it_names_a_model(self) -> None:
        assert model_pool_is_needed("inprocess", topology_of(ELSEWHERE_CHAIN)) is False

    def test_the_sibling_that_resolves_that_name_here_does_need_one(self) -> None:
        """One attribute apart from the case above, and the answer flips."""
        assert model_pool_is_needed("inprocess", topology_of(HERE_CHAIN)) is True

    def test_no_engine_is_built_for_it(self, tmp_path: Path, monkeypatch) -> None:
        """The predicate's consequence, through ``run`` rather than through the helper.

        ``RefusingEngine`` fails on the constructor, so a regression is reported at the line
        that made the mistake instead of after a repository has been read.
        """
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RefusingEngine)

        assert run(write_chain(tmp_path, ELSEWHERE_CHAIN), runner="recording-pool") == 0

        assert EVENTS == ["runner.start", "wait", "runner.stop"]
        (built,) = RecordingRunner.instances
        assert built.models is None, "a chain whose model runs elsewhere was handed a pool"


class TestWhoNeedsImageOps:
    """The same shape one dependency over, and the same two declarations.

    ``get_image_ops`` is not free: under a non-``AUTO`` provider it constructs a torch
    implementation bound to a device. So a chain of mocks must resolve none, and a fleet
    launcher must resolve none — its shards each resolve one bound to their own GPU, and one
    resolved here would be bound to a device this process does not own.
    """

    def test_a_chain_with_a_pool_detector_on_a_runner_that_opens_it_here(self) -> None:
        assert image_ops_are_needed("inprocess", topology_of(POOL_CHAIN)) is True

    def test_a_chain_of_mocks_needs_none_although_its_kinds_are_the_same(self) -> None:
        assert image_ops_are_needed("inprocess", topology_of(MOCK_CHAIN)) is False

    def test_a_pool_chain_that_reads_no_pixels_needs_none(self) -> None:
        """The half that would be lost by reusing ``needs_model``: a ``pool`` element that
        forwards its payload submits a tensor somebody else shaped, so it needs the pool and
        no pre-processing at all.

        ``segment`` and not ``embed``: since C8 an embedder cuts one crop per detection out of
        the source frame and therefore *does* need ops
        (``tests/topology/test_pool_embed_crops.py``), which makes it the wrong witness for
        "needs the pool, needs no ops" — but not for the property, which is that the two
        declarations are independent.
        """
        segment_only = POOL_CHAIN.replace(
            "detect: {impl: pool, model: detector, params: {input: images}}",
            "segment: {impl: pool, model: detector}",
        ).replace("name: pool_chain", "name: segment_chain")

        assert model_pool_is_needed("inprocess", topology_of(segment_only)) is True
        assert image_ops_are_needed("inprocess", topology_of(segment_only)) is False

    def test_the_fleet_needs_none_because_each_shard_resolves_its_own(self) -> None:
        assert image_ops_are_needed("fleet", topology_of(POOL_CHAIN)) is False

    def test_an_unknown_runner_is_refused_by_the_registry_that_would_build_it(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown runner"):
            image_ops_are_needed("nope", topology_of(POOL_CHAIN))


class TestTheTwoPredicatesAreOneFunctionOverATable:
    """``models=`` and ``ops=`` are the same two questions about different attributes.

    They were two copies of four lines with one attribute name changed, and phase D adds a
    third dependency (a DataPool) — at which point the copies would be the design. The named
    predicates stay, because they are what the call site reads like; what they share is a
    table, so a new dependency is a row rather than a function.
    """

    def test_both_names_are_the_same_function_over_their_own_row(self) -> None:
        chain = topology_of(POOL_CHAIN)

        assert model_pool_is_needed("inprocess", chain) == dependency_is_needed(
            "models", "inprocess", chain
        )
        assert image_ops_are_needed("inprocess", chain) == dependency_is_needed(
            "ops", "inprocess", chain
        )

    def test_the_table_names_the_keyword_and_the_element_attribute(self) -> None:
        """The two ends the row ties together: the ``build_runner`` keyword an operator never
        sees, and the declaration an element makes. Both are read by name elsewhere, so a
        typo in either is silent -- ``getattr`` on a missing attribute would be an
        ``AttributeError`` per element and a missing keyword would simply never be built."""
        from shipinfer.runners.base import Runner
        from shipinfer.topology.base import Element

        for keyword, attribute in run_module._HANDED_IN.items():
            assert hasattr(Element, attribute), f"no element declares {attribute!r}"
            assert keyword in inspect.signature(Runner.__init__).parameters

    def test_a_runner_that_runs_the_chain_elsewhere_needs_no_row_at_all(self) -> None:
        """The first of the two questions short-circuits the second, for every row."""
        chain = topology_of(POOL_CHAIN)

        assert [dependency_is_needed(key, "fleet", chain) for key in run_module._HANDED_IN] == [
            False,
            False,
        ]


class TestThePoolIsUpBeforeTheChainOpensAndDownAfterItStops:
    def test_the_engine_brackets_the_runner(self, pool_chain_file: Path, monkeypatch) -> None:
        """The whole contract in one sequence.

        `built.start()` is what calls `open()` on every element, and a `pool` element resolves
        its model there -- so a pool started after it is a pool that was never there. On the
        way down the order is reversed for the mirror reason: a worker still walking a frame
        submits into the pool, and stopping the pool first would take the models out from
        under a request that is already in flight.
        """
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        assert run(pool_chain_file, runner="recording-pool") == 0

        assert EVENTS == [
            "engine.start",
            "runner.start",
            "wait",
            "runner.stop",
            "engine.stop",
        ]

    def test_the_runner_is_handed_the_engine_it_started(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """`models=` is the whole fix: without it `ElementContext.models` is `None` and the
        first `pool` element refuses at `open()`."""
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        run(pool_chain_file, runner="recording-pool")

        (engine,) = RecordingEngine.instances
        (built,) = RecordingRunner.instances
        assert built.models is engine, "the runner was built with no model pool"
        assert engine.started and engine.stopped

    def test_the_element_context_carries_the_pool_to_every_element(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """One step further than the constructor: what an element is actually opened with."""
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        run(pool_chain_file, runner="recording-pool")

        (built,) = RecordingRunner.instances
        assert built.element_context().models is RecordingEngine.instances[0]

    def test_the_runner_is_handed_image_ops_and_puts_them_on_the_context(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """`ops=` is the C3 half of the same fix, and it goes all the way to the element.

        Asserting the *context* rather than only the constructor keyword, because that is what
        `PoolDetect._do_open` reads; a runner that accepted the keyword and dropped it on the
        floor would pass the constructor assertion and refuse every real chain at `open()`.
        """
        from shipinfer.runtime.ops import ThreadLocalImageOps

        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        run(pool_chain_file, runner="recording-pool")

        (built,) = RecordingRunner.instances
        assert built.ops is not None, "a chain with a `pool` detector was handed no image ops"
        assert isinstance(built.ops, ThreadLocalImageOps)
        assert built.element_context().ops is built.ops


class TestEveryWorkerThreadGetsItsOwnImageOps:
    """The blocking half of the same wiring: *one instance per thread*, not per process.

    `pipeline.workers` threads walk one chain over one shared `PoolDetect`
    (`runners/inprocess.py`), and every implementation `get_image_ops` can return is per-thread
    by contract -- `NativeImageOps` keeps a staging ring inside the extension, `TorchImageOps`
    binds a device on the constructing thread and caches an event and a ping-pong staging pair
    on the instance. Handing one object to four workers is CONVENTIONS 2.8's buffer overwritten
    mid-DMA: plausible pixels, no error, and nothing an offline suite can see, because
    `NumpyImageOps` is stateless. So the property is asserted of the *decorator* the command
    hands over, which is checkable with no driver at all.

    The second test is the other half of the same defect: a single-process run on an 8-GPU box
    letterboxed every camera on `cuda:0`, which is this project's founding bug one layer up.
    """

    def test_four_threads_through_the_handed_in_ops_get_four_delegates(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        run(pool_chain_file, runner="recording-pool")

        (built,) = RecordingRunner.instances
        # The *objects*, not their ids: a delegate is dropped when its thread exits and
        # CPython hands the next one the same address, which makes an id-only set flaky.
        delegates: list[object] = []
        lock = threading.Lock()

        def touch() -> None:
            delegate = built.ops._ops
            with lock:
                delegates.append(delegate)

        threads = [threading.Thread(target=touch) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)

        assert len(delegates) == 4
        assert len({id(d) for d in delegates}) == 4, "four workers shared one ImageOps"

    def test_the_workers_are_spread_over_the_engine_s_devices(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """Eight threads over the four GPUs the engine reports, two each.

        The device list is read off the engine's `DeviceManager` and not resolved by building
        one here: `DeviceManager.__init__` validates every visible device, which costs a CUDA
        primary context per GPU that this process never gives back. `FourGpuEngine` is that
        manager's shape and nothing more -- `visible_gpus` plus `has_accelerator`, which is all
        `get_thread_local_image_ops` reads.
        """
        monkeypatch.setattr("shipinfer.engine.InferenceServer", FourGpuEngine)

        run(pool_chain_file, runner="recording-pool")

        (built,) = RecordingRunner.instances

        def touch() -> None:
            built.ops.on_device  # noqa: B018 - first touch assigns the device

        threads = [threading.Thread(target=touch) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)

        assert built.ops.assignments() == {0: 2, 1: 2, 2: 2, 3: 2}


class TestAChainThatNeedsOpsAndNoPool:
    """The two dependencies coming apart in the direction nothing ships yet.

    `dependency_is_needed` asks its two questions separately precisely so this chain is
    possible, and it is the shape of phase C's `crop` element. What it gets today is recorded
    here rather than assumed, because the recording is the honest half: with no engine there
    is no `DeviceManager`, so `get_thread_local_image_ops` binds no thread (ADR-002) and
    claims no pinned pool, and the device list comes from what the operator pinned.

    That is safe -- `TorchImageOps` and `NativeImageOps` are each constructed *with* their
    device index and act on it rather than on the ambient current device -- and it is not the
    full arrangement. Building a `DeviceManager` here to close the gap is the trade `run.py`
    refuses on purpose (a CUDA primary context per visible GPU, ~200 MiB each, that nothing in
    this process gives back), so the gap is documented at both ends and pinned here.
    """

    def test_it_resolves_ops_and_builds_no_engine_at_all(
        self, ops_only_chain_file: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RefusingEngine)
        handed = object()
        recorded: dict[str, Any] = {}

        def fake_ops(provider, **kwargs: Any) -> object:
            recorded.update(kwargs, provider=provider)
            return handed

        monkeypatch.setattr("shipinfer.runtime.ops.get_thread_local_image_ops", fake_ops)

        # `--gpus 0,1` so the fallback has something to fall back to, and so `_fill_in_gpus`
        # asks no driver: the whole point is that this path resolves a device list without an
        # engine to read one off.
        assert run(ops_only_chain_file, runner="recording-pool", gpus="0,1") == 0

        (built,) = RecordingRunner.instances
        assert built.models is None, "a chain that runs no model here was handed a pool"
        assert built.ops is handed
        assert recorded["devices"] == (0, 1), "the operator's `devices.visible_gpus`"
        assert recorded["device_manager"] is None, (
            "there is no engine, so there is no DeviceManager -- no thread is bound and no "
            "staging pool is claimed. Documented at both ends; change this and change those"
        )
        assert recorded["memory"] is None


class TestWhenNoPoolIsNeededNoneIsBuilt:
    """Three runs that must touch no repository, no device and no engine.

    Costly in the wrong direction if this regresses: an `InferenceServer` resolves the device
    list at construction and loads every selected model at `start()`, so a chain of mocks that
    quietly built one turns a laptop test run into an engine load.
    """

    def test_a_chain_of_mocks_builds_no_engine(
        self, mock_chain_file: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RefusingEngine)

        assert run(mock_chain_file, runner="recording-pool") == 0

        assert EVENTS == ["runner.start", "wait", "runner.stop"]
        (built,) = RecordingRunner.instances
        assert built.models is None, "a mock chain was handed a model pool"
        assert built.ops is None, "a mock chain was handed image ops"
        assert built.element_context().ops is None

    def test_a_dry_run_builds_no_engine_even_for_a_pool_chain(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """`--dry-run` resolves everything and spawns nothing; loading a repository to print a
        plan would be the same broken promise as spawning a shard to print one."""
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RefusingEngine)

        assert run(pool_chain_file, runner="recording-pool", dry_run=True) == 0

        assert EVENTS == [], "a dry run started something"

    def test_the_fleet_launcher_builds_no_engine(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """The launcher is the one process in the deployment that must hold no CUDA context:
        its shards each build their own engine (`cli/shard.py`), and one here would be a
        second copy of every model, loaded by a process that runs no inference.

        `_do_start`/`_do_stop` are stubbed because the point is what the *launcher* builds, not
        that sixteen shard processes can be spawned inside the offline tier.
        """
        from shipinfer.runners.fleet import FleetRunner

        monkeypatch.setattr("shipinfer.engine.InferenceServer", RefusingEngine)
        monkeypatch.setattr(FleetRunner, "_do_start", lambda self: EVENTS.append("fleet.start"))
        monkeypatch.setattr(
            FleetRunner, "_do_stop", lambda self, timeout_s: EVENTS.append("fleet.stop")
        )

        assert run(pool_chain_file, runner="fleet", gpus="0") == 0

        assert EVENTS == ["fleet.start", "wait", "fleet.stop"]


class TestWhenThePoolWillNotStart:
    @pytest.mark.parametrize(
        "failure",
        [
            ServerStateError("model 'echo' has no live instance"),
            ConfigurationError("visible_gpus names device(s) [7] but torch reports none"),
        ],
        ids=["server-state", "configuration"],
    )
    def test_the_typed_refusal_escapes_and_nothing_was_started(
        self, pool_chain_file: Path, monkeypatch, failure: Exception
    ) -> None:
        """Unwrapped, because the engine's own message names the model that would not load.

        It escapes ``run()`` the way every other refusal in this command does -- the CLI never
        constructs a ``typer.Exit`` for a call that raised, and click's standalone mode does
        not swallow a ``ShipInferError``, so the process exits non-zero rather than reporting a
        healthy deployment. What must not happen is a deployment that came up around a pool
        that did not.
        """
        RecordingEngine.fail_to_start = failure
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        with pytest.raises(type(failure), match=str(failure)[:20]):
            run(pool_chain_file, runner="recording-pool")

        assert EVENTS == ["engine.start", "engine.stop"], (
            "either something came up around a pool that had not, or the failed start was "
            "left holding whatever it had taken"
        )
        assert not RecordingRunner.instances[0].is_running

    def test_the_pool_is_given_back_and_the_runner_is_not_stopped(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """What matters is that the GPU goes back, once, and that nothing else is touched.

        This used to assert ``"engine.stop" not in EVENTS`` -- "it stops nothing it did not
        start" -- and that is the wrong property to pin. ``InferenceServer.stop`` is
        documented safe on a server whose ``start`` raised half-way (``engine/pool.py``), and
        a failed start is exactly the case where something *may* still be held: a strict start
        that fails on model 3 of 5 leaves two loaded. Forbidding the stop forbade the only
        thing that could give those back, so the assertion is now the honest one -- the engine
        is stopped exactly once, and the runner, which never started, is not stopped at all.
        """
        RecordingEngine.fail_to_start = ServerStateError("no")
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        with pytest.raises(ServerStateError):
            run(pool_chain_file, runner="recording-pool")

        assert EVENTS == ["engine.start", "engine.stop"], "the engine was stopped twice, or not"
        assert not RecordingRunner.instances[0].is_running


#: What an arming function returns: the keyword arguments ``run()`` needs for its case.
#: A callable rather than a table of settings because each refusal is armed differently --
#: one patches an import, one sets an environment variable, one changes the ``--runner``.
Arm = Callable[[pytest.MonkeyPatch, Path], dict[str, Any]]


def arm_a_host_without_the_server_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    """``--http`` where ``pip install "shipinfer[server]"`` was never run.

    The one an operator actually hits. Asked inside ``_wait`` -- after ``built.start()`` --
    it cost a fleet sixteen spawned shards and a full shutdown to learn it.
    """
    monkeypatch.setattr(
        "shipinfer.api.require_server_extra",
        lambda: (_ for _ in ()).throw(ConfigurationError("install shipinfer[server]")),
    )
    return {"http": True}


def arm_an_unreadable_camera_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    """``ingest.camera_db`` naming a file that is not there.

    ``cameras_to_place`` reads it off the disk (``ingest/camera/db.py`` raises for a missing
    file, unparseable JSON or a duplicate id), and it is a fact about the settings tree that
    needs no accelerator to establish. The environment variable is how an operator sets it and
    the only way in through ``run()``'s signature, which takes no ingest keyword.
    """
    monkeypatch.setenv("SHIPINFER_INGEST__CAMERA_DB", str(tmp_path / "no-such-fleet.json"))
    return {}


def arm_a_runner_that_manages_no_cameras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    """Cameras on a runner that owns no ingest plane, over a chain that wants a pool.

    ``refuse_if_it_manages_no_cameras`` reads two ``ClassVar``s off the registered class --
    ``manages_cameras`` and ``name`` -- so it needs no runner instance, and therefore no
    engine either. It is the combination that makes the ordering matter:
    :class:`CameralessPoolRunner` is the one runner in the registry that would build a pool
    *and* refuse the cameras, which in production is nothing today and is exactly the shape a
    service-style runner would have.
    """
    return {"runner": CameralessPoolRunner.name, "inputs": ["a.mp4"]}


class TestTheGpuGoesBackWhateverFails:
    """One guard over the whole bring-up, because every line of it can leak an engine.

    The window this class exists for is not a race: it spans the whole of ``built.start()``,
    which for a real chain opens decoders, sockets and a thread pool while an
    :class:`~shipinfer.engine.InferenceServer` is already up. An exception there used to leave
    that engine at ``_started = True`` with its worker threads alive and *nothing holding a
    reference to it* -- the unreachable-object-holding-CUDA-contexts failure that
    ``InferenceServer.start`` and ``cli/shard.py`` each spend a paragraph refusing in their own
    scope, on a box CLAUDE.md's hygiene rule says is shared.
    """

    def test_a_runner_that_will_not_start_gives_the_pool_back(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """The blocking case, in the order it happens.

        ``runner.stop`` between the two is ``Runner.start``'s own unwind, which closes the
        elements it had opened before re-raising; the engine's stop comes *after* it, which is
        the same reverse order the successful path uses -- a half-open chain may still be
        holding a model handle when it closes.
        """
        RecordingRunner.fail_to_start = RuntimeError("element 'detect' failed to open")
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        with pytest.raises(RuntimeError, match="failed to open"):
            run(pool_chain_file, runner="recording-pool")

        assert EVENTS == ["engine.start", "runner.start", "runner.stop", "engine.stop"]
        (engine,) = RecordingEngine.instances
        assert engine.stopped, "an engine was left running with nothing holding it"

    def test_a_keyboard_interrupt_while_the_chain_opens_gives_the_pool_back(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """``BaseException``, not ``Exception``, and this is why.

        A Ctrl-C during a chain that is opening decoders is the likeliest way this path is
        taken by hand, and :func:`~shipinfer.launch.signals.forward_signals` is not installed
        until ``_wait`` -- two lines further on. It must reach the operator, so the guard
        re-raises rather than converting it to an exit code.
        """
        RecordingRunner.fail_to_start = KeyboardInterrupt()
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        with pytest.raises(KeyboardInterrupt):
            run(pool_chain_file, runner="recording-pool")

        # The whole sequence, not just its last entry: `EVENTS[-1] == "engine.stop"` also
        # holds when the guard stops the engine *twice*, which is the one property this class
        # exists to pin. `runner.stop` is there for the sibling test's reason -- `Runner.start`
        # catches `BaseException` and unwinds under `contextlib.suppress` (`runners/base.py`),
        # so a `KeyboardInterrupt` closes the elements it had opened exactly as a
        # `RuntimeError` does.
        assert EVENTS == ["engine.start", "runner.start", "runner.stop", "engine.stop"]
        assert RecordingEngine.instances[0].stopped

    def test_a_runner_whose_stop_raises_still_gives_the_pool_back(
        self, pool_chain_file: Path, monkeypatch
    ) -> None:
        """The nested-stop guard on the shutdown path, which no test covered.

        ``built.stop()`` and ``engine.stop()`` are nested (``try``/``finally``) rather than
        written as two statements precisely so that a runner whose stop raises still frees the
        device: a crash must not be what frees it. Un-nesting them turns this test red and
        nothing else.
        """
        RecordingRunner.fail_to_stop = RuntimeError("a decoder thread would not join")
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RecordingEngine)

        with pytest.raises(RuntimeError, match="would not join"):
            run(pool_chain_file, runner="recording-pool")

        assert EVENTS == ["engine.start", "runner.start", "wait", "runner.stop", "engine.stop"]
        assert RecordingEngine.instances[0].stopped

    @pytest.mark.parametrize(
        ("arm", "message"),
        [
            (arm_a_host_without_the_server_extra, r"shipinfer\[server\]"),
            (arm_an_unreadable_camera_db, "does not exist"),
            (arm_a_runner_that_manages_no_cameras, "manages no cameras"),
        ],
        ids=["http-without-the-extra", "unreadable-camera-db", "runner-manages-no-cameras"],
    )
    def test_a_refusal_the_command_can_make_without_a_gpu_builds_no_engine(
        self, pool_chain_file: Path, tmp_path: Path, monkeypatch, arm: Arm, message: str
    ) -> None:
        """The cheap half: construction is not free, so it happens after the cheap refusals.

        ``InferenceServer.__init__`` validates every visible device, which takes one CUDA
        primary context per GPU (~200 MiB each on this box), and **nothing gives those back
        inside the process** -- ``stop()`` on a server that was never started returns without
        reaching ``_release``. So a refusal that needs no accelerator to make must be made
        before the constructor runs, and the guard above cannot substitute for that ordering.

        One case per refusal that was moved above the constructor, because they are moved
        independently and a revert of any one of them costs the same 200 MiB per device. Each
        is pinned on its own line: with only the ``--http`` case here, ``cameras_to_place``
        could be dropped back below ``InferenceServer(settings)`` and the whole suite stayed
        green.
        """
        monkeypatch.setattr("shipinfer.engine.InferenceServer", RefusingEngine)
        keywords: dict[str, Any] = {"runner": "recording-pool", **arm(monkeypatch, tmp_path)}

        with pytest.raises(ConfigurationError, match=message):
            run(pool_chain_file, **keywords)

        assert EVENTS == [], "something was built for a run that was refused"


class TestOverARealEngine:
    """The ledger item, closed: a `pool` element opens against a pool this command built.

    Everything above this class would pass over a double that forwarded a keyword. This one
    starts a real :class:`~shipinfer.engine.InferenceServer` on the mock backend, runs a real
    :class:`~shipinfer.runners.inprocess.InprocessRunner` over a chain with a `pool` element
    in it, and looks at the element while the chain is open -- which is the thing that used to
    raise ``ConfigurationError`` before the first frame.

    Offline: :func:`detector_repository` declares one ``platform: mock`` model on a
    ``KIND_CPU`` instance group, so no accelerator is touched.
    """

    def test_the_pool_element_resolves_its_model_while_the_chain_is_open(
        self, pool_chain_file: Path, detector_repository: Path, monkeypatch
    ) -> None:
        opened: dict[str, Any] = {}

        def probe(built: Runner, **kwargs: Any) -> None:
            # Read while the chain is up: `run()`'s `finally` closes every element, so after
            # the call there is nothing left to look at.
            element = built.topology.node("detect").element
            opened["is_open"] = element.is_open
            opened["model"] = element.model
            opened["resolved"] = element._handle is not None
            opened["dst_size"] = element._dst_size

        monkeypatch.setattr(run_module, "_wait", probe)

        assert run(pool_chain_file, runner="inprocess", repository=detector_repository) == 0

        # `dst_size` is in here because it is the half a forwarded keyword cannot fake: it
        # came from the artefact's own `dims: [3, 8, 8]`, read off a repository this command
        # loaded, which is the resolution every real deployment relies on.
        assert opened == {
            "is_open": True,
            "model": "detector",
            "resolved": True,
            "dst_size": (8, 8),
        }
