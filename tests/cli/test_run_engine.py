"""``shipinfer run`` builds the model pool its chain needs, and only then.

The defect this file pins: ``run`` built a runner with no ``models=``, so
:class:`~shipinfer.topology.base.ElementContext` carried ``models=None`` and the first
``pool`` element refused at ``open()`` — ``--runner inprocess`` could not execute any real
topology, only a chain of mocks. The fleet path had always been right: a shard builds an
:class:`~shipinfer.engine.InferenceServer`, starts it, and passes it in (``cli/shard.py``).

What is asserted here is the *order*, because that is where the remaining mistakes live: the
pool has to be up before the chain opens against it, and it has to outlive the workers that
are still submitting to it. So the double records rather than mocks, and the tests compare a
whole sequence instead of counting calls.

Everything is offline. The engine is a recording double in every test but the last, which
starts a real ``InferenceServer`` over the mock backend and a two-model repository — the one
that proves a ``pool`` element actually opens, rather than that a keyword was forwarded.
"""

from __future__ import annotations

import importlib
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

from shipinfer.cli.commands.run import model_pool_is_needed, run
from shipinfer.core.errors import ConfigurationError, ServerStateError
from shipinfer.core.request import ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.runners import RUNNERS
from shipinfer.runners.base import Runner
from shipinfer.topology import ChainItem, ChainSpec, Topology

#: The module object, not the ``run`` function ``shipinfer.cli.commands`` re-exports under the
#: same name -- ``monkeypatch.setattr("shipinfer.cli.commands.run._wait", ...)`` resolves the
#: dotted path by attribute and lands on the function.
run_module = importlib.import_module("shipinfer.cli.commands.run")

#: A chain with a `pool` element: the shape of every real topology, and the one that used to
#: fail at `open()`. `echo` is the model the offline repository fixture declares.
POOL_CHAIN = textwrap.dedent("""
    name: pool_chain
    elements:
      decode: {impl: mock}
      detect: {impl: pool, model: echo}
      output: {impl: mock}
    """)

#: The same chain with the mock detector: a model *kind*, naming a model, needing no pool.
#: Asking `node.kind in MODEL_KINDS` instead of the element would load a repository for this.
MOCK_CHAIN = POOL_CHAIN.replace("{impl: pool", "{impl: mock").replace(
    "name: pool_chain", "name: mock_chain"
)

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


class RefusingEngine:
    """An ``InferenceServer`` that fails the test if anybody builds one.

    The only honest way to assert "no engine was built": a spy that records a construction
    still lets the run continue, so a regression would have to be caught by an assertion at
    the end, and the run would meanwhile have loaded a repository that was never meant to be
    read. This fails on the line that made the mistake.
    """

    def __init__(self, settings: ServerSettings | None = None) -> None:
        raise AssertionError("an engine was built for a run that needs none")


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
def mock_chain_file(tmp_path: Path) -> Path:
    return write_chain(tmp_path, MOCK_CHAIN)


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

    Offline: the repository fixture declares two ``platform: mock`` models on ``KIND_CPU``
    instance groups, so no accelerator is touched.
    """

    def test_the_pool_element_resolves_its_model_while_the_chain_is_open(
        self, pool_chain_file: Path, tmp_repository: Path, monkeypatch
    ) -> None:
        opened: dict[str, Any] = {}

        def probe(built: Runner, **kwargs: Any) -> None:
            # Read while the chain is up: `run()`'s `finally` closes every element, so after
            # the call there is nothing left to look at.
            element = built.topology.node("detect").element
            opened["is_open"] = element.is_open
            opened["model"] = element.model
            opened["resolved"] = element._handle is not None

        monkeypatch.setattr(run_module, "_wait", probe)

        assert run(pool_chain_file, runner="inprocess", repository=tmp_repository) == 0

        assert opened == {"is_open": True, "model": "echo", "resolved": True}
