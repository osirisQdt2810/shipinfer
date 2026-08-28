"""What ``stop()`` guarantees when something else is happening at the same time.

Two failures live here, both found by review of the start-unwind work and both invisible to
a suite that only ever stops an idle server.

**A load racing the stop.** ``load_model`` checks that the server is started, then spends
seconds building a backend and starting one worker thread per instance, and only then
publishes the model. ``stop()`` used to take the *table* lock for the microseconds of the
drain and nothing else, so the two interleaved: the drain found an empty table and the load
published a fully started model into it afterwards. The result is a running model — live
threads, a backend, in production a CUDA context per device — on a server that answers
``is_started == False`` and holds no reference anyone could stop it through. That is the
leak ``test_start_unwind.py`` pins for the failed start, arriving by a different door.

**The trace sink after the stop.** ``stop()`` closed the sink and left the field pointing at
it, and ``stats()`` still reads that field. A metrics scrape does not stop when the server
does, so the first scrape after a shutdown asked a closed sink for numbers. The reset that
fixed it then threw the run's totals away, which is its own wrong answer: a dashboard's last
sample read ``recorded: 0`` for a run that traced thousands of requests.

Three more arrived with the review of that work, all the same shape — a caller that has
already passed a check, waiting on the control lock:

* **An unload racing the stop** got ``ModelNotFoundError`` ("no such model") for a model that
  existed when it asked and was drained while it waited. ``load_model`` re-checks under the
  lock; ``unload_model`` did not.
* **A second ``stop()``** returned before any teardown had happened: the flags are cleared
  *before* the lock is taken, so the second thread saw "already stopped" while the first was
  still queued behind a slow load. A caller pairing ``stop()`` with an immediate ``start()``
  then gets the first thread's teardown landing on its fresh server.
* **A stop waiting out a long start.** ``stop()`` queues behind a ``load_model`` that may be
  minutes inside TensorRT, and the fleet drains its shards on a shared deadline — so the
  polite wait is what gets the shard SIGKILLed. ``Model.start`` now polls an abort.

And four more from the review of *that* work, all of them about the ``start()`` side of the
same window:

* **``start()`` took no lifecycle lock at all.** A ``stop()`` concurrent with the *initial*
  ``start()`` cleared the flags, drained the table and closed the sink under a start that
  then published ``_started = True`` regardless — ``is_started`` true with an empty model
  table, which every readiness probe reads as "up and serving nothing".
* **A teardown that outlives its grace period.** ``_await_teardown`` expires and returns; a
  ``start()`` on the other side of it used to re-arm the barrier and get the still-queued
  teardown landing on the fresh server. The start is refused now, and the teardown checks its
  own generation before touching anything.
* **``stats()`` re-read the trace state.** "Is ``_last_trace_stats`` None?" and "load
  ``_traces``" are two bytecodes, and ``_release`` swapping in the null sink between them
  publishes ``{"sink": "none", "recorded": 0}`` for a run that traced thousands of requests —
  the exact answer ``_last_trace_stats`` was added to remove. It also handed the stored dict
  out by reference, so a scraper could edit what every later scrape reported.
* **Non-strict start-up swallowed the abort.** ``strict_startup=false`` logged "failed to
  load model 'm0'; continuing" once per remaining model for one shutdown, and went on
  building models the teardown had already been past.

Offline throughout — the mock backend, ``KIND_CPU`` instances and real worker threads,
because the evidence for the first one is ``threading.enumerate()``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import ServerStateError
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.tracing import NullTraceSink, RequestTrace, TraceSink
from shipinfer.core.types import Tensor
from shipinfer.engine import InferenceServer, pool
from shipinfer.engine.instance import ModelInstance
from shipinfer.engine.model import Model

_MODEL = """
platform: mock
max_batch_size: 4
inputs: [{name: x, data_type: FP32, dims: [2]}]
outputs: [{name: y, data_type: FP32, dims: [2]}]
instance_groups: [{kind: KIND_CPU, count: 2}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.0}
"""

#: How long any thread here may wait for its counterpart. Long enough that a loaded CI box
#: does not flake, short enough that a genuine deadlock fails the run instead of hanging it.
_TIMEOUT = 10.0


@pytest.fixture()
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "echo" / "1").mkdir(parents=True)
    (root / "echo" / "config.yaml").write_text(_MODEL.lstrip())
    return root


def _settings(root: Path) -> ServerSettings:
    """Explicit control, nothing loaded at start-up: the mode ``load_model`` exists for."""
    return ServerSettings(
        model_repository=root,
        devices={"visible_gpus": []},
        execution={"warmup_iterations": 0},
        model_control="explicit",
        load_all_models=False,
        startup_models=[],
    )


def _started(root: Path) -> Iterator[InferenceServer]:
    """A started server that always gets stopped, however the test ends.

    ``stop()`` is idempotent, so a test that stops it itself — most of them do, that being
    the subject — is not stopping it twice by accident here.
    """
    server = InferenceServer(_settings(root)).start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture()
def explicit(repository: Path) -> Iterator[InferenceServer]:
    yield from _started(repository)


def _startup_settings(
    root: Path, *, strict: bool = True, grace: float | None = None
) -> ServerSettings:
    """Every model in the repository loaded by ``start()`` itself.

    The other settings above run under explicit control with nothing loaded at start-up,
    which is what ``load_model`` needs. The races below are between ``stop()`` and the
    *start's own* loading, so the models have to belong to the start.
    """
    settings: dict[str, Any] = {
        "model_repository": root,
        "devices": {"visible_gpus": []},
        "execution": {"warmup_iterations": 0},
        "strict_startup": strict,
        "load_all_models": True,
    }
    if grace is not None:
        settings["shutdown_grace_s"] = grace
    return ServerSettings(**settings)


def _live_workers(model: str) -> list[str]:
    """The instance worker threads of ``model`` that are still alive.

    ``ModelInstance.start`` names its thread ``shipinfer-<model>_<ordinal>_<device>``, so
    this is the direct observation of the leak rather than a proxy for it.
    """
    return [t.name for t in threading.enumerate() if t.name.startswith(f"shipinfer-{model}_")]


def _await_stop_flags(server: InferenceServer) -> None:
    """Block until a ``stop()`` on another thread has cleared both lifecycle flags.

    The point at which the interesting window opens: from here the server reads as stopped
    to everyone, while its teardown is still queued behind whatever holds the control lock.
    Both flags, and polled rather than assumed, because ``_started = False`` and
    ``_starting = False`` are two statements — a test that only watched ``is_started`` would
    sometimes race into the gap between them and exercise a different interleaving while
    still passing.
    """
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if not (server.is_started or server._starting):
            return
        time.sleep(0.001)
    raise AssertionError("stop() never cleared the lifecycle flags")


class _WatchedEvent(threading.Event):
    """A ``threading.Event`` that records the name of every thread that waits on it.

    Installed over a server's ``_torn_down`` barrier so a test can tell that a second
    ``stop()`` really blocked on it. ``time.sleep(0.1)`` "long enough to get there" cannot:
    on a loaded runner the thread is descheduled past that interval, the test releases the
    first stop anyway, and the ordering assertion then holds for a run in which nothing ever
    blocked -- a pass, not a flake, which is worse. Waiting for a name to appear here fails
    on a timeout instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self._waiters_lock = threading.Lock()
        self.waiters: list[str] = []

    def wait(self, timeout: float | None = None) -> bool:
        with self._waiters_lock:
            self.waiters.append(threading.current_thread().name)
        return super().wait(timeout)


def _watch_barrier(server: InferenceServer) -> _WatchedEvent:
    """Swap in a barrier that announces its waiters, preserving whether it was already set."""
    watched = _WatchedEvent()
    if server._torn_down.is_set():
        watched.set()
    server._torn_down = watched
    return watched


def _await_barrier_waiter(barrier: _WatchedEvent, name: str) -> None:
    """Block until thread ``name`` is inside the teardown barrier's ``wait``."""
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if name in barrier.waiters:
            return
        time.sleep(0.001)
    raise AssertionError(f"{name} never reached the teardown barrier")


class _Outcome:
    """What ``load_model`` did on the other thread: a model, or the error it raised."""

    def __init__(self) -> None:
        self.model: Any = None
        self.error: BaseException | None = None

    def run(self, server: InferenceServer, name: str) -> None:
        try:
            self.model = server.load_model(name)
        # BaseException, not Exception: the test asserts on the type either way, and a
        # thread that swallowed nothing would report the failure as a timeout instead.
        except BaseException as exc:
            self.error = exc

    def run_unload(self, server: InferenceServer, name: str) -> None:
        try:
            server.unload_model(name)
        except BaseException as exc:
            self.error = exc

    def run_start(self, server: InferenceServer) -> None:
        """``start()`` on another thread. What it returned lands in ``model`` — the same
        "what came back" slot the loads use, because the assertions are all on ``error``."""
        try:
            self.model = server.start()
        except BaseException as exc:
            self.error = exc


class TestALoadRacingAStop:
    """Either the load wins and ``stop()`` drains it, or it is refused. Never a third state.

    Both interleavings are forced deterministically rather than hammered for, because a
    threaded test that only *sometimes* reproduces reads as coverage while proving nothing —
    the lesson ``TestTheModelTableIsNeverIteratedLive`` in ``test_model_control.py`` writes
    down at length.
    """

    def test_a_load_holding_the_lock_is_drained_by_the_stop_that_waited(
        self, explicit: InferenceServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The load wins the lock: it completes, and the stop that queued behind it drains
        the model it published — worker threads included.

        The block sits *after* ``Model.start`` returns, not before it. Before it is now the
        abort's window (``TestALongStartAbortsWhenTheServerIsStopping`` below): a stop
        arriving while the instances are still coming up refuses the load rather than
        finishing it. What is left here is the narrower race the control lock exists for —
        a model fully started, not yet published, and a drain that has already run.
        """
        inside_start = threading.Event()
        release = threading.Event()
        original_start = Model.start

        def blocking_start(self: Model, **kwargs: Any) -> None:
            # Called from `_build_and_start`, so the control lock is held for the whole of
            # this wait: a `stop()` on another thread cannot get past it until we return.
            original_start(self, **kwargs)
            inside_start.set()
            assert release.wait(_TIMEOUT), "the test never released the model's start"

        monkeypatch.setattr(Model, "start", blocking_start)

        outcome = _Outcome()
        loader = threading.Thread(target=outcome.run, args=(explicit, "echo"), name="loader")
        stopper = threading.Thread(target=explicit.stop, name="stopper")
        loader.start()
        assert inside_start.wait(_TIMEOUT), "the load never reached the model's start"
        stopper.start()
        # Long enough for the stopper to reach the control lock and block on it. The
        # assertions below hold under either interleaving; this only makes the interesting
        # one the one that runs.
        time.sleep(0.1)
        release.set()
        loader.join(_TIMEOUT)
        stopper.join(_TIMEOUT)

        assert not loader.is_alive() and not stopper.is_alive()
        assert outcome.error is None, f"the load failed: {outcome.error!r}"
        assert _live_workers("echo") == [], "the loaded model's threads outlived stop()"
        assert not outcome.model.is_ready
        assert not explicit.is_started
        assert explicit.models() == []

    def test_a_load_that_arrives_after_the_stop_is_refused_typed(
        self, explicit: InferenceServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stop wins: the load has already passed its started check, and the re-check
        under the lock is the only thing that keeps it from building anything."""
        passed_the_check = threading.Event()
        release = threading.Event()
        original_require = InferenceServer._require_control

        def gated_require(self: InferenceServer, action: str, name: str) -> None:
            original_require(self, action, name)
            # The window the fix exists for: started when checked, stopped by the time the
            # control lock is taken.
            passed_the_check.set()
            assert release.wait(_TIMEOUT), "the test never released the load"

        monkeypatch.setattr(InferenceServer, "_require_control", gated_require)

        outcome = _Outcome()
        loader = threading.Thread(target=outcome.run, args=(explicit, "echo"), name="loader")
        loader.start()
        assert passed_the_check.wait(_TIMEOUT), "the load never reached its started check"
        explicit.stop()
        release.set()
        loader.join(_TIMEOUT)

        assert not loader.is_alive()
        assert isinstance(outcome.error, ServerStateError), f"got {outcome.error!r}"
        assert _live_workers("echo") == [], "a refused load still started worker threads"
        assert explicit.models() == []

    def test_an_uncontended_load_and_stop_are_unaffected(
        self, explicit: InferenceServer
    ) -> None:
        """The lock must not have turned the ordinary path into a deadlock: `stop()` also
        runs from `start()`'s failure path and from `__exit__`, neither of which holds it."""
        model = explicit.load_model("echo")
        assert model.is_ready

        explicit.stop()
        explicit.stop()  # still idempotent

        assert _live_workers("echo") == []
        assert not explicit.is_started


class TestAnUnloadRacingAStop:
    """The mirror of ``TestALoadRacingAStop``, and the one that was missing.

    ``unload_model`` passes ``_require_control`` on a started server, then blocks on the
    control lock while ``stop()`` drains the table under it. Without a re-check it reaches
    ``self.model(name)`` and reports ``ModelNotFoundError`` — "no such model", listing the
    now-empty model set — for a model that existed when the operator asked for it. That
    sends them to check their spelling; what happened is that the server stopped.
    """

    def test_it_is_refused_as_a_state_error_not_as_a_missing_model(
        self, explicit: InferenceServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        explicit.load_model("echo")
        passed_the_check = threading.Event()
        release = threading.Event()
        original_require = InferenceServer._require_control

        def gated_require(self: InferenceServer, action: str, name: str) -> None:
            original_require(self, action, name)
            # The window the fix exists for: started when checked, stopped by the time the
            # control lock is taken. Gated on the action so the `load_model` above — which
            # runs before the patch — and the fixture's own teardown are unaffected.
            if action == "unload":
                passed_the_check.set()
                assert release.wait(_TIMEOUT), "the test never released the unload"

        monkeypatch.setattr(InferenceServer, "_require_control", gated_require)

        outcome = _Outcome()
        unloader = threading.Thread(
            target=outcome.run_unload, args=(explicit, "echo"), name="unloader"
        )
        unloader.start()
        assert passed_the_check.wait(_TIMEOUT), "the unload never reached its started check"
        explicit.stop()
        release.set()
        unloader.join(_TIMEOUT)

        assert not unloader.is_alive()
        assert isinstance(outcome.error, ServerStateError), f"got {outcome.error!r}"
        assert "stopped while the request was waiting" in str(outcome.error)
        # The stop still drained it, which is the other half of "this is not a missing model".
        assert _live_workers("echo") == []
        assert explicit.models() == []

    def test_an_unload_on_a_live_server_is_untouched(self, explicit: InferenceServer) -> None:
        """The re-check must not have made the ordinary unload conditional on anything."""
        explicit.load_model("echo")

        explicit.unload_model("echo")

        assert explicit.models() == []
        assert _live_workers("echo") == []
        assert explicit.is_started


class TestASecondStopWaitsForTheFirstOnesTeardown:
    """``stop()`` clears both flags *before* it takes the control lock, so from that moment a
    second ``stop()`` on another thread sees a stopped server — and used to return on that
    alone, while nothing had been released yet. A caller pairing ``stop()`` with an immediate
    ``start()`` would then have the first thread's teardown drain the *new* server's models
    and close its trace sink.
    """

    def test_the_second_stop_does_not_return_before_the_teardown_ran(
        self, explicit: InferenceServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        inside_start = threading.Event()
        release = threading.Event()
        original_start = Model.start
        original_release = InferenceServer._release

        def blocking_start(self: Model, **kwargs: Any) -> None:
            """Holds the control lock, so both stops queue behind it."""
            inside_start.set()
            assert release.wait(_TIMEOUT), "the test never released the model's start"
            original_start(self, **kwargs)

        def recording_release(self: InferenceServer, generation: int) -> None:
            # Patched at `_release`, not `_teardown`: `_teardown`'s own `finally` sets the
            # barrier before it returns, so an append AFTER `_teardown` would race the waiter
            # it releases. Inside `_release` the record lands before the barrier by construction.
            original_release(self, generation)
            order.append("teardown")

        monkeypatch.setattr(Model, "start", blocking_start)
        monkeypatch.setattr(InferenceServer, "_release", recording_release)

        barrier = _watch_barrier(explicit)
        loader = threading.Thread(target=_Outcome().run, args=(explicit, "echo"))
        loader.start()
        assert inside_start.wait(_TIMEOUT), "the load never reached the model's start"

        first = threading.Thread(target=explicit.stop, name="stop-a")
        first.start()
        # The window: the first stop has cleared the flags and is queued on the control lock.
        _await_stop_flags(explicit)

        def second_stop() -> None:
            explicit.stop()
            order.append("second-stop-returned")

        second = threading.Thread(target=second_stop, name="stop-b")
        second.start()
        # Not a sleep: the load is released only once the second stop is provably inside the
        # barrier, so the assertion below cannot hold by the second stop having skipped it.
        _await_barrier_waiter(barrier, "stop-b")
        release.set()
        for thread in (loader, first, second):
            thread.join(_TIMEOUT)
            assert not thread.is_alive(), f"{thread.name} never finished"

        assert order == ["teardown", "second-stop-returned"], order

    def test_a_stop_on_an_already_stopped_server_still_returns_at_once(
        self, explicit: InferenceServer
    ) -> None:
        """The barrier must not have turned idempotence into a grace-period wait: `stop()`
        is called twice by ordinary code (a `finally:` around a context manager), and both
        `cli/shard.py` and the failure path of `start()` do it."""
        explicit.stop()

        began = time.monotonic()
        explicit.stop()
        elapsed = time.monotonic() - began

        assert elapsed < 1.0, f"the second stop() waited {elapsed:.1f}s on a stopped server"

    def test_a_stop_on_a_server_that_was_never_started_returns_at_once(
        self, repository: Path
    ) -> None:
        """Nothing has been torn down, and nothing ever will be — the barrier starts set so
        this is not a `shutdown_grace_s` wait for an event no one will fire."""
        server = InferenceServer(_settings(repository))

        began = time.monotonic()
        server.stop()

        assert time.monotonic() - began < 1.0


class TestALongStartAbortsWhenTheServerIsStopping:
    """A ``load_model`` holding the control lock can be minutes inside TensorRT. ``stop()``
    queues behind it, and the fleet's supervisor drains every shard on one shared deadline —
    so the shard that waits politely is the shard that gets SIGKILLed, in-flight requests
    unresolved. ``Model.start`` polls the server's stopping flag between instances instead.
    """

    def test_the_remaining_instances_are_never_started_and_the_first_is_stopped(
        self, explicit: InferenceServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        attempted: list[str] = []
        original_start = ModelInstance.start

        def gated_start(self: ModelInstance) -> None:
            attempted.append(self.name)
            if len(attempted) == 1:
                # Instance 1 is slow — the TensorRT deserialisation, in miniature.
                entered.set()
                assert release.wait(_TIMEOUT), "the test never released instance 1"
            original_start(self)

        monkeypatch.setattr(ModelInstance, "start", gated_start)

        outcome = _Outcome()
        loader = threading.Thread(target=outcome.run, args=(explicit, "echo"), name="loader")
        loader.start()
        assert entered.wait(_TIMEOUT), "the load never reached the first instance's start"
        stopper = threading.Thread(target=explicit.stop, name="stopper")
        stopper.start()
        _await_stop_flags(explicit)
        release.set()
        loader.join(_TIMEOUT)
        stopper.join(_TIMEOUT)

        assert not loader.is_alive() and not stopper.is_alive()
        # The model declares two KIND_CPU instances; only the first was ever attempted.
        assert len(attempted) == 1, f"the abort did not stop instance 2: {attempted}"
        assert isinstance(outcome.error, ServerStateError), f"got {outcome.error!r}"
        assert "start aborted" in str(outcome.error)
        assert _live_workers("echo") == [], "the aborted start left instance 1 running"
        assert explicit.models() == []

    def test_an_ordinary_load_does_not_trip_the_abort(self, explicit: InferenceServer) -> None:
        """The predicate reads both lifecycle flags; a load on a running server, and the
        models a `start()` itself loads (``_starting`` set, ``_started`` still false), must
        both read as *not* stopping or nothing would ever load."""
        model = explicit.load_model("echo")

        assert model.is_ready
        assert len(_live_workers("echo")) == 2

    def test_a_start_with_no_abort_predicate_is_unchanged(self, repository: Path) -> None:
        """`Model.start()` is called without one by tests and by the ensemble path."""
        with InferenceServer(_settings(repository)) as server:
            server.load_model("echo")
            model = server.model("echo")
            assert model.is_ready


def _await_recorded(sink: _RecordingSink, count: int) -> None:
    """Block until the worker thread has recorded ``count`` traces.

    ``ModelInstance._complete`` resolves the future *before* it records the trace -- on
    purpose, so tracing can never delay the caller's answer -- so ``infer_sync`` returning does
    not mean the trace exists yet. A test that reads the sink's totals straight after the call
    is racing the worker's next few bytecodes; this is the forcing wait, not a sleep.
    """
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if sink.recorded >= count:
            return
        time.sleep(0.001)
    raise AssertionError(f"only {sink.recorded} of {count} traces were recorded")


class _RecordingSink(TraceSink):
    """A sink that remembers being used after it was closed, instead of tolerating it.

    ``TraceSink.record`` already answers ``False`` once closed and ``stats()`` still returns
    numbers, so a closed sink is *quiet* about being used — which is exactly why a server
    holding one went unnoticed. This double is loud.
    """

    name = "recording"

    def __init__(self) -> None:
        super().__init__(rate=1)
        self.traces: list[RequestTrace] = []
        self.uses_after_close: list[str] = []

    def _do_record(self, trace: RequestTrace) -> None:
        if self.is_closed:
            self.uses_after_close.append("record")
        self.traces.append(trace)

    def stats(self) -> dict[str, Any]:
        if self.is_closed:
            self.uses_after_close.append("stats")
        return super().stats()


class TestStopLeavesAUsableTraceSink:
    """A stopped server is still scraped; it must not read from what it just closed."""

    @pytest.fixture()
    def sinks(self, monkeypatch: pytest.MonkeyPatch) -> list[_RecordingSink]:
        """Every sink the server builds, in order.

        Patched rather than registered under a name: this sink is a fixture of the test, not
        of the process, and a registration outlives the test that made it. A list because
        ``start()`` builds one per run and one test below starts the server twice.
        """
        built: list[_RecordingSink] = []

        def build(*_args: object, **_kwargs: object) -> _RecordingSink:
            built.append(_RecordingSink())
            return built[-1]

        monkeypatch.setattr(pool, "build_trace_sink", build)
        return built

    @pytest.fixture()
    def explicit(
        self, sinks: list[_RecordingSink], repository: Path
    ) -> Iterator[InferenceServer]:
        """The module's server, with the recording sink installed *before* it starts.

        Overridden here so `sinks` is built first: `start()` is what asks for the sink, so a
        patch applied after it would arrive too late to be the server's.
        """
        yield from _started(repository)

    def test_stats_on_a_stopped_server_does_not_touch_the_closed_sink(
        self, explicit: InferenceServer, sinks: list[_RecordingSink]
    ) -> None:
        assert explicit.traces is sinks[0]  # the fixture really is the server's sink

        explicit.stop()
        stats = explicit.stats()

        assert sinks[0].is_closed
        assert sinks[0].uses_after_close == [], "stats() read a sink that stop() had closed"
        assert isinstance(explicit.traces, NullTraceSink), "the field kept the closed sink"
        # The name the run actually traced under, not the null sink that replaced it.
        assert stats["tracing"]["sink"] == "recording"

    def test_the_field_is_the_null_sink_again_and_a_restart_builds_a_new_one(
        self, explicit: InferenceServer, sinks: list[_RecordingSink]
    ) -> None:
        """Which is also what `start()` assumes: a server started again must not inherit the
        previous run's closed sink and trace into it."""
        explicit.stop()

        assert isinstance(explicit.traces, NullTraceSink)

        explicit.start()

        assert explicit.traces is sinks[1]
        assert not explicit.traces.is_closed

    def test_the_runs_totals_survive_the_reset(
        self, explicit: InferenceServer, sinks: list[_RecordingSink]
    ) -> None:
        """The reset must not be a way to lose the numbers.

        A dashboard scrapes on its own schedule, so the last sample it takes of a shard is
        usually taken *after* the shutdown. Reading the fresh null sink there reports
        ``recorded: 0``, which is not "we stopped measuring" — it is a claim that the run
        traced nothing, and it is wrong.
        """
        explicit.load_model("echo")
        explicit.infer_sync(
            InferenceRequest(
                model_name="echo",
                inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
                context=RequestContext(camera_id="cam0", frame_id=7),
            ),
            timeout=_TIMEOUT,
        )
        _await_recorded(sinks[0], 1)
        live = explicit.stats()["tracing"]
        assert live["recorded"] == 1

        explicit.stop()

        assert explicit.stats()["tracing"] == live
        assert sinks[0].uses_after_close == [], "the totals were read from the closed sink"

    def test_a_restart_reports_the_new_runs_sink_again(
        self, explicit: InferenceServer, sinks: list[_RecordingSink]
    ) -> None:
        """Keeping the last totals must not freeze `stats()` on them: a server started again
        is a new run, and its scrape has to follow the live sink."""
        explicit.stop()
        assert explicit.stats()["tracing"]["sink"] == "recording"

        explicit.start()

        assert explicit.traces is sinks[1]
        sinks[1].recorded = 3
        assert explicit.stats()["tracing"]["recorded"] == 3

    def test_the_totals_are_handed_out_as_a_copy(
        self, explicit: InferenceServer, sinks: list[_RecordingSink]
    ) -> None:
        """`stats()` used to return the stored dict itself. Its two callers are a metrics
        exporter and the KServe stats route, and either of them editing what came back edits
        what every later scrape reports — a wrong number with no way to trace it back."""
        explicit.stop()

        first = explicit.stats()["tracing"]
        first["recorded"] = 999
        first["sink"] = "edited"

        assert explicit.stats()["tracing"]["recorded"] != 999
        assert explicit.stats()["tracing"]["sink"] == "recording"

    def test_the_last_traces_still_land_and_the_sink_is_still_closed(
        self, explicit: InferenceServer, sinks: list[_RecordingSink]
    ) -> None:
        """Resetting the field must not become a way to skip the close, or a buffered sink
        loses whatever it had not flushed."""
        explicit.load_model("echo")
        explicit.infer_sync(
            InferenceRequest(
                model_name="echo",
                inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
                context=RequestContext(camera_id="cam0", frame_id=1),
            ),
            timeout=_TIMEOUT,
        )

        explicit.stop()

        assert [trace.frame_id for trace in sinks[0].traces] == [1]
        assert sinks[0].is_closed
        assert sinks[0].uses_after_close == []


class TestTwoStopsThatBothWinTheStartedCheck:
    """Two ``stop()`` calls arriving together must still produce exactly one teardown.

    The barrier above covers the second stop that arrives *after* the first has cleared the
    flags. This is the other interleaving: both threads read a started server before either
    clears it. The read and the clear used to sit either side of the ``_LOG.info`` line -- an
    emit that formats its arguments, may take a handler lock and may write to a file, so a
    guaranteed GIL switch point rather than a couple of bytecodes -- and both threads then
    fell through to ``_teardown`` -> ``_release``.

    Every step of the release is idempotent except one: it captures the run's trace totals
    *before* closing the sink, so a second pass captured them from the ``NullTraceSink`` the
    first pass had just installed and published ``recorded: 0`` for a run that traced
    thousands of requests -- verbatim the wrong answer ``TestStopLeavesAUsableTraceSink``
    exists to remove, restored by exactly the two-thread ``stop()`` that
    ``TestASecondStopWaitsForTheFirstOnesTeardown`` is about. And ``_torn_down`` was set by the
    first thread's ``finally`` while the second was still inside its release, so a third
    ``stop()`` returned claiming a teardown that was still running.
    """

    def test_one_release_runs_the_totals_survive_and_a_third_stop_waits(
        self, repository: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sinks: list[_RecordingSink] = []

        def build(*_args: object, **_kwargs: object) -> _RecordingSink:
            sinks.append(_RecordingSink())
            return sinks[-1]

        monkeypatch.setattr(pool, "build_trace_sink", build)
        server = InferenceServer(_settings(repository)).start()
        try:
            server.load_model("echo")
            for frame_id in range(3):
                server.infer_sync(
                    InferenceRequest(
                        model_name="echo",
                        inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
                        context=RequestContext(camera_id="cam0", frame_id=frame_id),
                    ),
                    timeout=_TIMEOUT,
                )
            _await_recorded(sinks[0], 3)
            live = server.stats()["tracing"]
            assert live["recorded"] == 3

            order: list[str] = []
            first_at_the_gate = threading.Event()
            both_past_the_check = threading.Event()
            inside_release = threading.Event()
            finish_release = threading.Event()
            gate_lock = threading.Lock()
            barrier = _watch_barrier(server)
            original_info = pool._LOG.info
            original_await = InferenceServer._await_teardown
            original_release = InferenceServer._release

            def gated_info(msg: Any, *args: Any, **kwargs: Any) -> None:
                """The forcing gate, sitting exactly where the old window was.

                The first thread to log the stop banner parks here until a second thread is
                provably past the started check, so the race is exercised on every run rather
                than on an unlucky one.
                """
                if isinstance(msg, str) and msg.startswith("stopping shipinfer"):
                    with gate_lock:
                        is_first = not first_at_the_gate.is_set()
                        first_at_the_gate.set()
                    if is_first:
                        assert both_past_the_check.wait(
                            _TIMEOUT
                        ), "the second stop never committed"
                    else:
                        # Reachable only before the fix: a second thread that also won the
                        # started check gets here instead of going to the barrier.
                        both_past_the_check.set()
                original_info(msg, *args, **kwargs)

            def committed_await(self: InferenceServer) -> None:
                """After the fix, this is how the second thread is past the check."""
                both_past_the_check.set()
                original_await(self)

            def gated_release(self: InferenceServer, generation: int) -> None:
                inside_release.set()
                assert finish_release.wait(_TIMEOUT), "the test never released the teardown"
                original_release(self, generation)
                order.append("released")

            monkeypatch.setattr(pool._LOG, "info", gated_info)
            monkeypatch.setattr(InferenceServer, "_await_teardown", committed_await)
            monkeypatch.setattr(InferenceServer, "_release", gated_release)

            def stop_and_record(label: str) -> None:
                server.stop()
                order.append(label)

            first = threading.Thread(target=stop_and_record, args=("a",), name="stop-a")
            first.start()
            assert first_at_the_gate.wait(_TIMEOUT), "the first stop never reached the gate"

            second = threading.Thread(target=stop_and_record, args=("b",), name="stop-b")
            second.start()
            assert both_past_the_check.wait(
                _TIMEOUT
            ), "the second stop never got past the check"
            assert inside_release.wait(_TIMEOUT), "no teardown ever started"

            # A third stop, issued while that release is still running: it must wait for the
            # real teardown, not for a barrier a first thread set while a second released.
            third = threading.Thread(target=stop_and_record, args=("c",), name="stop-c")
            third.start()
            _await_barrier_waiter(barrier, "stop-c")

            finish_release.set()
            for thread in (first, second, third):
                thread.join(_TIMEOUT)
                assert not thread.is_alive(), f"{thread.name} never finished"

            assert order.count("released") == 1, order
            assert order[0] == "released", order
            assert server.stats()["tracing"] == live
            assert sinks[0].uses_after_close == []
        finally:
            server.stop()


# -- the start side of the same window ---------------------------------------------------


@pytest.fixture()
def three_models(tmp_path: Path) -> Path:
    """Three start-up models, so "the remaining models" is a set a test can look at."""
    root = tmp_path / "fleet"
    for name in ("a_first", "b_second", "c_third"):
        (root / name / "1").mkdir(parents=True)
        (root / name / "config.yaml").write_text(_MODEL.lstrip())
    return root


def _await_torn_down(server: InferenceServer) -> None:
    """Block until the teardown running on another thread has finished.

    The forcing wait that makes "the drain happened *before* the start published" a fact
    rather than a hope: the barrier is set in ``_teardown``'s ``finally``, so once it is set
    the model table has provably been emptied.
    """
    assert server._torn_down.wait(_TIMEOUT), "the teardown never finished"


class TestAStopRacingTheInitialStart:
    """``stop()`` and the *initial* ``start()`` must not both believe they own this server.

    ``stop()`` has taken the lifecycle lock for its entry transition since the two-stops fix
    above; ``start()`` took no lock at all. So a stop arriving mid-start cleared the flags,
    drained the model table and closed the trace sink, and the start then set
    ``_started = True`` on top of it — a server answering ``is_started`` with an empty model
    table, which is what a readiness probe reads as "up, and serving nothing". Either the
    start completes and a later ``stop()`` drains what it loaded, or the start is refused
    with a typed error and leaves nothing running. There is no third state.
    """

    def test_a_stop_that_drains_the_table_under_a_start_refuses_it_typed(
        self, repository: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact interleaving, forced: the models are up and published, the start has not
        claimed ``_started`` yet, and a stop runs its whole teardown in that window."""
        server = InferenceServer(_startup_settings(repository))
        entered = threading.Event()
        release = threading.Event()
        original_join = InferenceServer._join_service_tier

        def gated_join(self: InferenceServer) -> Any:
            # The last step of the start before it publishes itself, so everything the start
            # loaded is in the table and nothing has been claimed.
            entered.set()
            assert release.wait(_TIMEOUT), "the test never released the start"
            return original_join(self)

        monkeypatch.setattr(InferenceServer, "_join_service_tier", gated_join)

        outcome = _Outcome()
        starter = threading.Thread(target=outcome.run_start, args=(server,), name="starter")
        starter.start()
        assert entered.wait(_TIMEOUT), "the start never reached the service tier"
        assert server.models() == ["echo"], "the start had not published its models yet"

        stopper = threading.Thread(target=server.stop, name="stopper")
        stopper.start()
        _await_stop_flags(server)
        _await_torn_down(server)
        release.set()
        for thread in (starter, stopper):
            thread.join(_TIMEOUT)
            assert not thread.is_alive(), f"{thread.name} never finished"

        assert isinstance(outcome.error, ServerStateError), f"got {outcome.error!r}"
        assert "stopped while it was starting" in str(outcome.error)
        # The property, stated as the disjunction the class docstring gives.
        assert not (server.is_started and server.models() == [])
        assert not server.is_started
        assert server.models() == []
        assert _live_workers("echo") == [], "the abandoned start left worker threads running"

    def test_a_model_published_after_the_drain_is_released_by_the_start_that_lost(
        self, repository: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The narrower half: the stop's drain runs *before* the start publishes, so the
        teardown never sees the model and only the losing start can release it.

        Without that, the model is running — two worker threads, a backend, in production a
        CUDA context — under a server that reports itself stopped and holds no reference to
        it. That is the leak the whole of this module is about, entered from the start side.
        """
        server = InferenceServer(_startup_settings(repository))
        inside_start = threading.Event()
        release = threading.Event()
        original_start = Model.start

        def blocking_start(self: Model, **kwargs: Any) -> None:
            original_start(self, **kwargs)
            # Started, not yet published: `_build_and_start` puts it in the table after this
            # returns, which is after the drain below has been and gone.
            inside_start.set()
            assert release.wait(_TIMEOUT), "the test never released the model's start"

        monkeypatch.setattr(Model, "start", blocking_start)

        outcome = _Outcome()
        starter = threading.Thread(target=outcome.run_start, args=(server,), name="starter")
        starter.start()
        assert inside_start.wait(_TIMEOUT), "the start never reached the model's start"
        assert server.models() == [], "the model reached the table before the drain"

        stopper = threading.Thread(target=server.stop, name="stopper")
        stopper.start()
        _await_stop_flags(server)
        _await_torn_down(server)
        release.set()
        for thread in (starter, stopper):
            thread.join(_TIMEOUT)
            assert not thread.is_alive(), f"{thread.name} never finished"

        assert isinstance(outcome.error, ServerStateError), f"got {outcome.error!r}"
        assert _live_workers("echo") == [], "the model published after the drain still runs"
        assert server.models() == []
        assert not server.is_started

    def test_a_second_start_while_one_is_in_progress_is_refused_typed(
        self, repository: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two starts are the same collision as a start and a stop: unclaimed, both loaded
        every model and the second one's copies were the leak."""
        server = InferenceServer(_startup_settings(repository))
        entered = threading.Event()
        release = threading.Event()
        original_join = InferenceServer._join_service_tier

        def gated_join(self: InferenceServer) -> Any:
            entered.set()
            assert release.wait(_TIMEOUT), "the test never released the start"
            return original_join(self)

        monkeypatch.setattr(InferenceServer, "_join_service_tier", gated_join)

        outcome = _Outcome()
        starter = threading.Thread(target=outcome.run_start, args=(server,), name="starter")
        starter.start()
        try:
            assert entered.wait(_TIMEOUT), "the start never reached the service tier"
            with pytest.raises(ServerStateError, match="already starting"):
                server.start()
        finally:
            release.set()
            starter.join(_TIMEOUT)
            server.stop()

        assert outcome.error is None, f"the first start failed: {outcome.error!r}"
        assert len(_live_workers("echo")) == 0, "the stop left the models running"

    def test_a_start_while_a_teardown_is_in_flight_is_refused_typed(
        self, repository: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented downgrade, closed. ``_await_teardown`` expires after
        ``shutdown_grace_s`` and returns; a ``start()`` on the other side of that used to
        re-arm the barrier and take the still-queued teardown's release on the fresh
        server's models. It waits the same budget and then refuses.
        """
        server = InferenceServer(_startup_settings(repository, grace=0.2)).start()
        inside_release = threading.Event()
        finish_release = threading.Event()
        original_release = InferenceServer._release

        def gated_release(self: InferenceServer, generation: int) -> None:
            inside_release.set()
            assert finish_release.wait(_TIMEOUT), "the test never released the teardown"
            original_release(self, generation)

        monkeypatch.setattr(InferenceServer, "_release", gated_release)

        stopper = threading.Thread(target=server.stop, name="stopper")
        stopper.start()
        try:
            assert inside_release.wait(_TIMEOUT), "no teardown ever started"
            began = time.monotonic()
            with pytest.raises(ServerStateError, match="still tearing this server down"):
                server.start()
            waited = time.monotonic() - began
        finally:
            finish_release.set()
            stopper.join(_TIMEOUT)
            server.stop()

        # It waited the grace period rather than refusing on sight: a teardown that finishes
        # inside its own budget must let the start through, which is the restart below.
        assert waited >= 0.2, f"the start refused after {waited:.3f}s without waiting"
        assert _live_workers("echo") == []

    def test_a_start_after_the_teardown_finished_is_let_through(self, repository: Path) -> None:
        """The refusal is on the teardown being *in flight*, not on there having been one:
        stop-then-start is what an operator does after fixing a config, and the barrier is
        already set by then."""
        server = InferenceServer(_startup_settings(repository)).start()
        server.stop()

        server.start()
        try:
            assert server.is_ready
            assert server.models() == ["echo"]
        finally:
            server.stop()
        assert _live_workers("echo") == []


class TestANonStrictStartRacingAStop:
    """``strict_startup=false`` logs and continues past a model that will not load — and used
    to do the same for the *abort*, which is not a model failing but the server going away.

    One shutdown produced one ERROR with a full traceback per remaining model ("failed to
    load model 'b_second'; continuing"), and the start then went on building models the
    teardown had already been past — the leak ``stop()`` takes the control lock to prevent,
    arriving through the door non-strict start-up left open. "Continuing" is meaningless once
    a stop has begun: there is nothing to continue towards.
    """

    def test_the_abort_ends_the_start_instead_of_being_logged_per_model(
        self,
        three_models: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        server = InferenceServer(_startup_settings(three_models, strict=False))
        entered = threading.Event()
        release = threading.Event()
        attempted: list[str] = []
        original_start = ModelInstance.start

        def gated_start(self: ModelInstance) -> None:
            attempted.append(self.name)
            if len(attempted) == 1:
                # Instance 1 of model 1 is slow — the TensorRT deserialisation, in miniature.
                entered.set()
                assert release.wait(_TIMEOUT), "the test never released instance 1"
            original_start(self)

        monkeypatch.setattr(ModelInstance, "start", gated_start)

        outcome = _Outcome()
        starter = threading.Thread(target=outcome.run_start, args=(server,), name="starter")
        with caplog.at_level(logging.DEBUG, logger="shipinfer.engine"):
            starter.start()
            assert entered.wait(_TIMEOUT), "the start never reached the first instance"
            stopper = threading.Thread(target=server.stop, name="stopper")
            stopper.start()
            _await_stop_flags(server)
            release.set()
            for thread in (starter, stopper):
                thread.join(_TIMEOUT)
                assert not thread.is_alive(), f"{thread.name} never finished"

        # The abort itself came out, once, naming the model it gave up in the middle of --
        # not the generic refusal that would mean the start had run to the end of the loop.
        assert isinstance(outcome.error, ServerStateError), f"got {outcome.error!r}"
        assert "start aborted: the server is stopping" in str(outcome.error)
        assert "model a_first" in str(outcome.error)
        continuing = [r.getMessage() for r in caplog.records if "continuing" in r.getMessage()]
        assert continuing == [], continuing
        # Only the first model was ever attempted; the abort is checked before each instance,
        # so the two later models never reached one.
        assert len(attempted) == 1, attempted
        for name in ("a_first", "b_second", "c_third"):
            assert _live_workers(name) == [], f"{name} was left running"
        assert server.models() == []
        assert not server.is_started

    def test_a_model_that_will_not_load_is_still_skipped(self, tmp_path: Path) -> None:
        """The re-raise is gated on the server stopping, not on the mode: a heterogeneous
        fleet where one node genuinely cannot host one model is what the mode is for."""
        root = tmp_path / "mixed"
        for name, config in (("a_first", _MODEL), ("b_second", _MODEL), ("c_third", _MODEL)):
            (root / name / "1").mkdir(parents=True)
            text = config if name != "b_second" else config.replace("mock", "no_such_runtime")
            (root / name / "config.yaml").write_text(text.lstrip())

        server = InferenceServer(_startup_settings(root, strict=False))
        with server:
            assert server.models() == ["a_first", "c_third"]
            assert server.is_ready


# -- the scrape that races the teardown ---------------------------------------------------


class _WatchedLock:
    """A stand-in for ``threading.Lock`` that records every thread which had to *wait*.

    ``threading.Lock`` cannot be subclassed, and a name recorded on entry to ``acquire``
    would not tell "took it" from "blocked on it" — which is the whole of the distinction
    the scrape test turns on: it has to know the stop is parked *outside* the lock, not
    merely that it reached it. The same reason ``_WatchedEvent`` exists, one primitive down.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names_lock = threading.Lock()
        self.blocked: list[str] = []

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        if not blocking:
            return False
        with self._names_lock:
            self.blocked.append(threading.current_thread().name)
        return self._lock.acquire(True, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


class _GatedStatsServer(InferenceServer):
    """A server whose ``stats()`` can be suspended *inside* the trace lookup.

    ``_last_trace_stats`` becomes a property so a test can park a scrape at the exact point
    the unlocked version checked it — between "is it None?" and the load of ``_traces``. That
    window is two bytecodes wide: no sleep can be placed in it, and nothing observable from
    outside ``stats()`` distinguishes a scrape that is in it from one that is not. So the
    hook goes where the read is.

    Class attributes rather than an ``__init__``: ``InferenceServer.__init__`` assigns
    ``_last_trace_stats`` itself, and that assignment has to find the setter already there.
    """

    _trace_stats_value: dict[str, Any] | None = None
    _trace_stats_gate: Callable[[], None] | None = None

    @property
    def _last_trace_stats(self) -> dict[str, Any] | None:  # type: ignore[override]
        # The value is taken *before* the gate and returned after it, because that is what
        # the window is: the read happened, and then the thread lost the GIL holding what it
        # had read. A getter that re-read afterwards would hand the caller the teardown's own
        # totals and quietly repair the bug it is here to expose.
        value = self._trace_stats_value
        gate, self._trace_stats_gate = self._trace_stats_gate, None
        if gate is not None:
            gate()
        return value

    @_last_trace_stats.setter
    def _last_trace_stats(self, value: dict[str, Any] | None) -> None:
        self._trace_stats_value = value


def _await_swap_or_block(server: InferenceServer, lock: _WatchedLock, name: str) -> None:
    """Block until the teardown has either swapped the sink in or blocked on ``lock``.

    Two outcomes, one wait, because the point of the test is which of them happens. With the
    lookup serialised the stop cannot get past the lifecycle lock the scrape is holding, so
    it is recorded as blocked; without it the stop runs to completion and the null sink is
    the observable. Waiting for the *swap* rather than for the totals is what makes the
    second branch deterministic: the totals are published before the sink is replaced, and
    resuming between the two would let the scrape read the live sink and pass by luck.
    """
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if name in lock.blocked or isinstance(server.traces, NullTraceSink):
            return
        time.sleep(0.001)
    raise AssertionError(f"{name} neither blocked on the lifecycle lock nor swapped the sink")


class TestAScrapeRacingTheTeardownsSwap:
    """``stats()`` reads two fields that ``_release`` rewrites, so it must read them as one.

    Unlocked it was a check and an act: a scrape that found ``_last_trace_stats`` still None
    and then loaded ``_traces`` after the swap reports ``{"sink": "none", "recorded": 0}`` —
    verbatim the wrong answer ``_last_trace_stats`` was added to remove, arriving through the
    door the fix left open. A scrape does not stop when the server does, so that zero is the
    last sample a dashboard takes of the shard.
    """

    def test_a_scrape_parked_in_the_lookup_never_reports_the_null_sinks_zeros(
        self, repository: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sinks: list[_RecordingSink] = []

        def build(*_args: object, **_kwargs: object) -> _RecordingSink:
            sinks.append(_RecordingSink())
            return sinks[-1]

        monkeypatch.setattr(pool, "build_trace_sink", build)
        server = _GatedStatsServer(_settings(repository))
        lifecycle = _WatchedLock()
        # Installed before the start, so every acquisition of it is recorded.
        server._lifecycle_lock = lifecycle  # type: ignore[assignment]
        server.start()
        try:
            server.load_model("echo")
            server.infer_sync(
                InferenceRequest(
                    model_name="echo",
                    inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
                    context=RequestContext(camera_id="cam0", frame_id=11),
                ),
                timeout=_TIMEOUT,
            )
            _await_recorded(sinks[0], 1)
            live = server.stats()["tracing"]
            assert live["recorded"] == 1

            parked = threading.Event()
            resume = threading.Event()
            scraped: dict[str, Any] = {}

            def gate() -> None:
                parked.set()
                assert resume.wait(_TIMEOUT), "the test never resumed the scrape"

            def scrape() -> None:
                scraped.update(server.stats()["tracing"])

            server._trace_stats_gate = gate
            scraper = threading.Thread(target=scrape, name="scraper")
            scraper.start()
            assert parked.wait(_TIMEOUT), "the scrape never reached the trace lookup"

            stopper = threading.Thread(target=server.stop, name="stopper")
            stopper.start()
            _await_swap_or_block(server, lifecycle, "stopper")
            resume.set()
            for thread in (scraper, stopper):
                thread.join(_TIMEOUT)
                assert not thread.is_alive(), f"{thread.name} never finished"

            assert scraped == live, "the scrape read across the teardown's swap"
            assert scraped["sink"] == "recording"
            # And the run's totals are what a scrape after the shutdown reports, as before.
            assert server.stats()["tracing"] == live
            assert sinks[0].uses_after_close == []
        finally:
            server.stop()


class TestATeardownThatWasOvertakenByANewRun:
    """A teardown carries the generation it was started for, and stands down if it lost it.

    :meth:`InferenceServer._begin_start` refuses to start while a teardown is in flight, so
    this state cannot be reached through the public API any more — that refusal is the first
    line of defence and this is the second. It exists because the first one is a *timed*
    wait: ``_await_teardown`` gives up after ``shutdown_grace_s``, and a teardown wedged on
    an instance that will not drain outlives it. Everything a release does is destructive —
    it empties the model table, closes the trace sink and closes the memory pool — so a
    release that has fallen behind must do none of it.

    The overtaking is therefore forced by hand: the barrier is set while the release is still
    inside, which is precisely the one step the fix makes unreachable. What is under test is
    what the release does *next*.
    """

    def test_it_releases_none_of_the_new_runs_models_sink_or_totals(
        self,
        repository: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sinks: list[_RecordingSink] = []

        def build(*_args: object, **_kwargs: object) -> _RecordingSink:
            sinks.append(_RecordingSink())
            return sinks[-1]

        monkeypatch.setattr(pool, "build_trace_sink", build)
        server = InferenceServer(_startup_settings(repository, grace=0.2)).start()
        inside_release = threading.Event()
        finish_release = threading.Event()
        original_release = InferenceServer._release

        def gated_release(self: InferenceServer, generation: int) -> None:
            inside_release.set()
            assert finish_release.wait(_TIMEOUT), "the test never released the teardown"
            original_release(self, generation)

        monkeypatch.setattr(InferenceServer, "_release", gated_release)

        stopper = threading.Thread(target=server.stop, name="stopper")
        stranded: list[Any] = []
        with caplog.at_level(logging.WARNING, logger="shipinfer.engine"):
            stopper.start()
            try:
                assert inside_release.wait(_TIMEOUT), "no teardown ever started"
                # Run 1's models, held while they are still reachable. They are stranded by
                # what follows: `_models` is keyed by name, so run 2 publishes its "echo"
                # over run 1's entry and no reference to run 1's copy survives inside the
                # server. That is *why* `_begin_start` refuses to enter this state — the
                # guard's job is to keep the live run intact, not to make the state good —
                # and it is why the test has to hold them itself to stop them at the end.
                stranded.extend(server)
                # The forced step: a wedged teardown outliving its grace period, seen from
                # the inside. From here a start is let through with a release still running.
                server._torn_down.set()
                server.start()
                assert server.models() == ["echo"], "the new run did not load its models"
                assert server.model("echo") is not stranded[0]
                sinks[1].recorded = 5

                finish_release.set()
                stopper.join(_TIMEOUT)
                assert not stopper.is_alive(), "the stale teardown never finished"

                assert server.models() == ["echo"], "the stale teardown drained the live run"
                assert server.model("echo") is not stranded[0]
                assert server.is_ready, "the stale teardown stopped the live run's instances"
                assert server.traces is sinks[1]
                assert not sinks[1].is_closed, "the stale teardown closed the live sink"
                assert server.stats()["tracing"] == {**sinks[1].stats(), "recorded": 5}
            finally:
                finish_release.set()
                stopper.join(_TIMEOUT)
                server.stop()
                for model in stranded:
                    model.stop()

        overtaken = [
            r.getMessage()
            for r in caplog.records
            if "abandoning the teardown" in r.getMessage()
        ]
        assert len(overtaken) == 1, overtaken
        assert "run 1" in overtaken[0] and "run 2" in overtaken[0]
        assert _live_workers("echo") == []
