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

Offline throughout — the mock backend, ``KIND_CPU`` instances and real worker threads,
because the evidence for the first one is ``threading.enumerate()``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
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

        def recording_release(self: InferenceServer) -> None:
            # Patched at `_release`, not `_teardown`: `_teardown`'s own `finally` sets the
            # barrier before it returns, so an append AFTER `_teardown` would race the waiter
            # it releases. Inside `_release` the record lands before the barrier by construction.
            original_release(self)
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

            def gated_release(self: InferenceServer) -> None:
                inside_release.set()
                assert finish_release.wait(_TIMEOUT), "the test never released the teardown"
                original_release(self)
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
