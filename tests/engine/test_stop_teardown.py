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
does, so the first scrape after a shutdown asked a closed sink for numbers.

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
        the model it published — worker threads included."""
        inside_start = threading.Event()
        release = threading.Event()
        original_start = Model.start

        def blocking_start(self: Model) -> None:
            # Called from `_build_and_start`, so the control lock is held for the whole of
            # this wait: a `stop()` on another thread cannot get past it until we return.
            inside_start.set()
            assert release.wait(_TIMEOUT), "the test never released the model's start"
            original_start(self)

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
        assert stats["tracing"]["sink"] == "none"

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
