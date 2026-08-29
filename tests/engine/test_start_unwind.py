"""A start-up that fails must not leave the models it already loaded running.

The failure this pins is silent and expensive. With ``strict_startup`` on, the first model
that will not load aborts the whole start — and the models loaded before it are *running*:
one worker thread per instance, each bound to a device, each holding a backend and, in
production, a CUDA context of ~250 MiB. ``start()`` raised, so it returned nothing and the
caller assigned nothing; those threads belong to an object no reference reaches. On this
shared box that is the next person's job failing to fit, with nothing anywhere saying why.

``stop()`` could not help, either: it returned early on ``not self._started``, and
``_started`` is set only once *every* model is up — so the one state in which models were
running and unreachable was the exact state ``stop()`` declined to handle.

Offline throughout: real TorchScript fixtures, ``KIND_CPU`` instances, and real worker threads —
which is the point, because the evidence here is ``threading.enumerate()``.
"""

from __future__ import annotations

import itertools
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from shipinfer.backends.torch_backend import TorchScriptBackend
from shipinfer.core.errors import ServerStateError, ShipInferError
from shipinfer.core.metrics import ServerMetrics
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.engine import InferenceServer
from shipinfer.engine.model import Model
from shipinfer.repository import ModelRepository
from shipinfer.runtime.device import DeviceManager
from shipinfer.runtime.memory import MemoryPool
from tests.support.models import materialise

_OK = """
platform: pytorch
max_batch_size: 4
inputs: [{name: x, data_type: FP32, dims: [2]}]
outputs: [{name: y, data_type: FP32, dims: [2]}]
instance_groups: [{kind: KIND_CPU, count: 2}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.0}
"""

#: Fails *late*: the config is valid, the backend loads, and the instance threads are
#: already spawned when the declared warm-up sample cannot be built. That is the shape that
#: leaks the most — the failing model has live threads of its own on top of its
#: predecessors'.
_BROKEN_WARMUP = _OK + """model_warmup:
  - {name: probe, batch_size: 1, count: 1, inputs: {x: {input_data_file: missing.bin}}}
"""

#: Fails *early*: there is no such backend, so the model raises while it is being
#: constructed and never reaches ``Model.start``. This is the one a non-strict start skips,
#: because it is an error in both modes (a warm-up that cannot run is only fatal under
#: ``strict_startup`` — see ``Model.start``).
_UNKNOWN_PLATFORM = _OK.replace("platform: pytorch", "platform: no_such_runtime")

#: One instance, so "nothing is ready" and "this one instance is not ready" are the same
#: statement and the test cannot pass by accident on a second instance.
_ONE_INSTANCE = _OK.replace("count: 2", "count: 1")


def _repository(root: Path, second: str) -> Path:
    """Three start-up models in load order, with the middle one broken.

    The names are alphabetical on purpose: ``_startup_names`` preserves the repository's
    order, so ``a_first`` really is loaded before ``b_second``.
    """
    for name, config in (("a_first", _OK), ("b_second", second), ("c_third", _OK)):
        (root / name / "1").mkdir(parents=True)
        (root / name / "config.yaml").write_text(config.lstrip())
    materialise(root)
    return root


def _live_workers(model: str) -> list[str]:
    """The instance worker threads of ``model`` that are still alive.

    ``ModelInstance.start`` names its thread ``shipinfer-<model>_<ordinal>_<device>``, so
    this is the direct observation of the leak rather than a proxy for it.
    """
    return [t.name for t in threading.enumerate() if t.name.startswith(f"shipinfer-{model}_")]


def _settings(root: Path, *, strict: bool) -> ServerSettings:
    return ServerSettings(
        model_repository=root,
        devices={"visible_gpus": []},
        strict_startup=strict,
        execution={"warmup_iterations": 0},
    )


def _one_model(root: Path, name: str, config: str) -> ServerSettings:
    """A repository holding exactly ``name``, and the settings that read it."""
    (root / name / "1").mkdir(parents=True)
    (root / name / "config.yaml").write_text(config.lstrip())
    materialise(root)
    return _settings(root, strict=False)


def _model_alone(root: Path, name: str, config: str) -> Model:
    """Build one :class:`Model` the way the pool builds it, without the pool.

    ``InferenceServer`` starts its models on ``Model.start``'s 120 s default, and a test
    for what happens *at* the readiness deadline cannot wait that out. Constructing the
    model here is what makes ``timeout_s`` reachable.
    """
    settings = _one_model(root, name, config)
    return Model(
        artifact=ModelRepository(root).resolve(name),
        settings=settings,
        devices=DeviceManager(settings.devices),
        memory=MemoryPool(settings.memory),
        metrics=ServerMetrics(),
    )


def _request(model: str) -> InferenceRequest:
    return InferenceRequest(
        model_name=model,
        inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
        context=RequestContext(camera_id="cam0", frame_id=0),
    )


class TestAFailedStrictStartReleasesWhatItTook:
    def test_it_raises_the_reason_the_model_would_not_load(self, tmp_path: Path) -> None:
        """The load error, not a teardown error from the clean-up that followed it."""
        root = _repository(tmp_path / "repo", _BROKEN_WARMUP)
        server = InferenceServer(_settings(root, strict=True))

        with pytest.raises(ServerStateError, match=r"missing\.bin"):
            server.start()

    def test_the_models_it_already_started_are_stopped(self, tmp_path: Path) -> None:
        """The assertion the fix exists for: no worker thread survives the failed start."""
        root = _repository(tmp_path / "repo", _BROKEN_WARMUP)
        server = InferenceServer(_settings(root, strict=True))

        with pytest.raises(ShipInferError):
            server.start()

        assert _live_workers("a_first") == [], "the first model's instances are still running"
        assert _live_workers("b_second") == [], "the failing model's own instances survived"
        assert server.models() == []
        assert not server.is_started

    def test_a_model_that_fails_while_being_built_unwinds_the_same_way(
        self, tmp_path: Path
    ) -> None:
        """The early failure too: nothing about the unwind may depend on *how* far the
        broken model got, because the models before it are running either way."""
        root = _repository(tmp_path / "repo", _UNKNOWN_PLATFORM)
        server = InferenceServer(_settings(root, strict=True))

        with pytest.raises(ShipInferError):
            server.start()

        assert _live_workers("a_first") == []
        assert not server.is_started

    def test_stop_afterwards_is_a_no_op_that_does_not_raise(self, tmp_path: Path) -> None:
        """A caller that wraps `start()` in a `finally: stop()` — which `cli/shard.py` now
        effectively does — must not get a second teardown, or a raise, out of it."""
        root = _repository(tmp_path / "repo", _BROKEN_WARMUP)
        server = InferenceServer(_settings(root, strict=True))

        with pytest.raises(ShipInferError):
            server.start()

        server.stop()
        server.stop()

        assert not server.is_started
        assert _live_workers("a_first") == []

    def test_the_server_can_be_started_again_once_the_repository_is_fixed(
        self, tmp_path: Path
    ) -> None:
        """The unwind leaves a usable object, not a wedged one: an operator who fixes the
        config and retries is the ordinary case."""
        root = _repository(tmp_path / "repo", _UNKNOWN_PLATFORM)
        server = InferenceServer(_settings(root, strict=True))

        with pytest.raises(ShipInferError):
            server.start()
        (root / "b_second" / "config.yaml").write_text(_OK.lstrip())
        materialise(root)  # the fixed config names a real runtime, so give it a real model

        with server:
            assert server.models() == ["a_first", "b_second", "c_third"]
        assert _live_workers("a_first") == []


class TestStopOnAServerThatWasNeverStarted:
    def test_it_does_nothing_and_does_not_raise(self, tmp_path: Path) -> None:
        """Removing the `not self._started` early return must not turn `stop()` into a
        teardown of a server that has nothing to tear down."""
        root = _repository(tmp_path / "repo", _OK)
        server = InferenceServer(_settings(root, strict=True))

        server.stop()

        assert not server.is_started
        assert server.models() == []
        # The memory pool was not closed out from under a server that may still start.
        server.start()
        try:
            assert server.is_ready
        finally:
            server.stop()


class TestNonStrictStartUpIsUnchanged:
    def test_the_other_models_still_load(self, tmp_path: Path) -> None:
        """`strict_startup=false` is for a heterogeneous fleet where one node genuinely
        cannot host one model. The unwind must not turn that into a refusal to start."""
        root = _repository(tmp_path / "repo", _UNKNOWN_PLATFORM)
        server = InferenceServer(_settings(root, strict=False))

        with server:
            assert server.models() == ["a_first", "c_third"]
            assert server.is_ready

    def test_the_skipped_model_leaves_no_threads_behind(self, tmp_path: Path) -> None:
        """A model skipped by a non-strict start is skipped, not half-run."""
        root = _repository(tmp_path / "repo", _BROKEN_WARMUP)
        server = InferenceServer(_settings(root, strict=False))

        with server:
            pass

        assert _live_workers("b_second") == []


class TestAModelWithZeroReadyInstancesIsRefused:
    """A model whose instances have *all* failed to start can never serve a request.

    Keeping it in the model table advertises capacity that does not exist, and the failure
    it hides is a start-up failure — it belongs at start-up, where an operator is watching,
    and not at the first request. The gate is therefore "every instance has settled and
    failed", not "nothing is ready yet": an instance still deserialising an engine when the
    shared deadline runs out is slow, not broken, and must be left alone to come up.
    """

    def test_non_strict_skips_it_like_a_load_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every instance of ``b_second`` dies on its declared warm-up sample, so the model
        is refused — and the pool's non-strict path treats that refusal exactly like any
        other failed load: logged at ERROR, skipped, the rest of the repository served."""
        root = _repository(tmp_path / "repo", _BROKEN_WARMUP)
        server = InferenceServer(_settings(root, strict=False))

        with server:
            assert server.models() == ["a_first", "c_third"], "zero-ready must not publish"
            assert server.is_ready

        skipped = [
            record
            for record in caplog.records
            if record.getMessage().startswith("failed to load model 'b_second'")
        ]
        assert skipped, "the refusal must be the pool's skip, or the operator sees nothing"
        cause = skipped[0].exc_info
        assert cause is not None, "the skip is logged with its traceback, not bare"
        assert "missing.bin" in str(cause[1]), "the refusal must carry the instance's reason"

    def test_a_model_still_loading_at_the_deadline_is_not_torn_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The instance misses the deadline with ``start_error`` unset — a slow TensorRT
        deserialise, not a broken one. The load is held on an event until the deadline
        verdict is in, so this cannot pass by lucky timing. Nothing is ready when
        ``start()`` returns, and that must not be read as a refusal: the model stays, and
        the instance comes up late and serves — the case a ``not any(is_ready)`` gate
        destroys."""
        may_finish = threading.Event()
        initialize = TorchScriptBackend._do_initialize

        def held_until_released(backend: TorchScriptBackend) -> None:
            """Slow, not broken: it waits, then really loads, so it can serve afterwards."""
            may_finish.wait(timeout=30.0)
            initialize(backend)

        monkeypatch.setattr(TorchScriptBackend, "_do_initialize", held_until_released)
        model = _model_alone(tmp_path / "repo", "slow", _ONE_INSTANCE)

        try:
            model.start(timeout_s=0.2)  # must not raise: still loading is not failed

            assert not model.is_ready, "the instance really did miss the deadline"
            may_finish.set()  # the load finishes only after the deadline verdict is in
            deadline = time.monotonic() + 30.0
            while not model.is_ready and time.monotonic() < deadline:
                time.sleep(0.05)
            assert model.is_ready, "the instance that was still loading never came up"
            assert set(model.infer(_request("slow")).result(timeout=30).outputs) == {"y"}
        finally:
            may_finish.set()  # a failing run must not hold the worker for the full wait
            model.stop()

    def test_a_partially_ready_model_stays_published(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One instance of two fails: that is degraded capacity, not zero, and the model
        keeps serving on the other. Guards the ``all`` in the gate — under ``any`` this
        model would be refused and the server would serve nothing at all."""
        started = itertools.count()
        lock = threading.Lock()
        initialize = TorchScriptBackend._do_initialize

        def only_the_first_fails(backend: TorchScriptBackend) -> None:
            with lock:
                first = next(started) == 0
            if first:
                raise RuntimeError("no device for this instance")
            initialize(backend)

        monkeypatch.setattr(TorchScriptBackend, "_do_initialize", only_the_first_fails)
        settings = _one_model(tmp_path / "repo", "half_up", _OK)  # two instances

        with InferenceServer(settings) as server:
            assert server.models() == ["half_up"], "a half-ready model is still capacity"
            assert server.model("half_up").is_ready
            assert set(server.infer(_request("half_up")).result(timeout=30).outputs) == {"y"}

    def test_one_dead_instance_does_not_condemn_a_loading_sibling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mixed case: one instance fails fast while its sibling is still loading.
        The model must wait for the sibling, not be refused — the gate reads "EVERY
        instance failed", and the plausible `failed and not any(is_ready)` spelling
        tears the healthy loading instance down mid-deserialise and abandons it."""
        may_finish = threading.Event()
        started = itertools.count()
        lock = threading.Lock()
        initialize = TorchScriptBackend._do_initialize

        def one_dies_one_loads_slowly(backend: TorchScriptBackend) -> None:
            with lock:
                first = next(started) == 0
            if first:
                raise RuntimeError("no device for this instance")
            may_finish.wait(timeout=30.0)
            initialize(backend)

        monkeypatch.setattr(TorchScriptBackend, "_do_initialize", one_dies_one_loads_slowly)
        model = _model_alone(tmp_path / "repo", "mixed", _OK)  # two instances

        try:
            model.start(timeout_s=0.3)  # must not raise: the sibling is merely slow

            assert not model.is_ready, "the sibling really was still loading"
            may_finish.set()
            deadline = time.monotonic() + 30.0
            while not model.is_ready and time.monotonic() < deadline:
                time.sleep(0.05)
            assert model.is_ready, "the loading sibling never came up"
            assert set(model.infer(_request("mixed")).result(timeout=30).outputs) == {"y"}
        finally:
            may_finish.set()
            model.stop()
        assert _live_workers("mixed") == []

    def test_strict_still_aborts_the_whole_start(self, tmp_path: Path) -> None:
        """A regression guard for behaviour that predates this gate, not evidence for it:
        under ``strict_startup`` the per-instance check above already raises, so the model
        gate is never reached. It is here so a change to the gate cannot quietly turn the
        strict path into a skip."""
        root = _repository(tmp_path / "repo", _BROKEN_WARMUP)
        server = InferenceServer(_settings(root, strict=True))

        with pytest.raises(ServerStateError):
            server.start()
        assert not server.is_started
