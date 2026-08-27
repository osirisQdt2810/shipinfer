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

Offline throughout: the mock backend, ``KIND_CPU`` instances, and real worker threads —
which is the point, because the evidence here is ``threading.enumerate()``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from shipinfer.core.errors import ServerStateError, ShipInferError
from shipinfer.core.settings import ServerSettings
from shipinfer.engine import InferenceServer

_OK = """
platform: mock
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
_UNKNOWN_PLATFORM = _OK.replace("platform: mock", "platform: no_such_runtime")


def _repository(root: Path, second: str) -> Path:
    """Three start-up models in load order, with the middle one broken.

    The names are alphabetical on purpose: ``_startup_names`` preserves the repository's
    order, so ``a_first`` really is loaded before ``b_second``.
    """
    for name, config in (("a_first", _OK), ("b_second", second), ("c_third", _OK)):
        (root / name / "1").mkdir(parents=True)
        (root / name / "config.yaml").write_text(config.lstrip())
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
