"""Ensemble steps are scheduled independently, not walked on one thread.

The DAG used to run inside a single pool task per request: a loop that called
``future.result()`` at every step. That thread was therefore blocked for the whole time
some *other* model was computing, which is nearly all of a DAG's wall time — so the size of
the ensemble's pool, not the capacity of the step models, decided how many frames could be
in flight, and one slow step held threads away from frames that were already past it.

Every test here drives :class:`EnsembleModel` against fake step models rather than through
:class:`InferenceServer`. That is deliberate: the properties under test are "does a waiting
step occupy a worker" and "do two independent steps overlap", and neither can be asserted
against a model that merely sleeps — a sleeping mock looks identical whether it was waited
on by a held thread or by a callback. A step that blocks on a :class:`threading.Barrier`
answers both without a timing assertion: if the scheduling is wrong the barrier is never
reached by enough parties and the test fails outright instead of flaking.

``tests/engine/test_ensemble.py`` keeps the end-to-end coverage over real TorchScript
models, including the conditional-branch behaviour these tests must not regress.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import InferenceError, QueueFullError, RequestCancelledError
from shipinfer.core.metrics import ServerMetrics
from shipinfer.core.request import (
    InferenceRequest,
    InferenceResponse,
    RequestContext,
    ResponseFuture,
)
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.engine.ensemble import EnsembleModel
from shipinfer.repository import ModelArtifact, ModelConfig

_ROW = 2


def _io(names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"name": name, "data_type": "FP32", "dims": [_ROW]} for name in names]


def _config(name: str, inputs: Sequence[str], outputs: Sequence[str]) -> ModelConfig:
    return ModelConfig(
        name=name,
        platform="pytorch",
        max_batch_size=4,
        inputs=_io(inputs),
        outputs=_io(outputs),
        instance_groups=[{"kind": "KIND_CPU", "count": 1}],
        dynamic_batching={"enabled": False},
    )


def _rows(count: int = 1, value: float = 1.0) -> Tensor:
    return Tensor.from_numpy(np.full((count, _ROW), value, dtype=np.float32))


def _waits_on(barrier: threading.Barrier) -> Callable[[InferenceRequest], Mapping[str, Tensor]]:
    """A step that only returns once ``barrier.parties`` requests are inside it at once.

    This is what replaces a timing assertion: if the scheduler serialises what should
    overlap, the barrier is never satisfied and the step raises instead of merely being
    slow, so the test fails loudly rather than flaking on a busy machine.
    """

    def work(_request: InferenceRequest) -> Mapping[str, Tensor]:
        barrier.wait()
        return {}

    return work


def _waits_for(
    event: threading.Event, timeout: float = 10.0
) -> Callable[[InferenceRequest], Mapping[str, Tensor]]:
    """A step held open until the test releases it."""

    def work(_request: InferenceRequest) -> Mapping[str, Tensor]:
        event.wait(timeout)
        return {}

    return work


def _eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _FakeModel:
    """A step model that resolves its futures on its own threads.

    Implements exactly the surface an ensemble step needs — ``name``, ``artifact``,
    ``is_ready``, ``infer`` — so the ensemble cannot tell it from a real
    :class:`~shipinfer.engine.model.Model`. ``work`` decides when the step finishes, which
    is the whole point: a real backend cannot be asked to stay open until three other
    requests have arrived.
    """

    is_ready = True

    def __init__(
        self,
        name: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        *,
        work: Callable[[InferenceRequest], Mapping[str, Tensor]] | None = None,
        workers: int = 8,
    ) -> None:
        self.name = name
        self.artifact = ModelArtifact(
            name=name, version=1, path=Path(), config=_config(name, inputs, outputs)
        )
        self._outputs = tuple(outputs)
        self._work = work
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"fake-{name}")
        self.calls = 0

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        self.calls += 1
        future = ResponseFuture(request)
        self._pool.submit(self._run, request, future)
        return future

    def _run(self, request: InferenceRequest, future: ResponseFuture) -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            produced = dict(self._work(request)) if self._work else {}
        except Exception as exc:  # a backend raising is exactly what this models
            future.set_exception(exc)
            return
        outputs = {name: produced.get(name, _rows()) for name in self._outputs}
        future.set_result(
            InferenceResponse(
                request_id=request.request_id,
                model_name=self.name,
                model_version=1,
                outputs=outputs,
                context=request.context,
                timings=request.timings,
            )
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def _ensemble(
    models: Mapping[str, _FakeModel],
    steps: Sequence[dict[str, Any]],
    *,
    inputs: Sequence[str],
    outputs: Sequence[str],
    max_workers: int = 8,
    max_pending: int = 0,
) -> EnsembleModel:
    artifact = ModelArtifact(
        name="pipe",
        version=1,
        path=Path(),
        config=ModelConfig(
            name="pipe",
            platform="ensemble",
            max_batch_size=0,
            inputs=_io(inputs),
            outputs=_io(outputs),
            dynamic_batching={"enabled": False},
            ensemble={"steps": list(steps)},
        ),
    )
    ensemble = EnsembleModel(
        artifact=artifact,
        settings=ServerSettings(devices={"visible_gpus": []}),
        metrics=ServerMetrics(),
        resolve=lambda name: models[name],
        max_workers=max_workers,
        max_pending=max_pending,
    )
    ensemble.start()
    return ensemble


def _request() -> InferenceRequest:
    return InferenceRequest(
        model_name="pipe",
        inputs={"images": _rows()},
        context=RequestContext(camera_id="cam7", frame_id=3),
    )


class TestWorkerOccupancy:

    def test_a_waiting_step_does_not_hold_a_pool_worker(self) -> None:
        """The regression test. Four requests, one ensemble worker: they can only all be
        inside the step at once if none of them is holding that worker while it waits."""
        barrier = threading.Barrier(4, timeout=3)
        slow = _FakeModel("slow", ["images"], ["out"], work=_waits_on(barrier), workers=4)
        ensemble = _ensemble(
            {"slow": slow},
            [
                {
                    "model": "slow",
                    "input_map": {"images": "images"},
                    "output_map": {"out": "out"},
                }
            ],
            inputs=["images"],
            outputs=["out"],
            max_workers=1,
            max_pending=8,
        )
        try:
            futures = [ensemble.infer(_request()) for _ in range(4)]
            for future in futures:
                assert set(future.result(timeout=10).outputs) == {"out"}
        finally:
            barrier.abort()
            ensemble.stop()
            slow.close()

    def test_the_pool_no_longer_bounds_requests_in_flight(self) -> None:
        """``max_pending`` is the bound on frames in flight, and it is now allowed to be far
        above ``max_workers`` — which is the capacity that used to be unreachable."""
        held = threading.Event()
        gate = _FakeModel("gate", ["images"], ["out"], work=_waits_for(held), workers=16)
        ensemble = _ensemble(
            {"gate": gate},
            [
                {
                    "model": "gate",
                    "input_map": {"images": "images"},
                    "output_map": {"out": "out"},
                }
            ],
            inputs=["images"],
            outputs=["out"],
            max_workers=2,
            max_pending=12,
        )
        try:
            futures = [ensemble.infer(_request()) for _ in range(12)]
            # Every one of them is inside the step model *before* anything is released.
            # With the DAG walked on pool threads this could never exceed `max_workers`.
            assert _eventually(
                lambda: gate.calls == 12
            ), f"only {gate.calls}/12 reached the step"
            with pytest.raises(QueueFullError):
                ensemble.infer(_request())  # 13 > max_pending, and that is the only bound
            held.set()
            for future in futures:
                future.result(timeout=10)
        finally:
            held.set()
            ensemble.stop()
            gate.close()


class TestIndependentSteps:

    def test_two_steps_that_share_only_the_input_run_concurrently(self) -> None:
        """Both steps read the ensemble's input and neither reads the other's output, so
        nothing in the graph says they must take turns. Walking the step list said it
        anyway."""
        barrier = threading.Barrier(2, timeout=3)
        left = _FakeModel("left", ["images"], ["left_out"], work=_waits_on(barrier))
        right = _FakeModel("right", ["images"], ["right_out"], work=_waits_on(barrier))
        ensemble = _ensemble(
            {"left": left, "right": right},
            [
                {
                    "model": "left",
                    "input_map": {"images": "images"},
                    "output_map": {"left_out": "left_out"},
                },
                {
                    "model": "right",
                    "input_map": {"images": "images"},
                    "output_map": {"right_out": "right_out"},
                },
            ],
            inputs=["images"],
            outputs=["left_out", "right_out"],
        )
        try:
            response = ensemble.infer(_request()).result(timeout=10)
            assert set(response.outputs) == {"left_out", "right_out"}
            assert ensemble.stats()["peak_parallel_steps"] >= 2
        finally:
            barrier.abort()
            ensemble.stop()
            left.close()
            right.close()

    def test_a_dependent_step_still_waits_for_its_producer(self) -> None:
        """The other half: concurrency is a consequence of the edges, not a replacement for
        them. A step reading an earlier step's output must not start early."""
        order: list[str] = []
        lock = threading.Lock()

        def record(name: str) -> Callable[[InferenceRequest], Mapping[str, Tensor]]:
            def work(_request: InferenceRequest) -> Mapping[str, Tensor]:
                with lock:
                    order.append(name)
                return {}

            return work

        head = _FakeModel("head", ["images"], ["mid"], work=record("head"))
        tail = _FakeModel("tail", ["mid"], ["out"], work=record("tail"))
        ensemble = _ensemble(
            {"head": head, "tail": tail},
            [
                {
                    "model": "head",
                    "input_map": {"images": "images"},
                    "output_map": {"mid": "mid"},
                },
                {"model": "tail", "input_map": {"mid": "mid"}, "output_map": {"out": "out"}},
            ],
            inputs=["images"],
            outputs=["out"],
        )
        try:
            ensemble.infer(_request()).result(timeout=10)
            assert order == ["head", "tail"]
            assert ensemble.stats()["steps"][1]["depends_on"] == ["mid"]
        finally:
            ensemble.stop()
            head.close()
            tail.close()


class TestConditionalBranches:
    """A skipped branch must stay distinguishable from a failed one — and, now that steps
    wait on a *relation* rather than on the previous line, a tensor that will never arrive
    has to resolve as absent rather than leave its readers parked forever."""

    def _pipeline(self, flag: float) -> tuple[EnsembleModel, list[_FakeModel]]:
        router = _FakeModel(
            "router",
            ["images"],
            ["crops", "flag"],
            work=lambda _r: {"flag": _rows(value=flag)},
        )
        branch = _FakeModel("branch", ["crops"], ["embedding"])
        head = _FakeModel("head", ["embedding"], ["score"])
        ensemble = _ensemble(
            {"router": router, "branch": branch, "head": head},
            [
                {
                    "model": "router",
                    "input_map": {"images": "images"},
                    "output_map": {"crops": "crops", "flag": "flag"},
                },
                {
                    "model": "branch",
                    "input_map": {"crops": "crops"},
                    "output_map": {"embedding": "embedding"},
                    "condition": "flag",
                },
                {
                    "model": "head",
                    "input_map": {"embedding": "embedding"},
                    "output_map": {"score": "score"},
                    "condition": "embedding",
                },
            ],
            inputs=["images"],
            outputs=["embedding", "score"],
        )
        return ensemble, [router, branch, head]

    def test_a_skipped_branch_leaves_its_readers_decidable(self) -> None:
        """``head`` is conditional on a tensor only the skipped ``branch`` produces. It must
        skip — not wait for a producer that will never run."""
        ensemble, models = self._pipeline(flag=0.0)
        try:
            response = ensemble.infer(_request()).result(timeout=10)
            assert set(response.outputs) == {"embedding", "score"}
            assert response.outputs["embedding"].shape == (0, _ROW)
            assert response.outputs["score"].shape == (0, _ROW)
            assert ensemble.stats()["skipped_steps"] == 2
            assert models[1].calls == 0 and models[2].calls == 0
        finally:
            ensemble.stop()
            for model in models:
                model.close()

    def test_a_taken_branch_runs_the_whole_chain(self) -> None:
        ensemble, models = self._pipeline(flag=1.0)
        try:
            response = ensemble.infer(_request()).result(timeout=10)
            assert response.outputs["score"].shape == (1, _ROW)
            assert ensemble.stats()["skipped_steps"] == 0
        finally:
            ensemble.stop()
            for model in models:
                model.close()


class TestFailurePaths:

    def _failing(self, **kwargs: Any) -> tuple[EnsembleModel, _FakeModel]:
        def boom(_request: InferenceRequest) -> Mapping[str, Tensor]:
            raise InferenceError("step exploded")

        model = _FakeModel("boom", ["images"], ["out"], work=boom)
        ensemble = _ensemble(
            {"boom": model},
            [
                {
                    "model": "boom",
                    "input_map": {"images": "images"},
                    "output_map": {"out": "out"},
                }
            ],
            inputs=["images"],
            outputs=["out"],
            **kwargs,
        )
        return ensemble, model

    def test_a_failing_step_fails_the_request_with_its_own_error(self) -> None:
        ensemble, model = self._failing()
        try:
            future = ensemble.infer(_request())
            with pytest.raises(InferenceError, match="step exploded"):
                future.result(timeout=10)
            # The tag rides the error path too: a failure nobody can attribute to a camera
            # and a frame is a failure nobody can act on.
            assert future.context.camera_id == "cam7"
            assert future.context.frame_id == 3
        finally:
            ensemble.stop()
            model.close()

    def test_a_step_that_refuses_work_fails_the_dag_rather_than_hanging_it(self) -> None:
        """A saturated step model raises from ``infer`` itself, before there is any future
        to attach a callback to. Nothing else is watching that execution — it holds no
        thread — so the dispatch site is the only place that can resolve the caller."""

        class _Saturated(_FakeModel):
            def infer(self, request: InferenceRequest) -> ResponseFuture:
                raise QueueFullError("step:full", 4, 4)

        model = _Saturated("full", ["images"], ["out"])
        ensemble = _ensemble(
            {"full": model},
            [
                {
                    "model": "full",
                    "input_map": {"images": "images"},
                    "output_map": {"out": "out"},
                }
            ],
            inputs=["images"],
            outputs=["out"],
            max_pending=2,
        )
        try:
            for _ in range(6):  # six, so a leaked slot would surface as a QueueFullError
                with pytest.raises(QueueFullError, match="step:full"):
                    ensemble.infer(_request()).result(timeout=10)
        finally:
            ensemble.stop()
            model.close()

    def test_a_failed_dag_gives_its_queue_slot_back(self) -> None:
        """Capacity is released on exactly one terminal transition. Leaking it on the error
        path would make an ensemble refuse everything after ``max_pending`` failures — and
        the failures are exactly when it must keep accepting work."""
        ensemble, model = self._failing(max_pending=2)
        try:
            for _ in range(6):
                with pytest.raises(InferenceError):
                    ensemble.infer(_request()).result(timeout=10)
        finally:
            ensemble.stop()
            model.close()


class TestShutdown:

    def test_stop_resolves_a_request_parked_mid_dag(self) -> None:
        """An execution waiting on a step holds no thread, so nothing else would ever
        notice it. ``stop`` has to know about it explicitly or the caller waits forever."""
        held = threading.Event()
        gate = _FakeModel("gate", ["images"], ["out"], work=_waits_for(held, timeout=5))
        ensemble = _ensemble(
            {"gate": gate},
            [
                {
                    "model": "gate",
                    "input_map": {"images": "images"},
                    "output_map": {"out": "out"},
                }
            ],
            inputs=["images"],
            outputs=["out"],
        )
        try:
            future = ensemble.infer(_request())
            ensemble.stop()
            with pytest.raises(RequestCancelledError):
                future.result(timeout=10)
        finally:
            held.set()
            gate.close()
