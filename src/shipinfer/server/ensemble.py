"""Ensembles: a DAG of models addressed as one model.

Triton's ensemble idea, kept because it earns its place here. The alternative is that every
client orchestrates the pipeline itself, which means the intermediate tensors — crops,
masks, embeddings — round-trip to the caller and back on every frame. Declaring the graph
server-side keeps them where they were produced.

Two things this adds to the Triton vocabulary, both because this pipeline needs them:

* ``condition`` — a step runs only when a named tensor is present and non-empty. That is
  how "segment only where a ship was detected" is expressed declaratively, instead of in a
  hand-written orchestration loop each caller would have to reimplement.
* **DAG validation at load time.** Every step's inputs must be either an ensemble input or
  produced by an earlier step. A repository whose wiring is wrong fails at start-up with
  the missing tensor named, rather than at the first inference.

**Steps are scheduled, not walked.** This used to run the DAG on one pool thread per
request: a ``for step in steps`` loop that blocked on ``future.result()`` at every step. A
thread therefore sat idle for the entire time some *other* model was computing, which is
most of a DAG's wall time — so a pool of eight bounded the whole ensemble to eight frames
in flight no matter how many instances the step models had, and one slow step held threads
away from the frames that were already past it. Triton's ensemble scheduler does not work
that way (``core/src/ensemble_scheduler``) and neither does this one any more: an execution
is a state machine advanced by the completion callback of whichever step just finished, so
a pool thread is occupied only while a step is being *dispatched*, never while one is being
waited for. The pool now bounds how much DAG bookkeeping runs at once, which is microseconds
of work, rather than how many frames may be in flight.

Falling out of that: steps whose inputs are all satisfied run **concurrently**. A step is
ready when every step that could produce a tensor it reads has finished or skipped, which is
the same relation the load-time validation already computes — so two branches that share
only the ensemble's input no longer serialise behind each other.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from shipinfer.core.errors import (
    ConfigurationError,
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    ServerStateError,
)
from shipinfer.core.logging import get_logger
from shipinfer.core.metrics import ServerMetrics
from shipinfer.core.request import InferenceRequest, InferenceResponse, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.tracing import NullTraceSink, RequestTrace, TraceSink
from shipinfer.core.types import Tensor, TensorSpec, validate_against
from shipinfer.repository import EnsembleStep, ModelArtifact
from shipinfer.server.statistics import ModelStatistics

__all__ = ["EnsembleModel"]

_LOG = get_logger("server.ensemble")

_NO_PRODUCERS: frozenset[int] = frozenset()


class _Servable(Protocol):
    """The slice of a model an ensemble step needs. Both ``Model`` and ``EnsembleModel``
    satisfy it, so ensembles nest without a special case."""

    name: str

    def infer(self, request: InferenceRequest) -> ResponseFuture: ...

    @property
    def is_ready(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class _StepPlan:
    """One node of the DAG, with its edges resolved into the ensemble's own namespace.

    Computed once at start-up from exactly the walk that validates the graph, so the
    scheduler and the validator can never disagree about what an edge is.
    """

    index: int
    step: EnsembleStep
    #: Ensemble-namespace names this step reads, including its ``condition``.
    reads: frozenset[str]
    #: Ensemble-namespace names it writes.
    writes: frozenset[str]


class _Execution:
    """One in-flight DAG.

    Deliberately not a thread and not a generator: it is state, advanced from whichever
    thread happens to finish a step. ``lock`` guards every field below it, and is held for
    dictionary updates only — never across a call into a step model, because that call
    enqueues work that another thread will complete.
    """

    __slots__ = (
        "finished",
        "future",
        "lock",
        "namespace",
        "request",
        "settled",
        "started",
        "written_by",
    )

    def __init__(self, request: InferenceRequest, future: ResponseFuture) -> None:
        self.request = request
        self.future = future
        self.namespace: dict[str, Tensor] = dict(request.inputs)
        self.lock = threading.Lock()
        #: Step indices dispatched (or skipped). A step enters this set exactly once, under
        #: the lock, which is what stops two threads dispatching the same step.
        self.started: set[int] = set()
        #: Step indices that will never produce anything more — completed or skipped.
        self.finished: set[int] = set()
        #: name -> the index of the step whose value is currently in `namespace`. Steps that
        #: write the same name now run concurrently, so "who wrote this" has to be recorded
        #: rather than inferred from arrival order — see `_step_done`.
        self.written_by: dict[str, int] = {}
        #: Whether the caller's future has been resolved. The terminal transition is
        #: claimed once so the semaphore slot is released exactly once.
        self.settled = False


class EnsembleModel:
    """Executes a declared DAG of other models.

    Presents the same surface as :class:`~shipinfer.server.model.Model` — ``start``,
    ``stop``, ``infer``, ``stats``, ``is_ready`` — so :class:`InferenceServer` holds both
    in one table and callers cannot tell which they are talking to.
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        settings: ServerSettings,
        metrics: ServerMetrics,
        resolve: Callable[[str], _Servable],
        *,
        traces: TraceSink | None = None,
        max_workers: int = 8,
        max_pending: int = 0,
    ) -> None:
        self._artifact = artifact
        self._settings = settings
        self._metrics = metrics
        self._resolve = resolve
        # The DAG's own span. Each step already traces its own model, and the thing that
        # joined them was invisible — which is the one span an operator debugging an ensemble
        # actually wants, because a slow ensemble is usually slow *between* its steps.
        self._traces = traces if traces is not None else NullTraceSink()
        self._steps = tuple(artifact.config.ensemble.steps)  # type: ignore[union-attr]
        self._plans: tuple[_StepPlan, ...] = ()
        self._producers: dict[str, frozenset[int]] = {}
        self._waits_on: dict[int, frozenset[int]] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"ensemble-{artifact.name}"
        )
        # A ThreadPoolExecutor's queue is unbounded, so an ensemble under load accumulates
        # work forever and never applies the backpressure every other path in this system
        # applies. The semaphore is the bound; exceeding it raises QueueFullError, exactly
        # as a saturated instance queue would. It bounds *requests in flight*, which since
        # steps stopped holding threads is no longer the same quantity as `max_workers`.
        self._capacity = max(1, max_pending if max_pending > 0 else max_workers * 4)
        self._slots = threading.Semaphore(self._capacity)
        self._live: set[_Execution] = set()
        self._live_lock = threading.Lock()
        self._started = False
        # A leaf lock over the counters. They are incremented from step-completion
        # callbacks, which now run on several worker threads at once, and `+= 1` on an
        # attribute is a read-modify-write that can drop an increment between them.
        self._counter_lock = threading.Lock()
        self._executions = 0
        self._skipped_steps = 0
        self._running_steps = 0
        self._peak_parallel_steps = 0
        self._statistics = ModelStatistics()

    # -- properties ----------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._artifact.name

    @property
    def version(self) -> int:
        return self._artifact.version

    @property
    def artifact(self) -> ModelArtifact:
        return self._artifact

    @property
    def instances(self) -> tuple:
        """An ensemble owns no instances of its own; its steps' models do."""
        return ()

    @property
    def statistics(self) -> ModelStatistics:
        """Cumulative per-model counters, in Triton's vocabulary."""
        return self._statistics

    @property
    def is_ready(self) -> bool:
        return self._started and all(self._resolve(s.model).is_ready for s in self._steps)

    @property
    def total_depth(self) -> int:
        return 0

    # -- lifecycle -----------------------------------------------------------------------

    def start(self, timeout_s: float = 120.0) -> None:
        """Validate the DAG against the loaded models, then accept work.

        Validation is the whole value of doing this at start-up: a mis-wired ensemble is a
        configuration mistake, and a configuration mistake should stop a deploy, not
        produce a confusing ``KeyError`` on the thousandth frame. The same walk yields the
        step plans the scheduler runs on, so the graph the server executes is by
        construction the graph it checked.
        """
        self._plans = self._validate_graph()
        self._refuse_late_producers()
        producers: dict[str, set[int]] = {}
        for plan in self._plans:
            for name in plan.writes:
                producers.setdefault(name, set()).add(plan.index)
        self._producers = {name: frozenset(idx) for name, idx in producers.items()}
        # Per step, the producers it must wait for: **only those declared before it**.
        #
        # Waiting on every producer of a name deadlocks a graph where a step reads a name a
        # later step also writes — `detect(images -> boxes)` then `refine(boxes -> boxes)`.
        # `refine` is itself a producer of `boxes`, so its precondition can never be
        # satisfied: it waits for itself, the future never resolves, and its semaphore slot
        # is never released. That shape is not exotic — refine-in-place is the obvious way to
        # write a second-pass stage — and the sequential walk it replaced ran it happily.
        #
        # Declaration order is the semantics the sequential walk had: a reader saw whatever
        # the last *earlier* writer left, and a later writer was simply not its business.
        self._waits_on = {
            plan.index: frozenset(
                index
                for name in plan.reads
                for index in self._producers.get(name, _NO_PRODUCERS)
                if index < plan.index
            )
            for plan in self._plans
        }
        self._started = True
        _LOG.info(
            "ensemble %s ready: %s",
            self.name,
            " -> ".join(step.model for step in self._steps),
        )

    def _refuse_late_producers(self) -> None:
        """Refuse a graph where a step reads a name some *later* step also writes.

        The scheduler runs steps as soon as their inputs are settled, so two steps with no
        dependency between them dispatch in the same pass. If one of them writes a name a
        third step reads, the read can see either value depending on which landed first — and
        the sequential walk this replaced always produced the earlier writer's.

        Concretely, with `0 detect(images->boxes)`, `1 slow(images->tmp)`,
        `2 embed(boxes->emb)`, `3 refine(images->boxes)`: steps 0, 1 and 3 all dispatch
        together, and if `refine` lands first then `embed` reads `refine`'s boxes rather than
        `detect`'s. `written_by` keeps the *namespace* deterministic, but it cannot make a
        reader wait for a writer it was never told about.

        This is refused at load rather than scheduled around, for the reason CONVENTIONS
        gives for the rest of this validation: a mis-wired ensemble is a configuration
        mistake, and a configuration mistake should stop a deploy rather than produce a
        different answer on the thousandth frame. Ordering it correctly instead would mean a
        reader waiting on every later producer, which is the deadlock this scheduler was just
        fixed for — the two requirements are genuinely incompatible, and refusing is the
        honest resolution.

        Raises:
            ConfigurationError: naming both steps and the tensor, because "step 2 is
                ambiguous" without saying against what is a message that costs an afternoon.
        """
        writers: dict[str, list[int]] = {}
        for plan in self._plans:
            for name in plan.writes:
                writers.setdefault(name, []).append(plan.index)
        for plan in self._plans:
            for name in plan.reads:
                late = [index for index in writers.get(name, ()) if index > plan.index]
                if not late:
                    continue
                raise ConfigurationError(
                    f"ensemble {self.name!r}: step {plan.index} ({plan.step.model}) reads "
                    f"{name!r}, which step(s) {late} also write. Steps run as soon as their "
                    f"inputs settle, so which value it reads would depend on which finished "
                    f"first. Reorder the steps so every producer of {name!r} is declared "
                    f"before every reader of it, or give the later writer its own name."
                )

    def _validate_graph(self) -> tuple[_StepPlan, ...]:
        """Type-check the wiring: every edge must exist, and both ends must agree.

        Names alone are not enough. A crop sized for a 512x512 segmenter fed to a 256x128
        embedder has a perfectly valid *name* and is still wrong — it would surface as a
        validation error on the first frame, from inside a worker thread, three models away
        from the config line that caused it. Checking dtype and shape here turns that into
        a start-up failure that names both ends of the edge.

        Returns:
            The step plans, in declaration order, each carrying the ensemble-namespace
            names it reads and writes.
        """
        # tensor name -> the spec that produced it
        available: dict[str, TensorSpec] = {
            spec.name: spec for spec in self._artifact.config.input_specs
        }
        # Tensors that only exist when some branch ran. Consuming one unconditionally is a
        # wiring error: it works on every frame that happens to take the branch and fails
        # on the first one that does not.
        conditional: set[str] = set()
        plans: list[_StepPlan] = []

        for index, step in enumerate(self._steps):
            model = self._resolve(step.model)  # raises ModelNotFoundError if missing
            step_config = getattr(model, "artifact", self._artifact).config
            where = f"ensemble {self.name!r} step {index} ({step.model})"
            reads: set[str] = set()

            for spec in step_config.input_specs:
                source = step.input_map.get(spec.name, spec.name)
                reads.add(source)
                produced = available.get(source)
                if produced is None:
                    if spec.optional:
                        continue
                    raise ConfigurationError(
                        f"{where}: input {spec.name!r} maps to {source!r}, which no earlier "
                        f"step produces (available: {sorted(available)})"
                    )
                if produced.dtype is not spec.dtype:
                    raise ConfigurationError(
                        f"{where}: input {spec.name!r} is {spec.dtype.value} but {source!r} "
                        f"is produced as {produced.dtype.value}"
                    )
                if not spec.matches(produced.shape):
                    raise ConfigurationError(
                        f"{where}: input {spec.name!r} expects {spec.describe()} but "
                        f"{source!r} is produced as {produced.describe()}"
                    )
                if source in conditional and not step.condition:
                    raise ConfigurationError(
                        f"{where}: input {spec.name!r} reads {source!r}, which only exists "
                        f"when an earlier conditional step runs, but this step has no "
                        f"`condition:` of its own. Give it one, or the graph is only valid "
                        f"for the frames that happen to take that branch."
                    )

            if step.condition:
                if step.condition not in available:
                    raise ConfigurationError(
                        f"{where}: condition {step.condition!r} is not produced by any "
                        f"earlier step (available: {sorted(available)})"
                    )
                reads.add(step.condition)

            writes: set[str] = set()
            for spec in step_config.output_specs:
                name = step.output_map.get(spec.name, spec.name)
                writes.add(name)
                available[name] = TensorSpec(
                    name=name, dtype=spec.dtype, shape=spec.shape, optional=spec.optional
                )
                if step.condition:
                    conditional.add(name)

            plans.append(
                _StepPlan(
                    index=index,
                    step=step,
                    reads=frozenset(reads),
                    writes=frozenset(writes),
                )
            )

        for spec in self._artifact.config.output_specs:
            produced = available.get(spec.name)
            if produced is None:
                raise ConfigurationError(
                    f"ensemble {self.name!r} declares output {spec.name!r} that no step produces"
                )
            if produced.dtype is not spec.dtype or not spec.matches(produced.shape):
                raise ConfigurationError(
                    f"ensemble {self.name!r} declares output {spec.describe()} but the graph "
                    f"produces {produced.describe()}"
                )
        return tuple(plans)

    def stop(self) -> None:
        """Stop accepting work and drain.

        ``cancel_futures=True`` would drop queued items on the floor, and the futures it
        drops are the *caller's* — they would never resolve and every waiter would block
        forever. So every execution still in flight is failed explicitly, with a typed
        error, before the pool is told to shut down. That now includes executions parked
        mid-DAG waiting on a step: they hold no thread, so nothing else would ever notice
        them.
        """
        self._started = False
        with self._live_lock:
            live = list(self._live)
        for state in live:
            # Not `_fail`: a shutdown is not the model failing, and counting it as one
            # would put a spike of `requests_failed` on every restart for an operator to
            # explain. The caller still learns, with a typed error rather than a hang.
            if self._claim(state):
                _settle(
                    state.future,
                    exception=RequestCancelledError(f"ensemble {self.name!r} was stopped"),
                )
        self._pool.shutdown(wait=True)

    # -- inference -----------------------------------------------------------------------

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        if not self._started:
            raise ServerStateError(f"ensemble {self.name!r} has not been started")

        request.timings.received_ns = time.monotonic_ns()
        try:
            validate_against(request.inputs, self._artifact.config.input_specs, what="input")
        except ValueError as exc:
            from shipinfer.core.errors import ValidationError

            raise ValidationError(f"{self.name}: {exc}") from exc

        if not self._slots.acquire(blocking=False):
            self._metrics.requests_rejected.inc(model=self.name)
            raise QueueFullError(f"ensemble:{self.name}", self._capacity, self._capacity)

        state = _Execution(request, ResponseFuture(request))
        with self._live_lock:
            self._live.add(state)
        self._submit(state, self._begin)
        return state.future

    # -- the scheduler -------------------------------------------------------------------

    def _submit(self, state: _Execution, work: Callable[[_Execution], None]) -> None:
        """Hand one unit of DAG bookkeeping to the pool.

        A shut-down pool raises rather than silently dropping the work, and dropping it is
        what would leave the caller's future unresolved forever. The request is failed
        instead — which, if :meth:`stop` already failed it, is a no-op.
        """
        try:
            self._pool.submit(work, state)
        except RuntimeError:
            self._fail(state, RequestCancelledError(f"ensemble {self.name!r} was stopped"))

    def _begin(self, state: _Execution) -> None:
        with state.lock:
            if state.settled:
                return
        try:
            running = state.future.set_running_or_notify_cancel()
        except RuntimeError:
            # `stop()` settled this execution in the window between the check above and
            # here. The future is already resolved; there is nothing left to begin.
            return
        if not running:
            self._retire(state)  # the caller cancelled while this waited for a thread
            return
        self._advance(state)

    def _advance(self, state: _Execution) -> None:
        """Pump the DAG, and turn any escape into a failed request rather than a hang.

        This runs as a pool task and as a step's done-callback, and in both places an
        escaping exception is swallowed by machinery that has no idea a caller is waiting
        on it. A hang is the one failure mode a server must not have, so anything the
        scheduler itself gets wrong is reported to the caller as its own failure.
        """
        try:
            self._pump(state)
        except Exception as exc:
            _LOG.exception("ensemble %s failed to schedule a step", self.name)
            self._fail(state, exc)

    def _pump(self, state: _Execution) -> None:
        """Dispatch everything the DAG can run right now, then let go of the thread.

        Loops rather than recurses because a skipped step resolves synchronously and can
        unlock the next one; it exits as soon as a pass dispatches nothing new, which
        means either the DAG is finished or at least one step is in flight and its
        completion callback will bring us back here.
        """
        while True:
            with state.lock:
                if state.settled:
                    return
                ready = self._take_ready(state)
                complete = not ready and len(state.finished) == len(self._plans)
            if complete:
                self._complete(state)
                return
            if not ready:
                return
            progressed = False
            for plan in ready:
                progressed |= self._dispatch(plan, state)
            if not progressed:
                return

    def _take_ready(self, state: _Execution) -> list[_StepPlan]:
        """Claim every step whose inputs can no longer change. Caller holds ``state.lock``.

        "Can no longer change" rather than "is present": a step waits for every producer
        **declared before it** to have finished or been skipped. Two things follow.

        It gives an answer for a tensor that will never arrive — every earlier producer
        skipped — which is what lets a conditional step downstream of a skipped branch decide
        instead of hang.

        And it does **not** deadlock on a step that writes a name it also reads. Waiting on
        *all* producers did: `refine(boxes -> boxes)` is itself a producer of `boxes`, so its
        precondition could never be satisfied. See :meth:`start` for why declaration order is
        the right cut.

        What this does **not** do on its own is preserve last-writer semantics — two earlier
        steps writing the same name still finish in whatever order they finish. That is
        handled where the write happens, in :meth:`_step_done`, not here; an earlier version
        of this docstring claimed it and was wrong.
        """
        ready: list[_StepPlan] = []
        for plan in self._plans:
            if plan.index in state.started:
                continue
            if not (self._waits_on.get(plan.index, _NO_PRODUCERS) - state.finished):
                state.started.add(plan.index)
                ready.append(plan)
        return ready

    def _dispatch(self, plan: _StepPlan, state: _Execution) -> bool:
        """Run one step, or decide it does not run.

        Returns:
            ``True`` when the step resolved without leaving the thread — a skipped branch —
            so the caller knows to look for newly-unlocked steps.
        """
        step = plan.step
        with state.lock:
            if state.settled:
                return False
            condition = state.namespace.get(step.condition) if step.condition else None
            namespace = dict(state.namespace)

        if step.condition is not None and not _is_truthy(condition):
            with self._counter_lock:
                self._skipped_steps += 1
            with state.lock:
                state.finished.add(plan.index)
            return True

        try:
            model = self._resolve(step.model)
            step_request = InferenceRequest(
                model_name=step.model,
                inputs=self._step_inputs(step, model, namespace),
                model_version=step.model_version,
                context=state.request.context,  # the (camera, frame) tag rides the whole DAG
                priority=state.request.priority,
                deadline_ns=state.request.deadline_ns,
            )
            with self._counter_lock:
                self._running_steps += 1
                self._peak_parallel_steps = max(self._peak_parallel_steps, self._running_steps)
            future = model.infer(step_request)
        except Exception as exc:  # a refused enqueue, an unloaded model, a bad namespace
            with self._counter_lock:
                self._running_steps = max(0, self._running_steps - 1)
            self._fail(state, exc)
            return False

        future.add_done_callback(lambda done: self._step_done(state, plan, done))
        return False

    def _step_inputs(
        self, step: EnsembleStep, model: _Servable, namespace: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        inputs: dict[str, Tensor] = {}
        for spec in getattr(model, "artifact", self._artifact).config.input_specs:
            source = step.input_map.get(spec.name, spec.name)
            tensor = namespace.get(source)
            if tensor is None:
                if spec.optional:
                    continue
                raise InferenceError(
                    f"ensemble {self.name}: step {step.model} needs {source!r}, "
                    f"which is not in the namespace ({sorted(namespace)})"
                )
            inputs[spec.name] = tensor
        return inputs

    def _step_done(self, state: _Execution, plan: _StepPlan, done: Future) -> None:
        """A step finished. Runs on whichever worker thread finished it, so it does the
        smallest possible amount of work and hands the DAG back to the pool."""
        with self._counter_lock:
            self._running_steps = max(0, self._running_steps - 1)
        try:
            response = done.result()
        except Exception as exc:
            self._fail(state, exc)
            return
        try:
            with state.lock:
                if state.settled:
                    return
                for name, tensor in response.outputs.items():
                    target = plan.step.output_map.get(name, name)
                    # Last writer by **declaration order**, not by completion time. Two
                    # independent steps that map an output onto the same ensemble name now
                    # run concurrently and race to write here; whichever finished last would
                    # win, non-deterministically, where the sequential walk always gave the
                    # later-declared step. Concretely: step 0 writes 1.0 slowly, step 1 writes
                    # 2.0 quickly, the ensemble outputs that name — sequential says 2.0, and
                    # completion order said 1.0 about half the time.
                    if state.written_by.get(target, -1) <= plan.index:
                        state.namespace[target] = tensor
                        state.written_by[target] = plan.index
                state.finished.add(plan.index)
        except Exception as exc:  # a callback that raises here would strand the caller
            self._fail(state, exc)
            return
        self._submit(state, self._advance)

    # -- terminal transitions ------------------------------------------------------------

    def _claim(self, state: _Execution) -> bool:
        """Take the one terminal transition, releasing the queue slot with it.

        Every path out of an execution goes through here, which is what makes the semaphore
        balanced: a request that failed on its third step must give its slot back exactly
        as one that completed does, or the ensemble leaks capacity until it refuses
        everything.
        """
        with state.lock:
            if state.settled:
                return False
            state.settled = True
        with self._live_lock:
            self._live.discard(state)
        self._slots.release()
        return True

    def _complete(self, state: _Execution) -> None:
        with state.lock:
            if state.settled:
                return
            outputs = self._collect_outputs(state.namespace)
        if not self._claim(state):
            return
        request = state.request
        request.timings.completed_ns = time.monotonic_ns()
        with self._counter_lock:
            self._executions += 1
        self._record(request)
        response = InferenceResponse(
            request_id=request.request_id,
            model_name=self.name,
            model_version=self.version,
            outputs=outputs,
            context=request.context,
            timings=request.timings,
        )
        # Recorded before settling: a caller woken by `_settle` can be reading the sink on the
        # next line, and a trace that lands after its own response is a trace nobody finds.
        self._traces.record(RequestTrace.from_response(response))
        _settle(state.future, result=response)

    def _fail(self, state: _Execution, exc: BaseException) -> None:
        if not self._claim(state):
            return  # some other step already failed this DAG; first exception wins
        self._metrics.requests_failed.inc(model=self.name)
        self._statistics.record_failure(1)
        _settle(state.future, exception=exc)

    def _retire(self, state: _Execution) -> None:
        """Give up an execution whose caller already cancelled it. No counters move: the
        request neither succeeded nor failed, and charging it to either would be a lie."""
        self._claim(state)

    def _record(self, request: InferenceRequest) -> None:
        """Credit one completed DAG to this ensemble's statistics.

        Three of Triton's five spans are **zero here, and that is a statement rather than a
        gap**: an ensemble has no queue of its own, assembles no batch and scatters no
        outputs — its steps do all three, and each step's own ``/v2/models/{step}/stats``
        holds those numbers. Attributing the steps' queue waits to the ensemble as well
        would double-count the one quantity an operator uses to decide whether a stage is
        backed up.

        ``compute_infer`` is the whole DAG, which is the only span this object actually
        measures, and ``execution_count`` equals ``inference_count`` because one request is
        one DAG — the steps inside it may now run concurrently, but they are not batched
        with another request's.
        """
        span_ns = max(0, request.timings.completed_ns - request.timings.received_ns)
        self._statistics.record_execution(
            requests=1,
            batch_size=1,
            queue_ns=0,
            compute_input_ns=0,
            compute_infer_ns=span_ns,
            compute_output_ns=0,
            total_ns=span_ns,
        )

    def _collect_outputs(self, namespace: dict[str, Tensor]) -> dict[str, Tensor]:
        """Every declared output, always.

        A branch that did not run yields a tensor with **zero rows** rather than a missing
        key. Both are "no ships in this frame", but only one of them is distinguishable
        from "the ship branch raised and something swallowed it" — and a response whose
        shape depends on the content of the frame forces every consumer to write the same
        defensive lookup.
        """
        outputs: dict[str, Tensor] = {}
        for spec in self._artifact.config.output_specs:
            produced = namespace.get(spec.name)
            if produced is not None:
                outputs[spec.name] = produced
                continue
            row_shape = tuple(dim if dim > 0 else 0 for dim in spec.shape)
            outputs[spec.name] = Tensor.from_numpy(
                np.zeros((0, *row_shape), dtype=spec.dtype.numpy_dtype)
            )
        return outputs

    # -- introspection -------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._counter_lock:
            executions, skipped = self._executions, self._skipped_steps
            running, peak = self._running_steps, self._peak_parallel_steps
        with self._live_lock:
            in_flight = len(self._live)
        return {
            "name": self.name,
            "version": self.version,
            "platform": "ensemble",
            "ready": self.is_ready,
            "steps": self._step_summaries(),
            "executions": executions,
            "skipped_steps": skipped,
            # Evidence that steps are scheduled rather than walked: a DAG whose steps only
            # ever ran one at a time cannot show a peak above 1, whatever the load.
            "running_steps": running,
            "peak_parallel_steps": peak,
            "in_flight": in_flight,
            "capacity": self._capacity,
            "instances": [],
        }

    def _step_summaries(self) -> list[dict[str, Any]]:
        """The DAG as the scheduler sees it.

        ``depends_on`` names only tensors some *step* produces, because those are the ones
        that decide when a step may run; an ensemble input is there from the first
        instruction and waiting on it would never mean anything. Before ``start()`` there
        are no plans yet, so the declaration is all there is to report.
        """
        if not self._plans:
            return [{"model": s.model, "condition": s.condition} for s in self._steps]
        return [
            {
                "model": plan.step.model,
                "condition": plan.step.condition,
                "depends_on": sorted(n for n in plan.reads if n in self._producers),
            }
            for plan in self._plans
        ]

    def model_stats(self) -> dict[str, Any]:
        """This ensemble's entry of Triton's ``model_stats`` array.

        Same shape as a plain model's, so ``/v2/models/{name}/stats`` answers for either and
        a client never has to know which it is talking to.
        """
        return self._statistics.as_dict(self.name, self.version)

    def __repr__(self) -> str:
        return f"<EnsembleModel {self.name} steps={[s.model for s in self._steps]}>"


def _settle(
    future: ResponseFuture,
    *,
    result: InferenceResponse | None = None,
    exception: BaseException | None = None,
) -> None:
    """Resolve a caller's future, tolerating a caller that cancelled it first.

    ``Future.set_exception`` raises on an already-cancelled future, and that raise would
    happen on a step model's worker thread — killing a thread that has nothing to do with
    this request over a race the caller is entitled to run.
    """
    try:
        if exception is not None:
            future.set_exception(exception)
        else:
            future.set_result(result)
    except InvalidStateError:  # pragma: no cover - the caller cancelled between checks
        _LOG.debug("ensemble future for %s was already resolved", future.request_id)


def _is_truthy(tensor: Tensor | None) -> bool:
    """Whether a condition tensor says "run this step".

    Empty means no: a detector that found no ships emits a zero-row crop tensor, and
    running a segmenter on zero crops is a wasted launch, not an error.
    """
    if tensor is None:
        return False
    if 0 in tensor.shape:
        return False
    try:
        return bool(np.any(tensor.numpy()))
    except RuntimeError:
        return True  # device-resident: assume present rather than force a readback
