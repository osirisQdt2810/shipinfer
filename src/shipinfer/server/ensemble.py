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
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
from shipinfer.core.types import Tensor, TensorSpec, validate_against
from shipinfer.repository import EnsembleStep, ModelArtifact

__all__ = ["EnsembleModel"]

_LOG = get_logger("server.ensemble")


class _Servable(Protocol):
    """The slice of a model an ensemble step needs. Both ``Model`` and ``EnsembleModel``
    satisfy it, so ensembles nest without a special case."""

    name: str

    def infer(self, request: InferenceRequest) -> ResponseFuture: ...

    @property
    def is_ready(self) -> bool: ...


class EnsembleModel:
    """Executes a declared DAG of other models.

    Presents the same surface as :class:`~shipinfer.server.model.Model` — ``start``,
    ``stop``, ``infer``, ``stats``, ``is_ready`` — so :class:`InferenceServer` holds both
    in one table and callers cannot tell which they are talking to.

    Steps run **sequentially** on a small thread pool. Sequential is not a limitation of
    the implementation but of the graph: in this pipeline every step consumes the previous
    step's output. What provides the parallelism is that each step's *own* model batches
    across all the frames in flight, so while frame A is embedding, frame B is detecting.
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        settings: ServerSettings,
        metrics: ServerMetrics,
        resolve: Callable[[str], _Servable],
        *,
        max_workers: int = 8,
        max_pending: int = 0,
    ) -> None:
        self._artifact = artifact
        self._settings = settings
        self._metrics = metrics
        self._resolve = resolve
        self._steps = tuple(artifact.config.ensemble.steps)  # type: ignore[union-attr]
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"ensemble-{artifact.name}"
        )
        # A ThreadPoolExecutor's queue is unbounded, so an ensemble under load accumulates
        # work forever and never applies the backpressure every other path in this system
        # applies. The semaphore is the bound; exceeding it raises QueueFullError, exactly
        # as a saturated instance queue would.
        self._capacity = max(1, max_pending if max_pending > 0 else max_workers * 4)
        self._slots = threading.Semaphore(self._capacity)
        self._pending: list[tuple[InferenceRequest, ResponseFuture]] = []
        self._pending_lock = threading.Lock()
        self._started = False
        self._executions = 0
        self._skipped_steps = 0

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
        produce a confusing ``KeyError`` on the thousandth frame.
        """
        self._validate_graph()
        self._started = True
        _LOG.info(
            "ensemble %s ready: %s",
            self.name,
            " -> ".join(step.model for step in self._steps),
        )

    def _validate_graph(self) -> None:
        """Type-check the wiring: every edge must exist, and both ends must agree.

        Names alone are not enough. A crop sized for a 512x512 segmenter fed to a 256x128
        embedder has a perfectly valid *name* and is still wrong — it would surface as a
        validation error on the first frame, from inside a worker thread, three models away
        from the config line that caused it. Checking dtype and shape here turns that into
        a start-up failure that names both ends of the edge.
        """
        # tensor name -> the spec that produced it
        available: dict[str, TensorSpec] = {
            spec.name: spec for spec in self._artifact.config.input_specs
        }
        # Tensors that only exist when some branch ran. Consuming one unconditionally is a
        # wiring error: it works on every frame that happens to take the branch and fails
        # on the first one that does not.
        conditional: set[str] = set()

        for index, step in enumerate(self._steps):
            model = self._resolve(step.model)  # raises ModelNotFoundError if missing
            step_config = getattr(model, "artifact", self._artifact).config
            where = f"ensemble {self.name!r} step {index} ({step.model})"

            for spec in step_config.input_specs:
                source = step.input_map.get(spec.name, spec.name)
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

            if step.condition and step.condition not in available:
                raise ConfigurationError(
                    f"{where}: condition {step.condition!r} is not produced by any earlier "
                    f"step (available: {sorted(available)})"
                )

            for spec in step_config.output_specs:
                name = step.output_map.get(spec.name, spec.name)
                available[name] = TensorSpec(
                    name=name, dtype=spec.dtype, shape=spec.shape, optional=spec.optional
                )
                if step.condition:
                    conditional.add(name)

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

    def stop(self) -> None:
        """Stop accepting work and drain.

        ``cancel_futures=True`` would drop queued items on the floor, and the futures it
        drops are the *caller's* — they would never resolve and every waiter would block
        forever. So the queued requests are cancelled explicitly, with a typed error,
        before the pool is told to shut down.
        """
        self._started = False
        with self._pending_lock:
            pending, self._pending = list(self._pending), []
        for _request, future in pending:
            if future.set_running_or_notify_cancel():
                future.set_exception(
                    RequestCancelledError(f"ensemble {self.name!r} stopped before this ran")
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

        future = ResponseFuture(request)
        with self._pending_lock:
            self._pending.append((request, future))
        self._pool.submit(self._run, request, future)
        return future

    def _run(self, request: InferenceRequest, future: ResponseFuture) -> None:
        with self._pending_lock:
            entry = (request, future)
            if entry in self._pending:
                self._pending.remove(entry)
        try:
            self._execute_graph(request, future)
        finally:
            self._slots.release()

    def _execute_graph(self, request: InferenceRequest, future: ResponseFuture) -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            namespace = dict(request.inputs)
            for step in self._steps:
                self._run_step(step, request, namespace)
            outputs = self._collect_outputs(namespace)
            request.timings.completed_ns = time.monotonic_ns()
            self._executions += 1
            future.set_result(
                InferenceResponse(
                    request_id=request.request_id,
                    model_name=self.name,
                    model_version=self.version,
                    outputs=outputs,
                    context=request.context,
                    timings=request.timings,
                )
            )
        except Exception as exc:
            self._metrics.requests_failed.inc(model=self.name)
            future.set_exception(exc)

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

    def _run_step(
        self, step: EnsembleStep, request: InferenceRequest, namespace: dict[str, Tensor]
    ) -> None:
        if step.condition is not None and not _is_truthy(namespace.get(step.condition)):
            self._skipped_steps += 1
            return

        model = self._resolve(step.model)
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

        step_request = InferenceRequest(
            model_name=step.model,
            inputs=inputs,
            model_version=step.model_version,
            context=request.context,  # the (camera, frame) tag rides the whole DAG
            priority=request.priority,
            deadline_ns=request.deadline_ns,
        )
        response = model.infer(step_request).result()
        for name, tensor in response.outputs.items():
            namespace[step.output_map.get(name, name)] = tensor

    # -- introspection -------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "platform": "ensemble",
            "ready": self.is_ready,
            "steps": [{"model": s.model, "condition": s.condition} for s in self._steps],
            "executions": self._executions,
            "skipped_steps": self._skipped_steps,
            "instances": [],
        }

    def __repr__(self) -> str:
        return f"<EnsembleModel {self.name} steps={[s.model for s in self._steps]}>"


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
