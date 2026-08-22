"""One model instance: a backend copy, a queue, and the worker thread that drains it.

This is where the design's central invariant is enforced: **one worker thread, one CUDA
context, one GPU**. The thread binds itself to its device once at start-up and never
changes. Nothing in this system ever reaches across into another GPU's memory; work moves
between GPUs by being *queued* somewhere else, which is a CPU-side decision made by the
dispatcher on metadata alone.

Getting that right is what makes multi-GPU load balancing simple rather than a lesson in
peer-to-peer topology (ADR-002).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from shipinfer.backends.base import ModelBackend
from shipinfer.core.errors import InferenceError, RequestCancelledError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.metrics import ServerMetrics
from shipinfer.core.request import InferenceResponse
from shipinfer.core.settings import SchedulerSettings
from shipinfer.core.types import Device
from shipinfer.runtime.device import DeviceManager
from shipinfer.scheduling.batching import AssembledBatch, Batcher
from shipinfer.scheduling.queues import BatchWindow, RequestQueue
from shipinfer.scheduling.work import WorkItem

__all__ = ["ModelInstance"]

_LOG = get_logger("server.instance")


class ModelInstance:
    """A backend copy pinned to one device, fed by its own bounded queue.

    Satisfies :class:`~shipinfer.scheduling.policies.Placeable`, which is all the placement
    policies are allowed to see of it.
    """

    def __init__(
        self,
        name: str,
        backend: ModelBackend,
        queue: RequestQueue,
        batcher: Batcher,
        window: BatchWindow,
        devices: DeviceManager,
        metrics: ServerMetrics,
        scheduler: SchedulerSettings,
    ) -> None:
        self.name = name
        self._backend = backend
        self._queue = queue
        self._batcher = batcher
        self._window = window
        self._devices = devices
        self._metrics = metrics
        self._alpha = scheduler.latency_ewma_alpha

        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._ready = threading.Event()
        self._ewma_latency_us = 0.0
        self._executed_batches = 0
        self._executed_requests = 0
        self._failed_batches = 0

    # -- Placeable -----------------------------------------------------------------------

    @property
    def device(self) -> Device:
        return self._backend.device

    @property
    def depth(self) -> int:
        return self._queue.depth

    @property
    def ewma_latency_us(self) -> float:
        return self._ewma_latency_us

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    # -- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread and block until the backend is loaded.

        Blocking here is deliberate: ``server.start()`` must not return while a model is
        still warming up, or the first client request races the engine load and the whole
        point of warm-up is lost.
        """
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name=f"shipinfer-{self.name}", daemon=True
        )
        self._thread.start()

    def wait_ready(self, timeout: float = 120.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self, grace_s: float = 10.0) -> None:
        """Stop accepting work, drain, and join. Safe to call twice."""
        self._running.clear()
        self._ready.clear()
        self._queue.close()
        if self._thread is not None:
            self._thread.join(timeout=grace_s)
            if self._thread.is_alive():
                _LOG.warning("instance %s did not stop within %.1fs", self.name, grace_s)
            self._thread = None
        self._backend.finalize()
        self._metrics.instances_ready.dec(model=self._model_label())

    # -- producer side -------------------------------------------------------------------

    def enqueue(self, item: WorkItem) -> None:
        """Hand a request to this instance.

        Raises:
            QueueFullError: which the dispatcher catches and turns into a spill.
        """
        self._queue.put(item)
        self._metrics.queue_depth.set(
            self._queue.depth, model=self._model_label(), device=str(self.device)
        )

    # -- worker --------------------------------------------------------------------------

    def _run(self) -> None:
        label = self._model_label()
        try:
            self._devices.bind_current_thread(self.device)
            self._backend.initialize()
            warmup = self._backend.context.execution.warmup_iterations
            self._backend.warmup(warmup)
        except Exception:
            _LOG.exception("instance %s failed to start", self.name)
            self._queue.close(InferenceError(f"instance {self.name} failed to start"))
            return

        self._ready.set()
        self._metrics.instances_ready.inc(model=label)
        _LOG.info(
            "instance %s ready",
            self.name,
            extra=log_context(model=label, device=str(self.device), instance=self.name),
        )

        while self._running.is_set():
            items = self._queue.get_batch(self._window)
            if not items:
                continue  # closed, or a spurious wake-up
            self._execute(items)

        # Drain anything that arrived between the flag clearing and the queue closing.
        for item in self._queue.close():
            item.fail(RequestCancelledError(f"instance {self.name} stopped"))

    def _execute(self, items: list[WorkItem]) -> None:
        label = self._model_label()
        device = str(self.device)
        batched_ns = time.monotonic_ns()

        try:
            batch = self._batcher.assemble(items)
        except Exception as exc:
            # An unassemblable batch is one bad *request*, not a broken instance. Fail them
            # all with the same reason rather than killing the worker.
            self._fail_batch(items, exc)
            return

        for item in items:
            item.request.timings.batched_ns = batched_ns
            self._metrics.queue_wait_us.observe(item.request.timings.queue_us, model=label)

        start_ns = time.monotonic_ns()
        try:
            outputs = self._backend.execute(batch.inputs, batch.size)
        except Exception as exc:
            self._failed_batches += 1
            self._fail_batch(items, exc)
            return
        end_ns = time.monotonic_ns()

        compute_us = (end_ns - start_ns) / 1_000.0
        self._observe(compute_us, batch, label, device)

        try:
            scattered = self._batcher.scatter(batch, outputs)
        except Exception as exc:
            self._fail_batch(items, exc)
            return

        completed_ns = time.monotonic_ns()
        for item, per_request in zip(batch.items, scattered, strict=True):
            timings = item.request.timings
            timings.compute_start_ns = start_ns
            timings.compute_end_ns = end_ns
            timings.completed_ns = completed_ns
            self._complete(item, per_request, completed_ns)

    def _observe(
        self, compute_us: float, batch: AssembledBatch, label: str, device: str
    ) -> None:
        self._executed_batches += 1
        self._executed_requests += batch.request_count
        # EWMA rather than a running mean: the placement policies want "how loaded is this
        # instance *now*", and a lifetime average stops responding after an hour of uptime.
        self._ewma_latency_us = (
            compute_us
            if self._ewma_latency_us == 0.0
            else (1 - self._alpha) * self._ewma_latency_us + self._alpha * compute_us
        )
        self._metrics.batches_total.inc(model=label, device=device)
        self._metrics.batch_size.observe(batch.size, model=label)
        self._metrics.compute_us.observe(compute_us, model=label, device=device)
        self._metrics.queue_depth.set(self._queue.depth, model=label, device=device)

    def _complete(self, item: WorkItem, outputs: dict, completed_ns: int) -> None:
        if not item.future.set_running_or_notify_cancel():
            return  # the caller gave up while we computed
        artifact = self._backend.context.artifact
        response = InferenceResponse(
            request_id=item.request.request_id,
            model_name=artifact.name,
            model_version=artifact.version,
            outputs=outputs,
            context=item.request.context,
            timings=item.request.timings,
            executed_on=self.device,
        )
        item.future.set_result(response)
        self._metrics.e2e_us.observe(
            item.request.timings.total_us, model=artifact.name, device=str(self.device)
        )

    def _fail_batch(self, items: list[WorkItem], error: BaseException) -> None:
        label = self._model_label()
        _LOG.error(
            "batch of %d failed on %s: %s",
            len(items),
            self.name,
            error,
            extra=log_context(model=label, device=str(self.device), batch_size=len(items)),
        )
        for item in items:
            item.fail(error)
        self._metrics.requests_failed.inc(len(items), model=label, device=str(self.device))

    # -- introspection -------------------------------------------------------------------

    def _model_label(self) -> str:
        return self._backend.context.artifact.name

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": str(self.device),
            "ready": self.is_ready,
            "queue": self._queue.stats().as_dict(),
            "batches": self._executed_batches,
            "requests": self._executed_requests,
            "failed_batches": self._failed_batches,
            "ewma_latency_us": round(self._ewma_latency_us, 1),
            "backend": self._backend.stats(),
        }

    def __repr__(self) -> str:
        return f"<ModelInstance {self.name} on {self.device} depth={self.depth}>"
