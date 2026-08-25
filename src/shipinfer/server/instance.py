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
from shipinfer.core.tracing import NullTraceSink, RequestTrace, TraceSink
from shipinfer.core.types import Device
from shipinfer.runtime.device import DeviceManager
from shipinfer.runtime.profiling import PhaseTimer, phase_timer
from shipinfer.scheduling.batching import AssembledBatch, Batcher
from shipinfer.scheduling.limits import RateLimiter, UnlimitedRateLimiter
from shipinfer.scheduling.queues import BatchWindow, RequestQueue
from shipinfer.scheduling.work import WorkItem
from shipinfer.server.statistics import ModelStatistics

__all__ = ["ModelInstance"]

_LOG = get_logger("server.instance")

#: How long a worker waits for an execution slot before re-checking that it should still be
#: running. The wait has to be bounded or a stopping server would sit inside the limiter
#: until some other instance finished its batch; 50 ms is short against a shutdown grace of
#: seconds and long enough that the re-check costs nothing measurable. A wait that expires is
#: an ordinary "not yet" and the batch is retried, never dropped — the queue is the only
#: place in this system that sheds work, on purpose (ADR-005).
_SLOT_WAIT_S = 0.05


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
        statistics: ModelStatistics | None = None,
        limiter: RateLimiter | None = None,
        traces: TraceSink | None = None,
    ) -> None:
        self.name = name
        self._backend = backend
        self._queue = queue
        self._batcher = batcher
        self._window = window
        self._devices = devices
        self._metrics = metrics
        self._alpha = scheduler.latency_ewma_alpha
        # All three default to the inert implementation rather than to None, so the hot path
        # has no `if x is not None` on it and no caller can forget one. `Model` supplies the
        # real objects; the statistics object is per model and shared with the model's other
        # instances, which is what makes the totals a model's totals.
        self._statistics = statistics if statistics is not None else ModelStatistics()
        self._limiter = limiter if limiter is not None else UnlimitedRateLimiter()
        self._traces = traces if traces is not None else NullTraceSink()
        self._rate_limit_waits = 0

        self._thread: threading.Thread | None = None
        # stop() may be called from the server's shutdown thread while the worker is still
        # running, and twice if a caller retries; the lock makes it idempotent rather than
        # merely usually-idempotent.
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._abandoned = False
        #: Whether this instance's readiness was counted, so shutdown cannot decrement a
        #: gauge it never incremented and drive it negative.
        self._counted_ready = False
        self._running = threading.Event()
        self._ready = threading.Event()
        # Set when the worker has *settled* either way — ready, or failed to start. Without
        # it `wait_ready` cannot tell "still loading" from "already dead", so a model whose
        # backend raised on the first line held start-up for the whole 120 s timeout before
        # reporting a failure it had known about immediately. A typo in `model_warmup` is
        # enough to reach that path.
        self._settled = threading.Event()
        self._start_error: BaseException | None = None
        self._ewma_latency_us = 0.0
        self._executed_batches = 0
        self._executed_requests = 0
        # One timer for the life of the instance, not one per batch. `PhaseTimer` documents
        # that its CUDA events are reused *because* allocating a pair per phase per batch
        # would be ~5000 allocations a second at the design point and would make the
        # instrument part of what it measures — and then `_execute` rebuilt it every batch,
        # so the cache was empty every time. One instance is one thread, so there is no
        # sharing to worry about.
        self._phase_timer = phase_timer()
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
        """Block until the worker is ready or has failed. True only if it is ready.

        Returns as soon as either is known: a failed start is an answer, not something to
        keep waiting for.
        """
        self._settled.wait(timeout)
        return self._ready.is_set()

    @property
    def start_error(self) -> BaseException | None:
        """Why the worker never became ready, or ``None``.

        Kept so the caller can say *what* went wrong rather than "did not become ready
        within 120s", which describes the symptom and hides the cause.
        """
        return self._start_error

    def stop(self, grace_s: float = 10.0) -> None:
        """Stop accepting work, drain, and join.

        Idempotent and callable from any thread.

        The one thing this must never do is tear down a backend that is still in use. A
        worker stuck inside ``execute_async_v3`` still holds the TensorRT execution context
        and every binding tensor; finalising underneath it frees memory the GPU is actively
        reading, which is a use-after-free that presents as corrupted output or a driver
        fault somewhere unrelated. So a join that times out means the backend is
        **abandoned, not finalised** — leaking it is strictly better, and the leak is
        recorded rather than hidden.

        Queued work is failed by :meth:`RequestQueue.close`. Work already in flight belongs
        to the worker thread and is resolved by it; racing to complete those futures from
        here would double-resolve them the moment the thread finished.
        """
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._running.clear()
            self._ready.clear()
            thread, self._thread = self._thread, None

        self._queue.close()

        if thread is not None:
            thread.join(timeout=grace_s)
            if thread.is_alive():
                self._abandoned = True
                _LOG.error(
                    "instance %s did not stop within %.1fs; abandoning its backend rather "
                    "than finalising it underneath a running batch (the memory is leaked "
                    "for the life of the process)",
                    self.name,
                    grace_s,
                )

        if not self._abandoned:
            self._backend.finalize()

        if self._counted_ready:
            self._counted_ready = False
            self._metrics.instances_ready.dec(model=self._model_label())

    @property
    def is_abandoned(self) -> bool:
        """True when :meth:`stop` gave up on the worker and left its backend alive."""
        return self._abandoned

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
        except Exception as exc:
            _LOG.exception("instance %s failed to start", self.name)
            self._start_error = exc
            self._queue.close(InferenceError(f"instance {self.name} failed to start: {exc}"))
            self._settled.set()
            return

        self._ready.set()
        self._settled.set()
        self._counted_ready = True
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
            self._execute_when_permitted(items)

        # Drain anything that arrived between the flag clearing and the queue closing.
        for item in self._queue.close():
            item.fail(RequestCancelledError(f"instance {self.name} stopped"))

    def _execute_when_permitted(self, items: list[WorkItem]) -> None:
        """Hold one of the model's execution slots for the length of the batch.

        The rate limiter bounds *concurrency*, which is a different bound from the queue's:
        the queue says how much work may be waiting, this says how much may be running. Eight
        instances whose batching windows close together otherwise enter compute at the same
        instant, on one memory bus and one PCIe root complex.

        The batch has already left the queue by the time this runs, so it is never dropped
        for want of a slot — it waits, which converts the burst back into queue depth, a
        quantity the rest of the system already measures and sheds at the edge. The only
        outcome that fails the batch is the server stopping while it waits.

        With the default ``off`` limiter every acquire succeeds on the first attempt, so this
        costs two method calls per *batch* — not per request.

        The acquire is attempted **before** the running flag is consulted, and that order is
        deliberate: a batch that already has a slot is executed even if the server began
        stopping between the queue handing it over and this line. Checking the flag first
        would widen the shutdown race — a request that used to be served would start failing
        with a cancellation, for every model, including the ones with no limiter at all.
        """
        while True:
            if self._limiter.acquire(_SLOT_WAIT_S):
                try:
                    self._execute(items)
                finally:
                    self._limiter.release()
                return
            self._rate_limit_waits += 1
            if not self._running.is_set():
                break
        self._fail_batch(
            items,
            RequestCancelledError(
                f"instance {self.name} stopped while waiting for an execution slot"
            ),
        )

    def _execute(self, items: list[WorkItem]) -> None:
        label = self._model_label()
        device = str(self.device)
        batched_ns = time.monotonic_ns()
        timer = self._phase_timer

        try:
            # Triton counts batch assembly and the host-to-device copy together as
            # `compute_input`; the backend owns the copy, so this span is assembly plus
            # whatever staging the batcher does.
            with timer.phase("compute_input"):
                batch = self._batcher.assemble(items)
        except Exception as exc:
            # An unassemblable batch is one bad *request*, not a broken instance. Fail them
            # all with the same reason rather than killing the worker.
            self._fail_batch(items, exc)
            return

        queue_ns = 0
        for item in items:
            timings = item.request.timings
            timings.batched_ns = batched_ns
            queue_ns += max(0, batched_ns - timings.queued_ns)
            self._metrics.queue_wait_us.observe(timings.queue_us, model=label)

        start_ns = time.monotonic_ns()
        try:
            # The backend owns the copies, so this is the finest split available without
            # reaching inside it. `execute` is annotated for NVTX either way, which is what
            # makes an `nsys` timeline readable at all.
            with timer.phase("compute_infer"):
                outputs = self._backend.execute(batch.inputs, batch.size)
        except Exception as exc:
            self._failed_batches += 1
            self._fail_batch(items, exc)
            return
        end_ns = time.monotonic_ns()

        compute_us = (end_ns - start_ns) / 1_000.0
        self._observe(compute_us, batch, label, device)

        try:
            with timer.phase("compute_output"):
                scattered = self._batcher.scatter(batch, outputs)
        except Exception as exc:
            self._fail_batch(items, exc)
            return

        completed_ns = time.monotonic_ns()
        self._observe_phases(timer, batched_ns, completed_ns, label, device)
        total_ns = 0
        for item, per_request in zip(batch.items, scattered, strict=True):
            timings = item.request.timings
            timings.compute_start_ns = start_ns
            timings.compute_end_ns = end_ns
            timings.completed_ns = completed_ns
            total_ns += max(0, completed_ns - timings.received_ns)
            self._complete(item, per_request, completed_ns)

        # Host spans, not device spans, and the difference is worth stating: the backend owns
        # the copies, so `compute_infer` is wall-clock around `execute` rather than a CUDA
        # event pair. That is what Triton reports too for a synchronous backend, and the
        # event-timed split is available separately behind SHIPINFER_PROFILE_PHASES.
        self._statistics.record_execution(
            requests=batch.request_count,
            batch_size=batch.size,
            queue_ns=queue_ns,
            compute_input_ns=start_ns - batched_ns,
            compute_infer_ns=end_ns - start_ns,
            compute_output_ns=completed_ns - end_ns,
            total_ns=total_ns,
        )

    def _observe_phases(
        self,
        timer: PhaseTimer,
        batched_ns: int,
        completed_ns: int,
        label: str,
        device: str,
    ) -> None:
        """Record the phase split, when it was measured.

        Gated on ``is_measured`` rather than on the env var: with phase timing off,
        ``device_busy_us`` is zero and the idle fraction would read 100% — a confident
        falsehood about precisely the quantity this exists to establish. An unmeasured batch
        contributes nothing rather than contributing a wrong zero.
        """
        timings = timer.finish((completed_ns - batched_ns) / 1_000.0)
        if not timings.is_measured:
            return
        for phase, micros in timings.per_phase.items():
            self._metrics.phase_us.observe(micros, model=label, device=device, phase=phase)
        self._metrics.device_idle_ratio.observe(
            timings.idle_fraction, model=label, device=device
        )

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
        # After the future resolves, so tracing can never delay the caller's answer, and
        # behind `should_record` so a server with tracing off (the default) does not build a
        # record it would immediately discard.
        if self._traces.should_record():
            self._traces.record(RequestTrace.from_response(response))

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
        self._statistics.record_failure(len(items))

    # -- introspection -------------------------------------------------------------------

    def _model_label(self) -> str:
        return self._backend.context.artifact.name

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": str(self.device),
            "ready": self.is_ready,
            "abandoned": self._abandoned,
            "queue": self._queue.stats().as_dict(),
            "batches": self._executed_batches,
            "requests": self._executed_requests,
            "failed_batches": self._failed_batches,
            #: Times this worker had to wait for an execution slot. Non-zero means the
            #: model's rate limiter is actually binding, which is the only way to tell a
            #: limiter that is shaping a burst from one that is configured and never reached.
            "rate_limit_waits": self._rate_limit_waits,
            "ewma_latency_us": round(self._ewma_latency_us, 1),
            "backend": self._backend.stats(),
        }

    def __repr__(self) -> str:
        return f"<ModelInstance {self.name} on {self.device} depth={self.depth}>"
