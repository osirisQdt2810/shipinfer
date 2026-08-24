"""The whole flow in one object: cameras in, events out.

Read this file to understand the pipeline. Everything else is a piece of it::

    camera actor ──put(frame)──► QueueFrameSink ──WorkItem──► fair, bounded queue
                                                                     │
                                              ┌──────────────────────┘
                                              ▼  N worker threads
                          FrameState ──► PipelineGraph.execute ──► stage results
                                              │                        │
                                              ▼                        ▼
                                        FrameCollector  ◄── planned / delivered
                                              │
                                    complete · sealed · timed out
                                              ▼
                                     PerceptionEvent ──► ResultSink

Four choices in that picture are worth defending.

**The queue in front of the pipeline is the fair one.** It is the same
:class:`~shipinfer.scheduling.queues.FairPriorityQueue` the model instances use, bucketed by
``camera_id`` with priority lanes above it, and it is where a camera actor is told "no" by
:class:`~shipinfer.core.errors.QueueFullError`. There is no second queue and no second
fairness mechanism (ADR-005).

**A worker runs one frame's graph to completion.** Stages are sequential because in this DAG
each one consumes the previous one's output; the concurrency comes from having several
workers and from each *model* batching across every frame in flight. That is the same
reasoning :class:`shipinfer.server.ensemble.EnsembleModel` gives for its steps, and keeping
the two consistent means one mental model for the whole server.

**Emission belongs to the collector, not to the worker.** The worker's last act is to *seal*
the frame; the collector decides whether that frame was complete, and it is also what emits a
frame whose worker died inside a stage. One exit path per frame, wherever the frame died.

**The runner does not own the cameras.** It exposes :attr:`frame_sink` and takes an optional
factory, so ingest is constructed *against* the pipeline rather than inside it. That keeps
the direction of dependency the same as everywhere else — ``pipeline`` supplies the sink that
``ingest`` publishes into (ADR-011) — and it is what lets the 50-camera bench, the CLI and a
test each supply their own producer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from shipinfer.core.errors import (
    ConfigurationError,
    InferenceError,
    RequestCancelledError,
    ServerStateError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.request import ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Device
from shipinfer.pipeline.graph import (
    FrameState,
    PipelineGraph,
    StageOutcome,
    StageStatus,
    build_perception_graph,
)
from shipinfer.pipeline.graph.ops import ThreadLocalImageOps
from shipinfer.pipeline.metrics import PipelineMetrics
from shipinfer.pipeline.reassembly import EVICTED, FrameCollector, FrameResult
from shipinfer.pipeline.schema import PerceptionEvent
from shipinfer.pipeline.sink import QueueFrameSink
from shipinfer.pipeline.sinks import RESULT_SINKS, ResultSink
from shipinfer.runtime.ops import ImageOps, get_image_ops
from shipinfer.scheduling.queues import QUEUES, BatchWindow, RequestQueue
from shipinfer.scheduling.work import WorkItem

__all__ = ["FrameProducer", "PipelineRunner"]

_LOG = get_logger("pipeline.runner")


class FrameProducer(Protocol):
    """Anything that feeds the pipeline and can be started and stopped.

    :class:`shipinfer.ingest.IngestManager` satisfies it; so does a bench's synthetic
    generator. Structural rather than an import, for the same reason
    :class:`~shipinfer.pipeline.sink.TaggedFrame` is: the layering DAG gives ``pipeline`` no
    edge to ``ingest``, and a two-method protocol satisfies the dependency without one.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...


class _CollectorObserver:
    """Bridges the graph's stage events onto the collector. One per frame, in flight only."""

    __slots__ = ("_collector", "_key", "_metrics")

    def __init__(
        self, collector: FrameCollector, key: tuple[str, int], metrics: PipelineMetrics
    ) -> None:
        self._collector = collector
        self._key = key
        self._metrics = metrics

    def planned(self, stages: tuple[str, ...] | list[str]) -> None:
        self._collector.expect(self._key, tuple(stages))

    def finished(self, outcome: StageOutcome) -> None:
        if outcome.status is StageStatus.SKIPPED:
            # Never expected, so never missing: the collector is told which stages *will*
            # run, and a skipped branch was never one of them.
            self._metrics.stages_skipped.inc(stage=outcome.stage)
            return
        if outcome.status is StageStatus.FAILED:
            self._metrics.stages_failed.inc(stage=outcome.stage)
            _LOG.warning(
                "stage %s failed for %s: %r",
                outcome.stage,
                self._key,
                outcome.error,
                extra=log_context(camera_id=self._key[0], frame_id=self._key[1]),
            )
            # Deliberately *not* delivered. The stage was expected and did not produce a
            # result, so it stays in `missing` and the emitted event names it.
            return
        self._metrics.stages_run.inc(stage=outcome.stage)
        self._metrics.stage_latency_us.observe(outcome.elapsed_us, stage=outcome.stage)
        self._collector.deliver(self._key, outcome.stage)


class PipelineRunner:
    """Owns the ingest queue, the workers, reassembly and the result sink.

    Args:
        server: a **started** :class:`shipinfer.server.InferenceServer`. Injected rather than
            constructed here: two runners over one server is a legitimate thing to want, and
            the server's lifecycle is longer than the pipeline's.
        settings: the deployment settings. Defaults to the server's own, which is what a
            single-process deployment wants.
        graph: override the DAG. A test builds a two-stage graph; production uses
            :func:`~shipinfer.pipeline.graph.build_perception_graph`.
        ops: image operations for pre-processing and cropping. Defaults to the best available
            for this host — fused kernels, then torch, then numpy (ADR-003, ADR-007).
        sink: where events go. Defaults to the registered sink named in settings.
        frames: ``factory(frame_sink) -> FrameProducer``, started last and stopped first. A
            factory rather than an instance because the producer needs the sink this runner
            owns, and passing a half-built object between them is how initialisation order
            becomes a bug.
    """

    def __init__(
        self,
        server: Any,
        *,
        settings: ServerSettings | None = None,
        graph: PipelineGraph | None = None,
        ops: ImageOps | None = None,
        sink: ResultSink | None = None,
        metrics: PipelineMetrics | None = None,
        queue: RequestQueue | None = None,
        frames: Callable[[QueueFrameSink], FrameProducer] | None = None,
    ) -> None:
        # Every default below is `if x is None`, never `x or default`, and that is not
        # pedantry: a queue and a sink both define `__len__`, so a freshly injected one is
        # *falsy* and `x or default` silently throws the caller's object away. It cost an
        # afternoon here — an injected queue's capacity was ignored and an injected sink
        # received nothing — and the shape of the bug is invisible at the call site.
        self._server = server
        if settings is None:
            settings = getattr(server, "settings", None) or ServerSettings()
        self._settings = settings
        pipeline = self._settings.pipeline
        self._metrics = PipelineMetrics() if metrics is None else metrics
        self._ops = self._build_ops() if ops is None else ops
        self._graph = (
            build_perception_graph(pipeline, resolve=server.model, ops=self._ops)
            if graph is None
            else graph
        )
        self._queue = (
            QUEUES.create(
                pipeline.queue_type,
                "pipeline",
                pipeline.queue_capacity,
                overflow=pipeline.overflow_policy,
                drop_expired=True,
            )
            if queue is None
            else queue
        )
        self._frame_sink = QueueFrameSink(self._queue, settings=self._settings.ingest)
        self._sink = (
            RESULT_SINKS.create(pipeline.result_sink, **pipeline.result_sink_options)
            if sink is None
            else sink
        )
        self._collector = FrameCollector(
            self._emit,
            settings=pipeline.reassembly,
            metrics=self._metrics,
        )
        self._producer_factory = frames
        self._producer: FrameProducer | None = None
        # Per-camera frame rate for the emitted event's `img_fps`. Ingest configuration, so
        # resolved once here rather than carried on a thousand frames a second.
        self._fps: Mapping[str, float] = {
            camera.camera_id: camera.fps for camera in self._settings.ingest.cameras
        }
        self._awaiting: dict[tuple[str, int], ResponseFuture] = {}
        self._workers: list[threading.Thread] = []
        self._sweeper: threading.Thread | None = None
        self._stopping = threading.Event()
        self._started = False
        self._input_name = self._settings.ingest.input_name
        self._budget_ns = pipeline.frame_budget_ms * 1_000_000
        self._window = BatchWindow(max_batch_size=pipeline.frames_per_wakeup)
        self.frames_accepted = 0

    def _build_ops(self) -> ImageOps:
        """One image-ops instance per worker thread, bound to one device for the thread's life.

        Three defects this avoids, all of them found by running the end-to-end test on a
        machine that has GPUs, and all of them silent or misleading in their raw form:

        * **not one shared instance.** An ``ImageOps`` owns a per-instance staging ring, so
          sharing one across workers overwrites a pinned buffer while its DMA is in flight.
          It surfaces as ``GpuError: crop_kernel failed: invalid argument`` from inside a
          stage, which reads like a bad bounding box.
        * **not all on device 0.** Letterboxing every frame on ``cuda:0`` while the models
          are spread over sixteen GPUs re-creates, one layer up, the exact imbalance this
          project exists to fix.
        * **bound, once, on the thread that will use it.** A worker whose current device is
          0 holding ops built for ``cuda:1`` gets ``GpuError: invalid resource handle`` —
          the events and the stream belong to another context. ADR-002's rule is one thread,
          one context, one GPU for the thread's whole life, and a pipeline worker that does
          pre-processing on a GPU is no exception.
        """
        provider = self._settings.execution.provider
        manager = getattr(self._server, "devices", None)
        devices = tuple(getattr(manager, "visible_gpus", ()) or (0,))

        def build(index: int) -> ImageOps:
            if manager is not None and getattr(manager, "has_accelerator", False):
                # Called lazily, on the worker thread itself: a CUDA context belongs to the
                # thread that created it, so binding from `start()` would bind the wrong one.
                manager.bind_current_thread(Device.cuda(index))
            return get_image_ops(provider, device_index=index)

        return ThreadLocalImageOps(build, devices=devices)

    # -- properties ----------------------------------------------------------------------

    @property
    def frame_sink(self) -> QueueFrameSink:
        """Where a camera actor publishes. The ingest plane's only view of the pipeline."""
        return self._frame_sink

    @property
    def graph(self) -> PipelineGraph:
        return self._graph

    @property
    def queue(self) -> RequestQueue:
        return self._queue

    @property
    def collector(self) -> FrameCollector:
        return self._collector

    @property
    def sink(self) -> ResultSink:
        return self._sink

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopping.is_set()

    # -- lifecycle -----------------------------------------------------------------------

    def start(self) -> PipelineRunner:
        """Validate the wiring, start the workers, then start the cameras. Idempotent.

        Validation first and cameras last, both deliberately. A mis-wired graph must stop a
        deploy rather than surface on the thousandth frame, and starting the producer before
        the consumers would drop frames for no reason other than start-up order.
        """
        if self._started:
            return self
        if not getattr(self._server, "is_started", False):
            raise ServerStateError(
                "the pipeline needs a started server; call InferenceServer.start() first"
            )
        self._graph.validate(self._server.model)
        self._check_entry_model()

        pipeline = self._settings.pipeline
        self._stopping.clear()
        for index in range(pipeline.workers):
            worker = threading.Thread(
                target=self._work, name=f"pipeline-worker-{index}", daemon=True
            )
            worker.start()
            self._workers.append(worker)
        self._sweeper = threading.Thread(
            target=self._sweep, name="pipeline-sweeper", daemon=True
        )
        self._sweeper.start()
        self._started = True

        if self._producer_factory is not None:
            self._producer = self._producer_factory(self._frame_sink)
            self._producer.start()
        _LOG.info(
            "pipeline ready: %s | %d worker(s) | sink=%s",
            " -> ".join(self._graph.stage_names),
            pipeline.workers,
            self._sink.name,
        )
        return self

    def _check_entry_model(self) -> None:
        """The frames ingest submits must be aimed at the graph's first stage.

        Two settings have to agree — ``ingest.target_model`` and the graph's entry model —
        and they live apart because one is video configuration and the other is the DAG. A
        mismatch means every frame arrives addressed to a model the pipeline never runs, so
        it is checked once, here, rather than being silently ignored per frame.
        """
        entry = self._graph.entry_model
        target = self._settings.ingest.target_model
        if entry is not None and target != entry:
            raise ConfigurationError(
                f"ingest.target_model is {target!r} but the graph's first stage runs "
                f"{entry!r}; a frame would be submitted to a model this pipeline does not "
                f"start with. Set ingest.target_model={entry!r} or change the graph."
            )

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop the cameras, drain, and publish what was still in flight. Idempotent.

        The order is the reverse of :meth:`start` and every step is there for a reason:
        cameras first so nothing new arrives; the queue next, which fails everything still
        queued with a typed error rather than dropping it; then the workers; then reassembly
        is drained so a half-finished frame is published as ``shutdown`` instead of
        disappearing; then the sink is flushed and closed.
        """
        if not self._started:
            return
        self._stopping.set()
        if self._producer is not None:
            try:
                self._producer.stop()
            except Exception:
                _LOG.exception("frame producer failed to stop cleanly")
            self._producer = None

        lost = self._queue.close()
        for item in lost:
            self._awaiting.pop(item.request.context.key, None)
        for worker in self._workers:
            worker.join(timeout_s)
        self._workers.clear()
        if self._sweeper is not None:
            self._sweeper.join(timeout_s)
            self._sweeper = None

        drained = self._collector.drain()
        for key, future in list(self._awaiting.items()):
            self._awaiting.pop(key, None)
            if future.set_running_or_notify_cancel():
                future.set_exception(RequestCancelledError("the pipeline stopped"))
        self._sink.close()
        self._started = False
        _LOG.info(
            "pipeline stopped: %d queued frame(s) failed, %d in-flight frame(s) published",
            len(lost),
            drained,
        )

    def __enter__(self) -> PipelineRunner:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- the worker loop -----------------------------------------------------------------

    def _work(self) -> None:
        """Drain the ingest queue and run the graph, one frame at a time.

        ``get_batch`` returning an empty list means the queue closed, which is how a worker
        learns to exit without a separate sentinel.
        """
        while not self._stopping.is_set():
            items = self._queue.get_batch(self._window, poll_s=0.05)
            if not items:
                if self._queue.is_closed:
                    return
                continue
            for item in items:
                try:
                    self._run_frame(item)
                except Exception as exc:
                    # A worker thread that dies stops serving every camera, so the loop
                    # survives one bad frame. The graph already contains per-stage failures;
                    # reaching here means something outside a stage went wrong.
                    self._metrics.frames_failed.inc(camera=item.request.context.camera_id)
                    _LOG.exception("pipeline worker failed on %s", item.request.context.key)
                    # Fail the future *here*, because the two cases that reach this handler
                    # both raise before `_run_frame` registers the key in `_awaiting`:
                    # `FrameState.from_request` rejects a request with no tensor under
                    # `ingest.input_name`, and one carrying an FP32 NCHW batch instead of a
                    # `(1, H, W, 3)` uint8 frame. With the key unregistered, `stop()` cannot
                    # cancel it either, so any producer holding the `ResponseFuture` — the
                    # bench harness, an HTTP handler, an injected `queue=` — waited forever
                    # on one malformed frame and the process exited with it pending. The
                    # `(camera_id, frame_id)` tag has to survive the error path, and a frame
                    # that vanishes with no typed failure delivered is the opposite of that.
                    if not item.future.done():
                        item.fail(
                            InferenceError(
                                f"pipeline failed on {item.request.context.key}: {exc}"
                            )
                        )

    def _run_frame(self, item: WorkItem) -> None:
        request = item.request
        camera = request.context.camera_id
        self.frames_accepted += 1
        self._metrics.frames_accepted.inc(camera=camera)

        if request.is_expired():
            # Too late to be worth a GPU. The queue drops expired requests on the way out
            # too; this catches one that expired while a worker was busy.
            self._metrics.frames_expired.inc(camera=camera)
            item.fail(RequestCancelledError("frame deadline passed before the pipeline ran"))
            return

        state = FrameState.from_request(
            request, self._input_name, fps=self._fps.get(camera, 0.0)
        )
        if self._budget_ns:
            budget = time.monotonic_ns() + self._budget_ns
            state.deadline_ns = min(state.deadline_ns, budget) if state.deadline_ns else budget

        key = state.key
        # `setdefault`, not assignment: if a frame with this tag is somehow already in flight
        # the collector will refuse the newcomer, and clobbering the entry here would leave
        # the *first* frame's caller waiting on a future nobody will ever resolve.
        self._awaiting.setdefault(key, item.future)
        if not self._collector.open(state):
            if self._awaiting.get(key) is item.future:
                del self._awaiting[key]
            self._metrics.frames_failed.inc(camera=camera)
            item.fail(
                RequestCancelledError(
                    f"reassembly refused {key}: it is either full and unable to evict, or a "
                    f"frame with this tag is already in flight"
                )
            )
            return

        observer = _CollectorObserver(self._collector, key, self._metrics)
        try:
            self._graph.execute(state, observer)
        finally:
            # Always sealed, even if the graph raised: an unsealed frame would sit in
            # reassembly until the timeout, turning a fast failure into a 1500 ms one.
            self._collector.seal(key)

    def _sweep(self) -> None:
        """Emit timed-out frames, and refresh the gauges an operator watches."""
        interval_s = self._settings.pipeline.reassembly.sweep_interval_ms / 1000.0
        while not self._stopping.wait(interval_s):
            try:
                self._collector.sweep()
                self._metrics.queue_depth.set(self._queue.depth)
            except Exception:  # pragma: no cover - the sweeper must outlive a bad frame
                _LOG.exception("reassembly sweep failed")

    # -- emission ------------------------------------------------------------------------

    def _emit(self, result: FrameResult) -> None:
        """Turn one finished frame into an event and publish it. Never raises.

        Called from a worker thread on the normal path, from the sweeper on a timeout, and
        from :meth:`stop` on shutdown. It must not raise in any of those: the sink already
        contains its own failures, and this wrapper is what protects the sweeper — the thread
        whose survival is the guarantee that every frame is eventually reported.
        """
        future = self._awaiting.pop(result.key, None)
        # Every exit resolves it. The build-failure path popped the future and returned, so
        # a caller awaiting that frame blocked forever and `stop()` could no longer find it
        # to cancel. Harmless only while the shipped sink discards the future; the first
        # caller that awaits one would hang.
        try:
            self._emit_resolved(result, future)
        finally:
            if future is not None and not future.done():
                future.set_exception(
                    InferenceError(f"frame {result.key} was finished without an event")
                )

    def _emit_resolved(self, result: FrameResult, future: Any) -> None:
        """The emission itself. Split out so the future's resolution is a `finally`."""
        try:
            if result.reason == EVICTED:
                # Counted by the collector and named by camera, not published: the buffer is
                # full because the system is behind, and an event that is mostly empty moves
                # the overload onto a consumer instead of resolving it.
                if future is not None and future.set_running_or_notify_cancel():
                    future.set_exception(
                        RequestCancelledError("reassembly evicted this frame while overloaded")
                    )
                return
            event = self._build_event(result)
        except Exception:
            # Building the event is ours; the sink has not been touched yet. Counting this
            # as a sink failure sent operators to an innocent broker while the real fault
            # was a field-map typo raising once per frame.
            self._metrics.build_failures.inc(camera=result.state.camera_id)
            _LOG.exception("failed to build the event for %s", result.key)
            return
        # `emit` returns whether it published, and never raises — so this has to be a
        # return-value check, not an `except`. It was an `except`, which made
        # `pipeline_sink_failures_total` unreachable and let a dropped event fall through to
        # `_record` and `future.set_result(event)`. The `# pragma: no cover` on that handler
        # was the author noticing it was dead and marking it instead of asking why.
        if not self._sink.emit(event):
            self._metrics.sink_failures.inc(sink=self._sink.name)
            if future is not None and future.set_running_or_notify_cancel():
                future.set_exception(
                    InferenceError(f"sink {self._sink.name!r} dropped {result.key}")
                )
            return
        try:
            self._record(result, event)
            if future is not None and future.set_running_or_notify_cancel():
                future.set_result(event)
        except Exception:  # pragma: no cover
            # After a successful emit. Also not the sink's fault, and the distinction
            # matters: the event *was* published.
            self._metrics.build_failures.inc(camera=result.state.camera_id)
            _LOG.exception("post-emit bookkeeping failed for %s", result.key)

    def _build_event(self, result: FrameResult) -> PerceptionEvent:
        state = result.state
        return PerceptionEvent.build(
            camera_id=state.camera_id,
            frame_id=state.frame_id,
            source_id=self._settings.pipeline.source_id,
            # Built here, from the capture the collector took under its lock, and *not*
            # under that lock: this is the most expensive per-object work in the pipeline
            # and the mutex it used to run inside is the one every worker takes on every
            # stage. Only a directly-constructed FrameResult (a test) lacks the capture, and
            # there nothing else is touching the state.
            objects=(
                result.inputs.records(self._graph.field_map)
                if result.inputs is not None
                else self._graph.objects(state)
            ),
            width=state.width,
            height=state.height,
            fps=state.fps,
            captured_ns=state.context.captured_ns,
            captured_unix_ns=state.context.captured_unix_ns,
            missing_stages=result.missing,
            reason=result.reason,
        )

    def _record(self, result: FrameResult, event: PerceptionEvent) -> None:
        camera = event.camera_id
        self._metrics.objects_per_frame.observe(len(event.objects), camera=camera)
        # From the capture, not from `result.state`. Reading the live state here was the same
        # ADR-002 race one level down: the sweeper finishes a frame with 3 detections, the
        # wedged stage answers, the owning worker calls `set_detections(12)`, and the event
        # correctly carries the 3 from the capture while `objects_total` is charged 12. The
        # per-camera counts then overstate reality on exactly the timed-out frames an
        # operator is investigating. The capture already holds what this needs.
        detections = (
            result.inputs.detections if result.inputs is not None else result.state.detections
        )
        for class_name, count in detections.counts().items():
            self._metrics.objects_total.inc(count, camera=camera, object_class=class_name)
        if event.latency_us:
            self._metrics.frame_latency_us.observe(event.latency_us, camera=camera)

    # -- observability ---------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """One snapshot of the whole flow — what a ``/health`` handler returns."""
        return {
            "running": self.is_running,
            "workers": len([w for w in self._workers if w.is_alive()]),
            "graph": {
                "name": self._graph.name,
                "stages": list(self._graph.stage_names),
                "models": list(self._graph.models()),
            },
            "queue": self._queue.stats().as_dict(),
            "frames_accepted": self.frames_accepted,
            "reassembly": self._collector.stats(),
            "pending_per_camera": self._collector.pending_per_camera(),
            "sink": self._sink.stats(),
            "ops": self._ops.describe(),
        }

    def __repr__(self) -> str:
        state = "running" if self.is_running else ("started" if self._started else "stopped")
        return (
            f"<PipelineRunner {state} graph={self._graph.name} "
            f"frames={self.frames_accepted} sink={self._sink.name}>"
        )
