"""A servable model: its instances, its dispatcher, its batcher, its cache."""

from __future__ import annotations

import time
from typing import Any

from shipinfer.backends.base import BackendContext
from shipinfer.backends.registry import build_backend
from shipinfer.core.errors import (
    ConfigurationError,
    QueueFullError,
    ServerStateError,
    ValidationError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.metrics import ServerMetrics
from shipinfer.core.request import InferenceRequest, InferenceResponse, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor, TensorSpec, validate_against
from shipinfer.repository import ModelArtifact
from shipinfer.runtime.device import DeviceManager
from shipinfer.runtime.graphs import GRAPH_CACHES
from shipinfer.runtime.memory import MemoryPool
from shipinfer.runtime.stream import StreamPool
from shipinfer.scheduling.batching import StackingBatcher
from shipinfer.scheduling.dispatcher import Dispatcher
from shipinfer.scheduling.policies import build_policy
from shipinfer.scheduling.queues import QUEUES, BatchWindow
from shipinfer.scheduling.work import WorkItem
from shipinfer.server.cache import RESPONSE_CACHES, NullResponseCache, ResponseCache
from shipinfer.server.instance import ModelInstance

__all__ = ["Model"]

_LOG = get_logger("server.model")


class Model:
    """Everything needed to serve one model version.

    Construction expands ``instance_group`` into concrete instances the same way Triton
    does — ``count`` per device, an empty ``gpus`` meaning "all visible" — which is what
    lets one config file run unchanged on a 2-GPU dev box and a 16-GPU node.
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        settings: ServerSettings,
        devices: DeviceManager,
        memory: MemoryPool,
        metrics: ServerMetrics,
    ) -> None:
        self._artifact = artifact
        self._settings = settings
        self._devices = devices
        self._memory = memory
        self._metrics = metrics

        config = artifact.config
        self._batcher = StackingBatcher(
            config.input_specs, config.output_specs, config.effective_max_batch_size
        )
        self._window = self._build_window()
        self._cache = self._build_cache()
        self._instances: list[ModelInstance] = self._build_instances()
        self._dispatcher = Dispatcher(
            model_name=artifact.name,
            instances=self._instances,
            policy=build_policy(
                settings.scheduler.placement_policy,
                **settings.scheduler.placement_policy_options,
            ),
            on_spill=self._on_spill,
        )
        self._started = False

    # -- construction --------------------------------------------------------------------

    def _build_window(self) -> BatchWindow:
        config = self._artifact.config
        batching = config.dynamic_batching
        return BatchWindow(
            max_batch_size=config.effective_max_batch_size,
            max_delay_us=batching.max_queue_delay_us if batching.enabled else 0,
            preferred_sizes=tuple(batching.preferred_batch_sizes),
        )

    def _build_cache(self) -> ResponseCache:
        params = self._artifact.config.parameters
        spec = params.get("response_cache")
        if not spec:
            return NullResponseCache()
        if isinstance(spec, str):
            return RESPONSE_CACHES.create(spec)
        options = dict(spec)
        kind = options.pop("type", "lru")
        return RESPONSE_CACHES.create(kind, **options)

    def _build_instances(self) -> list[ModelInstance]:
        config = self._artifact.config
        scheduler = self._settings.scheduler
        execution = self._settings.execution
        placements = config.placements(self._devices.visible_gpus)

        instances: list[ModelInstance] = []
        for ordinal, placement in enumerate(placements):
            device = self._devices.require(placement.device)
            streams = execution.streams_per_instance or placement.streams
            stream_pool = StreamPool(device, streams) if device.is_cuda else None
            graphs = (
                GRAPH_CACHES.create(
                    execution.graph_cache,
                    device,
                    enabled=execution.cuda_graphs,
                    batch_sizes=tuple(execution.cuda_graph_batch_sizes),
                    max_failures=execution.cuda_graph_max_capture_failures,
                )
                if device.is_cuda
                else None
            )
            context = BackendContext(
                artifact=self._artifact,
                device=device,
                memory=self._memory,
                execution=execution,
                streams=stream_pool,
                graphs=graphs,
            )
            backend = build_backend(config.platform, context)
            name = f"{self._artifact.name}_{ordinal}_{device}"
            queue = QUEUES.create(
                "fair" if scheduler.fair_queueing else "fifo",
                name,
                scheduler.max_queue_size,
                overflow=scheduler.overflow_policy,
                block_timeout_ms=scheduler.enqueue_block_timeout_ms,
                drop_expired=scheduler.drop_expired_requests,
            )
            instances.append(
                ModelInstance(
                    name=name,
                    backend=backend,
                    queue=queue,
                    batcher=self._batcher,
                    window=self._window,
                    devices=self._devices,
                    metrics=self._metrics,
                    scheduler=scheduler,
                )
            )
        if not instances:
            raise ConfigurationError(f"model {self._artifact.name!r} expands to zero instances")
        return instances

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
    def instances(self) -> tuple[ModelInstance, ...]:
        return tuple(self._instances)

    @property
    def is_ready(self) -> bool:
        return self._started and any(i.is_ready for i in self._instances)

    @property
    def total_depth(self) -> int:
        return sum(i.depth for i in self._instances)

    # -- lifecycle -----------------------------------------------------------------------

    def start(self, timeout_s: float = 120.0) -> None:
        """Start every instance and wait for at least one to become ready."""
        for instance in self._instances:
            instance.start()
        deadline = time.monotonic() + timeout_s
        for instance in self._instances:
            remaining = max(0.0, deadline - time.monotonic())
            if not instance.wait_ready(remaining) and self._settings.strict_startup:
                raise ServerStateError(
                    f"instance {instance.name} did not become ready within {timeout_s:.0f}s"
                )
        self._started = True
        _LOG.info(
            "model %s v%d ready with %d instance(s) across %s",
            self.name,
            self.version,
            len(self._instances),
            sorted({str(i.device) for i in self._instances}),
        )

    def stop(self) -> None:
        for instance in self._instances:
            instance.stop(self._settings.shutdown_grace_s)
        self._cache.clear()
        self._started = False

    # -- inference -----------------------------------------------------------------------

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        """Validate, check the cache, then dispatch.

        Returns:
            A future that resolves to an :class:`InferenceResponse`.

        Raises:
            ValidationError: for tensors that do not match the model's inputs — raised
                synchronously, before a queue slot is consumed, so a malformed client
                cannot occupy capacity.
            QueueFullError: when every instance is saturated. That is the honest signal
                that the *pool* cannot keep up, not that one GPU is unlucky.
            ServerStateError: when no instance is ready.
        """
        if not self._started:
            raise ServerStateError(f"model {self.name!r} has not been started")

        request.timings.received_ns = time.monotonic_ns()
        try:
            validate_against(request.inputs, self._batcher_input_specs(), what="input")
        except ValueError as exc:
            # core.types speaks plain ValueError so it stays dependency-free; the server
            # boundary is where that becomes a typed, HTTP-mappable failure.
            raise ValidationError(f"{self.name}: {exc}") from exc
        self._metrics.requests_total.inc(model=self.name)

        future = ResponseFuture(request)

        # One hash per request, computed behind the cache object so a model with caching
        # off (every model by default) pays a virtual call instead of a BLAKE2b pass over
        # every input byte. `None` means "not cacheable": caching disabled, or inputs that
        # live on a device and would need a D2H copy just to be hashed.
        key = self._cache.key_for(self.name, self.version, request.inputs)
        if key is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self._metrics.cache_hits.inc(model=self.name)
                return _completed(future, self._response_from(request, cached))
            self._metrics.cache_misses.inc(model=self.name)
            self._store_when_done(key, future)

        try:
            self._dispatcher.dispatch(WorkItem(request, future), _enqueue)
        except QueueFullError:
            self._metrics.requests_rejected.inc(model=self.name)
            raise
        return future

    def _batcher_input_specs(self) -> tuple[TensorSpec, ...]:
        return self._artifact.config.input_specs

    def _response_from(
        self, request: InferenceRequest, outputs: dict[str, Tensor]
    ) -> InferenceResponse:
        """Wrap cached outputs in a response carrying *this* request's identity.

        The outputs are shared with every other hit on the key and sealed non-writeable by
        the cache; everything else — the request id, the (camera, frame) tag, the timings —
        belongs to this request and must not come from whichever request happened to
        populate the entry.
        """
        request.timings.completed_ns = time.monotonic_ns()
        return InferenceResponse(
            request_id=request.request_id,
            model_name=self.name,
            model_version=self.version,
            outputs=outputs,
            context=request.context,
            timings=request.timings,
        )

    def _store_when_done(self, key: str, future: ResponseFuture) -> None:
        """Populate the cache once the request has actually succeeded.

        A done-callback rather than a write at completion time inside the instance, because
        the instance is shared by every model and knows nothing about caching. It runs on
        the worker thread that finished the batch, so the copy `put` makes is charged to
        that thread — bounded, because caching is opt-in and only sound for the small,
        deterministic outputs it is enabled for (ADR-009).

        A failed or cancelled request is not cached: a stored exception would be served
        forever, and that is far worse than being slow.
        """

        def _store(done: ResponseFuture) -> None:
            if done.cancelled() or done.exception() is not None:
                return
            try:
                self._cache.put(key, done.result().outputs)
            except RuntimeError:
                # Device-resident outputs. Reading them here would mean a D2H copy on the
                # worker thread to fill a cache that may never be hit.
                pass
            except Exception:
                # A cache is an optimisation; it must never fail the request it served.
                _LOG.exception("response cache write failed for %s", self.name)

        future.add_done_callback(_store)

    def _on_spill(self, wanted: Any, actual: Any) -> None:
        self._metrics.spills_total.inc(model=self.name, device=str(actual.device))
        _LOG.debug(
            "spilled from %s to %s",
            wanted.device,
            actual.device,
            extra=log_context(model=self.name, device=str(actual.device)),
        )

    # -- introspection -------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "platform": self._artifact.config.platform,
            "ready": self.is_ready,
            "policy": repr(self._dispatcher.policy),
            "window": {
                "max_batch_size": self._window.max_batch_size,
                "max_delay_us": self._window.max_delay_us,
                "preferred": list(self._window.preferred_sizes),
            },
            "cache": self._cache.stats(),
            "instances": [i.stats() for i in self._instances],
        }

    def __repr__(self) -> str:
        return f"<Model {self.name}:{self.version} instances={len(self._instances)}>"


def _completed(future: ResponseFuture, response: InferenceResponse) -> ResponseFuture:
    """Resolve a future immediately. Used by the cache-hit path, which has no work to do."""
    future.set_result(response)
    return future


def _enqueue(instance: Any, item: WorkItem) -> None:
    """Adapter so the dispatcher depends only on the narrow ``Placeable`` protocol."""
    instance.enqueue(item)
