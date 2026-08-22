"""The inference server: repository in, ready models out."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from shipinfer.core.errors import ModelNotFoundError, ServerStateError
from shipinfer.core.logging import get_logger
from shipinfer.core.metrics import EXPORTERS, ServerMetrics
from shipinfer.core.request import InferenceRequest, InferenceResponse, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.repository import ModelRepository
from shipinfer.runtime.device import DeviceManager
from shipinfer.runtime.memory import MemoryPool
from shipinfer.runtime.native import is_native_available, native_version, resolve_provider
from shipinfer.server.ensemble import EnsembleModel
from shipinfer.server.model import Model

__all__ = ["InferenceServer"]

_LOG = get_logger("server")


class InferenceServer:
    """Owns the repository, the devices, the memory pool and the loaded models.

    Deliberately *not* a singleton and not a global. Two servers can coexist in one
    process — a test does exactly that — and each owns its own metrics registry, so their
    counters do not silently merge.

    Lifecycle is explicit: :meth:`start` loads and warms every selected model and blocks
    until they are ready, :meth:`stop` drains and releases. Use it as a context manager and
    neither can be forgotten.
    """

    def __init__(self, settings: ServerSettings | None = None) -> None:
        self._settings = settings or ServerSettings()
        self._metrics = ServerMetrics()
        self._devices = DeviceManager(self._settings.devices)
        self._memory = MemoryPool(self._settings.memory)
        self._repository: ModelRepository | None = None
        self._models: dict[str, Model | EnsembleModel] = {}
        self._lock = threading.Lock()
        self._started = False
        self._started_at = 0.0

    # -- properties ----------------------------------------------------------------------

    @property
    def settings(self) -> ServerSettings:
        return self._settings

    @property
    def metrics(self) -> ServerMetrics:
        return self._metrics

    @property
    def devices(self) -> DeviceManager:
        return self._devices

    @property
    def memory(self) -> MemoryPool:
        return self._memory

    @property
    def repository(self) -> ModelRepository:
        if self._repository is None:
            raise ServerStateError("the server has not been started")
        return self._repository

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_ready(self) -> bool:
        """Ready == started, and every loaded model has at least one live instance.

        A stricter definition than "the process is up" on purpose: a readiness probe that
        passes while a model is still deserialising sends traffic into a wall.
        """
        return self._started and all(m.is_ready for m in self._models.values())

    def models(self) -> list[str]:
        return sorted(self._models)

    def model(self, name: str) -> Model | EnsembleModel:
        try:
            return self._models[name]
        except KeyError:
            raise ModelNotFoundError(name, self.models()) from None

    def __iter__(self) -> Iterator[Model | EnsembleModel]:
        return iter(self._models.values())

    # -- lifecycle -----------------------------------------------------------------------

    def start(self) -> InferenceServer:
        """Scan the repository, load the selected models, and wait for readiness."""
        if self._started:
            return self

        provider = resolve_provider(self._settings.execution.provider)
        _LOG.info(
            "starting shipinfer | devices: %s | data plane: %s%s",
            self._devices.describe(),
            provider.value,
            f" (shipinfer._C {native_version()})" if is_native_available() else "",
        )

        self._repository = ModelRepository.load(self._settings.model_repository)
        names = (
            self._repository.names()
            if self._settings.load_all_models
            else self._settings.startup_models
        )

        # Plain models first, then ensembles: an ensemble validates its DAG against the
        # models it composes, so those have to exist before it starts.
        plain = [n for n in names if not self._repository.entry(n).config.is_ensemble]
        ensembles = [n for n in names if self._repository.entry(n).config.is_ensemble]
        for name in (*plain, *ensembles):
            self._load(name)

        self._started = True
        self._started_at = time.monotonic()
        _LOG.info("shipinfer ready: %d model(s) — %s", len(self._models), self.models())
        return self

    def _load(self, name: str) -> None:
        assert self._repository is not None
        artifact = self._repository.resolve(name)
        try:
            model: Model | EnsembleModel
            if artifact.config.is_ensemble:
                model = EnsembleModel(
                    artifact=artifact,
                    settings=self._settings,
                    metrics=self._metrics,
                    resolve=self.model,
                )
            else:
                model = Model(
                    artifact=artifact,
                    settings=self._settings,
                    devices=self._devices,
                    memory=self._memory,
                    metrics=self._metrics,
                )
            model.start()
        except Exception:
            if self._settings.strict_startup:
                raise
            # Non-strict start-up is for a heterogeneous fleet where one node genuinely
            # cannot host one model. It is logged at ERROR, never swallowed: a server
            # silently serving nine of ten models is a worse outage than not starting.
            _LOG.exception("failed to load model %r; continuing (strict_startup=false)", name)
            return
        with self._lock:
            self._models[name] = model

    def stop(self) -> None:
        """Drain and release. Safe to call twice, and never raises."""
        if not self._started:
            return
        _LOG.info("stopping shipinfer (%d model(s))", len(self._models))
        with self._lock:
            models = list(self._models.values())
            self._models.clear()
        for model in models:
            try:
                model.stop()
            except Exception:
                _LOG.exception("error stopping model %s", model.name)
        self._memory.close()
        self._started = False

    def __enter__(self) -> InferenceServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- inference -----------------------------------------------------------------------

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        """Submit one request. Returns immediately with a future.

        Raises:
            ServerStateError: before :meth:`start`. Checked here rather than in the model
                so the error says what is actually wrong — "the server is not started"
                rather than "no such model", which is what an empty model table looks like.
        """
        if not self._started:
            raise ServerStateError(
                "the server has not been started; call start() or use it as a context manager"
            )
        return self.model(request.model_name).infer(request)

    def infer_sync(
        self, request: InferenceRequest, timeout: float | None = None
    ) -> InferenceResponse:
        """Submit and wait.

        Convenience for scripts and tests. Real pipelines should submit many and join with
        ``concurrent.futures.wait`` — a blocking call per request cannot fill a batch, which
        gives up the throughput the batcher exists to provide.
        """
        return self.infer(request).result(timeout)

    # -- observability ---------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready,
            "uptime_s": round(time.monotonic() - self._started_at, 1) if self._started else 0.0,
            "devices": {
                "visible": list(self._devices.visible_gpus),
                "accelerator": self._devices.kind.value,
            },
            "native": {"available": is_native_available(), "version": native_version()},
            "memory": self._memory.stats(),
            "models": [m.stats() for m in self._models.values()],
        }

    def render_metrics(self, exporter: str | None = None) -> str:
        """Metrics in the configured wire format."""
        name = exporter or self._settings.observability.metrics_exporter
        return EXPORTERS.create(name).render(self._metrics.registry)

    def __repr__(self) -> str:
        state = "ready" if self.is_ready else ("started" if self._started else "stopped")
        return f"<InferenceServer {state} models={self.models()}>"
