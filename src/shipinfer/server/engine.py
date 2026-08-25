"""The inference server: repository in, ready models out."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from shipinfer.core.errors import ModelControlError, ModelNotFoundError, ServerStateError
from shipinfer.core.logging import get_logger
from shipinfer.core.metrics import EXPORTERS, ServerMetrics
from shipinfer.core.request import InferenceRequest, InferenceResponse, ResponseFuture
from shipinfer.core.settings import ModelControlMode, ServerSettings
from shipinfer.core.tracing import NullTraceSink, TraceSink, build_trace_sink
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
        # A second lock, and the two are not interchangeable. `_lock` guards the model table
        # for the microseconds a lookup takes; `_control_lock` serialises whole load/unload
        # operations, which start threads and load engines and can run for seconds. Holding
        # the table lock for that long would block every inference on the server.
        self._control_lock = threading.Lock()
        self._traces: TraceSink = NullTraceSink()
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
    def traces(self) -> TraceSink:
        """Where per-request traces go. The null sink unless ``observability.trace_sink`` says
        otherwise."""
        return self._traces

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_ready(self) -> bool:
        """Ready == started, and every loaded model has at least one live instance.

        A stricter definition than "the process is up" on purpose: a readiness probe that
        passes while a model is still deserialising sends traffic into a wall.

        Snapshotted under the lock rather than iterated live. ``unload_model`` pops from
        ``_models`` on a request thread, and FastAPI runs both of these plain-``def`` handlers
        in its threadpool — so an unload landing mid-iteration raises ``RuntimeError:
        dictionary changed size during iteration``. That surfaces as a 500 from the *readiness
        probe*, which is the one endpoint whose failure takes the pod out of rotation.
        """
        return self._started and all(m.is_ready for m in self._models_snapshot())

    def _models_snapshot(self) -> list[Model]:
        """Every loaded model, as a list taken under the lock.

        `_models` is mutated by `load_model`/`unload_model` on request threads. A caller that
        iterates it directly races those, and the failure is not a stale answer — it is
        `RuntimeError: dictionary changed size during iteration` from an endpoint that was
        working. `check_health` already did this by hand; the three that did not are why it is
        a method now.
        """
        with self._lock:
            return list(self._models.values())

    def models(self) -> list[str]:
        # Through the snapshot, like every other reader. `sorted(self._models)` iterated the live
        # dict, and it was the reader that turned a 404 into a 500: `infer` on an unloaded name
        # builds `ModelNotFoundError(name, self.models())` while a concurrent unload pops the
        # table, and the `RuntimeError` from the iteration is not a `ShipInferError`, so the
        # route's handler never sees it. The structural test grepped for `_models.values()` and
        # this reader spells it `sorted(self._models)` — blind to exactly the one it missed.
        return sorted(model.name for model in self._models_snapshot())

    def model(self, name: str) -> Model | EnsembleModel:
        try:
            return self._models[name]
        except KeyError:
            raise ModelNotFoundError(name, self.models()) from None

    def __iter__(self) -> Iterator[Model | EnsembleModel]:
        """Iterate the loaded models.

        Over a snapshot, because handing back a live view makes every caller responsible for
        a lock they cannot see. `check_health` was careful — it does `list(server)`, which
        consumes the iterator before an unload can land — and being correct only for the
        careful caller is not a property worth having.
        """
        return iter(self._models_snapshot())

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

        observability = self._settings.observability
        self._traces = build_trace_sink(
            observability.trace_sink, **observability.trace_sink_options
        )

        self._repository = ModelRepository.load(self._settings.model_repository)
        names = self._startup_names()

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

    def _startup_names(self) -> list[str]:
        """Which models to load at start-up.

        Explicit model control loads only what it was told to, even when
        ``load_all_models`` is left at its default — that is the whole point of the mode, and
        a server that loaded the entire repository and then waited to be told to unload it
        would have the memory problem the mode exists to avoid.
        """
        assert self._repository is not None
        if self._settings.model_control is ModelControlMode.EXPLICIT:
            return list(self._settings.startup_models)
        if self._settings.load_all_models:
            return self._repository.names()
        return list(self._settings.startup_models)

    def _load(self, name: str) -> None:
        """Load one model at start-up, honouring ``strict_startup``."""
        try:
            self._build_and_start(name)
        except Exception:
            if self._settings.strict_startup:
                raise
            # Non-strict start-up is for a heterogeneous fleet where one node genuinely
            # cannot host one model. It is logged at ERROR, never swallowed: a server
            # silently serving nine of ten models is a worse outage than not starting.
            _LOG.exception("failed to load model %r; continuing (strict_startup=false)", name)

    def _build_and_start(self, name: str) -> Model | EnsembleModel:
        """Construct one model, start it, and publish it. Raises rather than degrading.

        The model only reaches the table once it has started, so a half-built model is never
        reachable by an inference — a failed load leaves the server exactly as it was.
        """
        assert self._repository is not None
        artifact = self._repository.resolve(name)
        model: Model | EnsembleModel
        if artifact.config.is_ensemble:
            model = EnsembleModel(
                artifact=artifact,
                settings=self._settings,
                metrics=self._metrics,
                resolve=self.model,
                # An ensemble-only deployment traced nothing at the ensemble level: the steps
                # each traced their own model and the DAG that joined them was invisible,
                # which is the one span an operator debugging an ensemble actually wants.
                traces=self._traces,
            )
        else:
            model = Model(
                artifact=artifact,
                settings=self._settings,
                devices=self._devices,
                memory=self._memory,
                metrics=self._metrics,
                traces=self._traces,
            )
        model.start()
        with self._lock:
            self._models[name] = model
        return model

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
        # After the models, so a trace written by a worker finishing its last batch still
        # has somewhere to go; closing it flushes whatever the sink had buffered.
        self._traces.close()
        self._memory.close()
        self._started = False

    # -- explicit model control ------------------------------------------------------------

    def load_model(self, name: str) -> Model | EnsembleModel:
        """Load one model into a running server — Triton's ``/v2/repository/models/*/load``.

        The repository index is **re-scanned first**, which is what makes this useful for a
        repository that grows: a model copied in after start-up is found. That is not the
        polling mode Triton also offers and this server does not — the difference is that an
        operator asked at this moment, so a half-written config fails their call immediately
        with the file named, rather than being picked up by a timer minutes later with
        nothing pointing at the edit.

        Loading a model that is already loaded is **refused**, not treated as a reload. A
        reload has to stop the running copy before it can build the new one (two copies of a
        detector do not fit on one GPU), so a reload that failed halfway would take a working
        model down. ``unload`` then ``load`` says the same thing and says it deliberately.

        Raises:
            ModelControlError: when the server was not started with
                ``model_control='explicit'``, or the model is already loaded.
            ModelNotFoundError: when the repository has no such model.
            ServerStateError: before :meth:`start`.
            ConfigurationError: when the model's config or artefacts are wrong. The server
                keeps serving everything else.
        """
        self._require_control("load", name)
        with self._control_lock:
            if name in self._models:
                raise ModelControlError(
                    f"model {name!r} is already loaded; unload it first if you meant to "
                    "replace it"
                )
            self._repository = ModelRepository.load(self._settings.model_repository)
            model = self._build_and_start(name)
        _LOG.info(
            "loaded model %s on request (%d model(s) now loaded)", name, len(self._models)
        )
        return model

    def unload_model(self, name: str) -> None:
        """Unload one model — Triton's ``/v2/repository/models/*/unload``.

        Drains and releases it. In-flight requests are resolved or failed by the instances'
        own shutdown path, which is the same one :meth:`stop` uses, so unloading is not a
        second way for a request to disappear.

        Raises:
            ModelControlError: when explicit control is off, or a loaded ensemble still
                composes this model. Unloading a step out from under an ensemble would turn
                every one of its requests into a ``ModelNotFoundError`` from inside the DAG,
                which is a much worse way to find out.
            ModelNotFoundError: when it is not loaded.
        """
        self._require_control("unload", name)
        with self._control_lock:
            model = self.model(name)
            dependents = self._ensembles_depending_on(name)
            if dependents:
                raise ModelControlError(
                    f"model {name!r} is a step of loaded ensemble(s) {dependents}; "
                    "unload those first"
                )
            with self._lock:
                self._models.pop(name, None)
            model.stop()
        _LOG.info("unloaded model %s on request", name)

    def index(self) -> list[dict[str, str]]:
        """Every model the repository knows, and whether it is serving.

        Triton's ``/v2/repository/index`` shape: ``name``, ``version``, ``state``, ``reason``.
        Under explicit control the repository is re-scanned first, so a model added since
        start-up appears — an operator who cannot see a model cannot ask for it. Under
        ``none`` the start-up scan is reported unchanged, because that set is exactly what
        this server will ever serve.
        """
        if not self._started:
            raise ServerStateError("the server has not been started")
        if self._settings.model_control is ModelControlMode.EXPLICIT:
            # Non-blocking. `index` is what a readiness probe calls, and taking the control
            # lock put it behind an in-progress `unload_model` for up to
            # `shutdown_grace_s x instances` — so a rolling update that unloads one model
            # makes the whole server look unhealthy and gets itself restarted.
            #
            # A re-scan that loses the race reports the previous scan, which is a stale
            # answer rather than a wrong one: the *loaded* set below is read from
            # `self._models` either way, so a model that is serving always reads as READY.
            acquired = self._control_lock.acquire(blocking=False)
            if acquired:
                try:
                    self._repository = ModelRepository.load(self._settings.model_repository)
                finally:
                    self._control_lock.release()
        entries = []
        for entry in self.repository:
            loaded = self._models.get(entry.name)
            if loaded is None:
                state, reason = "UNAVAILABLE", "not loaded"
            elif loaded.is_ready:
                state, reason = "READY", ""
            else:
                state, reason = "LOADING", "no instance is ready yet"
            entries.append(
                {
                    "name": entry.name,
                    "version": str(entry.latest),
                    "state": state,
                    "reason": reason,
                }
            )
        return sorted(entries, key=lambda item: item["name"])

    def _require_control(self, action: str, name: str) -> None:
        if not self._started:
            raise ServerStateError(
                f"cannot {action} model {name!r}: the server has not been started"
            )
        if self._settings.model_control is not ModelControlMode.EXPLICIT:
            raise ModelControlError(
                f"cannot {action} model {name!r}: this server runs with "
                f"model_control={self._settings.model_control.value!r}; start it with "
                "model_control='explicit' to manage models over the API"
            )

    def _ensembles_depending_on(self, name: str) -> list[str]:
        """Loaded ensembles that name ``name`` as a step, so unloading it would break them.

        Snapshotted for the reason :meth:`is_ready` gives, and with an edge of its own: this is
        called *from* the unload path, so two concurrent unloads are exactly the case it has to
        survive.
        """
        dependents = []
        for model in self._models_snapshot():
            ensemble = model.artifact.config.ensemble
            if ensemble is not None and any(step.model == name for step in ensemble.steps):
                dependents.append(model.name)
        return sorted(dependents)

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
            "tracing": self._traces.stats(),
            # Snapshotted, not iterated: `Model.stats()` is enough Python that the
            # interpreter can switch mid-comprehension, and a concurrent unload then raises
            # from a scrape that worked a second earlier.
            "models": [m.stats() for m in self._models_snapshot()],
        }

    def render_metrics(self, exporter: str | None = None) -> str:
        """Metrics in the configured wire format."""
        name = exporter or self._settings.observability.metrics_exporter
        return EXPORTERS.create(name).render(self._metrics.registry)

    def __repr__(self) -> str:
        state = "ready" if self.is_ready else ("started" if self._started else "stopped")
        return f"<InferenceServer {state} models={self.models()}>"
