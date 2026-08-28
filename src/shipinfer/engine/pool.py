"""The inference server: repository in, ready models out."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from shipinfer.core.errors import (
    ConfigurationError,
    ModelControlError,
    ModelNotFoundError,
    ServerStateError,
)
from shipinfer.core.logging import get_logger
from shipinfer.core.metrics import EXPORTERS, ServerMetrics
from shipinfer.core.request import InferenceRequest, InferenceResponse, ResponseFuture
from shipinfer.core.settings import ModelControlMode, ServerSettings
from shipinfer.core.tracing import NullTraceSink, TraceSink, build_trace_sink
from shipinfer.engine.ensemble import EnsembleModel
from shipinfer.engine.model import Model
from shipinfer.repository import ModelRepository
from shipinfer.runtime.device import DeviceManager
from shipinfer.runtime.memory import MemoryPool
from shipinfer.runtime.native import is_native_available, native_version, resolve_provider

__all__ = ["InferenceServer"]

# The four "server…" logger names in this package become "engine…" here, with `server/`
# itself (A2 PR-6). They were kept through the move on purpose — an operator's log filter is
# behaviour, and a rename per PR would have changed it four times — so this is the one place
# the change is recorded: `shipinfer.server*` filters become `shipinfer.engine*`, and
# `shipinfer.server.api` becomes `shipinfer.api`.
_LOG = get_logger("engine")


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
        self._mesh: Any = None  # a ServiceMesh when this process joined the spill tier
        self._lock = threading.Lock()
        # A second lock, and the two are not interchangeable. `_lock` guards the model table
        # for the microseconds a lookup takes; `_control_lock` serialises whole load/unload
        # operations, which start threads and load engines and can run for seconds. Holding
        # the table lock for that long would block every inference on the server.
        self._control_lock = threading.Lock()
        # A third lock, and the smallest of the three: it covers every *lifecycle
        # transition* -- `start()`'s claim, `stop()`'s read-and-clear of the flags, and the
        # teardown's hand-off of the trace state -- and nothing else, so it is never held
        # for more than a few bytecodes and never contended in practice. `_control_lock`
        # cannot do this job -- `stop()` has to clear the flags *before* it takes that one,
        # for the reason spelled out there -- and without a lock the check and the act sit
        # either side of a log emit, which is a GIL switch point. See `stop()`.
        #
        # Lock order, where two are held at once: `_control_lock` then `_lifecycle_lock`
        # (`_release` runs under the control lock and takes this one twice). Nothing takes
        # them the other way round: `stop()` drops this one before taking the control lock,
        # and `_begin_start` never touches the control lock at all.
        self._lifecycle_lock = threading.Lock()
        self._traces: TraceSink = NullTraceSink()
        # The last run's trace totals, captured by `_release` before the sink is closed, or
        # None while a run is live. See `stats()` — a scrape after a shutdown wants the
        # numbers the run finished with, not the fresh null sink's zeros. Published under
        # `_lifecycle_lock` in the same breath as the null sink is installed, because those
        # two writes are one fact: a scrape that saw the swap without the totals would
        # report `recorded: 0` for the run it is the last sample of.
        self._last_trace_stats: dict[str, Any] | None = None
        # Which run each of the above belongs to. Bumped by `_begin_start` under the
        # lifecycle lock, carried by `stop()` into `_teardown`/`_release`, and checked there
        # before anything destructive: a teardown that has fallen behind a *newer* run must
        # not drain its models, close its sink or close its memory pool. `_begin_start`
        # refuses to start while a teardown is in flight, so this is the second line of
        # defence rather than the first -- but the first one is a timed wait, and a guard
        # that only holds while nothing takes too long is not a guard.
        self._generation = 0
        # Set once `_teardown` has finished, so a *second* `stop()` can tell "there is
        # nothing to release" from "another thread is releasing it right now". Both look
        # identical from the flags alone — `stop()` clears them before it takes the control
        # lock — and a caller pairing `stop()` with an immediate `start()` would otherwise
        # get the first thread's teardown landing on its fresh server. Starts set, because a
        # server that was never started has nothing to wait for.
        self._torn_down = threading.Event()
        self._torn_down.set()
        # The name of the thread that entered `_teardown`, so an `_await_teardown` whose
        # grace period expires can say who it gave up on. Written once per run, read without
        # a lock: it is diagnostic text in a WARNING, so a stale read is a wrong name in a
        # log line rather than a wrong decision. None until a teardown actually begins --
        # the owner may still be queued on the control lock.
        self._teardown_owner: str | None = None
        self._started = False
        # Distinct from `_started`, and the distinction is the whole of the unwind below:
        # `_started` means "the start finished", `_starting` means "the start got far enough
        # to be holding something". Between them is the window `stop()` used to refuse — a
        # strict start that failed on model 3 of 5 left models 1 and 2 running, and the only
        # handle on them was a server that answered `is_started == False`.
        self._starting = False
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

    def get(self, name: str) -> Model | EnsembleModel:
        """:meth:`model`, under the name the chain's ``pool`` elements ask for.

        This is the whole of the `ModelResolver` protocol a topology element needs
        (`topology/base.py`): one method, a name in, a handle out,
        :class:`~shipinfer.core.errors.ModelNotFoundError` when there is no such model. The
        satisfaction is **structural** and stays that way in both directions — the engine
        never imports `topology`, and `topology` never imports the engine — which is what lets
        a chain be loaded and validated on a host that has no accelerator, and what lets a
        test hand an element a dict instead of a server.

        A second name for one lookup, rather than renaming `model`: `server.model("x")` is the
        spelling every existing caller and the KServe routes use, and `get` is the spelling the
        protocol needs. Renaming either would only move the alias.
        """
        return self.model(name)

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
        """Scan the repository, load the selected models, and wait for readiness.

        **A start that fails releases what it had already taken.** Under
        ``strict_startup`` the first model that will not load aborts the whole start, and
        the models loaded before it are running: worker threads bound to devices, holding
        backends and CUDA contexts. The caller has no handle on them — ``start()`` raised,
        so nothing was returned and nothing was assigned — so unless this method cleans up
        after itself, those contexts are held for the life of the process by an object
        nothing references. On a shared box that is somebody else's job failing to fit.

        So a failure here calls :meth:`stop`, which stops every model that did start (and
        the trace sink and the memory pool), and then the original error is re-raised
        unchanged: the operator must still see *why* model 3 would not load, not a teardown
        error from the clean-up that followed.

        ``strict_startup=false`` is untouched — a model that fails is logged and skipped,
        and the server starts with the rest, which is what a heterogeneous fleet needs.

        **The entry and the exit are both atomic transitions**, under the same
        ``_lifecycle_lock`` :meth:`stop` uses, and that is what keeps the two apart. Without
        it a ``stop()`` arriving mid-start cleared the flags, drained the table and closed
        the sink underneath a start that then set ``_started = True`` regardless — a server
        answering ``is_started`` with an empty model table, which every readiness probe
        reads as "up and serving nothing". Now the start claims the server or is refused,
        and publishes its result only if it still owns the claim; if it does not, it
        releases whatever it had loaded and raises rather than reporting a server that is
        not there.

        Raises:
            ServerStateError: when another thread is already starting this server, when a
                teardown is still in flight after ``shutdown_grace_s``, and when a
                :meth:`stop` took the server over while this start was loading.
        """
        if self._started:
            return self
        generation = self._begin_start()
        if generation is None:
            return self  # another thread finished a start while this one was claiming

        try:
            provider = resolve_provider(self._settings.execution.provider)
            _LOG.info(
                "starting shipinfer | devices: %s | data plane: %s%s",
                self._devices.describe(),
                provider.value,
                f" (shipinfer._C {native_version()})" if is_native_available() else "",
            )

            observability = self._settings.observability
            sink = build_trace_sink(
                observability.trace_sink, **observability.trace_sink_options
            )
            with self._lifecycle_lock:
                # The new run's sink and the previous run's totals are one fact, so they
                # change together: until here a scrape reports the run that ended, from here
                # the live one, and never the gap between them — which reads as a sink named
                # "none" that recorded nothing. See `stats()`.
                self._traces = sink
                self._last_trace_stats = None

            self._repository = ModelRepository.load(self._settings.model_repository)
            names = self._startup_names()

            # Plain models first, then ensembles: an ensemble validates its DAG against the
            # models it composes, so those have to exist before it starts.
            plain = [n for n in names if not self._repository.entry(n).config.is_ensemble]
            ensembles = [n for n in names if self._repository.entry(n).config.is_ensemble]
            for name in (*plain, *ensembles):
                self._load(name)
            self._mesh = self._join_service_tier()
        except BaseException:
            # `BaseException`, not `Exception`: a KeyboardInterrupt during a two-minute
            # engine load is the *likeliest* way this path is taken by hand, and it leaks
            # exactly the same contexts.
            self.stop()
            # And then whatever landed *after* that teardown drained. A concurrent `stop()`
            # is what aborted this start, and the model whose instances were already coming
            # up when the abort was polled still finishes and publishes itself; the drain
            # has been and gone by then. On the ordinary failure path this finds an empty
            # table and does nothing.
            self._abandon_start()
            raise

        if not self._finish_start(generation):
            self._abandon_start()
            raise ServerStateError(
                "start aborted: this server was stopped while it was starting, so the "
                "models this start had loaded have been released rather than published"
            )
        _LOG.info("shipinfer ready: %d model(s) — %s", len(self._models), self.models())
        return self

    def _begin_start(self) -> int | None:
        """Claim this server for one start, atomically. Returns the run's generation.

        ``None`` means another thread completed a start while this one was waiting, which is
        the concurrent form of :meth:`start`'s ``if self._started: return self`` and is
        answered the same way.

        A teardown in flight is *waited for*, not raced: the barrier is bounded by
        ``shutdown_grace_s``, the same budget the teardown grants each instance, and an
        expiry is a typed refusal. Starting anyway is what the expiry used to mean, and it
        is the corruption this method exists to prevent — the queued teardown lands on the
        fresh server, draining its models and closing its sink.

        Raises:
            ServerStateError: when another start is already in progress, and when a teardown
                is still running after the grace period.
        """
        waited = False
        while True:
            with self._lifecycle_lock:
                if self._started:
                    return None
                if self._starting:
                    raise ServerStateError(
                        "cannot start: another thread is already starting this server"
                    )
                if self._torn_down.is_set():
                    # Set *before* the first thing that can fail, so `stop()` knows there is
                    # something to release even when the failure was the repository scan.
                    # The barrier is cleared in the same breath and for the same reason:
                    # from here on there is a teardown to wait for, and clearing it any
                    # earlier would leave a `start()` that failed in `resolve_provider` with
                    # a barrier nothing will ever set.
                    self._generation += 1
                    self._starting = True
                    self._torn_down.clear()
                    self._teardown_owner = None
                    return self._generation
                owner = self._teardown_owner
            if waited:
                raise ServerStateError(
                    f"cannot start: {owner or 'another thread'} is still tearing this "
                    f"server down after {self._settings.shutdown_grace_s:.1f}s (the "
                    "shutdown grace period); starting now would hand its teardown the "
                    "models this start loads"
                )
            self._torn_down.wait(self._settings.shutdown_grace_s)
            waited = True

    def _finish_start(self, generation: int) -> bool:
        """Publish a completed start, unless a :meth:`stop` took the server over first.

        The other half of :meth:`_begin_start`'s claim, and the reason it is a generation
        rather than a flag: ``_starting`` alone cannot tell "still mine" from "cleared by a
        stop, and a later start set it again".

        Returns:
            True when this start now owns a started server. False when it lost the claim —
            the caller must release what it loaded, because the teardown that took the
            server over has already been past the model table.
        """
        with self._lifecycle_lock:
            if self._generation != generation or not self._starting:
                return False
            self._started = True
            self._starting = False
            self._started_at = time.monotonic()
            return True

    def _abandon_start(self) -> None:
        """Release what a lost start published, after the teardown that beat it to the table.

        Under the control lock, which is also the wait: the concurrent ``stop()`` holds it
        for the whole of its teardown, so this runs after that has finished and drains only
        what it left behind. Deliberately narrower than :meth:`_release` — the trace sink,
        the run's totals and the memory pool belong to whoever owns the server now, and this
        start does not.

        A no-op when there is nothing left, which is the ordinary failed-start path: the
        caller's own :meth:`stop` has already drained the table.
        """
        with self._control_lock:
            self._leave_service_tier()
            self._drain_models()

    def _join_service_tier(self) -> Any:
        """Offer the shared models to this shard's peers and take theirs (ADR-015).

        The gate is the shard index alone: the launcher sets `runner.service.shard` for a
        child that is part of a tier, and nothing else does. It used to be gated on the
        topology *name* as well (`kind == "service"`), which stopped meaning anything when
        the placement classes were deleted and "topology" became the chain (A2 PR-6) — a
        second switch that could only ever disagree with the first.
        """
        service = self._settings.runner.service
        if service.shard is None:
            return None
        from shipinfer.engine.spill.mesh import ServiceMesh, wire_slot_bytes

        shared = {
            name: self._models[name] for name in service.shared_models if name in self._models
        }
        for name, candidate in shared.items():
            if not hasattr(candidate, "admit_local"):
                raise ConfigurationError(
                    f"shared model {name!r} is an ensemble; only plain models cross the tier "
                    f"— share the models it composes instead"
                )
        if not shared:
            _LOG.warning(
                "service tier: none of the shared models %s is loaded here; no tier joined",
                service.shared_models,
            )
            return None
        assert self._repository is not None  # start() loaded it before any model
        sizes = {
            name: wire_slot_bytes(self._repository.entry(name).config, service.slot_bytes)
            for name in shared
        }
        mesh = ServiceMesh(service, service.shard, shared, slot_bytes_by_model=sizes)
        mesh.create()
        try:
            mesh.connect()
        except BaseException:
            # A peer that never appeared (or a failed open) must not leak the rings this
            # shard already created: close and unlink them so a restart can recreate the
            # names, then let the start fail with the connect error.
            mesh.stop()
            raise
        return mesh

    @property
    def service_mesh(self) -> Any:
        """The tier this process joined, or ``None`` when it joined none."""
        return self._mesh

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
        """Load one model at start-up, honouring ``strict_startup`` — and the abort, which
        overrides it."""
        try:
            self._build_and_start(name)
        except Exception:
            if self._settings.strict_startup:
                raise
            if self._is_stopping():
                # "Continuing" is meaningless once a stop has begun, and it was not free: a
                # non-strict start met by a `stop()` logged one ERROR with a full traceback
                # per remaining model -- five for one shutdown -- and then went on building
                # models the teardown had already been past, which is the leak `stop()`
                # takes the control lock to prevent. The abort `Model.start` polls is a
                # decision about the *server*, not about this model, so it propagates.
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
        try:
            if isinstance(model, Model):
                # Only a plain model's start spawns worker threads and waits on engine
                # deserialisation, which is the start that can run for minutes; the
                # ensemble's is a DAG walk over models that are already up.
                model.start(should_abort=self._is_stopping)
            else:
                model.start()
        except BaseException:
            # A model starts its instances one at a time and then waits for each; a failure
            # on the third leaves the first two with live worker threads holding backends,
            # and this model is about to become unreachable. Stop it before letting the
            # error out, guarded so the teardown cannot replace the reason.
            try:
                model.stop()
            except Exception:
                _LOG.exception("error stopping model %s after a failed start", name)
            raise
        with self._lock:
            self._models[name] = model
        return model

    def _is_stopping(self) -> bool:
        """Whether a :meth:`stop` has begun — the abort signal a long model start polls.

        Both flags, not just ``_started``: an in-progress :meth:`start` runs with
        ``_starting`` set and ``_started`` still false, and a model loaded by *that* start
        must not read its own server as stopping.

        Read without a lock on purpose. It is a hint checked between instance starts, both
        flags are plain bools assigned under the GIL, and the worst a stale read can do is
        start one more instance that the unwind then stops.
        """
        return not (self._started or self._starting)

    def stop(self) -> None:
        """Drain and release. Never raises.

        Safe to call twice, on a server that was never started, and — the case that matters
        — on one whose :meth:`start` raised half-way. It used to return early on
        ``not self._started``, which is exactly false for a partial start: the flag is only
        set once every model is up, so the one situation in which models were running and
        unreachable was the one situation ``stop()`` refused to handle.

        Every step is guarded, because this runs on the failure path of :meth:`start` and a
        teardown error there would replace the load error the operator needs to read.

        **The teardown runs under the control lock**, so it cannot interleave with a
        :meth:`load_model` that is in progress. Without that, the two overlap in exactly the
        way that leaks: a load passes its started check, spends seconds building a backend
        and starting worker threads, and publishes the model into a table this method has
        already drained — a running model, with live threads, on a server that reports
        itself stopped and holds no reference to stop it. Taking the lock makes the two
        orderings the only two: the load finishes first and this drains it, or it finds
        ``_started`` false under the lock and is refused. The cost is that a stop arriving
        mid-load waits for that load to finish, which is bounded by model start-up and is
        the right trade against dropping a GPU context on the floor.

        **The entry is one atomic transition**, under ``_lifecycle_lock``, so exactly one
        caller ever moves this server from started to stopping and every other one waits on
        the barrier. That used to be a check and an act with a log emit between them; see
        the comment below for what the two stops that both won the check went on to do.
        """
        with self._lifecycle_lock:
            # Read and cleared as one step, because a second stop that also reads "started"
            # here is a second *teardown*. The two used to be separated by the `_LOG.info`
            # below -- an emit that formats its arguments, may take a handler lock and may
            # write to a file, which is a guaranteed GIL switch point rather than a couple
            # of bytecodes. Both threads then fell through, both cleared the flags, and both
            # ran `_teardown` -> `_release` in turn: the second `_release` re-read the trace
            # totals from the null sink the first had just installed and published
            # `recorded: 0` for a run that traced thousands of requests, and `_torn_down`
            # was set by the first while the second was still releasing, so a third `stop()`
            # returned claiming a teardown that was still running.
            already_stopped = not (self._started or self._starting)
            # Read with the flags, and handed to the teardown: it identifies the run being
            # released, so a teardown that falls behind a newer one can tell and stand down.
            generation = self._generation
            if not already_stopped:
                # Cleared here, *before* the control lock is taken rather than inside it, so
                # a `load_model` already blocked on the control lock sees the false flag the
                # moment it gets in. Clearing it also makes a second call a no-op even if a
                # step below raises something this method deliberately does not catch.
                self._started = False
                self._starting = False
        if already_stopped:
            # Either there was nothing to release, or another thread is releasing it right
            # now — and the flags cannot tell those apart, because the thread that got here
            # first cleared them *before* taking the control lock. Returning immediately in
            # the second case is what let a caller pair `stop()` with an immediate `start()`
            # and have the first thread's teardown land on the fresh server: its models
            # drained, its trace sink closed. So wait for the barrier instead.
            self._await_teardown()
            return
        _LOG.info("stopping shipinfer (%d model(s))", len(self._models))
        with self._control_lock:
            self._teardown(generation)

    def _await_teardown(self) -> None:
        """Block until whoever is tearing down has finished, or the grace period expires.

        Bounded by ``shutdown_grace_s`` — the same budget the teardown grants each instance
        to drain — because ``stop()`` is also what a supervisor calls on its own deadline,
        and a wait that could not expire would turn one wedged instance into a shard that is
        SIGKILLed instead of drained.

        Expiry is logged, not raised: :meth:`stop` never raises (it runs on
        :meth:`start`'s failure path, where a teardown error would replace the load error the
        operator needs to read), so the honest signal is a WARNING naming the thread that is
        still inside the teardown -- ``_teardown`` records its own name on the way in, which
        is what makes the log line point at a thread an operator can find in a stack dump.
        An owner still queued on the control lock has not reached the teardown yet and has
        no name to give, so the message says "another thread" for that case rather than
        inventing one.

        **What the expiry now costs.** It used to be a silent downgrade: this returned, the
        caller took that for "released", and a :meth:`start` on the other side of it re-armed
        the barrier and got the still-queued teardown landing on the fresh server — its
        models drained, its sink closed, under a server that reported itself ready.
        :meth:`_begin_start` closes that door. A start that arrives while a teardown is in
        flight waits on the same barrier and then **refuses**, typed, instead of proceeding;
        and a teardown that finishes after a newer run began stands down at the generation
        check in :meth:`_release` rather than releasing that run's state. So an expiry here
        now means "the shutdown is still running and this server is unusable until it
        finishes", which is the honest reading of the warning below, and the ``do not
        start() this instance again`` it asks for is enforced rather than requested.
        """
        if self._torn_down.wait(self._settings.shutdown_grace_s):
            return
        _LOG.warning(
            "stop() returned while %s is still tearing this server down (waited %.1fs, the "
            "shutdown grace period); models may still be draining, so do not start() this "
            "instance again until they are",
            self._teardown_owner or "another thread",
            self._settings.shutdown_grace_s,
        )

    def _teardown(self, generation: int) -> None:
        """Release everything the server holds. Called by :meth:`stop`, under the lock.

        Sets ``_torn_down`` on the way out, in a ``finally``: a concurrent second
        :meth:`stop` is blocked on that barrier, and a teardown that raised something this
        method deliberately does not catch must still release it rather than hang the
        caller for its whole grace period.

        Records the running thread's name first, so a waiter whose grace period expires can
        name whoever it gave up on instead of logging an anonymous warning.

        Args:
            generation: The run :meth:`stop` read off the flags it cleared. Carried straight
                through to :meth:`_release`, which will not touch a server that has moved on
                to a later one.
        """
        self._teardown_owner = threading.current_thread().name
        try:
            self._release(generation)
        finally:
            self._torn_down.set()

    def _release(self, generation: int) -> None:
        """The teardown proper. Every step is guarded; see :meth:`stop`.

        Every step is also *destructive* — the model table is drained, the trace sink is
        closed, the memory pool is closed — so each one is done on behalf of exactly one run.
        ``generation`` says which, and the check below is what stops a teardown that has
        fallen behind from tearing down the run that replaced it. :meth:`_begin_start`
        refuses to start while a teardown is in flight, so reaching the stale branch takes a
        teardown that outran its own grace period; the guard is the second line, not the
        first, and it is here because the first one is a timed wait.
        """
        if self._stale(generation):
            return
        self._leave_service_tier()
        self._drain_models()
        # After the models, so a trace written by a worker finishing its last batch still
        # has somewhere to go; closing it flushes whatever the sink had buffered. The totals
        # are read *first*, while the sink is still live — see `stats()`.
        with self._lifecycle_lock:
            sink = self._traces if self._generation == generation else None
        if sink is None:
            self._warn_overtaken(generation)
            return
        totals: dict[str, Any] | None
        try:
            totals = sink.stats()
        except Exception:
            _LOG.exception("error reading the trace sink's totals")
            totals = None
        with self._lifecycle_lock:
            # The totals and the swap are published as one transition, so no scrape can land
            # between them. Split, they are the wrong answer this pair exists to remove: a
            # scrape that read `_last_trace_stats` as None and then found the null sink
            # already installed reports `{"sink": "none", "recorded": 0}` for a run that
            # traced thousands of requests — and a scrape does not stop when the server does,
            # so that zero is the last sample a dashboard takes.
            #
            # Back to the null sink, rather than leaving the field pointing at a closed one.
            #
            # Not because a reader would fail: no sink in this tree reads its stream to
            # answer `stats()` (the counters are plain ints on the base class), and `start()`
            # rebinds this field unconditionally, so a restart does not inherit anything
            # either. The reason is narrower and it is enough: `traces` is a public property
            # and `self._traces` is what every model built in this run was handed, so leaving
            # a closed object there publishes a handle whose every `record()` silently
            # increments `failed` instead of writing. A field must not point at a closed
            # object once the owner is done with it.
            #
            # The cost of the reset is that the run's totals go with it — a scrape after the
            # shutdown would read the fresh null sink's zeros, and "0 traces recorded" is a
            # different claim from "we did not measure any more". `_last_trace_stats` keeps
            # them, and `stats()` reports those while the server is stopped.
            current = self._generation == generation
            if current:
                if totals is not None:
                    self._last_trace_stats = totals
                self._traces = NullTraceSink()
        if not current:
            self._warn_overtaken(generation)
            return
        # The close comes after the swap, not before it: a scrape holding the lifecycle lock
        # is either in front of the swap and reads the live sink, or behind it and reads the
        # totals, and neither of those can be a `stats()` call on a sink that is closing.
        try:
            sink.close()
        except Exception:
            _LOG.exception("error closing the trace sink")
        try:
            self._memory.close()
        except Exception:
            _LOG.exception("error closing the memory pool")

    def _stale(self, generation: int) -> bool:
        """Whether ``generation``'s teardown has been overtaken by a later run."""
        with self._lifecycle_lock:
            stale = self._generation != generation
        if stale:
            self._warn_overtaken(generation)
        return stale

    def _warn_overtaken(self, generation: int) -> None:
        """Say that a teardown stood down, and why. A WARNING because reaching it means a
        teardown outlived its own grace period — the state :meth:`_await_teardown` warns
        about, seen from the other side."""
        _LOG.warning(
            "abandoning the teardown of run %d: run %d has since started on this server, "
            "and releasing its models, trace sink or memory pool now would tear down a "
            "server that is up",
            generation,
            self._generation,
        )

    def _leave_service_tier(self) -> None:
        """Close and unlink this shard's rings, if it joined a tier. Guarded, like every
        teardown step.

        First of the teardown's steps, and shared with :meth:`_abandon_start`: peers see the
        closed rings and fail their in-flight requests to us with the tags, instead of
        waiting on a model that is stopping.
        """
        if self._mesh is None:
            return
        mesh, self._mesh = self._mesh, None
        try:
            mesh.stop()
        except Exception:
            _LOG.exception("error leaving the service tier")

    def _drain_models(self) -> None:
        """Empty the model table and stop what was in it. Guarded per model.

        Reverse of the order `start()` loaded them in, which is dependents before
        dependencies: `start()` loads plain models and then the ensembles that compose them,
        so an ensemble is stopped while every step it may still dispatch to is alive, rather
        than after they have gone.

        Shared with :meth:`_abandon_start`, which needs exactly this and none of the rest of
        the teardown — the table is the only thing a lost start can have left behind.
        """
        with self._lock:
            models = list(self._models.values())
            self._models.clear()
        for model in reversed(models):
            try:
                model.stop()
            except Exception:
                _LOG.exception("error stopping model %s", model.name)

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
            ServerStateError: before :meth:`start`, and when :meth:`stop` ran between this
                call's started check and the lock — see the re-check below.
            ConfigurationError: when the model's config or artefacts are wrong. The server
                keeps serving everything else.
        """
        self._require_control("load", name)
        with self._control_lock:
            # Checked twice, and the second one is the one that matters. Between
            # `_require_control` above and this lock the server can have stopped — the whole
            # of `stop()`'s teardown runs under this same lock — and building the model
            # anyway would publish live worker threads into a table nothing drains again.
            # Refuse instead, with the state error the caller can act on; the first check
            # stays because it gives the same answer without waiting behind a slow load.
            if not self._started:
                raise ServerStateError(
                    f"cannot load model {name!r}: the server stopped while the request was "
                    "waiting"
                )
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
            ServerStateError: when :meth:`stop` ran between this call's started check and
                the lock — the same window :meth:`load_model` re-checks for, and the same
                reason. Without it the wait ends at ``self.model(name)`` against a table
                ``_teardown`` has just cleared, so the caller is told "no such model" —
                which reads as "you asked for something that does not exist" and sends an
                operator to check their spelling, when what happened is that the server
                stopped underneath them and the model was drained on the way out.
        """
        self._require_control("unload", name)
        with self._control_lock:
            if not self._started:
                raise ServerStateError(
                    f"cannot unload model {name!r}: the server stopped while the request "
                    "was waiting"
                )
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
            "tracing": self._trace_stats(),
            # Snapshotted, not iterated: `Model.stats()` is enough Python that the
            # interpreter can switch mid-comprehension, and a concurrent unload then raises
            # from a scrape that worked a second earlier.
            "models": [m.stats() for m in self._models_snapshot()],
        }

    def _trace_stats(self) -> dict[str, Any]:
        """The live sink's numbers while a run is up; the totals the last run finished with
        once it is down.

        Both reads are under ``_lifecycle_lock``, and so is the teardown's swap, because
        this is a check *and* an act on state another thread rewrites. Unlocked it was two
        bytecodes with a window between them: a scrape that found ``_last_trace_stats`` still
        None and then loaded ``_traces`` after `_release` had installed the null sink reports
        ``{"sink": "none", "recorded": 0}`` for a run that traced thousands of requests. A
        scrape does not stop when the server does, so that zero is the last sample a
        dashboard takes of the shard — the exact wrong answer `_last_trace_stats` was added
        to remove, arriving by the door the fix left open.

        The lock is held across the sink's own ``stats()``, which is a dict of five ints on
        the base class, and that is deliberate too: a snapshot taken under the lock and read
        outside it could be read after :meth:`_release` closed it.

        Returns:
            A **copy**. The stored totals are handed to a metrics exporter and to the KServe
            stats route; returning the dict itself lets either of them mutate what every
            later scrape reports.
        """
        with self._lifecycle_lock:
            last = self._last_trace_stats
            return dict(last) if last is not None else self._traces.stats()

    def render_metrics(self, exporter: str | None = None) -> str:
        """Metrics in the configured wire format."""
        name = exporter or self._settings.observability.metrics_exporter
        return EXPORTERS.create(name).render(self._metrics.registry)

    def __repr__(self) -> str:
        state = "ready" if self.is_ready else ("started" if self._started else "stopped")
        return f"<InferenceServer {state} models={self.models()}>"
