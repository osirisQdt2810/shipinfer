"""The runner contract: how a validated topology is executed, and by whom.

A **runner** is the third of arch.md's three concepts (§1). The topology says *what* runs;
the runner says *where and how*: in this process with a pool of worker threads
(``inprocess``), across one shard process per GPU (``fleet``, phase A2f), or compiled into a
GStreamer graph (``deepstream``, phase E). One chain definition, three executions — which
only works if the three agree on a contract, and this file is it.

What a runner owns, and nothing else:

* **placement** — which shard, which GPU, which process. An element is *told*, through the
  :class:`~shipinfer.topology.base.ElementContext` it receives at ``open()``; it never
  chooses (``topology/base.py``).
* **admission** — the bounded, per-camera-fair lane in front of the chain (arch.md §5②) and
  the typed refusal when it is full. That refusal reaches the producer, because a producer
  that is told "no" can drop *its own* newest frame; a silent eviction downstream charges
  the drop to whichever camera happens to be in the buffer (ADR-005).
* **the walk** — one item through the chain's elements, in an order the loader has already
  proved legal, honouring each ``when:`` with skip-and-continue.
* **the camera set**, for the runners that have one. ``add_camera`` / ``remove_camera`` /
  ``drain`` are the control plane's per-camera RPCs (arch.md §2) as methods, so
  :mod:`shipinfer.runners.service` can answer them over *any* runner. The default is a typed
  refusal, because "no" is a real answer: the launcher places the camera on another shard.

What a runner must **not** own: batching (the engine's, arch.md §5④), what a cap means (the
loader's), or any per-element knowledge. A runner that special-cased a kind would make the
registry decorative.

``start`` / ``stop`` / ``submit`` are **template methods**, exactly as
:class:`~shipinfer.topology.base.Element` and
:class:`~shipinfer.ingest.base.FrameSource` do it, and for the same reason: the four
invariants below are the ones a second implementation would re-derive and eventually get
wrong.

* ``start`` is idempotent, and unwinds a partial start before re-raising — a runner that
  opened six of nine elements and then raised must not leave six decoder threads and a CUDA
  context behind on a shared box;
* ``stop`` is idempotent and safe before ``start``, so the shutdown path and the failed-start
  path are one path;
* ``submit`` before ``start`` is a typed refusal, never an implicit start: opening a chain on
  a producer's thread is how a CUDA context ends up on the wrong thread (ADR-002);
* ``health`` and ``stats`` answer *while running and while stopped*, because the first
  question asked of a runner that will not start is what state it thinks it is in;
* ``request_stop`` **records** and ``supervise`` **blocks**, so a signal handler never does
  the stopping — the invariant ``launch/signals.py`` was written around, moved onto the
  contract so ``forward_signals`` works over any runner and not only over a fleet.
"""

from __future__ import annotations

import abc
import contextlib
import threading
from collections.abc import Callable
from typing import Any, ClassVar

from shipinfer.core.errors import ServerStateError
from shipinfer.core.request import ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Device
from shipinfer.launch.control import CameraSpec
from shipinfer.topology import ChainItem, ElementContext, ModelResolver, Topology

__all__ = ["Runner"]


class Runner(abc.ABC):
    """Executes one :class:`~shipinfer.topology.chain.Topology`. Subclass, register, done.

    Args:
        topology: a **validated** chain. Injected rather than loaded here: a runner is given
            the chain to run, and ``Topology.from_spec`` is the single door through which a
            chain becomes trustworthy (ADR-017). One topology per runner — element instances
            are stateful, so two runners over one ``Topology`` would share a tracker.
        settings: the deployment settings. Defaults to ``ServerSettings()``, which is what a
            test and a laptop want; a shard is given the tree its launcher resolved.
        shard_id: which shard this runner is (arch.md §2). Handed to every element for logs,
            metrics and the one decision ``scope: global`` needs — never used as a key for
            per-camera state, which keys on the camera id.
        device: the GPU this runner owns, or ``None`` on a host with no accelerator. Carried,
            not chosen: the launcher decided.
        models: the model pool, for elements of kind ``pool``. A
            :class:`~shipinfer.topology.base.ModelResolver` — structural, so ``runners`` need
            not import the engine and a test satisfies it with a dict.
        chain_yaml: the text ``topology`` was loaded from, when the caller has it. Provenance
            for a runner that executes here; **required** by one that hands the chain to other
            processes, because the loader is the single door through which a chain becomes
            trustworthy (ADR-017) and on a fleet that door is on the shard. Re-rendering a
            ``Topology`` back to YAML would be a second writer of the format its loader is
            the only reader of.

    Subclasses implement :meth:`_do_start`, :meth:`_do_stop` and :meth:`_do_submit`, and may
    add to :meth:`health` and :meth:`stats` through the two optional hooks.
    """

    #: The registered implementation name, set by each subclass so the name in the settings
    #: tree and the name in a log line cannot drift. Mirrors
    #: :attr:`shipinfer.scheduling.queues.base.RequestQueue.name`.
    name: ClassVar[str] = "abstract"

    #: Whether :meth:`add_camera`, :meth:`remove_camera` and :meth:`drain` do anything.
    #: ``False`` here, and the three methods refuse; a runner that owns an ingest manager
    #: sets it True and implements them. It exists so a caller can tell "this runner does not
    #: manage cameras, so it abandoned none of them" from "this runner manages cameras and
    #: could not release them" — both arrive as a ``ServerStateError`` otherwise, and the
    #: shard's ``Stop`` reply has to report 0 for the first and a failure for the second
    #: (``runners/service.py``).
    manages_cameras: ClassVar[bool] = False

    #: Whether this runner opens the chain's elements in *this* process, and therefore wants
    #: the caller to build a model pool and hand it in as ``models=``. ``False`` here, and the
    #: default is the safe one: a runner that executes the chain somewhere else must not make
    #: the process that built it load engines. ``fleet`` is exactly that case and leaves it
    #: ``False`` -- each shard builds its own engine (``cli/shard.py``), and a launcher that
    #: built one too would hold a CUDA context on every device it can see while doing no
    #: inference at all, which is the one thing ``check_layers.py`` keeps ``launch`` clean of.
    #: Read off the *class*, before a runner is built, because ``models=`` is a constructor
    #: argument (``cli/commands/run.py``).
    needs_model_pool: ClassVar[bool] = False

    def __init__(
        self,
        topology: Topology,
        settings: ServerSettings | None = None,
        *,
        shard_id: int = 0,
        device: Device | None = None,
        models: ModelResolver | None = None,
        chain_yaml: str = "",
    ) -> None:
        self._topology = topology
        self._chain_yaml = chain_yaml
        self._settings = settings if settings is not None else ServerSettings()
        self._shard_id = shard_id
        self._device = device
        self._models = models
        self._running = False
        # Only ever taken by `start` and `stop`, so that "idempotent" survives two threads
        # asking at once — a supervisor stopping a runner while a health probe restarts it is
        # an ordinary race. Deliberately *not* taken by `submit`, `health` or `stats`: the
        # first is the hot path and the other two must answer while `stop` is joining
        # workers, which is exactly when someone wants to know what is happening.
        self._lifecycle = threading.RLock()
        #: Somebody asked this runner to stop, without doing any of the stopping. Set by
        #: :meth:`request_stop` and read by :meth:`supervise`; an `Event` because the setter
        #: is a signal handler and the only thing a handler may safely do is record
        #: (``launch/signals.py`` makes the whole argument). Cleared by `start`, so a runner
        #: that was asked to stop and then restarted supervises again instead of returning at
        #: once.
        self._stop_requested = threading.Event()

    # -- what the runner was told ------------------------------------------------------

    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def chain_yaml(self) -> str:
        """The text the topology was loaded from, or ``""`` when the caller had none."""
        return self._chain_yaml

    @property
    def settings(self) -> ServerSettings:
        return self._settings

    @property
    def shard_id(self) -> int:
        return self._shard_id

    @property
    def device(self) -> Device | None:
        """The GPU this runner owns, or ``None`` on a host with no accelerator."""
        return self._device

    @property
    def models(self) -> ModelResolver | None:
        return self._models

    @property
    def is_running(self) -> bool:
        return self._running

    def element_context(self) -> ElementContext:
        """What every element is told at ``open()``.

        One object for the whole chain, built once here rather than per element, because it
        is the *runner's* decision and two elements of one chain disagreeing about which GPU
        they are on is not a state worth being able to represent.

        The two settings-derived fields are **resolved here** rather than read by the element:
        ``topology`` is pure and may not import the settings tree, so a knob an operator turns
        reaches an element only if a runner carries it. ``pipeline.stage_timeout_ms`` becomes
        :attr:`~shipinfer.topology.base.ElementContext.stage_timeout_s` and
        ``ingest.input_name`` becomes ``input_name``; an element still prefers its own
        ``params:`` over both, because a tensor name belongs to the model and not to the
        deployment.
        """
        return ElementContext(
            shard_id=self._shard_id,
            device=self._device,
            models=self._models,
            stage_timeout_s=self._settings.pipeline.stage_timeout_ms / 1000.0,
            input_name=self._settings.ingest.input_name,
        )

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> Runner:
        """Open the chain and start executing it. Idempotent.

        A partial start leaks whatever it did manage to acquire — an open decoder, a socket, a
        worker thread, a CUDA context on a shared box. The subclass often cannot tell how far
        it got, so unwind unconditionally and best-effort, then re-raise the *original*
        failure, which is the one worth reading. Same shape as
        :meth:`shipinfer.topology.base.Element.open`, for the same reason.

        **This is the only unwind.** A subclass that catches its own partial start and
        released things itself would run the release twice, and the second pass would see the
        state the first one cleared — which is how a fleet came to report zero abandoned
        camera threads after unwinding six of them. :meth:`_do_stop` is the single owner, and
        :meth:`_unwind_timeout_s` is how a subclass whose release is not instantaneous says
        what budget that pass gets.

        Raises:
            ShipInferError: whatever the implementation needs to say — a missing model, a
                camera that will not open, a port already bound. Nothing is swallowed.
        """
        with self._lifecycle:
            if self._running:
                return self
            self._stop_requested.clear()
            try:
                self._do_start()
            except BaseException:
                with contextlib.suppress(Exception):
                    self._do_stop(self._unwind_timeout_s())
                raise
            self._running = True
            return self

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop executing and release everything :meth:`start` acquired. Idempotent.

        ``timeout_s`` is one deadline for the whole shutdown, not one per worker or per
        element — the same rule :meth:`shipinfer.ingest.manager.IngestManager.stop` follows:
        everyone is signalled at once, so a worker still unfinished at the deadline is
        genuinely stuck, and charging the budget per worker would turn one stuck element into
        thirty-two consecutive waits.

        The runner is marked stopped *before* the hook runs, so a hook that fails cannot
        leave behind a runner that still claims to be running — a supervisor's restart would
        then skip it and the shard would serve nothing forever.
        """
        with self._lifecycle:
            if not self._running:
                return
            self._running = False
            self._do_stop(timeout_s)

    def __enter__(self) -> Runner:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- supervision -------------------------------------------------------------------

    @property
    def stop_requested(self) -> bool:
        """Whether :meth:`request_stop` has been called since the last :meth:`start`."""
        return self._stop_requested.is_set()

    def request_stop(self) -> None:
        """Ask :meth:`supervise` to return. Records only; does none of the stopping.

        Thread-safe and non-blocking by construction, because the caller is a **signal
        handler**: :func:`shipinfer.launch.signals.forward_signals` routes Ctrl-C here, and a
        handler that called :meth:`stop` directly would block up to the whole shutdown budget
        inside the lock it takes — a second Ctrl-C then re-enters that frame and waits on
        itself. Setting an event cannot block. The stopping happens on the supervising thread,
        which is the one allowed to take its time.
        """
        self._stop_requested.set()

    def supervise(
        self, *, poll_s: float = 1.0, until: Callable[[], bool] | None = None
    ) -> None:
        """Block while this runner runs; return when it should not any more.

        Returns when :meth:`request_stop` is called, when ``until()`` says so, or when the
        runner has stopped. Does **not** stop anything: the caller's ``finally: stop()`` owns
        that, so "supervise returned" means the same thing on every runner.

        This default is the in-process shape — the workers are threads in this process, and a
        thread that dies takes the process with it, so there is nothing to watch for beyond
        being told to go. :class:`~shipinfer.runners.fleet.FleetRunner` overrides it, because
        for a fleet there *is*: a shard that exits while the rest keep reporting healthy is
        the state a supervisor exists to refuse to sit in.

        Args:
            poll_s: how often ``until()`` is consulted. A request to stop is not polled — it
                wakes the wait immediately.
            until: an extra reason to return, consulted every ``poll_s``.
        """
        while not self._stop_requested.is_set():
            if until is not None and until():
                return
            if not self._running:
                return
            self._stop_requested.wait(poll_s)

    def describe_plan(self) -> str:
        """What this runner would do, for ``shipinfer run --dry-run``. Spawns nothing.

        The default is the honest answer for a runner that places nothing: there is one
        process, and it is this one. A runner that *does* decide a placement overrides it with
        the plan itself — computed, not described a second time, so a dry run and a real start
        cannot disagree.
        """
        return "no plan: one process"

    # -- submission --------------------------------------------------------------------

    def submit(self, item: ChainItem) -> ResponseFuture:
        """Admit one item into the chain.

        Returns:
            A future that completes when the item has been walked to the end of the chain.
            Its result is the item as the last *non-None* element saw it — a sink consumes
            its item and returns ``None``, which is not the walk producing nothing. Its
            *exception* is how a caller learns the item was dropped — a deadline that passed,
            a shutdown, an element that failed. A caller that does not care may discard it,
            but there is always one: a frame that vanishes with no typed outcome delivered is
            the failure mode ADR-005 exists to prevent.

        Raises:
            ServerStateError: before :meth:`start`. A refusal, not an implicit start.
            QueueFullError: the lane for this item's camera is full. Propagated untouched so
                the producer can drop *its own* newest frame and charge it to the right
                camera (arch.md §5②).
        """
        if not self._running:
            raise ServerStateError(
                f"runner {self.name!r} was asked to submit before start(); "
                "call start() or use it as a context manager"
            )
        return self._do_submit(item)

    # -- cameras -----------------------------------------------------------------------
    #
    # The control plane's three per-camera RPCs (arch.md section 2), as runner methods. They
    # live on the ABC rather than only on the runner that implements them because the shard
    # servicer must be able to call them on *any* runner: `runners/service.py` is handed a
    # runner and answers `AddCamera` over it, and a servicer that first asked "are you the
    # kind of runner that..." would be the if/elif the registry exists to prevent
    # (CONVENTIONS 2.3).
    #
    # The default is a typed refusal rather than a silent success, because "no" is a real
    # answer here: the launcher places the camera on another shard. Returning None would
    # leave it believing a camera is being read that nobody is reading (ADR-005's failure
    # mode, one layer up).

    def add_camera(self, camera: CameraSpec) -> None:
        """Start one camera on this runner.

        Raises:
            ServerStateError: this runner does not manage cameras.
            ConfigurationError: a camera with this id is already running, or its source
                cannot be resolved. Both reach the launcher as ``accepted=False`` with the
                reason, because both mean "place it elsewhere" rather than "this shard is
                broken" (``ingest/manager.py``).
        """
        raise self._no_camera_management()

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool:
        """Stop and forget one camera.

        Returns:
            Whether its thread stopped within the deadline. ``False`` means it was abandoned
            and still references this runner's buffers — the caller's to know, not the log's
            to bury (``ingest/manager.py``).

        Raises:
            ServerStateError: this runner does not manage cameras.
            ConfigurationError: no such camera. Naming what is running turns a typo in an
                operator's call into an answer instead of a silent no-op.
        """
        raise self._no_camera_management()

    def drain(self, timeout_s: float = 20.0) -> int:
        """Stop reading cameras and let what is in flight finish.

        ``timeout_s`` is ONE deadline for the whole camera set, not one per camera: everyone
        is signalled at t0, so a camera still unfinished at the deadline is genuinely stuck,
        and charging the budget per camera would turn one stuck decoder into fifty
        consecutive waits (``ingest/manager.py``).

        Returns:
            How many camera threads had to be abandoned; ``0`` is the clean drain.

        Raises:
            ServerStateError: this runner does not manage cameras.
        """
        raise self._no_camera_management()

    def _no_camera_management(self) -> ServerStateError:
        return ServerStateError(
            f"this runner does not manage cameras ({self.name!r}); frames enter its chain "
            "through a decode element, and only a runner that owns an ingest manager can "
            "add, remove or drain one while it runs"
        )

    # -- observability -----------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """What state this runner is in, and whether its chain is open.

        Answers while stopped as well as while running, which is the point: the first
        question asked of a runner that will not start is what it thinks its state is.
        """
        health: dict[str, Any] = {
            "runner": self.name,
            "state": "running" if self._running else "stopped",
            "shard_id": self._shard_id,
            "device": str(self._device) if self._device is not None else None,
            "topology": self._topology.name,
            "elements": {node.name: node.element.is_open for node in self._topology},
        }
        health.update(self._do_health())
        return health

    def stats(self) -> dict[str, Any]:
        """Counters an operator would page on. Merged with the subclass's own."""
        stats: dict[str, Any] = {
            "runner": self.name,
            "topology": self._topology.name,
            "shard_id": self._shard_id,
        }
        stats.update(self._do_stats())
        return stats

    # -- subclass hooks ----------------------------------------------------------------

    @abc.abstractmethod
    def _do_start(self) -> None:
        """Open the chain and start whatever executes it. Called at most once per cycle."""

    def _unwind_timeout_s(self) -> float:
        """The budget :meth:`start` gives :meth:`_do_stop` when it unwinds a partial start.

        Zero for a runner whose release is local and immediate — closing elements, joining
        threads that were never handed work. A runner whose release is a *conversation*
        overrides it: a fleet's unwind is a ``Stop`` RPC to every shard that did come up, and
        those shards have frames in flight whether or not their sibling ever answered, so
        giving them no budget would abandon work a shutdown would have finished.
        """
        return 0.0

    @abc.abstractmethod
    def _do_stop(self, timeout_s: float) -> None:
        """Release everything :meth:`_do_start` acquired, within one shared deadline.

        Must tolerate being called after a *partial* start — :meth:`start` calls it to unwind
        one — and must not raise for work it never started.
        """

    @abc.abstractmethod
    def _do_submit(self, item: ChainItem) -> ResponseFuture:
        """Admit one item. See :meth:`submit` for the contract."""

    def _do_health(self) -> dict[str, Any]:
        """Implementation-specific health. Optional: queue depth, worker liveness."""
        return {}

    def _do_stats(self) -> dict[str, Any]:
        """Implementation-specific counters. Optional."""
        return {}

    def __repr__(self) -> str:
        state = "running" if self._running else "stopped"
        return (
            f"<{type(self).__name__} {self.name} topology="
            f"{self._topology.name or '<unnamed>'} shard={self._shard_id} {state}>"
        )
