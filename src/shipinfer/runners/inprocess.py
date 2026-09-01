# doc: long the five choices below each had a simpler-looking alternative that is wrong
"""The single-process runner: a fair lane per camera, N workers, one frame each.

arch.md section 5 with the process boundary removed -- the fair lanes and the pipeline
workers, over whatever model pool the runner was handed. One worker takes one item and walks
it through the whole chain, sleeping at every pool wait; concurrency comes from having
several workers and from each *model* batching across the frames in flight. The per-frame
walk itself is :mod:`shipinfer.runners.walk`.

Five choices, each with a simpler-looking alternative that is wrong:

* **The queue is the fair one, typed on** :class:`~shipinfer.scheduling.work.WorkItem`.
  Admission is the problem the engine's per-model queues already solve (ADR-005), so an item
  is *wrapped* rather than the queue generalised. The three fields it reads -- camera, band,
  deadline -- are request fields, so the wrap buys the per-camera lane for free.
* **One worker walks the topological order.** Not a thread per element: nine elements as nine
  queues puts eight hand-offs in every frame's path, and the loader already proved the order
  legal (ADR-017).
* **Fan-in merges deterministically or not at all** -- "whichever finished last" is no answer
  when the branches carry different metadata (:meth:`~shipinfer.runners.walk.ChainWalk.inbound`).
* **An element that raises loses one item, not the worker**, which would stop serving every
  camera on the shard.
* **A batch is as wide as the worker pool.** The walk is synchronous, so at
  ``pipeline.workers = 4`` a shard offers each model batches of at most 4 whatever
  ``max_batch_size`` says. A throughput ceiling, not a correctness bug; the fix is more
  workers, and the asynchronous walk is arch.md section 5.

**The cameras are the runner's; the decode element only names them.** A decode element that
opened its own camera would drag the camera set into ``topology``, which must stay pure
enough to validate a chain on a laptop. ``shipinfer.ingest`` is imported inside
:meth:`InprocessRunner._ingest` so ``import shipinfer.runners`` costs no decode runtime.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from shipinfer.core.errors import (
    ConfigurationError,
    QueueFullError,
    RequestCancelledError,
    ServerStateError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.request import InferenceRequest, Priority, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.ingest import CameraConfig
from shipinfer.core.types import Device
from shipinfer.launch.control import CameraSpec
from shipinfer.runners.base import Runner
from shipinfer.runners.frames import ChainFrameSink
from shipinfer.runners.metrics import RunnerMetrics
from shipinfer.runners.registry import RUNNERS
from shipinfer.runners.walk import ChainWalk, ChainWork
from shipinfer.scheduling.queues import QUEUES, BatchWindow, RequestQueue
from shipinfer.scheduling.work import WorkItem
from shipinfer.topology import (
    Caps,
    ChainItem,
    Element,
    ElementContext,
    ElementKind,
    ImageOpsLike,
    ModelResolver,
    Topology,
    WaiterBudget,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; `ingest` is imported inside `_ingest`
    from shipinfer.ingest import IngestManager, SourceFactory

__all__ = ["InprocessRunner"]

_LOG = get_logger("runners.inprocess")

#: The share of a stop's budget the cameras get before the workers are joined. Cameras are
#: released first because they are the producers, and joining workers while frames keep
#: arriving is a shutdown racing its own input; the share exists because one decoder wedged
#: inside a blocking read must not spend the whole deadline the workers need to resolve what
#: is already in flight. Both halves are measured against ONE deadline, not two budgets --
#: the same rule ``IngestManager.stop`` follows within the camera set.
_INGEST_STOP_SHARE = 0.5

#: What ``stats()["ingest"]`` says on a runner that has no camera set: before the first
#: start, and after a stop released it. Written out rather than taken from
#: :meth:`shipinfer.ingest.IngestSummary.as_dict`, whose keys these are, because reaching for
#: that class would import the whole ingest plane -- and with it a decode runtime and
#: ``shipinfer.runtime`` -- onto a runner that has never been given a camera.
#: ``tests/runners/test_camera_lifecycle``
#: asserts the two shapes still agree, which is the check that keeps a copy honest.
_NO_INGEST: Mapping[str, Any] = MappingProxyType(
    {
        "cameras": 0,
        "streaming": 0,
        "unhealthy": 0,
        "total_fps": 0.0,
        "frames_read": 0,
        "frames_published": 0,
        "frames_dropped": 0,
    }
)


@dataclass(frozen=True, slots=True)
class _Head:
    """What the chain's decode roots agree the frames entering them are.

    Resolved once per runner from the *loader's* answers, and both fields refuse a chain
    whose roots disagree: one ingest manager publishes into one sink, every root of the walk
    sees that one item (:meth:`~shipinfer.runners.walk.ChainWalk.run`), so two roots wanting different
    payloads or different decoders is a chain this runner cannot honour. Refusing at start
    beats stamping one root's answer on the other's frames.

    Args:
        caps: the negotiated cap of the edges leaving the roots -- read from
            :attr:`~shipinfer.topology.chain.Topology.edges`, never from the root element's
            own ``produces``. A cap belongs to an edge, and an element with two ``produces``
            hands a different one to each consumer (``topology/chain.py::Edge``).
        source: the name of an ingest source in :data:`shipinfer.ingest.SOURCES`, or ``None``
            when the chain did not say and ``ingest.backend`` (then the environment) decides.
    """

    caps: Caps
    source: str | None


# doc: long an Args block for nine constructor parameters is specification, not prose
@RUNNERS.register("inprocess", "single")
class InprocessRunner(Runner):
    """Runs the whole chain in this process, on a pool of worker threads.

    Args:
        topology: the validated chain.
        settings: deployment settings; ``pipeline.workers``, ``queue_type``,
            ``queue_capacity``, ``overflow_policy`` and ``frames_per_wakeup`` are read here.
        shard_id: passed to every element (arch.md section 2).
        device: the GPU this runner owns, or ``None``.
        models: the model pool for ``pool`` elements.
        ops: image pre-processing, handed in for the reason ``models`` is.
        queue: override the admission queue. An injected one is never replaced -- see
            :meth:`_do_start` for what that means for a restart.
        workers: override the worker count; ``None`` takes ``pipeline.workers``.
        metrics: share a registry with the rest of the process. ``None`` mints a private one.
            The ingest plane's handles go on the *same* registry, so one exporter carries
            both halves of a dropped frame.
        source_factory: overrides how every camera's source is built. The seam a test uses to
            run the camera lifecycle against a fake camera.

    Raises:
        ConfigurationError: ``workers`` below one -- a runner that accepts items and walks
            none of them looks exactly like a hung chain.

    **Not honoured yet: ``per:`` and ``scope:``.** One element instance is shared by every
    worker, so at ``workers > 1`` two frames of one camera can be inside a ``per: camera``
    element at once and its ordering can invert. Nothing stateful ships today, so this is a
    promise not yet kept rather than a live defect; a chain with a stateful element runs
    correctly at ``workers=1``. Resolved in phase C.
    """

    name: ClassVar[str] = "inprocess"
    #: This runner owns an ingest manager, so the control plane's three per-camera RPCs do
    #: something here (``runners/base.py`` says what ``False`` buys the ones that do not).
    manages_cameras: ClassVar[bool] = True
    #: The chain is walked here, so the ``pool`` elements in it resolve their models against
    #: a pool this process owns; whoever builds this runner builds that pool and passes it as
    #: ``models=``.
    needs_model_pool: ClassVar[bool] = True

    def __init__(
        self,
        topology: Topology,
        settings: ServerSettings | None = None,
        *,
        shard_id: int = 0,
        device: Device | None = None,
        models: ModelResolver | None = None,
        ops: ImageOpsLike | None = None,
        chain_yaml: str = "",
        queue: RequestQueue | None = None,
        workers: int | None = None,
        metrics: RunnerMetrics | None = None,
        source_factory: SourceFactory | None = None,
    ) -> None:
        super().__init__(
            topology,
            settings,
            shard_id=shard_id,
            device=device,
            models=models,
            ops=ops,
            chain_yaml=chain_yaml,
        )
        pipeline = self._settings.pipeline
        if workers is not None and workers < 1:
            raise ConfigurationError(
                f"runner workers must be >= 1, got {workers}; a runner with no workers "
                "accepts items and walks none of them"
            )
        self._wanted_workers = pipeline.workers if workers is None else workers
        #: The permits every waiting element in this process shares, sized once here. One
        #: object for the runner and not one per element: two elements that each waited
        #: `workers - 1` of their own would park every worker between them, and then no
        #: element could close on evidence -- the stall the guard exists to prevent. Handed
        #: out on `element_context()`.
        self._waiter_budget = WaiterBudget(max(0, self._wanted_workers - 1))
        # `is not None`, never `queue or ...`: a `RequestQueue` defines `__len__`, so an empty
        # injected queue is *falsy* and the truthy form silently throws the caller's object
        # away. It cost an afternoon in `pipeline/runner.py`; the comment there says so.
        self._injected_queue = queue is not None
        self._queue = self._build_queue() if queue is None else queue
        self._window = BatchWindow(max_batch_size=pipeline.frames_per_wakeup)
        # Measured from *capture*, as `QueueFrameSink` measures it: the gap between capture
        # and dequeue is precisely the queue latency a frame deadline exists to catch. A
        # context with no capture clock (a hand-built item in a test) gets no deadline rather
        # than one that expired before the process started.
        self._deadline_ns = self._settings.ingest.frame_deadline_ms * 1_000_000
        self._threads: list[threading.Thread] = []
        #: This start cycle's stop signal, and *only* where `_do_stop` finds it — a worker is
        #: handed the event as an argument and never reads this attribute. Rebuilt by
        #: `_do_start`, never cleared: an event a worker holds must stay set forever, because
        #: the worker holding it may be one abandoned at a shutdown deadline, and clearing it
        #: for the next cycle is what turned that stale thread back into a live consumer of
        #: the *new* cycle's queue. See `_work` for what that cost.
        self._stopping = threading.Event()
        self._metrics = RunnerMetrics() if metrics is None else metrics
        #: What each worker still owes an answer for, one slot per worker. Single writer (the
        #: owning worker), read only by `_do_stop` after the join deadline, so the hot path
        #: takes no lock. It exists so an abandoned worker does not take its items' futures
        #: with it -- an unresolved future is the frame ADR-005 exists to prevent.
        #:
        #: **One list per start cycle, handed to the workers as an argument**; this attribute
        #: is only where `_do_stop` finds the current one. A worker that read it instead wrote
        #: into whatever list it pointed at *now*, so abandon-restart-abandon cleared a live
        #: worker's slot in the new cycle.
        #:
        #: A tuple of the undelivered **remainder**, not the item being walked: the rest of a
        #: wake-up batch is off the queue too, so a single slot would strand
        #: `frames_per_wakeup - 1` futures nobody owns.
        self._inflight: list[tuple[WorkItem, ...]] = [()] * self._wanted_workers
        #: The bottom half: everything from here down is chain items and elements. See
        #: `runners/walk.py`, which is where the per-frame loop lives.
        self._walker = ChainWalk(
            topology,
            self._metrics,
            {(edge.producer, edge.consumer): edge.caps for edge in topology.edges},
        )
        #: The cameras, or ``None`` when this runner has none. Built on **first use**
        #: (:meth:`add_camera`) and never here, because constructing it imports the ingest
        #: plane and ``shipinfer.runtime`` behind it, which ``import shipinfer.runners`` must
        #: not pay for (``tests/test_architecture.py``). A runner nobody places a camera on
        #: never touches it. Rebuilt per start cycle: a manager stopped at a deadline may still
        #: hold an abandoned decoder thread, and reusing it would make that thread a live
        #: producer into the next cycle's queue.
        self._ingest_manager: IngestManager | None = None
        self._source_factory = source_factory
        #: negotiated-rate cache for `_camera_fps`; dropped on re-add (the rate may change).
        self._fps_cache: dict[str, float] = {}
        #: consecutive frames that asked and got nothing; caps the retries above.
        self._fps_misses: dict[str, int] = {}
        #: ``camera_id -> priority`` for the cameras **this process's own configuration**
        #: names, plus the default learned once for a camera it does not
        #: (:meth:`_learn_priority`). Filled from ``ingest.cameras`` when the ingest manager
        #: is built (:meth:`_ingest`) and never emptied, because it is a fact about the
        #: settings this process loaded rather than about any one placement: a camera removed
        #: and posted again is the same camera, and the operator's band for it did not change.
        #:
        #: Resolved by camera and not carried on a frame, exactly as ``pipeline/sink.py``
        #: resolves it: a frame is data and a priority is configuration, so there is one place
        #: it can be wrong and it is the config file.
        self._configured: dict[str, Priority] = {}
        #: ``camera_id -> priority`` for the band a **launcher** named on the spec of a camera
        #: held right now (:meth:`_admit_at`). Kept apart from :attr:`_configured` because the
        #: two have different lifetimes: one dict meant nothing could be un-said, so DELETE then
        #: POST-with-no-band left the camera in the removed one's ``tracking_critical`` -- the
        #: wrong direction to get wrong (ADR-005). Tracks the live camera set: written when a
        #: spec carries a band, dropped when it does not and when the placement ends.
        self._placed_bands: dict[str, Priority] = {}
        #: Only taken when a camera is seen for the first time, which is once per camera for
        #: the life of the process -- never on the steady-state submission path.
        self._priority_lock = threading.Lock()
        #: The chain's head, resolved on demand and then kept: a topology is immutable once
        #: built, so the answer cannot change under a running runner.
        self._head_resolved: _Head | None = None

    def _build_queue(self) -> RequestQueue:
        """The configured admission queue, named after the chain it fronts."""
        pipeline = self._settings.pipeline
        return QUEUES.create(
            pipeline.queue_type,
            f"chain:{self._topology.name or 'unnamed'}",
            pipeline.queue_capacity,
            overflow=pipeline.overflow_policy,
            drop_expired=True,
        )

    # -- introspection -----------------------------------------------------------------

    @property
    def queue(self) -> RequestQueue:
        """The admission queue. Read by a bench and by a test that provokes overflow."""
        return self._queue

    @property
    def workers(self) -> int:
        """How many worker threads this runner wants."""
        return self._wanted_workers

    @property
    def metrics(self) -> RunnerMetrics:
        """The counters, per camera. ``stats()`` is the rolled-up view of these."""
        return self._metrics

    def element_context(self) -> ElementContext:
        """The base context plus the three things only an in-process runner knows.

        * ``metrics`` is **this** runner's registry, so one exporter carries both halves of a
          dropped frame; a fresh one would publish counters nothing scrapes.
        * ``workers`` is the thread count actually started, not ``settings.pipeline.workers``.
          An MTMC barrier told "four" on a one-worker runner parks the only thread there is.
        * ``waiter_budget`` is one shared pool, not one per element: each element counts only
          its own waiters, so two waiting elements would each admit ``workers - 1``.

        :meth:`Runner.element_context` leaves all three ``None`` because a runner that does not
        execute the chain here has none of them to promise.
        """
        return replace(
            super().element_context(),
            metrics=self._metrics.registry,
            workers=self._wanted_workers,
            waiter_budget=self._waiter_budget,
        )

    @property
    def cameras(self) -> tuple[str, ...]:
        """The camera ids this runner is reading, sorted. Empty when it has none.

        Reads the ingest manager rather than building one, so asking a stopped runner what it
        is reading stays a cheap, side-effect-free question -- the same rule
        :meth:`Runner.health` follows.
        """
        manager = self._ingest_manager
        return () if manager is None else tuple(manager.camera_ids)

    # -- cameras -----------------------------------------------------------------------
    #
    # doc: long the lock rule below is the one thing that cannot be re-derived from the code
    # The three per-camera RPCs (arch.md section 2), over an ingest manager this runner owns
    # for one start cycle. Every invariant -- one actor per camera, the insert/start/re-check,
    # the one deadline for the whole camera set -- is the manager's; what is added here is the
    # mapping from the launcher's vocabulary to the ingest plane's.
    #
    # **All three take :attr:`_lifecycle`**, because they are called from a servicer's thread
    # pool, so an add racing a `stop()` is ordinary. Without it, an add that passed the
    # `_running` check just before `stop()` cleared the flag went on to build a *fresh*
    # manager on a torn-down runner and started a decoder thread into it that nothing stops.
    # Re-entrant on purpose: a camera method called from inside a lifecycle step must not
    # deadlock. Not taken by `submit`, `health`, `stats` or `cameras` -- the first is the hot
    # path, the rest must answer while a stop is joining threads.

    #: How many frames may ask a camera for a rate it has not negotiated. `actor()` takes
    #: `IngestManager._lock` -- the one `add_camera`/`remove_camera` hold -- so a source whose
    #: rate never resolves (RTSP negotiation failed, a container with no metadata) would
    #: otherwise take that mutex once per frame forever: 1000 acquisitions/s at the design
    #: sizing, serialising ingest on a camera that is already degraded.
    _FPS_ATTEMPTS = 8

    def _camera_fps(self, camera_id: str) -> float:
        """The camera's negotiated rate, cached once non-zero (constant per connect).

        A rate of zero is cached too, after :attr:`_FPS_ATTEMPTS` frames: it is the answer for
        a source that does not know its own rate, and re-asking costs a shared lock per frame.
        """
        cached = self._fps_cache.get(camera_id, 0.0)
        if cached:
            return cached
        if self._fps_misses.get(camera_id, 0) >= self._FPS_ATTEMPTS:
            return 0.0
        manager = self._ingest_manager
        if manager is None:
            return 0.0
        try:
            fps = manager.actor(camera_id).source_fps
        except ConfigurationError:
            return 0.0
        if fps:
            self._fps_cache[camera_id] = fps
            self._fps_misses.pop(camera_id, None)
        else:
            self._fps_misses[camera_id] = self._fps_misses.get(camera_id, 0) + 1
        return fps

    def add_camera(self, camera: CameraSpec) -> None:
        """Start one camera on this runner.

        Nothing is caught, including the *type*: ``IngestManager.add_camera`` raises a
        ``DuplicateCameraError`` or a ``ServerStateError`` and ``runners/service.py`` maps each
        to ``accepted=False`` with its reason. The duplicate is deliberately not flattened to
        ``ConfigurationError`` -- ``POST /streams`` re-mints an id on that refusal and no
        other.

        The state check and the add are **one atomic step**, which is why this holds
        :attr:`_lifecycle`: it is the one camera method that can *build* an ingest manager, so
        a check and an add either side of a ``stop()`` left a live decoder thread on a runner
        already torn down.

        **A refusal leaves the priority table as it found it.** The band must be recorded
        before :meth:`_camera_config` reads it back, so it is rolled back by
        :meth:`_restore_band` if the add raises; :meth:`_priority_for` is consulted per frame.

        Raises:
            ServerStateError: not running, or the fleet forgot the camera while starting.
            DuplicateCameraError: a camera with this id is already running.
            ConfigurationError: the source the chain names is not registered.
        """
        with self._lifecycle:
            if not self._running:
                raise ServerStateError(
                    f"runner {self.name!r} was asked to add camera {camera.camera_id!r} "
                    "while it is not running (before start(), or after a stop() on another "
                    "thread); the chain is not open, so the camera would decode into a "
                    "closed queue -- call start() first"
                )
            manager = self._ingest()
            # The band is recorded first because `_camera_config` reads it back (the record
            # and the lane are one resolution), and undone if the placement is refused --
            # `_placed_bands` is read per frame by `_priority_for`, so a band left behind by
            # an add that raised would move a *running* camera's lane on the strength of a
            # request the server rejected. `None` is a sound "there was none" here: the table
            # never holds `None`, which is the invariant `_priority_for`'s `is not None`
            # already rests on.
            previous = self._placed_band(camera.camera_id)
            self._admit_at(camera)
            try:
                self._fps_cache.pop(camera.camera_id, None)
                self._fps_misses.pop(camera.camera_id, None)
                manager.add_camera(self._camera_config(camera))
            except BaseException:
                self._restore_band(camera.camera_id, previous)
                raise
            # After the actor exists, not before, and the order is the decision: a refused
            # placement must announce nothing, and an element cannot tell a camera that was
            # refused from one that was placed and has yet to send a frame. See
            # `Element.camera_added` for what that costs.
            self._announce(Element.camera_added, camera.camera_id)

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool:
        """Stop and forget one camera.

        Returns:
            Whether its thread stopped within the deadline. ``False`` means it was abandoned
            and still holds a decoder and a reference to this runner's sink -- the caller's to
            know, not the log's to bury.

        Under :attr:`_lifecycle` so that the manager this reads is the manager it removes
        from: :meth:`_stop_ingest` drops the attribute, and a removal that had already read it
        would otherwise stop a camera on a manager the shutdown has stopped tracking. It does
        **not** raise on a stopped runner: a runner that is not running holds no cameras, so
        "there is no cam-7 here" is still the true answer, and it is the one
        ``runners/service.py`` turns into ``removed=False`` (a ``ServerStateError`` there means
        "this camera raised on its way out", which is a different thing to tell an operator).

        Raises:
            ConfigurationError: no such camera. Also the answer on a runner that has no camera
                set at all, because "there is no cam-7 here" is the same fact either way and a
                caller that has to tell the two apart is a caller writing an if/elif.
        """
        with self._lifecycle:
            manager = self._ingest_manager
            if manager is None:
                raise ConfigurationError(
                    f"camera {camera_id!r} is not running; runner {self.name!r} has no cameras"
                )
            removed = manager.remove_camera(camera_id, timeout_s=timeout_s)
            # The placement is over, so the band that came with it is over too. Without this
            # the entry outlived the camera and the *next* spec for the same id -- a re-`POST`
            # with no band, which means "leave it to the deployment" -- inherited a lane the
            # caller never asked for. Popped even when `removed` is `False`: an abandoned
            # decoder thread is still a camera this runner no longer holds a placement for.
            with self._priority_lock:
                self._placed_bands.pop(camera_id, None)
            # After the actor is stopped, which is the safe order: dropping a tracker's shard
            # while its decoder is still publishing would let the very next frame rebuild it.
            # `removed` being False means the thread was abandoned at the deadline rather than
            # joined, so a late frame is more than theoretical -- `Element.camera_removed`
            # says so, and an element must treat one as a first frame.
            self._announce(Element.camera_removed, camera_id)
            return removed

    def drain(self, timeout_s: float = 20.0) -> int:
        """Stop reading every camera and let what is already admitted finish.

        The workers, the queue and the chain are deliberately left running: a drain is how a
        shard is emptied without being torn down, and the items already in the lane still have
        producers waiting on their futures. The manager is kept rather than dropped, so a
        camera may be added again afterwards -- whether that is *allowed* is the shard
        servicer's decision and not this runner's (``runners/service.py`` owns the drained
        refusal).

        Under :attr:`_lifecycle`, which serialises it against a concurrent ``stop()`` rather
        than letting both call ``IngestManager.stop`` on the same manager and charge two
        deadlines to one camera set. A stopped runner is not refused for the reason
        :meth:`remove_camera` is not: it has no cameras, it therefore abandoned none of them,
        and ``0`` is what ``runners/service.py::_drain`` needs to hear -- an error there lands
        in ``DrainReply.detail`` and tells a launcher the drain *failed*.

        Returns:
            How many camera threads had to be abandoned; ``0`` is the clean drain, and ``0``
            is also the honest answer for a runner that had no cameras to release.
        """
        with self._lifecycle:
            manager = self._ingest_manager
            if manager is None:
                return 0
            # Snapshotted *before* the stop, because `IngestManager.stop` clears the actor map
            # and the ids would be gone by the time there was anything to announce. A drain is
            # a removal of every camera at once, so every element hears about every one of
            # them -- a shard drained and then placed on again must not keep the previous
            # deployment's per-camera state, which is the same argument ADR-018 makes for the
            # single-camera path.
            placed = tuple(manager.camera_ids)
            abandoned = manager.stop(timeout_s=timeout_s)
            # Every camera is released, so every band a launcher placed with one is stale --
            # the same fact :meth:`remove_camera` records one camera at a time, through the
            # door that empties the shard in one call.
            with self._priority_lock:
                self._placed_bands.clear()
            for camera_id in placed:
                self._announce(Element.camera_removed, camera_id)
            return abandoned

    def _announce(self, hook: Callable[[Element, str], None], camera_id: str) -> None:
        """Tell every element that a camera arrived or left. Best-effort, never fatal.

        One element that raises must not take the operation with it: a tracker that fails to
        drop a shard *and* blocks the removal is a shard that cannot be recovered, and ADR-018
        names remove + add as the only recovery there is. Logged with the element's name; the
        loop continues.

        Called under :attr:`Runner._lifecycle` by all three camera methods, so an element sees
        one camera's announcements in order. That lock orders them against each other and
        against ``start``/``stop`` -- and nothing else: this runs concurrently with
        :meth:`Element.process`, so an element with a per-camera table needs its own lock.

        Topological order both ways, unlike ``close``: there is nothing to unwind, so two
        orders would differ only in log sequence.

        Args:
            hook: the :class:`~shipinfer.topology.base.Element` method, named through the ABC
                so a typo is a ``NameError`` at import rather than a missing announcement.
        """
        for node in self._topology.nodes:
            # Resolved on the *instance*, not called as `hook(node.element, ...)`.
            # `Element.camera_added` is the ABC's own function object, and invoking it with
            # an instance runs the base no-op straight past every override -- an announcement
            # loop that reaches every element and tells none of them anything, with nothing
            # to see in a log. `getattr` is the ordinary bound lookup, so the override runs;
            # the argument stays the ABC's method so the name is still checked at import.
            announce = getattr(node.element, hook.__name__)
            try:
                announce(camera_id)
            except Exception:
                _LOG.exception(
                    "element %r raised in %s(%r); the camera lifecycle continues without it",
                    node.name,
                    hook.__name__,
                    camera_id,
                    extra=log_context(camera_id=camera_id),
                )

    # -- resolving the ingest plane ----------------------------------------------------

    def _head(self) -> _Head:
        """What the chain says frames entering it are, and which source produces them.

        Read off the **loader's** answers: the cap comes from ``Topology.edges``, never from
        the root element's ``produces``, because a cap belongs to an edge. The source comes
        from the decode slot's ``params:`` first, its class's ``source`` second.

        A decode root declaring **more than one** cap is refused rather than read. Every
        decode element hands the frame on untouched, so the cap the sink stamps is a claim
        about a buffer nothing converts: with two declarations the loader would pick the
        consumer's preference and the sink would stamp it on the same unconverted array. A
        converting decode is phase D.

        Resolved once and kept; a topology is immutable once built.

        Raises:
            ConfigurationError: roots disagree on the cap or the source; a root with no
                successor; a root declaring two caps; or a root that is not a decode element
                (said again here because a ``Topology`` may be built directly from parts).
        """
        if self._head_resolved is not None:
            return self._head_resolved
        caps: dict[str, Caps] = {}
        sources: dict[str, str | None] = {}
        for root in self._topology.roots:
            if root.kind is not ElementKind.DECODE:
                raise ConfigurationError(
                    f"element {root.name!r} is a root of topology "
                    f"{self._topology.name or '<unnamed>'} but is a {root.kind.value} "
                    "element; frames enter a chain through a decode element, and only a "
                    "decode element names the ingest source that produces them"
                )
            if len(root.element.output_caps) > 1:
                raise ConfigurationError(
                    f"decode element {root.name!r} declares "
                    f"{[str(cap) for cap in root.element.output_caps]}, but a pass-through "
                    "decode declares exactly one cap: the runner's frame sink stamps the "
                    "negotiated cap onto a buffer the element hands on unchanged, so a "
                    "second declaration is a claim about pixels nothing converts. A "
                    "converting decode -- `gstreamer-gpu`, which would keep NV12 in VRAM -- "
                    "is phase D, behind the DataPool"
                )
            outbound = {
                edge.caps for edge in self._topology.edges if edge.producer == root.name
            }
            if len(outbound) != 1:
                raise ConfigurationError(
                    f"decode element {root.name!r} leaves the chain under "
                    f"{sorted(str(cap) for cap in outbound) or 'no'} cap(s); a runner submits "
                    "one frame to it and needs exactly one answer for what that frame is"
                )
            caps[root.name] = outbound.pop()
            declared = root.element.params.get("source")
            # `getattr`, and the contract it is reading is named rather than duck-typed:
            # `topology/elements/decode.py` documents `source` as the class attribute every
            # decode element in that family carries, defaulting to `""` for "the chain did not
            # say". A decode element from outside that family -- a test's own probe, which
            # invents a frame handle instead of naming a decoder -- carries no such attribute,
            # and `""` is the right reading of that too. An `isinstance` against
            # `_IngestDecode` would import a private class to ask a question a documented
            # attribute already answers, and would refuse those chains outright.
            inherent = getattr(root.element, "source", "")
            sources[root.name] = str(declared) if declared else (str(inherent) or None)
        if not caps:
            raise ConfigurationError(
                f"topology {self._topology.name or '<unnamed>'} has no decode element; "
                "frames enter a chain through its roots"
            )
        if len(set(caps.values())) != 1:
            raise ConfigurationError(
                "the decode elements of topology "
                f"{self._topology.name or '<unnamed>'} disagree about what enters the chain: "
                f"{ {name: str(cap) for name, cap in caps.items()} }; every root sees the same "
                "submitted frame, so they must agree"
            )
        if len(set(sources.values())) != 1:
            raise ConfigurationError(
                "the decode elements of topology "
                f"{self._topology.name or '<unnamed>'} name different ingest sources: "
                f"{sources}; one ingest manager feeds every root, so they must agree"
            )
        self._head_resolved = _Head(
            caps=next(iter(caps.values())), source=next(iter(sources.values()))
        )
        return self._head_resolved

    def _camera_config(self, camera: CameraSpec) -> CameraConfig:
        """The launcher's five-field :class:`CameraSpec` as an ingest camera.

        Every other field stays ``None``, which means *inherit*: codec, transport, hardware
        decode, jitter buffer and the reconnect schedule are **deployment** settings a shard
        resolves from its own tree (CONVENTIONS 2.6). Three exceptions ride the spec:

        * ``source`` — what the *chain* asked for, and the line that makes
          ``decode: {impl: replay}`` mean anything. ``None`` lets ``ingest.backend`` decide.
        * ``loop`` — the only per-camera field that decides whether the camera *ends*. The
          ``--inputs`` camera never appears in ``ingest.cameras``, so without this the setting
          the help text names is unreachable for exactly the cameras that need it.
        * ``priority`` — read back from :meth:`_priority_for`, never ``camera.priority or
          NORMAL``: ``TRACKING_CRITICAL`` is ``0`` and therefore falsy, so ``or`` demotes it.
        """
        return CameraConfig(
            camera_id=camera.camera_id,
            uri=camera.url,
            fps=camera.fps,
            source=self._head().source,
            loop=camera.loop,
            priority=self._priority_for(camera.camera_id),
        )

    def _admit_at(self, camera: CameraSpec) -> None:
        """Record the band the *launcher* chose, outranking the config.

        The launcher wins because on a fleet shard it is the only one who knows: a shard's
        ingest config is stripped (:meth:`_ingest`), so every band an operator wrote would
        resolve to ``NORMAL``.

        A spec with no band **erases** any band a previous placement of the same id left, and
        falls back to :attr:`_configured`. Two independent sources, not one dict: ``None``
        means "leave it to the deployment", and after a DELETE and a re-POST the only way to
        honour that is to drop what the dead placement said.

        Under :attr:`_priority_lock` because :meth:`_priority_for` reads the same tables on
        the submit path.
        """
        with self._priority_lock:
            if camera.priority is None:
                self._placed_bands.pop(camera.camera_id, None)
            else:
                self._placed_bands[camera.camera_id] = camera.priority

    def _placed_band(self, camera_id: str) -> Priority | None:
        """The band a placement has recorded for this camera, or ``None`` if none has.

        Separate from :meth:`_priority_for`, which answers what *lane* a camera admits into
        by falling back through :attr:`_configured`; this one answers only what
        :attr:`_placed_bands` holds, because that is the single value
        :meth:`_restore_band` has to be able to put back.
        """
        with self._priority_lock:
            return self._placed_bands.get(camera_id)

    def _restore_band(self, camera_id: str, previous: Priority | None) -> None:
        """Put :attr:`_placed_bands` back the way :meth:`_placed_band` found it.

        The undo half of :meth:`_admit_at`. ``IngestManager.add_camera`` refuses a duplicate
        id *after* the band was written, so without this a rejected ``POST /streams`` would
        answer 400 and still move that camera's lane for the rest of its life.

        ``previous is None`` pops rather than writes, because :attr:`_placed_bands` never
        stores ``None``. Never ``if previous:`` -- ``TRACKING_CRITICAL`` is ``0`` (ADR-005).
        """
        with self._priority_lock:
            if previous is None:
                self._placed_bands.pop(camera_id, None)
            else:
                self._placed_bands[camera_id] = previous

    def _ingest(self) -> IngestManager:
        """This cycle's ingest manager, built on first use.

        **The import is inside this method and that is load-bearing.** ``shipinfer.ingest``
        reaches ``shipinfer.runtime`` through its source registry, so naming it at module
        scope would put torch behind ``import shipinfer.runners`` -- which
        ``tests/test_architecture.py`` refuses, because it is what lets a chain start on a
        host with no driver.

        The ingest plane's metric handles go on **this** runner's registry, so one exporter
        carries both halves of a dropped frame: the camera's and the admission door's.
        """
        manager = self._ingest_manager
        if manager is not None:
            return manager
        from shipinfer.ingest import IngestManager, IngestMetrics, configured_cameras

        # `_do_submit`, not the public `submit`: a stop clears `_running` and only then
        # releases the cameras, so every frame an actor publishes during `_do_stop` would meet
        # a `ServerStateError` -- an error outside the `FrameSink` contract, which the actor
        # reads as a bug, backs off from and logs a traceback for. The queue is still open at
        # that moment; once it is closed the actor gets the `RequestCancelledError` the
        # contract does name, which is what tells it to finish.
        sink = ChainFrameSink(self._do_submit, self._head().caps, fps_of=self._camera_fps)
        ingest = self._settings.ingest
        manager = IngestManager(
            sink,
            # WITHOUT the configured camera set -- a defence, not a tidy-up. A shard is an
            # `InprocessRunner` built from env-only settings, so it inherits the operator's
            # whole fleet; a manager holding that list is one `start()` away from 8 shards x
            # 50 cameras, duplicate `(camera_id, frame_id)` tags, and every later
            # `add_camera` refused as already running. `add_camera` is THE door.
            #
            # `model_copy` rather than a rebuilt `IngestSettings`: the fields are being
            # cleared, so there is nothing left for a validator to judge.
            settings=ingest.model_copy(update={"cameras": [], "camera_db": None}),
            metrics=IngestMetrics(registry=self._metrics.registry),
            source_factory=self._source_factory,
        )
        # From the FULL settings, before any actor exists: a band is configuration keyed by
        # camera id (CONVENTIONS 2.6), so a camera this process is *told* about is admitted
        # into the band its config names even if it does not start it.
        #
        # ON A FLEET SHARD THIS TABLE IS EMPTY -- `ingest.cameras` was cleared above -- which
        # is why the launcher sends the band on the `CameraSpec` instead. What is left for
        # this line is the single-process deployment, where the config IS the operator's.
        self._configured.update(
            {camera.camera_id: camera.priority for camera in configured_cameras(ingest)}
        )
        self._ingest_manager = manager
        return manager

    def _priority_for(self, camera_id: str) -> Priority:
        """The lane band this camera's items are admitted into.

        Three sources in precedence: the launcher's placement (:attr:`_placed_bands`), this
        process's ``ingest.cameras`` (:attr:`_configured`), then ``NORMAL``.

        Expressed as *lookups*, not write order into one dict: two dicts have two lifetimes
        and can disagree honestly, so a placement's band can be un-said. Write order meant a
        camera removed and re-posted with no band inherited the dead placement's lane.

        ``is not None`` and never ``or``: ``TRACKING_CRITICAL`` is ``0`` and therefore falsy,
        so the shorter spelling demotes the one camera priorities exist for (ADR-005).
        """
        placed = self._placed_bands.get(camera_id)
        if placed is not None:
            return placed
        configured = self._configured.get(camera_id)
        if configured is not None:
            return configured
        return self._learn_priority(camera_id)

    def _learn_priority(self, camera_id: str) -> Priority:
        """Resolve, log and memoise the band for a camera that is not in the config.

        A camera added over the control plane at runtime is a normal event -- a fifty-camera
        site gains cameras during commissioning -- so it gets the fleet default rather than an
        error. Logged **once**, because "my new camera is not being prioritised" is otherwise
        an invisible configuration gap, and memoised because paying for that discovery on
        every frame would make it a performance bug as well. The same shape, and the same
        argument, as ``pipeline/sink.py::_policy_for``.

        Memoised into :attr:`_configured` and not into :attr:`_placed_bands`, which is the
        difference between the two spelled out once more: what is being recorded is that
        *this process's configuration has nothing to say about this camera*, which is true for
        as long as those settings are, and is not undone by the camera being removed. A band a
        launcher named is undone by exactly that, which is why it lives in the other dict.
        """
        with self._priority_lock:
            existing = self._configured.get(camera_id)
            if existing is not None:
                return existing
            _LOG.info(
                "camera %s is not in the ingest config; admitting it at priority %s",
                camera_id,
                Priority.NORMAL.name,
                extra=log_context(camera_id=camera_id),
            )
            self._configured[camera_id] = Priority.NORMAL
            return Priority.NORMAL

    # -- lifecycle ---------------------------------------------------------------------

    def _do_start(self) -> None:
        """Open every element in topological order, then start the workers.

        Elements first: a worker that woke before the chain was open would meet a
        ``ServerStateError`` from ``Element.process`` on a real item -- a refusal the runner
        caused and the operator has to diagnose.

        An element failing at position five must leave one through four *closed*, and that
        unwind is :meth:`Runner.start`'s, which calls :meth:`_do_stop` on the way out. No
        second unwind loop here: ``Element.close`` is idempotent and a no-op on an element
        that never opened, so the shared path covers the partial case exactly.

        Raises:
            ServerStateError: the runner was given a queue that is already closed. Only
                reachable for an *injected* queue -- replacing the caller's object would
                silently ignore the capacity and policy they chose.
            ShipInferError: whatever an element raises from ``open()``.
        """
        # Before anything is opened, because it is the one start-up refusal that costs
        # nothing to check and everything to discover late: a chain whose roots disagree
        # about what enters it has no camera set this runner can serve, and finding that out
        # after nine elements are open means unwinding nine elements.
        head = self._head()
        if self._queue.is_closed:
            if self._injected_queue:
                raise ServerStateError(
                    f"runner {self.name!r} was given a closed queue "
                    f"({self._queue.name!r}); a closed queue fails every submission with "
                    "RequestCancelledError, so pass a fresh one to restart"
                )
            self._queue = self._build_queue()

        context = self.element_context()
        for index, node in enumerate(self._topology.nodes):
            try:
                node.element.open(context)
            except BaseException:
                _LOG.error(
                    "element %r failed to open; the %d already open are closed by the "
                    "unwind in Runner.start",
                    node.name,
                    index,
                )
                raise

        # Every piece of per-cycle state is built here and passed *by value* to this
        # cycle's workers, so an abandoned worker from the previous one cannot reach any of
        # it. Never `self._stopping.clear()` and never a shared queue attribute: those two
        # reads are what let a stale worker rejoin. See `_work` for the bug that shape fixes.
        stopping = threading.Event()
        self._stopping = stopping
        queue = self._queue
        inflight: list[tuple[WorkItem, ...]] = [()] * self._wanted_workers
        self._inflight = inflight
        for index in range(self._wanted_workers):
            thread = threading.Thread(
                target=self._work,
                args=(index, stopping, queue, inflight),
                name=f"chain-worker-{self._shard_id}-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        _LOG.info(
            "runner %s ready on shard %d: %s | %d worker(s) | queue=%s(%d) | head=%s",
            self.name,
            self._shard_id,
            " -> ".join(node.name for node in self._topology),
            self._wanted_workers,
            self._queue.name,
            self._queue.capacity,
            head.caps,
        )
        # No cameras. A start opens the chain and raises the workers; it reads nothing until
        # somebody places a camera, which is what `manages_cameras` being a control-plane
        # property means (arch.md section 2). Starting `settings.ingest.cameras` here is right
        # for one process and wrong for a shard, whose env-only settings inherit the operator's
        # whole fleet -- every shard would start every camera. `cli/commands/run.py` places
        # them through `add_camera` instead.

    def _do_stop(self, timeout_s: float) -> None:
        """Release the cameras, close the queue, join the workers, close the chain.

        The reverse of :meth:`_do_start`, and the order earns its place. Cameras first because
        they are the producers: joining while frames still arrive is a shutdown racing its own
        input. They get a share of the budget (:data:`_INGEST_STOP_SHARE`) against the same
        deadline, so one wedged decoder cannot spend the time the workers need. Closing the
        queue then *fails* what is still waiting with a typed error, and is how a worker
        blocked in ``get_batch`` learns to exit. One deadline for all workers, not one each.
        Elements close last, in reverse topological order, each guarded.

        The queue is left alone when no worker ever started -- the unwind of a failed
        ``start`` -- since nothing can have been submitted and closing would poison an
        injected queue.

        **An abandoned worker's items are failed, not forgotten.** Closing the queue resolves
        what is still queued; the wake-up batch a wedged worker already took is not, and it
        will never resolve any of it. Leaving them would break the promise ``submit`` makes,
        so every in-flight slot is drained here.

        The event, the queue and the slot list are **this cycle's**, which is what makes an
        abandoned worker terminal: it wakes to a set event and a closed queue and exits. While
        they came off ``self``, a restart handed the stale thread the new cycle's queue and it
        went back to work, publishing into a slot list no shutdown drains. The list is
        replaced rather than only drained, because an abandoned worker republishes its
        remainder on waking -- which left ``stats()["in_flight"]`` up forever after a stop.
        """
        deadline = time.monotonic() + timeout_s
        self._stop_ingest(timeout_s * _INGEST_STOP_SHARE)
        self._stopping.set()
        lost: int = 0
        if self._threads:
            lost = self._close_queue()
            abandoned = 0
            for thread in self._threads:
                thread.join(max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    abandoned += 1
            self._threads.clear()
            if abandoned:
                _LOG.warning(
                    "%d worker(s) did not stop within %.1fs; abandoning them",
                    abandoned,
                    timeout_s,
                )
            inflight = self._inflight
            stranded = self._fail_in_flight(inflight)
            # Drained *and replaced*. `_fail_in_flight` empties the slots, but an abandoned
            # worker still holds a reference to `inflight` and its `finally` writes the
            # remainder of its wake-up batch straight back into it -- so a `stats()` taken
            # after that store read a non-zero `in_flight` for a runner that had stopped, and
            # it never came back down. The stale thread goes on writing into the object it was
            # handed; nothing reads that object any more.
            self._inflight = [()] * self._wanted_workers
            if stranded:
                # Counted against `failed`, and an abandoned worker that later finishes its
                # walk counts the same item as `walked` too, so the two numbers can disagree
                # by the number of items abandoned. That is the honest shape — the item did
                # fail for its producer *and* did eventually run — and the alternative,
                # suppressing the walk's own count from a thread the runner has stopped
                # tracking, would need the hot path to check a flag.
                _LOG.warning(
                    "%d in-flight item(s) failed with the runner; their workers were "
                    "abandoned mid-walk",
                    stranded,
                )

        for node in reversed(self._topology.nodes):
            try:
                node.element.close()
            except Exception:
                _LOG.exception("element %r failed to close cleanly", node.name)

        totals = self._metrics.totals()
        _LOG.info(
            "runner %s stopped: %d item(s) failed in the queue, %d walked, %d failed",
            self.name,
            lost,
            totals["walked"],
            totals["failed"],
        )

    def _stop_ingest(self, timeout_s: float) -> int:
        """Release this cycle's cameras. Returns how many threads had to be abandoned.

        The manager is **dropped**, not merely stopped, for the same reason the stop signal
        and the queue are rebuilt per cycle: a decoder abandoned at this deadline is parked
        inside a blocking read, not gone, and it still holds a sink bound to this cycle's
        ``_do_submit``. Handing that manager to the next start would make the stale actor a
        live producer into the new cycle's queue -- the ingest-side spelling of exactly the
        bug ``_work`` documents on the worker side. A restarted runner therefore comes up
        with no cameras and waits to be placed on again, which is what the control plane's
        own vocabulary says a fresh shard is (``ShardState.READY``).

        Safe on the unwind path of a failed start, where there is no manager and this is a
        no-op costing one attribute read.
        """
        manager, self._ingest_manager = self._ingest_manager, None
        # For :meth:`drain`'s reason, and one more: the next cycle rebuilds the manager and
        # therefore re-reads :attr:`_configured` from settings, so a band left here from the
        # last cycle would outrank the operator's table on a runner that holds no cameras.
        with self._priority_lock:
            self._placed_bands.clear()
        if manager is None:
            return 0
        abandoned = manager.stop(timeout_s=timeout_s)
        if abandoned:
            _LOG.warning(
                "%d camera thread(s) did not stop within %.1fs; they still hold this "
                "runner's sink",
                abandoned,
                timeout_s,
            )
        return abandoned

    def _close_queue(self) -> int:
        """Close the admission queue and charge every item it failed. Returns how many.

        One call, not two, because the counter is only true if nothing can close the queue
        here without it: an outcome with no counter is the gap ``stats()`` exists to close.

        **A queue its owner closes is its owner's to count.** That close happens where this
        runner cannot see it, and attributing it from a stats delta would be a guess — a
        guessed counter being worse than an absent one.
        """
        drained = self._queue.close()
        for item in drained:
            self._metrics.items_queue_closed.inc(camera=item.request.context.camera_id)
        return len(drained)

    def _fail_in_flight(self, inflight: list[tuple[WorkItem, ...]]) -> int:
        """Fail everything the abandoned workers still owed an answer for. Returns how many.

        Each slot holds the item being walked *and* the rest of its wake-up batch, because
        neither is in the queue any more. An abandoned worker that finishes its current item a
        microsecond later is benign: the walker's ``fail`` and ``finish`` both refuse a future
        that is already resolved, whichever of the two arrived first.

        Args:
            inflight: the slot list of the cycle being stopped, passed rather than read off
                ``self`` so that this drains the same object its workers write to even when a
                restart has already published a newer one.
        """
        stranded = 0
        for slot, batch in enumerate(inflight):
            inflight[slot] = ()
            for work in batch:
                stranded += 1
                error = RequestCancelledError("the runner stopped")
                self._walker.count_failure(work, error)
                self._walker.fail(work, error)
        return stranded

    # -- submission --------------------------------------------------------------------

    def _do_submit(self, item: ChainItem) -> ResponseFuture:
        """Wrap the item in a work item and enqueue it.

        The request built here is a **carrier**: it exists so the fair queue can read the
        camera id, band and deadline it already knows how to read. No model sees it -- a
        ``pool`` element builds its own from the item it is handed.

        **The band is resolved per camera here**, and it is the only reason a priority lane
        does anything on this runner. It used to be left at the default, so every camera
        shared one lane and ``priority:`` applied to nothing.

        Raises:
            QueueFullError: this camera's lane is full. Propagated untouched and counted
                against **this** camera (ADR-005) as ``items_dropped`` -- refused at the door,
                so never one of ``accepted`` and not a term the ``stats()`` ledger names. A
                model queue refusing an item already inside the chain is ``items_backpressure``.
            RequestCancelledError: the queue is closed -- the runner is shutting down.
        """
        context = item.context
        request = InferenceRequest(
            model_name=self._topology.name or "chain",
            inputs={},
            context=context,
            priority=self._priority_for(context.camera_id),
            deadline_ns=(
                context.captured_ns + self._deadline_ns
                if self._deadline_ns and context.captured_ns
                else 0
            ),
        )
        work = ChainWork(request, ResponseFuture(request), item=item)
        try:
            self._queue.put(work)
        except QueueFullError:
            self._metrics.items_dropped.inc(camera=context.camera_id)
            raise
        self._metrics.items_accepted.inc(camera=context.camera_id)
        return work.future

    # -- the worker loop ---------------------------------------------------------------

    def _work(
        self,
        slot: int,
        stopping: threading.Event,
        queue: RequestQueue,
        inflight: list[tuple[WorkItem, ...]],
    ) -> None:
        """Drain the admission queue and walk one item at a time.

        An empty batch means the queue closed, which is how a worker learns to exit. The batch
        is a *wakeup* batch (``pipeline.frames_per_wakeup``), not an inference batch: batching
        for a GPU is the engine's job.

        **The stop signal is checked in front of every item**, not only every wake-up: a
        worker holds a whole batch, so a top-of-loop check alone let a worker abandoned at a
        shutdown deadline walk the rest of it through elements the runner had closed. The
        remainder is left in the slot, where :meth:`_fail_in_flight` has already failed it.

        **Every piece of per-cycle state is an argument, never read off** ``self``. An
        abandoned worker outlives its cycle, so what it reads from the runner belongs to
        whichever cycle is current when it wakes: ``_stopping`` was cleared by the next start,
        ``_queue`` was rebuilt so the stale thread took work off the new one, and ``_inflight``
        was rebound so what it published landed in the live worker's slot.
        """
        while not stopping.is_set():
            items = queue.get_batch(self._window, poll_s=0.05)
            if not items:
                if queue.is_closed:
                    return
                continue
            # Published before the first walk and narrowed in a `finally` after each item, so every
            # item this worker owes an answer for has an owner a shutdown can find. The
            # *remainder*, not the current item: the ones behind it left the queue in the same
            # drain. Both slices are free at the default `frames_per_wakeup = 1`.
            batch = tuple(items)
            inflight[slot] = batch
            for index, work in enumerate(batch):
                if stopping.is_set():
                    # Abandoned mid-batch. `_do_stop`'s drain already failed these futures,
                    # so walking them would emit ghost events through a closed chain.
                    #
                    # Deliberate on the CLEAN stop path too: `stop` is abort-shaped, and the
                    # remainder goes to `_fail_in_flight` for a typed cancellation inside the
                    # shutdown budget. Finishing the batch would be a *drain*, and a drain has
                    # no deadline -- one wedged element would hold the shutdown open.
                    break
                try:
                    self._walker.run(work)
                except Exception as exc:
                    # Reaching here means something outside an element failed. A worker that
                    # dies stops serving every camera on this shard, so the loop survives it
                    # — and the item's future is resolved, because a frame that vanishes with
                    # no typed outcome is the failure ADR-005 exists to prevent.
                    self._walker.count_failure(work, exc)
                    _LOG.exception("runner failed on %s", work.request.context.key)
                    self._walker.fail(work, self._walker.typed(exc, "the runner failed"))
                finally:
                    inflight[slot] = batch[index + 1 :]

    # -- failure and observability -----------------------------------------------------

    def _do_health(self) -> dict[str, Any]:
        """Queue, workers and one entry per camera.

        The camera map is **not optional decoration**: ``runners/service.py`` derives a
        shard's state from it, so a runner that manages cameras and reports none answers
        ``ready`` forever and never ``running``. An empty map is honest; a missing one is a
        runner lying about what it is.

        Carries :meth:`~shipinfer.ingest.CameraHealth.as_dict` plus one field ingest does not
        have: ``priority``, stamped here because a frame is data and a band is policy.
        """
        manager = self._ingest_manager
        return {
            "queue": self._queue.stats().as_dict(),
            "workers": {
                "wanted": self._wanted_workers,
                "alive": sum(1 for thread in self._threads if thread.is_alive()),
            },
            "cameras": (
                {}
                if manager is None
                else {
                    camera_id: {
                        **health.as_dict(),
                        "priority": self._priority_for(camera_id).name.lower(),
                    }
                    for camera_id, health in manager.health().items()
                }
            ),
        }

    def _do_stats(self) -> dict[str, Any]:
        """Every outcome an accepted item can reach, including the queue's own.

        The three ``queue_*`` terms and ``in_flight`` are here so that it adds up::

            accepted == walked + failed + expired + timed_out + backpressure
                        + queue_closed + queue_evicted + queue_expired + in_flight

        ``dropped`` is deliberately not a term: those submissions were never ``accepted``, so
        including it would over-count by exactly ``queue["rejected"]``.

        The identity holds within one start cycle on a runner that abandoned no worker. Two
        deliberate exceptions, named because a term pretending they are absent is worse: an
        abandoned worker finishing after :meth:`_do_stop` counts twice (failed and walked),
        and the ``queue_*`` terms reset on a restart while the runner's counters do not.
        """
        queue = self._queue.stats()
        items = self._metrics.totals()
        items["queue_evicted"] = queue.evicted
        items["queue_expired"] = queue.expired
        items["in_flight"] = queue.depth + sum(len(batch) for batch in self._inflight)
        manager = self._ingest_manager
        return {
            "items": items,
            "queue": queue.as_dict(),
            "workers": self._wanted_workers,
            # The producer's side: `accepted` is what got in, `ingest["frames_read"]` what was
            # offered, and while the camera set is unchanged the gap is backpressure.
            #
            # **They do not share a lifetime.** `IngestSummary` sums the actors that exist
            # *now*, so a drain or a remove takes that camera's `frames_read` with it while
            # `accepted` keeps every frame ever admitted -- subtracting across a drain reads as
            # negative backpressure. Per camera the honest pairing is the two
            # `*_dropped_total{camera}` counters, cumulative on both sides.
            "ingest": dict(_NO_INGEST) if manager is None else manager.summary().as_dict(),
        }
