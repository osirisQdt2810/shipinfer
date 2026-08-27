"""The single-process runner: a fair lane per camera, N workers, one frame each.

This is arch.md §5 with the process boundary removed — ② the fair lanes and ③ the pipeline
workers, over ④ whatever model pool the runner was handed. One worker takes one item and
walks it through the whole chain, sleeping at every pool wait; concurrency comes from having
several workers and from each *model* batching across every frame in flight. It is the
runner for dev, tests and a few cameras, and it is the shard-side executor the ``fleet``
runner will drive in phase A2f — which is why the walk lives here and not in a launcher.

Four choices are worth defending, because each one had an alternative that looks simpler.

**The queue is the fair one, and it stays typed on** :class:`~shipinfer.scheduling.work.
WorkItem`. Admission into the chain is the same problem the engine's per-model queues solve
— bounded, bucketed by ``camera_id``, drained round-robin, honest overflow — and it is
already solved (ADR-005). So an item submitted here is *wrapped* in a ``WorkItem`` exactly as
:class:`~shipinfer.pipeline.sink.QueueFrameSink` wraps a decoded frame, rather than the queue
being generalised over a second element type. The three fields the queue reads —
``request.context.camera_id`` for the lane, ``request.priority`` for the band,
``request.deadline_ns`` for expiry — are request fields, so the wrap is what buys a
per-camera fair lane for free. The :class:`~shipinfer.topology.base.ChainItem` rides on the
work item and is taken off it at the top of :meth:`InprocessRunner._walk`, the way
``FrameState`` is derived at the top of the pipeline's per-frame work.

**One worker walks the topological order, skipping what does not admit the item.** Not a
thread per element: a chain of nine elements as nine queues would put eight hand-offs and
eight context switches in the path of every frame, and the branch conditions would have to be
re-decided at each one. The loader has already proved the order legal (ADR-017), and
:meth:`~shipinfer.topology.chain.ElementNode.admits` already answers "should this element see
this item" identically for all three runners.

**Fan-in is merged deterministically or not at all.** Two branches rejoin at ``track``, and
"whichever finished last wins" is not an answer when the two carry different metadata. See
:meth:`InprocessRunner._inbound` for the rule.

**An element that raises loses one item, not the worker.** A worker thread that dies stops
serving every camera on the shard, so the loop survives one bad frame and the item's future
carries the typed failure — the same shape ``pipeline/runner.py`` uses, and for the same
reason.

**A batch is as wide as the worker pool, and that is the known ceiling.** The walk is
synchronous: a worker submits to a model and sleeps on the future, so the most frames that
can be sitting in one model's queue at once is the number of workers. With the default
``pipeline.workers = 4`` a shard therefore offers each model **batches of at most 4**, whatever
``max_batch_size`` the model declares. That is arch.md §5⑤'s asynchronous walk, deferred: it
is a throughput ceiling and not a correctness bug, the workaround is more workers, and the
number phase B's bench has to produce is the achieved batch size per model per shard against
this bound.

**On the duplication with** :mod:`shipinfer.pipeline.runner`. That module is the precedent
this one follows: the queue-and-workers shape, the expiry re-check, the per-frame error
handling and the shared shutdown deadline are all its ideas. The two coexist deliberately
until phase C, when the perception stages become chain elements and ``pipeline/graph/``'s
hard-coded DAG is superseded by a topology; whichever survives, it is one file then. Until
then a fix that applies to both belongs in both, and this paragraph is the reminder.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Mapping
from concurrent.futures import InvalidStateError
from dataclasses import dataclass
from typing import Any, ClassVar

from shipinfer.core.errors import (
    ConfigurationError,
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    RequestTimeoutError,
    RingClosedError,
    ServerStateError,
    ShipInferError,
    WireRefusedError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.request import InferenceRequest, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Device
from shipinfer.runners.base import Runner
from shipinfer.runners.metrics import RunnerMetrics
from shipinfer.runners.registry import RUNNERS
from shipinfer.scheduling.queues import QUEUES, BatchWindow, RequestQueue
from shipinfer.scheduling.work import WorkItem
from shipinfer.topology import (
    MODEL_KINDS,
    ChainItem,
    ElementNode,
    ModelResolver,
    Topology,
)

__all__ = ["InprocessRunner"]

_LOG = get_logger("runners.inprocess")


@dataclass(slots=True)
class _ChainWork(WorkItem):
    """A queued :class:`WorkItem` carrying the chain item it will start from.

    A subclass rather than a second queue type, and rather than smuggling the item through
    ``request.parameters``: it *is* a work item — the queue's fairness, priority and expiry
    all read the request it wraps — and a named field is greppable where a magic parameter
    key is not.

    Optional only because ``WorkItem.enqueued_ns`` has a default and a dataclass field after
    a defaulted one must have one too; :meth:`InprocessRunner._walk` refuses a ``None``
    rather than walking an item that does not exist.
    """

    item: ChainItem | None = None


@RUNNERS.register("inprocess", "single")
class InprocessRunner(Runner):
    """Runs the whole chain in this process, on a pool of worker threads.

    Args:
        topology: the validated chain.
        settings: deployment settings; ``pipeline.workers``, ``pipeline.queue_type``,
            ``pipeline.queue_capacity``, ``pipeline.overflow_policy`` and
            ``pipeline.frames_per_wakeup`` are the ones read here.
        shard_id: passed to every element (arch.md §2).
        device: the GPU this runner owns, or ``None``.
        models: the model pool for ``pool`` elements.
        queue: override the admission queue. A test injects a tiny one to provoke overflow;
            production takes the configured
            :class:`~shipinfer.scheduling.queues.fair.FairPriorityQueue`. An injected queue
            is never replaced — see :meth:`_do_start` for what that means for a restart.
        workers: override the worker count. ``None`` takes ``pipeline.workers``.
        metrics: share a :class:`~shipinfer.runners.metrics.RunnerMetrics` with the rest of
            the process. ``None`` mints a private registry, which is what a test wants; a
            shard passes the one its exporter scrapes.

    Raises:
        ConfigurationError: ``workers`` below one. A runner with no workers accepts items and
            walks none of them, which looks exactly like a hung chain.

    **What this runner does not honour yet: ``per:`` and ``scope:``.** One element instance is
    shared by every worker, so with ``workers > 1`` two frames of the same camera can be inside
    a ``per: camera`` element at the same time, and that element's per-camera ordering can
    invert — frame 8 finishing before frame 7 because a different worker got to it first. A
    ``scope: global`` element is likewise just an ordinary element here. Nothing stateful ships
    today (the mocks hold no state and a ``pool`` element holds only a model handle), so this
    is a promise not yet kept rather than a live defect; it is resolved in phase C, either with
    a per-camera element instance or with a camera-keyed lock around a ``per: camera`` element.
    Until then, a chain with a stateful element runs it correctly at ``workers=1``.
    """

    name: ClassVar[str] = "inprocess"

    def __init__(
        self,
        topology: Topology,
        settings: ServerSettings | None = None,
        *,
        shard_id: int = 0,
        device: Device | None = None,
        models: ModelResolver | None = None,
        queue: RequestQueue | None = None,
        workers: int | None = None,
        metrics: RunnerMetrics | None = None,
    ) -> None:
        super().__init__(topology, settings, shard_id=shard_id, device=device, models=models)
        pipeline = self._settings.pipeline
        if workers is not None and workers < 1:
            raise ConfigurationError(
                f"runner workers must be >= 1, got {workers}; a runner with no workers "
                "accepts items and walks none of them"
            )
        self._wanted_workers = pipeline.workers if workers is None else workers
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
        #: What each worker still owes an answer for, one slot per worker, indexed by the
        #: worker's own number. Written only by the worker that owns the slot and read only by
        #: `_do_stop` after the join deadline, so the hot path takes no lock — the same
        #: single-writer discipline `pipeline/runner.py` gets from `_awaiting` being keyed by
        #: the tag. It exists so that a worker abandoned at the deadline does not take its
        #: items' futures with it: an unresolved future is exactly the frame that vanishes
        #: with no typed outcome that ADR-005 exists to prevent, and `base.py`'s `submit`
        #: promises there is always one.
        #:
        #: **One list per start cycle, handed to the workers as an argument.** This attribute
        #: is only where `_do_stop` finds the *current* cycle's list; a worker never reads it.
        #: It used to, and an abandoned worker then wrote its `()` into whatever list the
        #: attribute pointed at *now* — so abandon, restart, abandon cleared a live worker's
        #: slot in the new cycle and the second shutdown had nothing left to fail. Binding the
        #: list at thread start is what makes a slot belong to one cycle.
        #:
        #: A **tuple of the undelivered remainder**, not the one item being walked. A worker
        #: takes a whole wake-up batch off the queue (`pipeline.frames_per_wakeup`), and the
        #: items behind the current one are no longer in the queue either — so with a single
        #: slot, `frames_per_wakeup: 4` and a worker wedged on item 0, closing the queue would
        #: resolve nothing and three producers would wait forever on futures nobody owns.
        self._inflight: list[tuple[WorkItem, ...]] = [()] * self._wanted_workers
        #: The cap the loader negotiated per edge, read by :meth:`_inbound` when a fan-in has
        #: to substitute for a donor that produced nothing. Snapshotted here because a
        #: topology is immutable once built, and the alternative is a dict comprehension per
        #: fan-in per frame.
        self._edge_caps = {(edge.producer, edge.consumer): edge.caps for edge in topology.edges}

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

    # -- lifecycle ---------------------------------------------------------------------

    def _do_start(self) -> None:
        """Open every element in topological order, then start the workers.

        Elements first and workers last: a worker that woke before the chain was open would
        meet :class:`~shipinfer.core.errors.ServerStateError` from
        :meth:`~shipinfer.topology.base.Element.process` on a real item, which is a refusal
        the runner caused and the operator would have to diagnose.

        An element that fails at position five must leave one through four *closed* rather
        than holding a decoder thread, a socket or a CUDA context on a shared box — and that
        unwind is :meth:`Runner.start`'s, which calls :meth:`_do_stop` on the way out for
        exactly this. There is deliberately no second unwind loop here: two paths that close
        the same elements in the same order are one path that can be fixed in one place and
        one that cannot, and ``Element.close`` is idempotent and a no-op on an element that
        never opened, so the shared path covers the partial case exactly.

        Raises:
            ServerStateError: the runner was given a queue that is already closed. Only
                reachable for an *injected* queue: a queue built here is rebuilt on a
                restart, but replacing the caller's object would silently ignore the capacity
                and overflow policy they chose.
            ShipInferError: whatever an element raises from ``open()`` — a model that is not
                in the pool, a camera that will not open.
        """
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
            "runner %s ready on shard %d: %s | %d worker(s) | queue=%s(%d)",
            self.name,
            self._shard_id,
            " -> ".join(node.name for node in self._topology),
            self._wanted_workers,
            self._queue.name,
            self._queue.capacity,
        )

    def _do_stop(self, timeout_s: float) -> None:
        """Close the queue, join the workers on one shared deadline, close the chain.

        The order is the reverse of :meth:`_do_start` and each step earns its place: closing
        the queue *fails* everything still waiting with a typed error rather than dropping it
        silently, and it is also how a worker blocked in ``get_batch`` learns to exit without
        a second sentinel. One deadline for every worker, not one each — everybody was
        signalled at the same instant, so a worker still running at the deadline is genuinely
        stuck and thirty-two consecutive waits would only delay the shutdown.

        Elements are closed last and in reverse topological order, each one guarded: a sink
        that fails to flush must not leave the decoder upstream of it open.

        The queue is left alone when no worker was ever started, which is the unwind path of
        a failed :meth:`Runner.start`. Nothing can have been submitted then — ``submit``
        refuses before ``start`` returns — so closing it would buy nothing and would poison
        an injected queue for the restart that follows the fix.

        **An abandoned worker's items are failed, not forgotten.** Closing the queue resolves
        what is still *queued*; the wake-up batch a worker took off it is not in the queue any
        more, and a worker still stuck at the deadline will never resolve any of it — not the
        item it is wedged inside, and not the ``frames_per_wakeup - 1`` behind that one.
        Leaving them is the one case where this runner would break the promise ``submit``
        makes — a future that never completes, on a producer that is waiting for it — so every
        in-flight slot is drained here and each item failed with the same typed cancellation
        the queue uses. The race with a worker that finishes a microsecond later is benign
        because :meth:`_fail` and :meth:`_finish` both refuse a future that is already
        resolved, whichever of the two got there first.

        The signal and the queue are **this cycle's** too, and that is load-bearing rather
        than tidy. Setting an event that no later start will clear, and closing a queue that no
        later start will hand out again, is what makes a worker abandoned here *terminal*: it
        wakes from whatever wedged it, sees a set event and a closed queue, and exits. While
        those two came off ``self``, a restart handed the stale thread a live event and the new
        cycle's queue, and it went back to work — taking cycle two's items and publishing them
        into cycle one's slot list, which no shutdown drains. Every one of those futures was
        lost after ``stop()`` had returned.

        The slots drained are this cycle's for the same reason: :attr:`_inflight` holds the
        list handed to the workers by the matching :meth:`_do_start`, a worker abandoned here
        goes on writing into that same list, and the next start publishes a different one.
        That list is also *replaced* here rather than only drained, because "goes on writing
        into it" is not hypothetical: an abandoned worker's ``finally`` republishes the
        remainder of its wake-up batch the moment it wakes, and while :attr:`_inflight` still
        pointed at that object ``stats()["items"]["in_flight"]`` came back up after the stop
        and stayed up forever — a stopped runner reporting work in flight that no shutdown
        would ever resolve, because every one of those futures had already been failed here.
        """
        self._stopping.set()
        lost: int = 0
        if self._threads:
            lost = self._close_queue()
            deadline = time.monotonic() + timeout_s
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

    def _close_queue(self) -> int:
        """Close the admission queue and charge every item it failed. Returns how many.

        The close and the count are one call because they were two, and the counter is only
        true if nothing can close the queue here without it: ``close`` resolves what is still
        queued with a typed error, and an outcome with no counter is precisely the gap
        ``stats()`` exists to close — ``accepted`` outran the sum of every outcome by exactly
        the number of items a shutdown caught in the lane.

        **An injected queue that its owner closes is its owner's to count.** The runner counts
        the items *it* failed; a caller who passes a queue in keeps the right to close it (and
        :meth:`_do_start` refuses to replace it for exactly that reason), and that close
        happens where this runner cannot see it — no callback, no drained list, and the queue's
        own ``stats()`` does not separate "failed by close" from the rest. Attributing it from
        a stats delta would be a guess, and a guessed counter is worse than an absent one, so
        the honest split is: the runner owns the closes it performs, the queue's owner owns
        the closes they perform. It costs nothing in the shape this runner ships with, where
        the queue it built is the queue it closes.
        """
        drained = self._queue.close()
        for item in drained:
            self._metrics.items_queue_closed.inc(camera=item.request.context.camera_id)
        return len(drained)

    def _fail_in_flight(self, inflight: list[tuple[WorkItem, ...]]) -> int:
        """Fail everything the abandoned workers still owed an answer for. Returns how many.

        Each slot holds the item being walked *and* the rest of its wake-up batch, because
        neither is in the queue any more. An abandoned worker that finishes its current item a
        microsecond later is benign: :meth:`_fail` and :meth:`_finish` both refuse a future
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
                self._count_failure(work, error)
                self._fail(work, error)
        return stranded

    # -- submission --------------------------------------------------------------------

    def _do_submit(self, item: ChainItem) -> ResponseFuture:
        """Wrap the item in a work item and enqueue it.

        The request built here is a **carrier**, not a model call: it exists so the fair
        queue can read the camera id, the priority band and the deadline it already knows how
        to read. No model ever sees it — a ``pool`` element builds its own request from the
        item it is handed. Phase B's frame sink will fill ``inputs`` with the decoded frame
        and this shape is what it will fill.

        Raises:
            QueueFullError: this camera's lane is full and the policy rejects. Propagated
                untouched, and counted against **this** camera on the way past (ADR-005): the
                number worth paging on is which camera flooded, not the shard total. The
                counter is ``items_dropped``, which this is the only writer of — the item was
                refused at the door and never became one of ``accepted``, so it is not an
                outcome the ``stats()`` ledger has to name. A model queue refusing an item
                already inside the chain is ``items_backpressure``; see :meth:`_count_failure`.
            RequestCancelledError: the queue is closed — the runner is shutting down.
        """
        context = item.context
        request = InferenceRequest(
            model_name=self._topology.name or "chain",
            inputs={},
            context=context,
            deadline_ns=(
                context.captured_ns + self._deadline_ns
                if self._deadline_ns and context.captured_ns
                else 0
            ),
        )
        work = _ChainWork(request, ResponseFuture(request), item=item)
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

        ``get_batch`` returning an empty list means the queue closed, which is how a worker
        learns to exit without a separate sentinel. The batch here is a *wakeup* batch
        (``pipeline.frames_per_wakeup``), not an inference batch: batching for a GPU is the
        engine's job, and a worker walks its items one after another.

        **The stop signal is checked in front of every item, not only every wake-up.** A
        worker holds a whole ``frames_per_wakeup`` batch, so a check at the top of the outer
        loop alone let a worker abandoned at a shutdown deadline finish its *entire* batch when
        whatever wedged it finally let go — up to ``frames_per_wakeup - 1`` further items,
        walked through elements the runner had closed and emitted through a sink a restart had
        re-opened. Breaking here instead leaves the remainder in the slot, which is precisely
        where :meth:`_fail_in_flight` has already found it: those futures were failed at the
        stop, so the item that is skipped is one whose producer has its answer.

        **Every piece of per-cycle state is an argument, and none of it is read off**
        ``self``. A worker abandoned at a shutdown deadline outlives its cycle — it is parked
        inside ``element.process()``, not gone — so whatever it reads from the runner it reads
        from whichever cycle is current when it finally wakes. All three attributes had to
        move for that to be safe, and each one cost a different bug:

        * ``self._stopping`` was *cleared* by the next :meth:`_do_start`, so the stale
          thread's loop condition came back true;
        * ``self._queue`` was rebuilt by that same start, so the stale thread called
          ``get_batch`` on the **new** cycle's queue and took real work off it;
        * ``self._inflight`` was rebound, so what it published landed in the new cycle's slot
          list — under the *live* worker's index, at the same slot number.

        Together those made an abandoned worker a second, untracked consumer: it drained cycle
        two's items into cycle one's slot list, which no shutdown drains, and their futures
        were lost after ``stop()`` had returned. It also walked those items through elements
        cycle one had closed. Bound to its own cycle's event and queue, it instead finds an
        event set forever and a queue closed forever, and exits on its next turn.

        Args:
            slot: this worker's index into ``inflight``. Owned exclusively by this thread,
                which is what lets the publish be a plain store: :meth:`_do_stop` reads the
                slots only after its join deadline has passed.
            stopping: this cycle's stop signal. Set by the matching :meth:`_do_stop` and never
                cleared again.
            queue: this cycle's admission queue, closed by that same :meth:`_do_stop`.
            inflight: this cycle's slot list, drained by that same :meth:`_do_stop`.
        """
        while not stopping.is_set():
            items = queue.get_batch(self._window, poll_s=0.05)
            if not items:
                if queue.is_closed:
                    return
                continue
            # The whole batch is published *before* the first walk and narrowed to the
            # remainder in a `finally` after each item, so that from the drain to the last
            # resolution every item this worker owes an answer for has an owner a shutdown
            # can find. The window where a frame's future could go unresolved is the two
            # stores between `get_batch` returning and this line.
            #
            # The *remainder*, not the item being walked: the ones behind it left the queue in
            # the same drain, so a slot holding only the current item would strand them —
            # `frames_per_wakeup: 4` with a worker wedged on item 0 loses three futures. Both
            # slices are free at the default `frames_per_wakeup = 1`: `batch[1:]` on a
            # one-tuple is the empty-tuple singleton, so the hot path allocates one tuple per
            # wake-up and none per item.
            batch = tuple(items)
            inflight[slot] = batch
            for index, work in enumerate(batch):
                if stopping.is_set():
                    # Abandoned mid-batch. The remainder is left in the slot exactly as the
                    # previous iteration's `finally` published it, so `_do_stop`'s drain --
                    # which ran after the join deadline, before this thread woke -- has
                    # already failed every one of these futures. Walking them anyway would
                    # emit ghost events through a chain this cycle closed.
                    break
                try:
                    self._walk(work)
                except Exception as exc:
                    # Reaching here means something outside an element failed. A worker that
                    # dies stops serving every camera on this shard, so the loop survives it
                    # — and the item's future is resolved, because a frame that vanishes with
                    # no typed outcome is the failure ADR-005 exists to prevent.
                    self._count_failure(work, exc)
                    _LOG.exception("runner failed on %s", work.request.context.key)
                    self._fail(work, self._typed(exc, "the runner failed"))
                finally:
                    inflight[slot] = batch[index + 1 :]

    def _walk(self, work: WorkItem) -> None:
        """Walk one item through every element that admits it, in topological order.

        The :class:`ChainItem` is taken off the work item here, at the top, exactly where
        ``pipeline/runner.py`` derives its ``FrameState`` from the queued request: everything
        above this line deals in work items and queues, everything below it deals in chain
        items and elements.

        Three rules, all of them the loader's and none of them re-decided per runner:

        * an element whose ``when:`` rejects the item is **skipped**, and the item it was
          given is handed to its successors unchanged (``ElementNode.admits``);
        * an element that returns ``None`` **consumed** the item — a sink, or a filter — so
          its successors receive nothing from it, which is different from a skip;
        * a fan-in merges its predecessors' contributions by :meth:`_inbound`.

        An element that raises costs this item and nothing else: it is counted, logged with
        the ``(camera, frame)`` tag, and its future carries the typed failure. Walking on
        would produce a plausible event with no boxes in it, which is worse than a reported
        failure.

        **The failure the submitter sees is the element's own, whenever the element raised
        one of ours.** A :class:`~shipinfer.core.errors.QueueFullError` from a ``pool``
        element carries the depth and the capacity of the model queue that refused it, a
        :class:`~shipinfer.core.errors.RequestTimeoutError` says the model never answered, and
        a :class:`~shipinfer.core.errors.ValidationError` says the payload was wrong — three
        different events with three different responses (shed load, add capacity, fix the
        chain). This wrapped all three in ``InferenceError`` and flattened them into one, which
        also made ``pool.py``'s "propagated untouched" promise false. Only a failure that is
        *not* ours — the ordinary bug, ``RuntimeError`` and friends — is wrapped, and then the
        wrapper names the element and the tag because nothing else will.
        """
        # Taken off the work item, and checked rather than assumed: the queue's element type
        # is `WorkItem`, so anything that reached it without going through `submit` — a test
        # or a caller putting into an injected queue directly — gets a typed refusal here
        # instead of an `AttributeError` from inside a worker thread.
        item = work.item if isinstance(work, _ChainWork) else None
        if item is None:
            raise InferenceError(
                f"the work item for {work.request.context.key} carries no chain item; "
                "items enter a runner through submit()"
            )
        if self._expired(work, "before the walk"):
            return

        produced: dict[str, ChainItem | None] = {}
        last = item
        for node in self._topology.nodes:
            try:
                incoming = item if node.is_root else self._inbound(node, produced)
            except InferenceError as exc:
                # A fan-in the loader's donor rule cannot answer for this item. Counted and
                # failed like an element failure, because it is the same shape: one item, a
                # typed reason naming the node, and the walk stops rather than handing a
                # payload on under a cap nobody negotiated.
                self._count_failure(work, exc)
                _LOG.error(
                    "fan-in %s could not be merged for %s: %s",
                    node.name,
                    work.request.context.key,
                    exc,
                    extra=log_context(
                        camera_id=work.request.context.camera_id,
                        frame_id=work.request.context.frame_id,
                    ),
                )
                self._fail(work, exc)
                return
            if incoming is None:
                continue
            if not node.admits(incoming):
                # Skip and continue: the item is not dropped and the walk does not stop, so
                # this element's successors receive *its* inbound item unchanged. The loader
                # negotiated that bypass pair, which is why it is safe to do here.
                produced[node.name] = incoming
                continue
            if node.kind in MODEL_KINDS and self._expired(
                work, f"before element {node.name!r}"
            ):
                # Re-checked in front of every element that can *wait*. The four model kinds
                # are the ones that submit to the pool and sleep on the answer, so a nine-step
                # chain can spend several stage timeouts between the check at the top of the
                # walk and this element — and a frame that is already too late to act on must
                # not be given another GPU. The other four kinds are local work with no wait
                # in them, so checking in front of them would only cost a clock read.
                return
            try:
                result = node.element.process(incoming)
            except Exception as exc:
                self._count_failure(work, exc)
                _LOG.exception(
                    "element %s failed on %s",
                    node.name,
                    incoming.key,
                    extra=log_context(camera_id=incoming.key[0], frame_id=incoming.key[1]),
                )
                self._fail(
                    work, self._typed(exc, f"element {node.name!r} failed on {incoming.key}")
                )
                return
            produced[node.name] = result
            if result is not None:
                last = result

        self._metrics.items_walked.inc(camera=work.request.context.camera_id)
        self._finish(work, last)

    def _inbound(
        self, node: ElementNode, produced: Mapping[str, ChainItem | None]
    ) -> ChainItem | None:
        """The item a non-root element receives, merged from its predecessors.

        Returns ``None`` when no predecessor contributed anything — every one of them either
        consumed its item or never saw one — in which case this element does not run. That is
        not a failure: it is what a branch that ended in a sink looks like.

        The merge rule, and it is deliberately boring, because "whichever branch finished
        last wins" is not an answer when two branches carry different metadata:

        * **metadata is the union, in ``node.inputs`` order, first writer wins.** So a fan-in
          at ``track`` sees the ship branch's ``identities`` *and* the person branch's
          ``vectors``. First-writer-wins rather than last, so the resolution of a genuine
          collision is a property of the chain file's declaration order and does not change
          between runs.
        * **payload and caps come from one donor**, the predecessor the *loader* nominated
          (:attr:`~shipinfer.topology.chain.ElementNode.donor`, resolved from the negotiated
          edge caps). A payload is a frame handle or a tensor, and half of one plus half of
          another is not a thing; the choice belongs where ``admits`` belongs, so that
          ``inprocess`` and ``fleet`` cannot come to disagree about which branch donated the
          frame.
        * **a skipped predecessor contributes its own inbound item**, because that is what
          skip-and-continue means — the walk stored it under that predecessor's name.

        Raises:
            InferenceError: the nominated donor produced nothing and no other contributor
                donates under the same negotiated cap. See :meth:`_substitute_donor`.
        """
        contributors = [
            (name, contributed)
            for name in node.inputs
            if (contributed := produced.get(name)) is not None
        ]
        if not contributors:
            return None
        donor = produced.get(node.donor) if node.donor is not None else None
        if donor is None:
            donor = self._substitute_donor(node, contributors)
        if len(contributors) == 1:
            # The common case — a straight line — allocates nothing.
            return donor

        meta: dict[str, Any] = {}
        for _, contributed in contributors:
            for key, value in contributed.meta.items():
                meta.setdefault(key, value)
        return ChainItem(
            context=donor.context, caps=donor.caps, payload=donor.payload, meta=meta
        )

    def _substitute_donor(
        self, node: ElementNode, contributors: list[tuple[str, ChainItem]]
    ) -> ChainItem:
        """Who donates payload and caps when the nominated donor did not, or a typed refusal.

        The loader nominated one predecessor (:attr:`~shipinfer.topology.chain.ElementNode.
        donor`) by walking this element's ``accepts`` in order against the negotiated cap of
        each inbound edge. Reached only when that one contributed nothing — it consumed its
        item, or it never received one — so this is off the ordinary path by construction.

        A **substitute is only legal if it donates under the same negotiated cap**. That cap
        is not decoration: it is what the loader resolved this element's own ``produces: *@*``
        from, what it checked every bypass pair against, and what every element downstream
        was validated against. Handing the payload over under a *different* one relabels it exactly the way
        a concrete ``produces`` on a ``pool`` element used to — the item claims a format and a
        location it does not have, and the device-to-host download arch.md §8 exists to refuse
        becomes invisible. This used to take ``contributors[0]`` unconditionally, which is the
        one place an item could travel under a cap nobody negotiated.

        So: the donor, else the first contributor whose edge into ``node`` carries the donor's
        cap, else a typed failure for this item alone. A chain where that failure is reachable
        on every frame is a chain whose fan-in the loader cannot resolve, and phase B's
        load-time check is where it will be caught before a deploy rather than per frame.

        Args:
            node: the fan-in being merged.
            contributors: ``(predecessor name, its contribution)``, in ``node.inputs`` order.

        Raises:
            InferenceError: no contributor donates under the donor's negotiated cap. Names the
                node, the donor and both caps, because the fix is in the chain file.
        """
        wanted = self._edge_caps.get((node.donor, node.name)) if node.donor else None
        if wanted is None:
            # No nominated donor, or no negotiated edge to read a cap from — a root, or a node
            # the loader wired without one. There is nothing to be inconsistent with, so the
            # first contributor donates, as it always did.
            return contributors[0][1]
        for name, contributed in contributors:
            if self._edge_caps.get((name, node.name)) == wanted:
                return contributed
        offered = ", ".join(
            f"{name} [{self._edge_caps.get((name, node.name))}]" for name, _ in contributors
        )
        raise InferenceError(
            f"fan-in {node.name!r} has no donor for this item: {node.donor!r} produced "
            f"nothing and none of the predecessors that did ({offered}) donates under the "
            f"negotiated cap [{wanted}]. Handing the payload on under another cap would "
            f"relabel it; declare {node.name!r}'s inbound edges with one cap, or make "
            f"{node.donor!r} produce for every item it admits"
        )

    # -- failure and observability -----------------------------------------------------

    def _expired(self, work: WorkItem, where: str) -> bool:
        """Fail and count ``work`` if its deadline has passed. Returns whether it did.

        One helper and not two checks written out, because the counter and the typed error
        have to stay identical wherever the walk asks: an expiry that was counted at the top
        of the walk and merely logged in the middle would make ``stats()["items"]["expired"]``
        answer a different question depending on where the frame died.

        Args:
            work: the queued item.
            where: named in the message, so an operator reading the failure can tell the
                deadline the queue enforces from the one the walk re-checks, and can tell
                *which* element the frame had already reached.
        """
        if not work.request.is_expired():
            return False
        self._metrics.items_expired.inc(camera=work.request.context.camera_id)
        self._fail(work, RequestCancelledError(f"the item's deadline passed {where}"))
        return True

    @staticmethod
    def _typed(error: Exception, context: str) -> BaseException:
        """The failure this item's future should carry.

        One of ours travels untouched: the submitter is meant to branch on it, and re-wrapping
        turns backpressure, a stage timeout and a bug into the same ``InferenceError``. A
        foreign exception is wrapped, because ``RuntimeError('the detector fell over')`` on its
        own says neither which element fell over nor for which frame.

        Args:
            error: whatever was raised.
            context: the prefix for the wrapper — the element and the tag, or the walk.
        """
        return (
            error
            if isinstance(error, ShipInferError)
            else InferenceError(f"{context}: {error}")
        )

    def _count_failure(self, work: WorkItem, error: BaseException | None = None) -> None:
        """Charge one lost item to the counter its *kind* of failure belongs on, per camera.

        Three destinations, because an operator does three different things about them:
        backpressure means shed load or add lanes, a stage timeout means the model is
        saturated, and anything else means read a stack trace. Counting all of them as
        ``failed`` — which is what this did — hid the first two behind the third, so a shard
        under sustained overload looked like a shard full of bugs.

        Backpressure lands on ``items_backpressure``, not on ``items_dropped``. Every item
        this method sees was ``accepted`` — it is mid-walk, off the queue and inside the chain
        — whereas ``items_dropped`` is the submission that never got in at all. One counter
        for both is a defensible operational graph and an indefensible ledger: ``accepted``
        could then only be reconciled against the outcomes by subtracting the queue's own
        ``rejected`` first. See :meth:`_do_stats` for the identity that buys.

        The backpressure family is narrowed by hand, because inheritance does not carry the
        operator's response. ``core/errors/launch.py`` makes :class:`RingClosedError` and
        :class:`WireRefusedError` :class:`QueueFullError` subclasses so that phase D's
        dispatcher spill loop treats them as a refusal and tries the next candidate — the
        right call there, and the wrong label here: a ring whose peer died and a wire that
        cannot carry the payload are both "file a ticket", not "shed load or add capacity".
        They count as ``failed``. :class:`RingFullError` stays with ``items_backpressure``,
        which is exactly what it is.

        Args:
            work: the item being failed; its camera is the label.
            error: the failure. ``None``, or anything outside the backpressure and timeout
                families, counts as ``items_failed``.
        """
        camera = work.request.context.camera_id
        if isinstance(error, (RingClosedError, WireRefusedError)):
            self._metrics.items_failed.inc(camera=camera)
        elif isinstance(error, QueueFullError):
            self._metrics.items_backpressure.inc(camera=camera)
        elif isinstance(error, RequestTimeoutError):
            self._metrics.items_timed_out.inc(camera=camera)
        else:
            self._metrics.items_failed.inc(camera=camera)

    def _fail(self, work: WorkItem, error: BaseException) -> None:
        """Resolve this item's future with a typed failure, unless it is already resolved.

        The ``done()`` guard is what makes shutdown and a slow worker safe to race: a future
        :meth:`_do_stop` has already cancelled is *finished*, and
        ``Future.set_running_or_notify_cancel`` raises ``RuntimeError`` on a finished future
        rather than answering ``False``. Without the guard, an abandoned worker that reached
        the end of its walk after the runner stopped would die inside the failure handler,
        which is a stack trace about the thing that was supposed to be handled cleanly.
        """
        if work.future.done():
            return
        with contextlib.suppress(RuntimeError, InvalidStateError):
            work.fail(error)

    def _finish(self, work: WorkItem, item: ChainItem) -> None:
        """Resolve this item's future with the walked item, unless it is already resolved.

        Same guard as :meth:`_fail`, and the same race: the loser of a stop-versus-finish is
        whoever arrives second, and the caller sees exactly one outcome either way.
        """
        if work.future.done():
            return
        with contextlib.suppress(RuntimeError, InvalidStateError):
            if work.future.set_running_or_notify_cancel():
                work.future.set_result(item)

    def _do_health(self) -> dict[str, Any]:
        return {
            "queue": self._queue.stats().as_dict(),
            "workers": {
                "wanted": self._wanted_workers,
                "alive": sum(1 for thread in self._threads if thread.is_alive()),
            },
        }

    def _do_stats(self) -> dict[str, Any]:
        """Every outcome an accepted item can reach, including the queue's own.

        ``items`` used to be :meth:`~shipinfer.runners.metrics.RunnerMetrics.totals` alone, and
        that under-reported: an item the *queue* resolved — failed by ``close()`` at shutdown,
        dropped at the drain because its deadline had passed, evicted to make room under
        ``DROP_OLDEST`` — got a typed future and no counter, so ``accepted`` outran the sum of
        every outcome and the difference was indistinguishable from work still in flight. The
        three ``queue_*`` terms and ``in_flight`` are here so that it adds up::

            accepted == walked + failed + expired + timed_out + backpressure
                        + queue_closed + queue_evicted + queue_expired + in_flight

        ``dropped`` is deliberately *not* a term: it counts the submissions this runner's own
        lane refused at the door, and those were never ``accepted``, so adding it would make
        the right-hand side over-count by exactly ``queue["rejected"]``. It used to be one
        counter with the mid-walk refusals, which is why that correction term used to be
        written down here instead; :attr:`~shipinfer.runners.metrics.RunnerMetrics.
        items_backpressure` is the half that *was* accepted, and both remain per camera because
        the camera lost a frame to backpressure either way (ADR-005).

        That identity holds within one start cycle on a runner that abandoned no worker. The
        two ways it does not are both deliberate, and naming them is cheaper than a term that
        pretends they are not there:

        * **an abandoned worker is counted twice** when it finishes its walk after
          :meth:`_do_stop` failed its items: once as ``failed``, once as ``walked``. See
          :meth:`_fail_in_flight`.
        * **the ``queue_*`` terms read from the queue reset on a restart** and the runner's
          counters do not — a queue this runner built is rebuilt by :meth:`_do_start`. So
          ``queue_evicted`` and ``queue_expired`` describe the current cycle while ``accepted``
          describes every cycle. ``queue_closed`` is a runner counter for exactly that reason:
          it is the outcome a *previous* cycle's shutdown earned, and it has to survive.

        ``in_flight`` is the queue's depth plus what **the current cycle's** workers have
        published: a gauge, read without a lock, and wrong by at most one wake-up batch in
        either direction. Short, for the two stores between a drain returning and
        :meth:`_work` publishing the batch; long, for the ones between an item's future being
        resolved at the end of its walk and the ``finally`` that narrows the slot. Poll it to
        zero before reading the ledger as an identity, which is what
        ``tests/runners/test_inprocess.py::settled`` does.

        *Current cycle's* is the load-bearing word. A worker abandoned at a shutdown deadline
        keeps writing into the slot list it was handed, and after :meth:`_do_stop` that list is
        no longer the one this reads — so a stopped runner reports zero even while a stale
        thread is still republishing a batch whose futures the stop already failed.
        """
        queue = self._queue.stats()
        items = self._metrics.totals()
        items["queue_evicted"] = queue.evicted
        items["queue_expired"] = queue.expired
        items["in_flight"] = queue.depth + sum(len(batch) for batch in self._inflight)
        return {
            "items": items,
            "queue": queue.as_dict(),
            "workers": self._wanted_workers,
        }
