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
    ServerStateError,
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
        self._stopping = threading.Event()
        self._metrics = RunnerMetrics() if metrics is None else metrics
        #: What each worker still owes an answer for, one slot per worker, indexed by the
        #: worker's own number. Sized once, written only by the worker that owns the slot and
        #: read only by `_do_stop` after the join deadline, so the hot path takes no lock —
        #: the same single-writer discipline `pipeline/runner.py` gets from `_awaiting` being
        #: keyed by the tag. It exists so that a worker abandoned at the deadline does not
        #: take its items' futures with it: an unresolved future is exactly the frame that
        #: vanishes with no typed outcome that ADR-005 exists to prevent, and `base.py`'s
        #: `submit` promises there is always one.
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

        self._stopping.clear()
        self._inflight = [()] * self._wanted_workers
        for index in range(self._wanted_workers):
            thread = threading.Thread(
                target=self._work,
                args=(index,),
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
        """
        self._stopping.set()
        lost: int = 0
        if self._threads:
            lost = len(self._queue.close())
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
            stranded = self._fail_in_flight()
            if stranded:
                # Counted against `failed`, and an abandoned worker that later finishes its
                # walk counts the same item as `walked` too: a restart re-arms the slots, so
                # the two numbers can disagree by the number of items abandoned. That is the
                # honest shape — the item did fail for its producer *and* did eventually run
                # — and the alternative, suppressing the walk's own count from a thread the
                # runner has stopped tracking, would need the hot path to check a flag.
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

    def _fail_in_flight(self) -> int:
        """Fail everything the abandoned workers still owed an answer for. Returns how many.

        Each slot holds the item being walked *and* the rest of its wake-up batch, because
        neither is in the queue any more. An abandoned worker that finishes its current item a
        microsecond later is benign: :meth:`_fail` and :meth:`_finish` both refuse a future
        that is already resolved, whichever of the two arrived first.
        """
        stranded = 0
        for slot, batch in enumerate(self._inflight):
            self._inflight[slot] = ()
            for work in batch:
                stranded += 1
                self._metrics.items_failed.inc(camera=work.request.context.camera_id)
                self._fail(work, RequestCancelledError("the runner stopped"))
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
                number worth paging on is which camera flooded, not the shard total.
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

    def _work(self, slot: int) -> None:
        """Drain the admission queue and walk one item at a time.

        ``get_batch`` returning an empty list means the queue closed, which is how a worker
        learns to exit without a separate sentinel. The batch here is a *wakeup* batch
        (``pipeline.frames_per_wakeup``), not an inference batch: batching for a GPU is the
        engine's job, and a worker walks its items one after another.

        Args:
            slot: this worker's index into :attr:`_inflight`. Owned exclusively by this
                thread, which is what lets the publish be a plain store: :meth:`_do_stop`
                reads the slots only after its join deadline has passed.
        """
        while not self._stopping.is_set():
            items = self._queue.get_batch(self._window, poll_s=0.05)
            if not items:
                if self._queue.is_closed:
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
            self._inflight[slot] = batch
            for index, work in enumerate(batch):
                try:
                    self._walk(work)
                except Exception as exc:
                    # Reaching here means something outside an element failed. A worker that
                    # dies stops serving every camera on this shard, so the loop survives it
                    # — and the item's future is resolved, because a frame that vanishes with
                    # no typed outcome is the failure ADR-005 exists to prevent.
                    self._count_failure(work)
                    _LOG.exception("runner failed on %s", work.request.context.key)
                    self._fail(work, InferenceError(f"the runner failed: {exc}"))
                finally:
                    self._inflight[slot] = batch[index + 1 :]

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
                self._count_failure(work)
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
                self._count_failure(work)
                _LOG.exception(
                    "element %s failed on %s",
                    node.name,
                    incoming.key,
                    extra=log_context(camera_id=incoming.key[0], frame_id=incoming.key[1]),
                )
                self._fail(
                    work,
                    InferenceError(f"element {node.name!r} failed on {incoming.key}: {exc}"),
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

    def _count_failure(self, work: WorkItem) -> None:
        self._metrics.items_failed.inc(camera=work.request.context.camera_id)

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
        return {
            "items": self._metrics.totals(),
            "queue": self._queue.stats().as_dict(),
            "workers": self._wanted_workers,
        }
