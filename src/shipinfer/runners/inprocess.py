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
from dataclasses import dataclass
from typing import Any, ClassVar

from shipinfer.core.errors import (
    ConfigurationError,
    InferenceError,
    RequestCancelledError,
    ServerStateError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.request import InferenceRequest, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Device
from shipinfer.runners.base import Runner
from shipinfer.runners.registry import RUNNERS
from shipinfer.scheduling.queues import QUEUES, BatchWindow, RequestQueue
from shipinfer.scheduling.work import WorkItem
from shipinfer.topology import Caps, ChainItem, ElementNode, ModelResolver, Topology

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


@dataclass(slots=True)
class _Counts:
    """The runner's counters, incremented under one lock.

    A plain ``+= 1`` from thirty-two worker threads is a read-modify-write that can lose an
    update, and these numbers are what a test asserts and an operator reads. One lock for all
    of them costs a fraction of what the queue's own lock already costs per item.
    """

    accepted: int = 0
    walked: int = 0
    failed: int = 0
    expired: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "walked": self.walked,
            "failed": self.failed,
            "expired": self.expired,
        }


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

    Raises:
        ConfigurationError: ``workers`` below one. A runner with no workers accepts items and
            walks none of them, which looks exactly like a hung chain.
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
        #: Which predecessor donates payload and caps at a fan-in, per node. Pure topology
        #: data, so it is resolved here — once per runner, not once per frame — and is
        #: available to `_inbound` before `start()`.
        self._donors = _donors(topology)
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._counts = _Counts()
        self._counts_lock = threading.Lock()

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

    # -- lifecycle ---------------------------------------------------------------------

    def _do_start(self) -> None:
        """Open every element in topological order, then start the workers.

        Elements first and workers last: a worker that woke before the chain was open would
        meet :class:`~shipinfer.core.errors.ServerStateError` from
        :meth:`~shipinfer.topology.base.Element.process` on a real item, which is a refusal
        the runner caused and the operator would have to diagnose.

        Opening walks forwards and unwinds backwards, so an element that fails at position
        five leaves one through four *closed* rather than holding a decoder thread, a socket
        or a CUDA context on a shared box. :meth:`Runner.start` also calls
        :meth:`_do_stop` on the way out; both paths are safe because ``Element.close`` is
        idempotent.

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
        opened: list[ElementNode] = []
        for node in self._topology.nodes:
            try:
                node.element.open(context)
            except BaseException:
                _LOG.error(
                    "element %r failed to open; closing the %d already open",
                    node.name,
                    len(opened),
                )
                for previous in reversed(opened):
                    with contextlib.suppress(Exception):
                        previous.element.close()
                raise
            opened.append(node)

        self._stopping.clear()
        for index in range(self._wanted_workers):
            thread = threading.Thread(
                target=self._work,
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

        for node in reversed(self._topology.nodes):
            try:
                node.element.close()
            except Exception:
                _LOG.exception("element %r failed to close cleanly", node.name)

        _LOG.info(
            "runner %s stopped: %d item(s) failed in the queue, %d walked, %d failed",
            self.name,
            lost,
            self._counts.walked,
            self._counts.failed,
        )

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
                untouched (ADR-005).
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
        self._queue.put(work)
        with self._counts_lock:
            self._counts.accepted += 1
        return work.future

    # -- the worker loop ---------------------------------------------------------------

    def _work(self) -> None:
        """Drain the admission queue and walk one item at a time.

        ``get_batch`` returning an empty list means the queue closed, which is how a worker
        learns to exit without a separate sentinel. The batch here is a *wakeup* batch
        (``pipeline.frames_per_wakeup``), not an inference batch: batching for a GPU is the
        engine's job, and a worker walks its items one after another.
        """
        while not self._stopping.is_set():
            items = self._queue.get_batch(self._window, poll_s=0.05)
            if not items:
                if self._queue.is_closed:
                    return
                continue
            for work in items:
                try:
                    self._walk(work)
                except Exception as exc:  # pragma: no cover - the walk handles its own
                    # Reaching here means something outside an element failed. A worker that
                    # dies stops serving every camera on this shard, so the loop survives it
                    # — and the item's future is resolved, because a frame that vanishes with
                    # no typed outcome is the failure ADR-005 exists to prevent.
                    self._count_failure()
                    _LOG.exception("runner failed on %s", work.request.context.key)
                    self._fail(work, InferenceError(f"the runner failed: {exc}"))

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
        if work.request.is_expired():
            # The queue drops expired items on the way out too (`drop_expired`); this catches
            # one that expired while a worker was busy with the frame before it. Spending a
            # GPU on a frame that is already too late to act on is pure waste.
            with self._counts_lock:
                self._counts.expired += 1
            self._fail(
                work, RequestCancelledError("the item's deadline passed before the walk")
            )
            return

        produced: dict[str, ChainItem | None] = {}
        last = item
        for node in self._topology.nodes:
            incoming = item if node.is_root else self._inbound(node, produced)
            if incoming is None:
                continue
            if not node.admits(incoming):
                # Skip and continue: the item is not dropped and the walk does not stop, so
                # this element's successors receive *its* inbound item unchanged. The loader
                # negotiated that bypass pair, which is why it is safe to do here.
                produced[node.name] = incoming
                continue
            try:
                result = node.element.process(incoming)
            except Exception as exc:
                self._count_failure()
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

        with self._counts_lock:
            self._counts.walked += 1
        if work.future.set_running_or_notify_cancel():
            work.future.set_result(last)

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
        * **payload and caps come from one donor**, the first predecessor whose edge carries
          the cap this element prefers (:func:`_donors`). A payload is a frame handle or a
          tensor, and half of one plus half of another is not a thing; picking by the
          negotiated cap is what makes the choice the loader's rather than the runner's.
        * **a skipped predecessor contributes its own inbound item**, because that is what
          skip-and-continue means — the walk stored it under that predecessor's name.
        """
        contributors = [
            contributed
            for name in node.inputs
            if (contributed := produced.get(name)) is not None
        ]
        if not contributors:
            return None
        donor_name = self._donors.get(node.name)
        donor = produced.get(donor_name) if donor_name is not None else None
        if donor is None:
            donor = contributors[0]
        if len(contributors) == 1:
            # The common case — a straight line — allocates nothing.
            return donor

        meta: dict[str, Any] = {}
        for contributed in contributors:
            for key, value in contributed.meta.items():
                meta.setdefault(key, value)
        return ChainItem(
            context=donor.context, caps=donor.caps, payload=donor.payload, meta=meta
        )

    # -- failure and observability -----------------------------------------------------

    def _count_failure(self) -> None:
        with self._counts_lock:
            self._counts.failed += 1

    def _fail(self, work: WorkItem, error: BaseException) -> None:
        """Resolve this item's future with a typed failure, unless the caller gave up."""
        work.fail(error)

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
            "items": self._counts.as_dict(),
            "queue": self._queue.stats().as_dict(),
            "workers": self._wanted_workers,
        }


def _donors(topology: Topology) -> dict[str, str]:
    """``node -> the predecessor whose payload and caps a fan-in adopts``.

    Resolved once per runner from topology data alone, so the per-frame merge is a dict
    lookup rather than a search over edges.

    The donor is the first predecessor, **in declaration order**, whose edge carries the cap
    this element most prefers — ``Element.accepts`` is a preference list and
    :func:`~shipinfer.topology.caps.negotiate` already treats it as one, so the runner reads
    the same order the loader did. A node whose predecessors all carry the same cap therefore
    adopts the first one declared, which is what a reader of the chain file would expect.
    """
    caps_by_edge: dict[tuple[str, str], Caps] = {
        (edge.producer, edge.consumer): edge.caps for edge in topology.edges
    }
    donors: dict[str, str] = {}
    for node in topology.nodes:
        if not node.inputs:
            continue
        donors[node.name] = _donor_for(node, caps_by_edge)
    return donors


def _donor_for(node: ElementNode, caps_by_edge: Mapping[tuple[str, str], Caps]) -> str:
    """One node's donor. Falls back to the first predecessor when no edge cap is known."""
    for declared in node.element.input_caps:
        for name in node.inputs:
            caps = caps_by_edge.get((name, node.name))
            if caps is not None and caps.matches(declared):
                return name
    return node.inputs[0]
