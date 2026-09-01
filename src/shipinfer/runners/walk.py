"""Walking one frame through the chain — the bottom half of a runner.

The seam this file *is*: above it a runner deals in work items, queues and threads; here
everything is a :class:`ChainItem` and an element. A runner owns the queue and the workers and
calls :meth:`ChainWalk.run` once per item; nothing in this file knows a thread exists.

Four rules, all the loader's and none re-decided per runner:

* an element whose ``when:`` rejects the item is **skipped**, and its successors receive the
  item it was given, unchanged;
* an element that returns ``None`` **consumed** the item — a sink, or a filter — so its
  successors receive nothing from it, which is not the same as a skip;
* a fan-in merges its predecessors' contributions (:meth:`ChainWalk.inbound`).
* an element that raises costs this item and nothing else.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from concurrent.futures import InvalidStateError
from dataclasses import dataclass
from typing import Any

from shipinfer.core.errors import (
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    RequestTimeoutError,
    RingClosedError,
    ShipInferError,
    WireRefusedError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.runners.metrics import RunnerMetrics
from shipinfer.scheduling.work import WorkItem
from shipinfer.topology import Caps, ChainItem, ElementNode, RowIndexed, Topology

_LOG = get_logger(__name__)


@dataclass(slots=True)
class ChainWork(WorkItem):
    """A queued :class:`WorkItem` carrying the chain item it will start from.

    A subclass rather than a second queue type: it *is* a work item — the queue's fairness,
    priority and expiry all read the request it wraps — and a named field is greppable where a
    magic ``request.parameters`` key is not. Optional only because a dataclass field after a
    defaulted one must have a default; :meth:`ChainWalk.run` refuses ``None``.
    """

    item: ChainItem | None = None


class ChainWalk:
    """Runs one :class:`ChainItem` through every element of a topology that admits it.

    Args:
        topology: the loaded chain. Its node order is topological, so walking it in order is
            all the sequencing this needs.
        metrics: the runner's counters, labelled by camera.
        edge_caps: ``(producer, consumer) -> Caps``, from ``Topology.edges``. Read only by
            :meth:`_substitute_donor`, to pick a fan-in donor whose cap matches the one the
            loader negotiated for the missing donor's edge.
    """

    def __init__(
        self,
        topology: Topology,
        metrics: RunnerMetrics,
        edge_caps: Mapping[tuple[str | None, str], Caps],
    ) -> None:
        self._topology = topology
        self._metrics = metrics
        self._edge_caps = edge_caps

    # -- the walk -------------------------------------------------------------------------

    def run(self, work: WorkItem) -> None:
        """Walk one item, then finish or fail its future exactly once."""
        item = work.item if isinstance(work, ChainWork) else None
        if item is None:
            raise InferenceError(
                f"the work item for {work.request.context.key} carries no chain item; "
                "items enter a runner through submit()"
            )
        if self.expired(work, "before the walk"):
            return

        produced: dict[str, ChainItem | None] = {}
        last = item
        for node in self._topology.nodes:
            try:
                incoming = item if node.is_root else self.inbound(node, produced)
            except InferenceError as exc:
                self._blame(
                    work,
                    exc,
                    f"fan-in {node.name} could not be merged for "
                    f"{work.request.context.key}",
                    with_traceback=False,
                )
                return
            if incoming is None:
                continue
            if not node.admits(incoming):
                produced[node.name] = incoming
                continue
            # Re-checked only in front of elements that can *wait*: those submit to the pool
            # and sleep, so a long chain can spend several stage timeouts between checks, and
            # a frame already too late to act on must not be given another GPU.
            if node.element.needs_model and self.expired(work, f"before element {node.name!r}"):
                return
            try:
                result = node.element.process(incoming)
            except Exception as exc:
                self._blame(
                    work,
                    exc,
                    f"element {node.name!r} failed on {incoming.key}",
                    with_traceback=True,
                )
                return
            produced[node.name] = result
            if result is not None:
                last = result

        self._metrics.items_walked.inc(camera=work.request.context.camera_id)
        self.finish(work, last)

    def inbound(
        self, node: ElementNode, produced: Mapping[str, ChainItem | None]
    ) -> ChainItem | None:
        """What a node receives: its donor's payload, carrying every branch's meta."""
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

        return ChainItem(
            context=donor.context,
            caps=donor.caps,
            payload=donor.payload,
            meta=self._merge_meta(node, contributors),
        )

    # -- outcomes -------------------------------------------------------------------------

    def expired(self, work: WorkItem, where: str) -> bool:
        """Count and fail an item past its deadline. One helper so the counter and the typed
        error stay identical wherever the walk asks."""
        if not work.request.is_expired():
            return False
        self._metrics.items_expired.inc(camera=work.request.context.camera_id)
        self.fail(work, RequestCancelledError(f"the item's deadline passed {where}"))
        return True

    def finish(self, work: WorkItem, item: ChainItem) -> None:
        """Publish the result, unless a stop already finished the future."""
        if work.future.done():
            return
        with contextlib.suppress(RuntimeError, InvalidStateError):
            if work.future.set_running_or_notify_cancel():
                work.future.set_result(item)

    def fail(self, work: WorkItem, error: BaseException) -> None:
        """Fail the future, unless a stop already finished it.

        The ``done()`` guard is what makes shutdown safe to race: a cancelled future is
        *finished*, and ``set_running_or_notify_cancel`` raises on a finished future rather
        than answering ``False``.
        """
        if work.future.done():
            return
        with contextlib.suppress(RuntimeError, InvalidStateError):
            work.fail(error)

    def count_failure(self, work: WorkItem, error: BaseException | None = None) -> None:
        """Route a failure to the counter an operator acts on.

        Three destinations because there are three responses: backpressure means shed load or
        add lanes, a stage timeout means the model is saturated, anything else means read a
        stack trace. ``RingClosedError``/``WireRefusedError`` are ``QueueFullError`` subclasses
        for phase D's spill loop but count as ``failed`` here — a dead peer is "file a ticket",
        not "add capacity".
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

    @staticmethod
    def typed(error: Exception, context: str) -> BaseException:
        """One of ours travels untouched; a foreign exception is wrapped and named.

        Re-wrapping ours would turn backpressure, a stage timeout and a bug into one
        ``InferenceError``, and the submitter is meant to branch on them.
        """
        return (
            error
            if isinstance(error, ShipInferError)
            else InferenceError(f"{context}: {error}")
        )

    # doc: long why `with_traceback` cannot be inferred from the error is the whole point
    def _blame(
        self, work: WorkItem, error: Exception, context: str, *, with_traceback: bool
    ) -> None:
        """Count, log with the ``(camera, frame)`` tag, and fail — the one exit for a bad item.

        Walking on would produce a plausible event with no boxes in it, which is worse than a
        reported failure.

        Args:
            with_traceback: whether to log ``exc_info``. **Not** inferable from the error:
                :meth:`typed` *builds* a fresh error rather than raising, so nothing chains a
                traceback onto what the submitter receives, and if this frame does not carry it
                the frames are gone from every channel. An element failure asks for it (the
                bug is inside somebody else's ``process``); a fan-in failure does not, because
                the walk raised that :class:`InferenceError` itself and its message is the
                whole story.
        """
        self.count_failure(work, error)
        _LOG.error(
            "%s: %s",
            context,
            error,
            exc_info=with_traceback,
            extra=log_context(
                camera_id=work.request.context.camera_id,
                frame_id=work.request.context.frame_id,
            ),
        )
        self.fail(work, self.typed(error, context))

    # -- fan-in ---------------------------------------------------------------------------

    def _merge_meta(
        self, node: ElementNode, contributors: list[tuple[str, ChainItem]]
    ) -> dict[str, Any]:
        """First writer wins, except that two :class:`RowIndexed` scatter-backs union."""
        merged: dict[str, Any] = {}
        for _, contributed in contributors:
            for key, value in contributed.meta.items():
                if key not in merged:
                    merged[key] = value
                    continue
                held = merged[key]
                if held is value:
                    # The same object down both branches, written before the fork: merging it
                    # into itself cannot be refused, so skipping only skips the copy.
                    continue
                if not (isinstance(held, RowIndexed) and isinstance(value, RowIndexed)):
                    # Not a declared union to take. Two plain mappings land here on purpose: a
                    # model's raw `{output name: Tensor}` response is a mapping, not an
                    # attribution.
                    continue
                union = RowIndexed(held)
                for entry, row in value.items():
                    if entry in union and union[entry] is not row:
                        raise InferenceError(self._collision(node, key, entry, contributors))
                    union[entry] = row
                merged[key] = union
        return merged

    def _collision(
        self,
        node: ElementNode,
        key: str,
        entry: Any,
        contributors: list[tuple[str, ChainItem]],
    ) -> str:
        """Name *every* branch that claimed the row, not the two the merge happened to hold.

        On a three-way rejoin the pair being merged need not be the pair at fault, and an
        operator sent to the wrong two slots is worse served than one sent to none.
        """
        claimants = [
            name
            for name, contributed in contributors
            if isinstance(inbound := contributed.meta.get(key), RowIndexed) and entry in inbound
        ]
        return (
            f"fan-in {node.name!r}: branches {', '.join(repr(name) for name in claimants)} "
            f"each filed a different value for meta[{key!r}][{entry!r}]. Rejoining branches "
            "merge a row-indexed scatter-back rather than one replacing the other, and two "
            "elements covering detection row "
            f"{entry!r} means the chain file asked both of them for it -- check their "
            "`params: classes:` do not overlap"
        )

    def _substitute_donor(
        self, node: ElementNode, contributors: list[tuple[str, ChainItem]]
    ) -> ChainItem:
        """A donor for a fan-in whose nominated donor produced nothing.

        It must donate under the cap the loader negotiated for the missing donor's edge:
        handing the payload on under another cap would relabel it.
        """
        wanted = self._edge_caps.get((node.donor, node.name)) if node.donor else None
        if wanted is None:
            # No nominated donor, or no negotiated edge to read a cap from — a root, or a node
            # the loader wired without one. Nothing to be inconsistent with.
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
