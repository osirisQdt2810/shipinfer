"""A peer's model instance, seen through a shared ring — and the two threads that serve it.

THE SHAPE
---------
`service` (ledger T3) is the fleet plus a cross-process inference tier: every shard process
keeps serving its own GPU's instances of the crop-stage models and *also* offers them to its
peers. This module is the three pieces that make a peer's instance appear in a local model's
dispatcher as just another candidate:

* :class:`RemoteInstance` — a :class:`~shipinfer.scheduling.policies.base.Placeable`. Its
  ``depth`` and ``ewma_latency_us`` are read off the owner's ring header without a lock, the
  way a local instance's are read off its queue; ``enqueue`` writes the request into a slot.
  The dispatcher and the placement policy never learn it is remote — that is what the
  ``Placeable`` protocol is for.
* :class:`ResultReader` — one thread per process, reading every result ring this process
  owns, resolving the futures of the requests it sent, and failing them with
  :class:`~shipinfer.core.errors.PeerLostError` when an owner stops stamping its heartbeat.
* :class:`RingIngress` — one thread per inbound ring on the owner's side: take a slot, decode
  the request, hand it to the local model through the same ``infer()`` a local caller uses (so
  it goes through the owner's dispatcher and fair queue under *the submitter's camera id*), and
  write the response back into the submitter's result ring when the future settles. The slot
  stays claimed until then, so the request's tensors can view it instead of copying it — the
  Python plane's equivalent of the C++ ``keepalive``.

Which is the vLLM shape (`multiproc_executor.py`: a broadcast queue in, a per-worker response
queue out) with the router replaced by our own dispatcher and the load published as two numbers
in a header rather than a message.

WHAT DOES NOT CROSS
-------------------
Nothing device-side. The Python plane hands its models host tensors, so a remote submit is a
memcpy into the slot and the owner's backend stages host → device exactly as for a local
request (ADR-002: no cross-device access; the payload crosses through host memory). The tag
crosses in the request head and comes back in the response, so reassembly keys on it as if
the work had never left the process.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass

from shipinfer.core.errors import (
    PeerLostError,
    QueueFullError,
    RequestTimeoutError,
    RingClosedError,
    RingFullError,
    ServerStateError,
)
from shipinfer.core.logging import get_logger
from shipinfer.core.request import InferenceRequest, InferenceResponse
from shipinfer.core.types import Device
from shipinfer.runtime.memory.shared_ring import SharedRing
from shipinfer.scheduling.work import WorkItem
from shipinfer.server import remote_wire as wire

__all__ = ["RemoteInstance", "ResultReader", "RingIngress"]

_LOG = get_logger(__name__)

#: A heartbeat older than this many seconds means the owner is gone. Five stamps at the
#: default 200 ms — one missed stamp is a scheduler hiccup, five is a dead process.
DEFAULT_LOST_AFTER_S = 1.0
#: How long a submit waits for a free slot before the ring is called full. Short on purpose:
#: the dispatcher's spill loop is right behind it with the next candidate.
DEFAULT_SUBMIT_TIMEOUT_S = 0.005


class RemoteInstance:
    """A peer shard's instance of one model, as a placement candidate.

    ``device`` is ``cpu``: a proxy is *not here*, and the one thing a policy does with a
    candidate's device is compare it to the request's resident device to keep work local —
    a proxy must never win that comparison. Where the work actually ran comes back in the
    response's ``executed_on``, which is the number a load-balance dashboard wants.
    """

    def __init__(
        self,
        *,
        owner: str,
        model_name: str,
        submit: SharedRing,
        reader: ResultReader,
        submit_timeout_s: float = DEFAULT_SUBMIT_TIMEOUT_S,
        lost_after_s: float = DEFAULT_LOST_AFTER_S,
    ) -> None:
        self.owner = owner
        self.model_name = model_name
        self._submit = submit
        self._reader = reader
        self._submit_timeout_s = submit_timeout_s
        self._lost_after_ns = int(lost_after_s * 1e9)
        reader.watch(owner, submit)

    # -- Placeable -------------------------------------------------------------------------

    @property
    def device(self) -> Device:
        return Device.cpu()

    @property
    def depth(self) -> int:
        return self._submit.load_signal()[0]

    @property
    def ewma_latency_us(self) -> float:
        return self._submit.load_signal()[1]

    @property
    def is_ready(self) -> bool:
        # `is_closed` is a one-byte read and `load_signal` one unpack: the dispatcher touches
        # every candidate's `is_ready` per request, so no header dataclass on this path.
        if self._submit.is_closed:
            return False
        return time.monotonic_ns() - self._submit.load_signal()[2] < self._lost_after_ns

    # -- the submit ------------------------------------------------------------------------

    def enqueue(self, item: WorkItem) -> None:
        """Write the request into the owner's ring.

        Raises:
            RingFullError: no slot came free within the submit timeout. A
                :class:`~shipinfer.core.errors.QueueFullError`, so the dispatcher's spill loop
                treats it exactly like a full local queue: the request is still with the
                caller and the next candidate is tried.
        """
        index = self._submit.claim(self._submit_timeout_s)
        try:
            wire.encode_request(item.request, self._submit.payload(index))
        except Exception:
            # The slot was claimed but never published: hand it straight back.
            self._submit.abandon(index)
            raise
        item.request.timings.queued_ns = time.monotonic_ns()
        # Registered before publish: a fast owner could answer before the registration.
        self._reader.expect(item, self.owner)
        self._submit.publish(index)

    def __repr__(self) -> str:
        return f"<RemoteInstance {self.model_name!r} at {self.owner!r} depth={self.depth}>"


class ResultReader(threading.Thread):
    """One per process: resolves the futures of the requests this process sent to its peers.

    Owns the result rings (this process is their reader), knows every pending request by id,
    and watches every owner's heartbeat so a dead peer fails its requests with the tags they
    carried rather than leaving them to time out one by one.
    """

    def __init__(
        self,
        *,
        poll_s: float = 0.0005,
        lost_after_s: float = DEFAULT_LOST_AFTER_S,
        pending_timeout_s: float = 60.0,
    ) -> None:
        super().__init__(name="shipinfer-result-reader", daemon=True)
        self._results: dict[str, SharedRing] = {}
        self._heartbeats: dict[str, SharedRing] = {}
        self._pending: dict[int, tuple[WorkItem, str, int]] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._poll_s = poll_s
        self._lost_after_ns = int(lost_after_s * 1e9)
        #: A live owner that *dropped* one reply (its patience ran out on a full result
        #: ring) never closes and never misses a heartbeat — without a deadline the pending
        #: entry would pin the WorkItem (and its ~6 MB of inputs) for the process's life.
        self._pending_deadline_ns = int(pending_timeout_s * 1e9)
        self._lost: set[str] = set()

    def add_result_ring(self, owner: str, ring: SharedRing) -> None:
        """The ring ``owner`` writes results into; this process created it and reads it."""
        with self._lock:
            self._results[owner] = ring

    def watch(self, owner: str, submit: SharedRing) -> None:
        """Read ``owner``'s heartbeat off the submit ring it stamps."""
        with self._lock:
            self._heartbeats[owner] = submit

    def expect(self, item: WorkItem, owner: str) -> None:
        with self._lock:
            self._pending[item.request.request_id] = (
                item,
                owner,
                time.monotonic_ns() + self._pending_deadline_ns,
            )

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        idle = 0
        last_bookkeeping = 0
        while not self._stopping.is_set():
            busy = self._drain_results()
            now = time.monotonic_ns()
            # Liveness and expiry decide on second-scale windows; checking them thousands of
            # times a second only contends `_lock` with the dispatch path's `expect`.
            if now - last_bookkeeping >= self._lost_after_ns // 4:
                self._check_heartbeats()
                last_bookkeeping = now
            if busy:
                idle = 0
                continue
            idle += 1
            if idle > 50:
                time.sleep(self._poll_s)

    def _drain_results(self) -> bool:
        busy = False
        with self._lock:
            rings = list(self._results.items())
        for owner, ring in rings:
            index = ring.take(timeout_s=0)
            if index is None:
                continue
            busy = True
            try:
                self._resolve(owner, ring.payload(index))
            except RingClosedError:
                # The ring closed between take and the payload read; its owner's pending
                # futures are the heartbeat watcher's to fail, not a reason to die.
                continue
            finally:
                ring.release(index)
        return busy

    def _resolve(self, owner: str, payload: memoryview) -> None:
        request_id = wire.peek_request_id(payload)
        with self._lock:
            entry = self._pending.pop(request_id, None)
        if entry is None:
            _LOG.warning("result for unknown request %d from %s; dropped", request_id, owner)
            return
        item, _, _ = entry
        try:
            response = wire.decode_response(payload, item.request.context)
        except wire.RemoteFailureError as exc:
            item.fail(ServerStateError(f"remote {owner!r}: {exc}"))
            return
        except Exception as exc:  # a protocol error is the peer's build disagreeing with ours
            item.fail(exc)
            return
        response.timings.completed_ns = response.timings.completed_ns or time.monotonic_ns()
        if item.future.set_running_or_notify_cancel():
            item.future.set_result(response)

    def _check_heartbeats(self) -> None:
        now = time.monotonic_ns()
        with self._lock:
            watched = list(self._heartbeats.items())
        for owner, submit in watched:
            alive = not submit.is_closed and now - submit.load_signal()[2] < self._lost_after_ns
            if alive:
                self._lost.discard(owner)
                continue
            if owner in self._lost:
                continue
            self._lost.add(owner)
            self._fail_owner(owner)
        self._expire_pending(now)

    def _expire_pending(self, now: int) -> None:
        """Fail entries whose owner is alive but never answered: a dropped reply (the owner's
        patience ran out on a full result ring) must not pin the WorkItem forever."""
        with self._lock:
            expired = [
                (rid, item, who)
                for rid, (item, who, deadline) in self._pending.items()
                if now > deadline
            ]
            for rid, _, _ in expired:
                del self._pending[rid]
        for rid, item, who in expired:
            item.fail(
                RequestTimeoutError(
                    f"remote {who!r} never answered request {rid} within the pending deadline"
                )
            )

    def _fail_owner(self, owner: str) -> None:
        with self._lock:
            stranded = [
                (rid, item) for rid, (item, who, _) in self._pending.items() if who == owner
            ]
            for rid, _ in stranded:
                del self._pending[rid]
        if not stranded:
            return
        tags = [
            (item.request.context.camera_id, item.request.context.frame_id)
            for _, item in stranded
        ]
        error = PeerLostError(owner, tags)
        _LOG.error("%s", error)
        for _, item in stranded:
            item.fail(error)


@dataclass
class IngressLane:
    """One (submitter, model) pair the sweeper serves: its rings and its model's doors."""

    submitter: str
    inbound: SharedRing
    results: SharedRing
    infer: Callable[[InferenceRequest], Future[InferenceResponse]]
    load: Callable[[], tuple[int, float]]


@dataclass
class _Reply:
    """A settled request waiting for a slot in its submitter's result ring.

    Holding it here (with its inbound slot still claimed) is the back-pressure: the GPU
    worker thread that settled the future appended this and returned — it never touches a
    ring, never waits, never stops draining its own queue.
    """

    lane: IngressLane
    inbound_index: int
    response: InferenceResponse | None
    request_id: int
    model_name: str
    error: BaseException | None
    deadline: float


class RingIngress(threading.Thread):
    """The owner's side of the tier: ONE thread sweeping every inbound ring.

    The shape `shared_ring`'s docstring prescribes — `take(timeout_s=0)` round-robin over all
    lanes with a backoff once a whole sweep is idle — so an idle fleet costs one mostly
    sleeping thread per process, not a hot spinner per (peer, model). Requests go through each
    lane's `infer` (the model's own local entry point) exactly as a local caller's would.

    **The GPU worker thread never touches a ring.** A future's completion callback runs on
    the instance's single worker thread; here it only appends the settled result to an
    internal queue and returns. This thread claims the result slot, encodes, publishes and
    releases the inbound slot — retrying a full result ring between sweeps (the submitter's
    reader back-pressure) until the reply's patience runs out, and dropping quietly when the
    ring is closed (the submitter is gone; its own machinery failed the future). Stamping
    continues every interval regardless, so a slow reply can never read as a dead owner.
    """

    def __init__(
        self,
        lanes: Sequence[IngressLane],
        *,
        stamp_every_s: float = 0.2,
        result_patience_s: float = 30.0,
        idle_sleep_s: float = 0.0005,
        idle_after: int = 50,
    ) -> None:
        super().__init__(name="shipinfer-ring-ingress", daemon=True)
        self._lanes = list(lanes)
        self._stamp_every_ns = int(stamp_every_s * 1e9)
        self._result_patience_s = result_patience_s
        self._idle_sleep_s = idle_sleep_s
        self._idle_after = idle_after
        self._stopping = threading.Event()
        self._replies: deque[_Reply] = deque()
        #: Requests the owner's model queue refused: retried next sweep with their inbound
        #: slots still claimed, so the ring fills and the submitter spills locally (ADR-005).
        self._deferred: deque[tuple[IngressLane, int, InferenceRequest, float]] = deque()
        self._last_stamp_ns = 0
        self.served = 0
        self.failed = 0
        #: Replies that could not land: the submitter's result ring stayed full past
        #: `result_patience_s`, or was closed (the submitter is gone). The submitter's own
        #: deadline machinery fails the future; this counter says it was us.
        self.dropped = 0

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        try:
            self._stamp(force=True)
            idle = 0
            while not self._stopping.is_set():
                busy = self._drain_replies()
                busy = self._retry_deferred() or busy
                # A sweep that landed replies or requeued work changed the depths it reads;
                # peers place on those numbers, so they go out now, not at the next timer tick.
                self._stamp(force=busy)
                deferred_lanes = {id(lane) for lane, *_ in self._deferred}
                open_lanes = 0
                for lane in self._lanes:
                    if lane.inbound.is_closed:
                        continue
                    open_lanes += 1
                    if id(lane) in deferred_lanes:
                        # This lane's model is saturated; taking more would only queue more
                        # refusals. Leaving the ring full is the back-pressure.
                        continue
                    index = lane.inbound.take(timeout_s=0)
                    if index is None:
                        continue
                    busy = True
                    self._serve(lane, index)
                    self._stamp(force=True)  # the depth just changed; peers decide on it
                if open_lanes == 0 and not self._replies and not self._deferred:
                    return
                if busy:
                    idle = 0
                    continue
                idle += 1
                time.sleep(self._idle_sleep_s if idle > self._idle_after else 0)
        finally:
            # However the loop ends: drop what could not be delivered, free the slots (inert
            # on closed rings), and close every inbound so writers stop and readers know.
            while self._replies:
                reply = self._replies.popleft()
                self.dropped += 1
                reply.lane.inbound.release(reply.inbound_index)
            while self._deferred:
                lane, index, _request, _cap = self._deferred.popleft()
                self.dropped += 1
                lane.inbound.release(index)
            for lane in self._lanes:
                lane.inbound.close()

    # -- the owner side of one request ---------------------------------------------------

    def _serve(self, lane: IngressLane, index: int) -> None:
        payload = lane.inbound.payload(index)
        try:
            request = wire.decode_request(payload, copy=False)
        except Exception as exc:
            request_id = wire.peek_request_id(payload) if len(payload) >= 16 else 0
            self.failed += 1
            self._replies.append(self._failure_reply(lane, index, request_id, "?", exc))
            return
        self._start(lane, index, request, time.monotonic() + self._result_patience_s)

    def _start(
        self, lane: IngressLane, index: int, request: InferenceRequest, cap: float
    ) -> None:
        try:
            future = lane.infer(request)
        except QueueFullError:
            # The owner's own model queue refused — saturation, not death. Failing the frame
            # here inverts ADR-005: the submitter (depth 5) could have run what it sent to a
            # peer whose header was 200 ms stale. Keep the inbound slot claimed and retry the
            # request next sweep; the ring fills, the submitter's next enqueue raises
            # RingFullError, and the dispatcher spills back to a local instance — the same
            # pressure propagation a full result ring already applies to replies.
            if time.monotonic() >= cap:
                self.failed += 1
                self._replies.append(
                    self._failure_reply(
                        lane,
                        index,
                        request.request_id,
                        request.model_name,
                        QueueFullError(
                            f"{request.model_name} at {lane.submitter}'s peer", 0, 0
                        ),
                    )
                )
                return
            self._deferred.append((lane, index, request, cap))
            return
        except Exception as exc:
            self.failed += 1
            self._replies.append(
                self._failure_reply(lane, index, request.request_id, request.model_name, exc)
            )
            return

        def done(
            settled: Future[InferenceResponse],
            *,
            lane: IngressLane = lane,
            index: int = index,
            request: InferenceRequest = request,
        ) -> None:
            # Runs on the instance's worker thread: append and return. Anything slower —
            # claiming a slot, waiting out a full ring — would stall the GPU it serves.
            error = settled.exception()
            self._replies.append(
                _Reply(
                    lane=lane,
                    inbound_index=index,
                    response=None if error is not None else settled.result(),
                    request_id=request.request_id,
                    model_name=request.model_name,
                    error=error,
                    deadline=time.monotonic() + self._result_patience_s,
                )
            )

        future.add_done_callback(done)

    def _failure_reply(
        self,
        lane: IngressLane,
        index: int,
        request_id: int,
        model_name: str,
        error: BaseException,
    ) -> _Reply:
        return _Reply(
            lane=lane,
            inbound_index=index,
            response=None,
            request_id=request_id,
            model_name=model_name,
            error=error,
            deadline=time.monotonic() + self._result_patience_s,
        )

    # -- the replies -----------------------------------------------------------------------

    def _retry_deferred(self) -> bool:
        """Requests an earlier sweep could not hand to a saturated model: try again, once
        each, keeping their slots claimed. Never blocks."""
        busy = False
        for _ in range(len(self._deferred)):
            try:
                lane, index, request, cap = self._deferred.popleft()
            except IndexError:
                break
            busy = True
            self._start(lane, index, request, cap)
        return busy

    def _drain_replies(self) -> bool:
        """Land what can land; requeue what is back-pressured; drop what expired or lost its
        submitter. Never blocks: a full ring is retried on the next sweep."""
        busy = False
        for _ in range(len(self._replies)):
            try:
                reply = self._replies.popleft()
            except IndexError:
                break
            if reply.lane.results.is_closed:
                # The submitter left; its reader failed every pending future already.
                self.dropped += 1
                reply.lane.inbound.release(reply.inbound_index)
                busy = True
                continue
            try:
                slot = reply.lane.results.claim(timeout_s=0)
            except RingClosedError:
                self.dropped += 1
                reply.lane.inbound.release(reply.inbound_index)
                busy = True
                continue
            except RingFullError:
                if time.monotonic() >= reply.deadline:
                    self.dropped += 1
                    _LOG.error(
                        "result ring to %s stayed full for %.0fs; reply for request %d dropped",
                        reply.lane.submitter,
                        self._result_patience_s,
                        reply.request_id,
                    )
                    reply.lane.inbound.release(reply.inbound_index)
                    busy = True
                else:
                    self._replies.append(reply)  # back-pressure: retry next sweep
                continue
            view = reply.lane.results.payload(slot)
            if reply.error is not None:
                wire.encode_failure(reply.request_id, reply.model_name, reply.error, view)
            else:
                assert reply.response is not None
                try:
                    wire.encode_response(reply.response, view)
                    self.served += 1
                except Exception as exc:
                    wire.encode_failure(reply.request_id, reply.model_name, exc, view)
                    self.failed += 1
            reply.lane.results.publish(slot)
            reply.lane.inbound.release(reply.inbound_index)
            busy = True
        return busy

    # -- the heartbeat -----------------------------------------------------------------------

    def _stamp(self, *, force: bool = False) -> None:
        """The timer is the *liveness* cadence; activity re-stamps immediately (`force`), so
        the depth a peer reads tracks queue transitions rather than the heartbeat interval —
        a placement decision made on a 200 ms-old depth herds bursts onto whichever peer
        stamped last."""
        now = time.monotonic_ns()
        if not force and now - self._last_stamp_ns < self._stamp_every_ns:
            return
        for lane in self._lanes:
            depth, ewma = lane.load()
            lane.inbound.stamp(depth=depth, ewma_latency_us=ewma)
        self._last_stamp_ns = now
