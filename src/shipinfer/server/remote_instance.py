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
from collections.abc import Callable
from concurrent.futures import Future

from shipinfer.core.errors import (
    PeerLostError,
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
        return self._submit.header().depth

    @property
    def ewma_latency_us(self) -> float:
        return self._submit.header().ewma_latency_us

    @property
    def is_ready(self) -> bool:
        header = self._submit.header()
        if header.closed:
            return False
        return time.monotonic_ns() - header.heartbeat_ns < self._lost_after_ns

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
        self, *, poll_s: float = 0.0005, lost_after_s: float = DEFAULT_LOST_AFTER_S
    ) -> None:
        super().__init__(name="shipinfer-result-reader", daemon=True)
        self._results: dict[str, SharedRing] = {}
        self._heartbeats: dict[str, SharedRing] = {}
        self._pending: dict[int, tuple[WorkItem, str]] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._poll_s = poll_s
        self._lost_after_ns = int(lost_after_s * 1e9)
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
            self._pending[item.request.request_id] = (item, owner)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        idle = 0
        while not self._stopping.is_set():
            busy = self._drain_results()
            self._check_heartbeats()
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
        item, _ = entry
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
            header = submit.header()
            alive = not header.closed and now - header.heartbeat_ns < self._lost_after_ns
            if alive:
                self._lost.discard(owner)
                continue
            if owner in self._lost:
                continue
            self._lost.add(owner)
            self._fail_owner(owner)

    def _fail_owner(self, owner: str) -> None:
        with self._lock:
            stranded = [
                (rid, item) for rid, (item, who) in self._pending.items() if who == owner
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


class RingIngress(threading.Thread):
    """The owner's side of one inbound ring: requests in, responses out.

    Requests go through ``infer`` — the local model's own entry point — so they are validated,
    counted, dispatched and queued exactly as a local caller's would be, under the submitter's
    camera id. The inbound slot stays claimed until the future settles, so the request's tensors
    view the slot rather than copy it; the response (or the failure) is written into the
    submitter's result ring and the slot is released.
    """

    def __init__(
        self,
        *,
        submitter: str,
        inbound: SharedRing,
        results: SharedRing,
        infer: Callable[[InferenceRequest], Future[InferenceResponse]],
        load: Callable[[], tuple[int, float]],
        stamp_every_s: float = 0.2,
        result_timeout_s: float = 1.0,
        result_patience_s: float = 30.0,
        poll_s: float = 0.0005,
    ) -> None:
        super().__init__(name=f"shipinfer-ring-ingress-{submitter}", daemon=True)
        self._submitter = submitter
        self._inbound = inbound
        self._results = results
        self._infer = infer
        self._load = load
        self._stamp_every_ns = int(stamp_every_s * 1e9)
        self._result_timeout_s = result_timeout_s
        self._result_patience_s = result_patience_s
        self._poll_s = poll_s
        self._stopping = threading.Event()
        self._last_stamp_ns = 0
        self.served = 0
        self.failed = 0
        #: Replies that could not land: the submitter's result ring stayed full past
        #: `result_patience_s`, or was already closed (the submitter is gone). The submitter's
        #: own timeout/heartbeat machinery fails the future; this counter says it was us.
        self.dropped = 0

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        self._stamp(force=True)
        while not self._stopping.is_set():
            self._stamp()
            index = self._inbound.take(timeout_s=self._poll_s)
            if index is None:
                if self._inbound.is_closed:
                    return
                continue
            self._serve(index)
        # Closing tells every writer to stop claiming and every reader that the owner is gone.
        self._inbound.close()

    def _stamp(self, *, force: bool = False) -> None:
        now = time.monotonic_ns()
        if not force and now - self._last_stamp_ns < self._stamp_every_ns:
            return
        depth, ewma = self._load()
        self._inbound.stamp(depth=depth, ewma_latency_us=ewma)
        self._last_stamp_ns = now

    def _serve(self, index: int) -> None:
        payload = self._inbound.payload(index)
        try:
            request = wire.decode_request(payload, copy=False)
        except Exception as exc:
            request_id = wire.peek_request_id(payload) if len(payload) >= 16 else 0
            self._reply_failure(request_id, "?", exc)
            self._inbound.release(index)
            return
        try:
            future = self._infer(request)
        except Exception as exc:
            self._reply_failure(request.request_id, request.model_name, exc)
            self._inbound.release(index)
            return

        def done(
            settled: Future[InferenceResponse],
            *,
            index: int = index,
            request: InferenceRequest = request,
        ) -> None:
            try:
                error = settled.exception()
                if error is not None:
                    self._reply_failure(request.request_id, request.model_name, error)
                else:
                    self._reply(settled.result())
            finally:
                self._inbound.release(index)

        future.add_done_callback(done)

    def _reply(self, response: InferenceResponse) -> None:
        index = self._claim_result_slot(response.request_id)
        if index is None:
            return
        try:
            wire.encode_response(response, self._results.payload(index))
        except Exception as exc:
            wire.encode_failure(
                response.request_id, response.model_name, exc, self._results.payload(index)
            )
            self.failed += 1
        else:
            self.served += 1
        self._results.publish(index)

    def _reply_failure(self, request_id: int, model_name: str, error: BaseException) -> None:
        self.failed += 1
        index = self._claim_result_slot(request_id)
        if index is None:
            return
        wire.encode_failure(request_id, model_name, error, self._results.payload(index))
        self._results.publish(index)

    def _claim_result_slot(self, request_id: int) -> int | None:
        """A slot in the submitter's result ring — waiting out a burst, never a corpse.

        A full result ring is back-pressure from the submitter's reader: hold the reply (and,
        upstream, the inbound slot) and retry, up to ``result_patience_s`` — the pressure is
        the design (ADR-005), and a reply must not be dropped for a burst. A *closed* ring is
        the submitter gone: drop quietly, because its own heartbeat machinery has already
        failed every future it was waiting on. Both outcomes count in ``dropped``.
        """
        deadline = time.monotonic() + self._result_patience_s
        while True:
            try:
                return self._results.claim(self._result_timeout_s)
            except RingClosedError:
                self.dropped += 1
                return None
            except RingFullError:
                if self._stopping.is_set() or time.monotonic() >= deadline:
                    self.dropped += 1
                    _LOG.error(
                        "result ring to %s stayed full for %.0fs; reply for request %d dropped",
                        self._submitter,
                        self._result_patience_s,
                        request_id,
                    )
                    return None
