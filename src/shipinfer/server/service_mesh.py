"""The rings and threads one shard runs to join the `service` tier.

A shard is both an owner and a submitter for every shared model. For each peer ``P`` and each
shared model ``M`` present in this process:

* it **creates** the request ring ``P → me`` (P writes, this process reads) and the result ring
  ``me ← P`` (this process wrote the requests, P writes the results back), both named by the
  run id and the two shard indices, so a peer can open them by name;
* it **opens** the mirror images that ``P`` created — ``me → P`` to submit and ``P ← me`` to
  answer — retrying until they appear, because peers start in no particular order;
* it runs one :class:`~shipinfer.server.remote_instance.RingIngress` per ``(P, M)`` over the
  requests P sends — into the model's *local* instances, so a request never crosses twice — one :class:`~shipinfer.server.remote_instance.ResultReader` for every result
  ring it owns, and hands each shared model a :class:`~shipinfer.server.remote_instance.RemoteInstance`
  per peer.

Two phases, because creation must precede any peer's open: :meth:`create` first, in every
process, then :meth:`connect`. A single-process `serve` (no shard index) has no tier and the
mesh is not built at all.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from shipinfer.core.errors import ConfigurationError, RingClosedError
from shipinfer.core.logging import get_logger
from shipinfer.core.settings.topology import ServiceSettings
from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing, reap_pending_closes
from shipinfer.server.remote_instance import (
    IngressLane,
    RemoteInstance,
    ResultReader,
    RingIngress,
)

__all__ = ["ServiceMesh", "ring_name", "wire_slot_bytes"]

_LOG = get_logger(__name__)


def ring_name(run_id: str, submitter: int, owner: int, model: str, kind: str) -> str:
    """The name of the ``submitter → owner`` ring for ``model``: ``req`` or ``res``.

    Deterministic and run-scoped, so both processes derive it without talking to each other
    and two fleets on one box cannot collide.
    """
    if kind not in ("req", "res"):
        raise ValueError(f"ring kind must be 'req' or 'res', got {kind!r}")
    return f"shipinfer-{run_id}-{submitter}-to-{owner}-{model}-{kind}"


#: Head room per slot for the request/response head and the per-tensor heads, which travel in
#: the slot ahead of the bytes.
_HEAD_ROOM = 64 * 1024
_PAGE = 4096


def wire_slot_bytes(config: Any, fallback: int) -> tuple[int, int]:
    """``(request, response)`` slot bytes for one model, from its own config.

    A request slot holds one full batch of the model's inputs, a response slot one batch of its
    outputs — plus the heads, rounded to pages. Both processes derive the same numbers from the
    same repository, which is what lets them open each other's rings without negotiating. A
    dynamic extent (``-1``) makes a side unsizeable, and that side gets ``fallback``
    (``ServiceSettings.slot_bytes``).
    """

    def side(specs: Any) -> int:
        total = 0
        for spec in specs:
            count = 1
            for dim in spec.shape:
                if dim < 0:
                    return -1
                count *= dim
            total += count * spec.dtype.itemsize
        return total

    batch = max(1, int(getattr(config, "max_batch_size", 0) or 1))

    def slot(nbytes: int) -> int:
        if nbytes < 0:
            return fallback
        raw = batch * nbytes + _HEAD_ROOM
        return ((raw + _PAGE - 1) // _PAGE) * _PAGE

    return slot(side(config.input_specs)), slot(side(config.output_specs))


class _SharedModel(Protocol):
    """What the mesh needs of a model: admission and dispatch, its load, and the attach."""

    @property
    def name(self) -> str: ...

    def admit_local(self, request: Any) -> Any: ...

    def try_dispatch_local(self, item: Any) -> bool: ...

    def count_local_rejection(self) -> None: ...

    @property
    def advertised_depth(self) -> int: ...

    @property
    def ewma_latency_us(self) -> float: ...

    def attach_remote(self, candidates: Any) -> None: ...


@dataclass
class ServiceMesh:
    """This shard's side of the tier. Built by the server when the topology is `service`."""

    settings: ServiceSettings
    shard: int
    models: dict[str, _SharedModel]
    #: Per-model ``(request, response)`` slot bytes (`wire_slot_bytes`); the setting when absent.
    slot_bytes_by_model: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    _owned: list[SharedRing] = field(default_factory=list, init=False, repr=False)
    _opened: list[SharedRing] = field(default_factory=list, init=False, repr=False)
    _ingress: RingIngress | None = field(default=None, init=False, repr=False)
    _reader: ResultReader | None = field(default=None, init=False, repr=False)
    _proxies: dict[str, list[RemoteInstance]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.settings.run_id:
            raise ConfigurationError(
                "service mesh: no run id — was this process started by `shipinfer fleet`?"
            )
        if self.shard not in self.settings.peers:
            raise ConfigurationError(
                f"service mesh: shard {self.shard} is not among the peers {self.settings.peers}"
            )

    @property
    def peers(self) -> list[int]:
        return [p for p in self.settings.peers if p != self.shard]

    def layout_for(self, model: str, kind: str) -> RingLayout:
        """One (model, direction) ring's layout: request slots sized from the model's inputs,
        response slots from its outputs, the setting when the model's shapes are dynamic."""
        sizes = self.slot_bytes_by_model.get(model)
        if sizes is None:
            nbytes = self.settings.slot_bytes
        else:
            nbytes = sizes[0] if kind == "req" else sizes[1]
        return RingLayout(slots=self.settings.slots_per_pair, slot_bytes=nbytes)

    # -- phase 1: what this shard owns ---------------------------------------------------

    def create(self) -> None:
        """Create the rings this shard reads: requests from each peer, results for each peer."""
        me = str(self.shard)
        for peer in self.peers:
            for model in self.models:
                self._owned.append(
                    SharedRing.create(
                        ring_name(self.settings.run_id, peer, self.shard, model, "req"),
                        self.layout_for(model, "req"),
                        owner=me,
                    )
                )
                self._owned.append(
                    SharedRing.create(
                        ring_name(self.settings.run_id, self.shard, peer, model, "res"),
                        self.layout_for(model, "res"),
                        owner=me,
                    )
                )
        _LOG.info(
            "service mesh: shard %d created %d ring(s) for %d peer(s)",
            self.shard,
            len(self._owned),
            len(self.peers),
        )

    # -- phase 2: what the peers own -----------------------------------------------------

    def connect(self, timeout_s: float | None = None) -> None:
        """Open the peers' rings, start the threads, attach the proxies. Blocks until every
        peer's rings exist or ``timeout_s`` passes — peers start in no particular order."""
        deadline = time.monotonic() + (
            self.settings.connect_timeout_s if timeout_s is None else timeout_s
        )
        reader = ResultReader(
            lost_after_s=self.settings.lost_after_ms / 1000.0,
            pending_timeout_s=self.settings.pending_timeout_ms / 1000.0,
        )
        self._reader = reader
        owned = {ring.name: ring for ring in self._owned}
        lanes: list[IngressLane] = []
        # ONE load closure per model, shared by every peer's lane for it: the ingress
        # memoizes its sweep stamp by callable identity, so each model's depth is computed
        # once per sweep instead of once per lane (#26 round 4).
        loads = {name: _load_of(model) for name, model in self.models.items()}
        for peer in self.peers:
            for name, model in self.models.items():
                # The result ring for my requests to `peer`: I created it, I read it.
                reader.add_result_ring(
                    f"{peer}:{name}",
                    owned[ring_name(self.settings.run_id, self.shard, peer, name, "res")],
                )
                # Their request ring for my submits, and their result ring for my answers.
                submit = self._open(
                    ring_name(self.settings.run_id, self.shard, peer, name, "req"),
                    self.layout_for(name, "req"),
                    deadline,
                )
                answers = self._open(
                    ring_name(self.settings.run_id, peer, self.shard, name, "res"),
                    self.layout_for(name, "res"),
                    deadline,
                )
                proxy = RemoteInstance(
                    owner=f"{peer}:{name}",
                    model_name=name,
                    submit=submit,
                    reader=reader,
                    submit_timeout_s=self.settings.submit_timeout_ms / 1000.0,
                    lost_after_s=self.settings.lost_after_ms / 1000.0,
                )
                self._proxies.setdefault(name, []).append(proxy)
                inbound = owned[ring_name(self.settings.run_id, peer, self.shard, name, "req")]
                lanes.append(
                    IngressLane(
                        submitter=str(peer),
                        inbound=inbound,
                        results=answers,
                        # Local only: work that crossed once is done here, never re-routed.
                        # Admission happens once; a saturated queue is retried dispatch-only,
                        # so the counters and validation cannot repeat per retry.
                        admit=model.admit_local,
                        dispatch=model.try_dispatch_local,
                        reject=model.count_local_rejection,
                        load=loads[name],
                    )
                )
        # ONE sweeper over every lane — the consumer shape the ring's docstring prescribes.
        # A thread per (peer, model) was a hot spinner each: 6 busy cores on an idle 4-shard
        # fleet, 30 at the box's ceiling.
        self._ingress = RingIngress(lanes, stamp_every_s=self.settings.heartbeat_ms / 1000.0)
        self._ingress.start()
        reader.start()
        for name, proxies in self._proxies.items():
            self.models[name].attach_remote(proxies)
        _LOG.info(
            "service mesh: shard %d joined the tier — %d proxy(ies), one sweeper over %d lane(s)",
            self.shard,
            sum(len(v) for v in self._proxies.values()),
            len(lanes),
        )

    def _open(self, name: str, layout: RingLayout, deadline: float) -> SharedRing:
        while True:
            try:
                ring = SharedRing.open(name, layout)
            except RingClosedError:
                # Not created yet (peers start in no order) — or already unlinked, which the
                # deadline turns into the same configuration answer.
                if time.monotonic() >= deadline:
                    raise ConfigurationError(
                        f"service mesh: peer ring {name!r} never appeared; is every shard up?"
                    ) from None
                time.sleep(0.05)
                continue
            self._opened.append(ring)
            return ring

    # -- lifecycle -------------------------------------------------------------------------

    @property
    def proxies(self) -> dict[str, list[RemoteInstance]]:
        return self._proxies

    def stop(self) -> None:
        if self._ingress is not None:
            self._ingress.stop()
        if self._reader is not None:
            self._reader.stop()
        # `connect` can fail before the threads start (a peer that never came up), and an
        # unstarted thread cannot be joined.
        if self._ingress is not None and self._ingress.is_alive():
            self._ingress.join(timeout=2.0)
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=2.0)
        for ring in self._opened:
            ring.close()
        for ring in self._owned:
            ring.close()
        self._ingress = None
        self._opened.clear()
        self._owned.clear()
        # Sweep any mapping a zero-copy payload view kept alive past its ring's close: by
        # now the threads are joined, so the views are dead and the mappings can go.
        reap_pending_closes()


def _load_of(model: _SharedModel) -> Callable[[], tuple[int, float]]:
    def load() -> tuple[int, float]:
        # Per-instance, not the sum over instances: every local candidate reports one
        # queue's depth, and a peer advertised as `count x` deeper than it is declines the
        # borrow in exactly the crowded-shard case the tier exists for. `min` is the queue
        # the owner's own dispatcher would hand the work to.
        return int(model.advertised_depth), float(model.ewma_latency_us)

    return load
