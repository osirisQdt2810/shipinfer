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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import get_logger
from shipinfer.core.settings.topology import ServiceSettings
from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing
from shipinfer.server.remote_instance import RemoteInstance, ResultReader, RingIngress

__all__ = ["ServiceMesh", "ring_name"]

_LOG = get_logger(__name__)


def ring_name(run_id: str, submitter: int, owner: int, model: str, kind: str) -> str:
    """The name of the ``submitter → owner`` ring for ``model``: ``req`` or ``res``.

    Deterministic and run-scoped, so both processes derive it without talking to each other
    and two fleets on one box cannot collide.
    """
    if kind not in ("req", "res"):
        raise ValueError(f"ring kind must be 'req' or 'res', got {kind!r}")
    return f"shipinfer-{run_id}-{submitter}-to-{owner}-{model}-{kind}"


class _SharedModel(Protocol):
    """What the mesh needs of a model: the local entry point, its load, and the attach."""

    @property
    def name(self) -> str: ...

    def infer(self, request: Any) -> Any: ...

    def infer_local(self, request: Any) -> Any: ...

    @property
    def total_depth(self) -> int: ...

    @property
    def ewma_latency_us(self) -> float: ...

    def attach_remote(self, candidates: Any) -> None: ...


@dataclass
class ServiceMesh:
    """This shard's side of the tier. Built by the server when the topology is `service`."""

    settings: ServiceSettings
    shard: int
    models: dict[str, _SharedModel]
    _owned: list[SharedRing] = field(default_factory=list, init=False, repr=False)
    _opened: list[SharedRing] = field(default_factory=list, init=False, repr=False)
    _ingresses: list[RingIngress] = field(default_factory=list, init=False, repr=False)
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

    def layout(self) -> RingLayout:
        return RingLayout(
            slots=self.settings.slots_per_pair, slot_bytes=self.settings.slot_bytes
        )

    # -- phase 1: what this shard owns ---------------------------------------------------

    def create(self) -> None:
        """Create the rings this shard reads: requests from each peer, results for each peer."""
        layout = self.layout()
        me = str(self.shard)
        for peer in self.peers:
            for model in self.models:
                self._owned.append(
                    SharedRing.create(
                        ring_name(self.settings.run_id, peer, self.shard, model, "req"),
                        layout,
                        owner=me,
                    )
                )
                self._owned.append(
                    SharedRing.create(
                        ring_name(self.settings.run_id, self.shard, peer, model, "res"),
                        layout,
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
        layout = self.layout()
        deadline = time.monotonic() + (
            self.settings.connect_timeout_s if timeout_s is None else timeout_s
        )
        reader = ResultReader(lost_after_s=self.settings.lost_after_ms / 1000.0)
        self._reader = reader
        owned = {ring.name: ring for ring in self._owned}
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
                    layout,
                    deadline,
                )
                answers = self._open(
                    ring_name(self.settings.run_id, peer, self.shard, name, "res"),
                    layout,
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
                self._ingresses.append(
                    RingIngress(
                        submitter=str(peer),
                        inbound=inbound,
                        results=answers,
                        # Local only: work that crossed once is done here, never re-routed.
                        infer=model.infer_local,
                        load=_load_of(model),
                        stamp_every_s=self.settings.heartbeat_ms / 1000.0,
                    )
                )
        for ingress in self._ingresses:
            ingress.start()
        reader.start()
        for name, proxies in self._proxies.items():
            self.models[name].attach_remote(proxies)
        _LOG.info(
            "service mesh: shard %d joined the tier — %d proxy(ies), %d ingress(es)",
            self.shard,
            sum(len(v) for v in self._proxies.values()),
            len(self._ingresses),
        )

    def _open(self, name: str, layout: RingLayout, deadline: float) -> SharedRing:
        while True:
            try:
                ring = SharedRing.open(name, layout)
            except FileNotFoundError:
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
        for ingress in self._ingresses:
            ingress.stop()
        if self._reader is not None:
            self._reader.stop()
        # `connect` can fail before the threads start (a peer that never came up), and an
        # unstarted thread cannot be joined.
        for ingress in self._ingresses:
            if ingress.is_alive():
                ingress.join(timeout=2.0)
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=2.0)
        for ring in self._opened:
            ring.close()
        for ring in self._owned:
            ring.close()
        self._ingresses.clear()
        self._opened.clear()
        self._owned.clear()


def _load_of(model: _SharedModel) -> Callable[[], tuple[int, float]]:
    def load() -> tuple[int, float]:
        return int(model.total_depth), float(model.ewma_latency_us)

    return load
