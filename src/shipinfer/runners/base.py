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
  question asked of a runner that will not start is what state it thinks it is in.
"""

from __future__ import annotations

import abc
import contextlib
import threading
from typing import Any, ClassVar

from shipinfer.core.errors import ServerStateError
from shipinfer.core.request import ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Device
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

    Subclasses implement :meth:`_do_start`, :meth:`_do_stop` and :meth:`_do_submit`, and may
    add to :meth:`health` and :meth:`stats` through the two optional hooks.
    """

    #: The registered implementation name, set by each subclass so the name in the settings
    #: tree and the name in a log line cannot drift. Mirrors
    #: :attr:`shipinfer.scheduling.queues.base.RequestQueue.name`.
    name: ClassVar[str] = "abstract"

    def __init__(
        self,
        topology: Topology,
        settings: ServerSettings | None = None,
        *,
        shard_id: int = 0,
        device: Device | None = None,
        models: ModelResolver | None = None,
    ) -> None:
        self._topology = topology
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

    # -- what the runner was told ------------------------------------------------------

    @property
    def topology(self) -> Topology:
        return self._topology

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
        """
        return ElementContext(shard_id=self._shard_id, device=self._device, models=self._models)

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> Runner:
        """Open the chain and start executing it. Idempotent.

        A partial start leaks whatever it did manage to acquire — an open decoder, a socket, a
        worker thread, a CUDA context on a shared box. The subclass often cannot tell how far
        it got, so unwind unconditionally and best-effort, then re-raise the *original*
        failure, which is the one worth reading. Same shape as
        :meth:`shipinfer.topology.base.Element.open`, for the same reason.

        Raises:
            ShipInferError: whatever the implementation needs to say — a missing model, a
                camera that will not open, a port already bound. Nothing is swallowed.
        """
        with self._lifecycle:
            if self._running:
                return self
            try:
                self._do_start()
            except BaseException:
                with contextlib.suppress(Exception):
                    self._do_stop(0.0)
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
