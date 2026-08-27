"""``fleet``: one shard process per GPU, driven over the gRPC control plane (arch.md §2).

The production runner. It owns three things and delegates everything else:

* **the plan** — which shard gets which GPUs, from
  :func:`~shipinfer.scheduling.sharding.plan_shards`, printed before anything is spawned;
* **the processes** — :class:`~shipinfer.launch.supervisor.Fleet` spawns and reaps them, which
  is the one thing that cannot be an RPC (there is no RPC into a process that does not exist);
* **the conversation** — a :class:`~shipinfer.launch.client.ShardClient` per shard: wait for
  ready, install the chain, place a camera, read health, stop on one deadline.

WHAT THE CHILD IS TOLD, AND WHERE
---------------------------------
``[python, -m, shipinfer.cli.shard, --shard-id, N, --control-port, P]`` — its identity, and
nothing else (:meth:`FleetRunner.shard_command`). The camera set, the topology and the device
sharing all arrive as RPCs after it reports ready. That is V140's decision and the whole point
of the change: a camera can be added to a live shard, and a shard's state is a typed answer
rather than something inferred from an exit code.

**One exception, and it is physics rather than taste:** ``CUDA_VISIBLE_DEVICES`` must be in
the child's environment before its interpreter imports torch, which is several frames below
the first RPC it could possibly answer. :class:`Fleet` sets it at ``exec``. Everything else
the environment used to carry — ``SHIPINFER_SHARD_CAMERAS``, the three
``SHIPINFER_DEVICES__*`` placement keys — is now on the wire, and ``Fleet`` deliberately
strips the inherited copies so a parent's own configuration cannot leak into a child that
sees renumbered devices.

WHAT THIS RUNNER DOES NOT DO
----------------------------
It does not execute a chain. :meth:`submit` refuses, typed: frames enter a shard through that
shard's own decode elements, and a fleet that accepted items here would be funnelling every
camera through the parent process — the single shared buffer this project exists to delete.
Reassembly, batching and the walk are the shard's (``runners/inprocess.py``).
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, ClassVar

from shipinfer.core.errors import ConfigurationError, ServerStateError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.request import ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Device
from shipinfer.launch import Fleet, ShardClient
from shipinfer.launch.control import CameraSpec
from shipinfer.runners.base import Runner
from shipinfer.runners.registry import RUNNERS
from shipinfer.scheduling.sharding import Shard, ShardPlan, plan_shards
from shipinfer.topology import ChainItem, ModelResolver, Topology

__all__ = ["FleetRunner"]

_LOG = get_logger("runners.fleet")

#: The first shard's control port; shard *n* gets ``base + n``. Above the ephemeral range's
#: usual floor on Linux (32768) would collide with the kernel's own allocations, so this sits
#: below it — and a fleet that wants another number says so, because two fleets on one box
#: need two bases.
DEFAULT_CONTROL_PORT_BASE = 50100

#: How long a shard gets to answer ``Ready``. It has to import torch and bind a port, which is
#: seconds on a warm page cache and tens on a cold one.
DEFAULT_READY_TIMEOUT_S = 120.0

#: How long ``UpdateTopology`` may take. Deliberately minutes, not the client's default ten
#: seconds: that one call is where a shard loads its models and deserialises its engines
#: (``cli/shard.py``), and a deadline that expired mid-load would kill a fleet that was
#: starting perfectly well.
DEFAULT_TOPOLOGY_TIMEOUT_S = 900.0


@RUNNERS.register(
    "fleet",
    "shards",
    description="one shard process per GPU, driven over the gRPC control plane",
)
class FleetRunner(Runner):
    """Runs one topology across N shard processes. The production default (arch.md §1).

    Args:
        topology: the validated chain. Held so this runner can answer for it and so a
            mismatch between what the parent believes and what a shard installed is visible;
            the shards each load it themselves, from ``chain_yaml``.
        settings: the deployment settings. ``runner.shards``, ``runner.drain_s`` and
            ``devices.visible_gpus`` are the ones read here.
        chain_yaml: the chain file's **text**, which is what a shard is sent. Required,
            because the loader is the single door through which a chain becomes trustworthy
            (ADR-017) and it lives on the shard: re-rendering a ``Topology`` back to YAML here
            would be a second writer of the format the shard's loader is the only reader of.
        shards: how many processes. ``None`` takes ``runner.shards``, which itself defaults to
            one per visible GPU (ADR-006).
        gpus: physical CUDA ordinals to spread the shards over. ``None`` takes
            ``devices.visible_gpus``, and an empty list there means "ask the driver".
        control_port_base: shard *n* is reached on ``base + n``.
        command: what argv to spawn for a shard. Injected so the offline tier can supervise a
            process that is not a shard — everything below is about *conversation*, and
            testing it against a real shard would test CUDA.
        client: how to reach a shard. Injected for the same reason, and typed as a factory so
            a test can hand back a double with no channel behind it.

    Raises:
        ConfigurationError: no chain text, or a plan that cannot be made (no GPUs).

    LOCKING. :attr:`_lock` guards the two maps this class owns — the clients and the
    placements — and is **never held across an RPC**. A camera is placed by reserving it under
    the lock, asking the shard with the lock released, and committing (or releasing the
    reservation) under it again. That is not decoration: ``AddCamera`` starts a decoder and
    can take seconds on an RTSP source that is not answering, and a ``health`` probe that
    waited behind it would make the one call an operator reaches for during an incident the
    one call that hangs.
    """

    name: ClassVar[str] = "fleet"
    #: The fleet is exactly the runner that *does* manage cameras: it places them.
    manages_cameras: ClassVar[bool] = True

    def __init__(
        self,
        topology: Topology,
        settings: ServerSettings | None = None,
        *,
        shard_id: int = 0,
        device: Device | None = None,
        models: ModelResolver | None = None,
        chain_yaml: str = "",
        shards: int | None = None,
        gpus: Sequence[int] | None = None,
        control_port_base: int = DEFAULT_CONTROL_PORT_BASE,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        topology_timeout_s: float = DEFAULT_TOPOLOGY_TIMEOUT_S,
        command: Callable[[Shard], Sequence[str]] | None = None,
        client: Callable[[Shard, int], ShardClient] | None = None,
    ) -> None:
        super().__init__(
            topology,
            settings,
            shard_id=shard_id,
            device=device,
            models=models,
            chain_yaml=chain_yaml,
        )
        if not chain_yaml.strip():
            raise ConfigurationError(
                "a fleet sends its shards the chain's *text*, and none was given. The loader "
                "runs on the shard (ADR-017), so `chain_yaml=` is how the file reaches it — "
                "`shipinfer run` reads it once and passes both it and the parsed topology"
            )
        self._wanted_shards = shards
        self._wanted_gpus = None if gpus is None else list(gpus)
        self._control_port_base = control_port_base
        self._ready_timeout_s = ready_timeout_s
        self._topology_timeout_s = topology_timeout_s
        self._command = command
        self._client_factory = client
        self._fleet: Fleet | None = None
        self._plan: ShardPlan | None = None
        self._clients: dict[int, ShardClient] = {}
        #: Where each camera was placed. The launcher's own map, and authoritative: it is the
        #: only thing that places a camera, so a count derived from it cannot lag behind a
        #: health probe. What the shards report is in :meth:`health`, beside it, so a
        #: disagreement is visible rather than resolved silently in favour of one of them.
        self._placed: dict[str, int] = {}
        #: How many camera threads this cycle has had to abandon — a lifetime signal, not a
        #: statistic (``launch/control.py``). Non-zero means a detached thread on some shard
        #: still references buffers nobody may unwind. **Accumulated**, never assigned: a
        #: drain and the stop that follows it each abandon their own, and a failed start
        #: unwinds shards that abandon theirs. Assigning made the last writer the only truth,
        #: and the last writer is the one with nothing left to release. Reset by
        #: :meth:`_do_start`, which is the only moment there is genuinely nothing detached.
        self._abandoned = 0
        #: Cameras with an ``AddCamera`` in flight. They are in :attr:`_placed` already — so a
        #: concurrent placement counts them and does not pile onto the same shard — but not
        #: yet confirmed, and :meth:`health` says which is which rather than reporting a
        #: camera as placed while its shard is still deciding.
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    # -- what the operator asked for ----------------------------------------------------

    @property
    def plan(self) -> ShardPlan | None:
        """The plan being run, or ``None`` before :meth:`start`."""
        return self._plan

    @property
    def abandoned(self) -> int:
        return self._abandoned

    @staticmethod
    def shard_command(shard_id: int, control_port: int) -> list[str]:
        """The argv for one shard: its identity, and nothing else (arch.md §2).

        ``sys.executable`` rather than ``shipinfer`` so a child lands in the same virtualenv
        as its parent — the console script may not be on ``PATH`` inside a container, and a
        shard running under a different interpreter is a debugging session nobody wants.

        Two flags is the contract, and the child refuses a third
        (``cli/shard.py::build_parser``). Everything a shard used to be told here — the
        repository, the HTTP port, its cameras — is a call on its control plane now.
        """
        return [
            sys.executable,
            "-m",
            "shipinfer.cli.shard",
            "--shard-id",
            str(shard_id),
            "--control-port",
            str(control_port),
        ]

    def control_port(self, shard_id: int) -> int:
        return self._control_port_base + shard_id

    # -- lifecycle ----------------------------------------------------------------------

    def _do_start(self) -> None:
        """Spawn every shard, wait for it, and install the chain on it. In parallel.

        All or nothing, in both halves. :meth:`Fleet.start` already unwinds a partial spawn;
        this adds the second half — a shard that never answers, or that refuses the chain,
        takes the whole fleet down rather than leaving a deployment where three quarters of
        the cameras have somewhere to go. Half a fleet is the state that is hardest to notice:
        every process that *is* up reports healthy.

        **The installs overlap.** Each one is a ``wait_ready`` poll (up to two minutes while a
        child imports torch) followed by an ``UpdateTopology`` that deserialises this shard's
        engines (minutes on a cold page cache), and both are waits on *another process*. Run
        one after another, sixteen shards would take sixteen times a single shard's start —
        an eight-minute deployment turned into two hours — while every GPU but one sat idle.
        The pool is joined before anything is inspected, so no channel is closed under an RPC
        that is still in flight, and the failure re-raised is the **first in plan order**
        rather than whichever thread happened to lose first: a fleet must fail the same way
        twice or the operator is debugging the scheduler instead of the deployment.

        There is no unwind here on purpose. :meth:`Runner.start` owns exactly one
        (``runners/base.py``), and :meth:`_unwind_timeout_s` is how this class gives it the
        budget a fleet's release needs. Two unwinds is how the abandonment count came to be
        overwritten with a zero by the pass that ran second.
        """
        plan = self._build_plan()
        _LOG.info("fleet plan\n%s", plan.describe())
        fleet = Fleet(
            plan=plan,
            command=self._command_for,
            drain_s=self._settings.runner.drain_s,
        )
        self._plan, self._fleet = plan, fleet
        with self._lock:
            self._placed.clear()
            self._pending.clear()
        self._abandoned = 0
        fleet.start()
        installs: dict[int, Future[None]] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, len(plan.shards)), thread_name_prefix="fleet-install"
        ) as pool:
            for shard in plan.shards:
                installs[shard.index] = pool.submit(self._install, plan, shard)
        for shard_id in sorted(installs):
            installs[shard_id].result()
        _LOG.info("fleet up: %d shard(s) running %r", len(plan), self._topology.name)

    def _install(self, plan: ShardPlan, shard: Shard) -> None:
        """Wait for one shard, then hand it the chain and its share of its devices."""
        client = self._client_for(shard)
        if not client.wait_ready(self._ready_timeout_s):
            raise ServerStateError(
                f"shard {shard.index} did not answer within {self._ready_timeout_s:.0f}s on "
                f"{client.address}; its cameras would be dark and the rest of the fleet would "
                "report healthy, so the fleet is stopped instead"
            )
        installed = client.update_topology(
            self._chain_yaml,
            shared_by=plan.sharing_for(shard),
            share_rank=plan.rank_for(shard),
            timeout_s=self._topology_timeout_s,
        )
        _LOG.info(
            "shard %d installed %r (gpus %s, shared_by=%s)",
            shard.index,
            installed,
            list(shard.gpus),
            list(plan.sharing_for(shard)),
        )

    def _do_stop(self, timeout_s: float) -> None:
        """``Stop`` every shard on ONE deadline, then take the processes down.

        The deadline is shared rather than per shard — the rule the ingest manager and
        :meth:`Fleet.stop` both already follow: everyone is signalled at t0, so a shard still
        unfinished at the deadline is genuinely stuck, and charging the budget per shard would
        turn one wedged shard into sixteen consecutive waits.

        The RPC comes first and the SIGTERM second, on purpose. ``Stop`` is what lets a shard
        finish the frames it has and *say* how many camera threads it could not release; a
        launcher that only ever sent a signal would get the same shutdown with none of the
        information. The abandonment counts are **added** to :attr:`abandoned` rather than
        assigned, because while that number is non-zero a detached thread somewhere still
        references a shard's buffers (arch.md §2) — and this method is also the unwind
        :meth:`Runner.start` runs after a failed start, where an assignment would replace what
        the failed cycle abandoned with the zero an already-emptied client map reports.
        """
        self._abandoned += self._stop_shards(timeout_s)
        self._release()

    def _unwind_timeout_s(self) -> float:
        """A fleet's unwind is a conversation, so it gets the shutdown budget, not zero.

        :meth:`Runner.start` calls :meth:`_do_stop` to unwind a partial start, and for this
        runner that means a ``Stop`` RPC to every shard that did come up. Those shards have
        frames in flight and cameras to release whether or not their sibling ever answered, so
        a zero budget would abandon work an ordinary shutdown would have finished — and report
        it, which is the one thing worse than not doing it.
        """
        return self._settings.runner.drain_s

    def _stop_shards(self, timeout_s: float) -> int:
        """Ask every shard to stop within what is left of one budget. Never raises."""
        deadline = time.monotonic() + timeout_s
        abandoned = 0
        for shard_id, client in self._shards():
            remaining = max(0.5, deadline - time.monotonic())
            try:
                result = client.stop(remaining, deadline_s=remaining)
            except Exception as exc:
                # A shard that cannot be asked is a shard that gets SIGTERM in a moment. Not
                # raising here is what keeps the *other* shards' stops from being skipped.
                _LOG.warning("shard %d could not be stopped over RPC: %s", shard_id, exc)
                continue
            abandoned += result.abandoned
            if not result.clean:
                _LOG.warning(
                    "shard %d stopped with %d camera(s) abandoned%s",
                    shard_id,
                    result.abandoned,
                    f": {result.detail}" if result.detail else "",
                )
        return abandoned

    def _release(self) -> None:
        """Close every channel and take the processes down. Idempotent, never raises."""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:  # pragma: no cover - a channel close that fails is not news
                _LOG.debug("closing a shard channel failed", exc_info=True)
        fleet, self._fleet = self._fleet, None
        if fleet is not None:
            fleet.stop()

    def request_stop(self) -> None:
        """Record the stop, here and on the fleet. Safe from a signal handler; never blocks.

        Forwarded to :class:`Fleet` as well as recorded on the runner because
        :meth:`Fleet.supervise` watches the fleet's own flag, and a handler that set only the
        runner's would be noticed a poll later at best and not at all if the loop is inside
        its ``sleep``.
        """
        super().request_stop()
        fleet = self._fleet
        if fleet is not None:
            fleet.request_stop()

    def supervise(
        self, *, poll_s: float = 1.0, until: Callable[[], bool] | None = None
    ) -> None:
        """Block until a shard dies, ``until()`` says to stop, or this runner is stopped.

        Delegated to :meth:`Fleet.supervise`, which is where the argument lives: three shards
        up and one down is a deployment reporting healthy while a quarter of the cameras go
        unread, and a supervisor exists to refuse to sit in that state. Overrides the ABC's
        wait-until-told default, which is right for a runner whose workers are threads in this
        process and wrong for one whose workers are processes that can die on their own.

        Raises:
            ServerStateError: this runner is not running.
            ShardExitedError: a shard exited. The fleet is stopped before it is raised.
        """
        self._require_running()
        assert self._fleet is not None  # _require_running checked the clients, which imply it
        stop = self._stop_requested.is_set
        self._fleet.supervise(
            poll_s=poll_s,
            until=stop if until is None else (lambda: stop() or until()),
        )

    # -- submission ---------------------------------------------------------------------

    def _do_submit(self, item: ChainItem) -> ResponseFuture:
        """Refused, typed. A fleet has no chain of its own to admit an item into.

        Frames enter a shard through that shard's decode elements, on the GPU that will
        process them. Funnelling them through the parent would rebuild the single shared
        buffer the whole design exists to delete (ADR-005), one process higher up.
        """
        raise ServerStateError(
            f"the fleet runner does not execute items ({item.key}); frames enter a shard "
            "through its own decode element. Add the camera with add_camera(), or use the "
            "`inprocess` runner to walk a chain in this process"
        )

    # -- cameras ------------------------------------------------------------------------

    def add_camera(self, camera: CameraSpec) -> None:
        """Place one camera on the least-loaded shard (arch.md §2).

        Least-loaded means **fewest cameras**, ties broken by shard index so a fleet fills
        deterministically. Not by offered fps: a camera's real rate is what its source
        delivers, and pretending to know it before it has connected is how the previous
        system's balance was decided on a number nobody measured.

        A refusal is not a failure — a shard that is draining, or that already knows this
        camera, answers ``accepted=False`` with a reason, and the camera is offered to the
        next-least-loaded shard. Only when *every* shard has refused is this an error, and
        then the message carries what each of them said.

        RESERVE, ASK, COMMIT. :attr:`_lock` is taken three times for microseconds each and
        never across the ``AddCamera`` itself. That call starts a decoder on the shard and can
        sit for seconds on an RTSP source that is not answering; holding the lock across it
        would block :meth:`health` and :meth:`stats` for exactly as long, which is to say it
        would hang the one call an operator makes while wondering why a camera is dark. The
        reservation is what keeps the placement honest in the meantime: the camera is in
        :attr:`_placed` from the moment a shard is chosen, so a second placement counts it and
        picks a different shard, and it is in :attr:`_pending` until the shard says yes, so
        :meth:`health` reports "being placed" rather than "placed".

        Raises:
            ServerStateError: the fleet is not running.
            ConfigurationError: the camera is already placed, or no shard would take it.
        """
        self._require_running()
        with self._lock:
            if camera.camera_id in self._placed:
                raise ConfigurationError(
                    f"camera {camera.camera_id!r} is already on shard "
                    f"{self._placed[camera.camera_id]}; remove it before placing it again"
                )
            order = self._by_load()
            self._placed[camera.camera_id] = order[0]
            self._pending.add(camera.camera_id)
        refusals: list[str] = []
        try:
            for shard_id in order:
                with self._lock:
                    self._placed[camera.camera_id] = shard_id
                    client = self._clients[shard_id]
                result = client.add_camera(camera)
                with self._lock:
                    if result.accepted:
                        self._pending.discard(camera.camera_id)
                if result.accepted:
                    _LOG.info(
                        "camera %s placed on shard %d",
                        camera.camera_id,
                        shard_id,
                        extra=log_context(camera_id=camera.camera_id),
                    )
                    return
                refusals.append(f"shard {shard_id}: {result.reason}")
        finally:
            # The reservation outlives this method only when it was committed. Every other
            # way out - every shard refusing, an RPC that raised, a channel that died - drops
            # it, because a camera nobody accepted must be placeable again.
            with self._lock:
                if camera.camera_id in self._pending:
                    self._pending.discard(camera.camera_id)
                    self._placed.pop(camera.camera_id, None)
        raise ConfigurationError(
            f"no shard would take camera {camera.camera_id!r} ({'; '.join(refusals)})"
        )

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool:
        """Stop one camera on the shard that holds it.

        The placement is dropped whatever the shard says about the *thread*: ``clean=False``
        means the decoder was abandoned, not that the camera is still being served, and a
        launcher that kept the placement would refuse to ever put that camera anywhere again.
        It is dropped **before** the RPC rather than after, and that is the same decision said
        twice: the launcher stops believing in the placement the moment it decides to remove
        it, and what the shard answers describes the thread and not the placement. It also
        means the lock is not held across a call that waits ``timeout_s`` on a decoder.

        Returns:
            Whether the camera's thread stopped within the deadline.

        Raises:
            ConfigurationError: no shard holds this camera.
        """
        self._require_running()
        with self._lock:
            shard_id = self._placed.get(camera_id)
            if shard_id is None or camera_id in self._pending:
                raise ConfigurationError(
                    f"no shard holds camera {camera_id!r}; this fleet has "
                    f"{sorted(set(self._placed) - self._pending) or 'none'}"
                )
            client = self._clients[shard_id]
            self._placed.pop(camera_id, None)
        return client.remove_camera(camera_id, timeout_s=timeout_s)

    def drain(self, timeout_s: float = 20.0) -> int:
        """Ask every shard to stop reading, on one shared deadline. Sums what was abandoned.

        Returns:
            How many camera threads across the whole fleet had to be abandoned; ``0`` is the
            clean drain. Non-zero is a lifetime signal — somewhere a detached thread still
            references a shard's buffers.
        """
        self._require_running()
        deadline = time.monotonic() + timeout_s
        abandoned = 0
        for shard_id, client in self._shards():
            remaining = max(0.5, deadline - time.monotonic())
            try:
                abandoned += client.drain(remaining, deadline_s=remaining)
            except Exception as exc:
                _LOG.warning("shard %d could not be drained: %s", shard_id, exc)
        with self._lock:
            self._placed.clear()
            self._pending.clear()
        # Added, not assigned. The stop that follows a drain abandons its own threads, and
        # both sets are still detached: this is a lifetime signal, and a lifetime signal that
        # the next writer can zero is worse than none (see :attr:`_abandoned`).
        self._abandoned += abandoned
        return abandoned

    # -- observability -------------------------------------------------------------------

    def _do_health(self) -> dict[str, Any]:
        """One entry per shard: what it says, plus what this launcher placed on it.

        A shard that cannot be reached is reported as ``unreachable`` with the reason rather
        than omitted. An absent entry reads as "no such shard", and the whole point of a fleet
        health report is to name the shard that stopped answering.
        """
        with self._lock:
            clients = sorted(self._clients.items())
            placed = dict(self._placed)
            pending = set(self._pending)
        shards: dict[str, Any] = {}
        for shard_id, client in clients:
            entry: dict[str, Any] = {"placed": _placed_on(placed, pending, shard_id)}
            try:
                report = client.health()
            except Exception as exc:
                entry.update(state="unreachable", detail=f"{type(exc).__name__}: {exc}")
            else:
                entry.update(
                    state=report.state,
                    cameras=report.cameras,
                    engine=report.engine,
                    detail=report.detail,
                )
            shards[str(shard_id)] = entry
        return {
            "shards": shards,
            "cameras": {
                camera: (
                    {"shard": shard}
                    if camera not in pending
                    else {"shard": shard, "pending": True}
                )
                for camera, shard in sorted(placed.items())
            },
            "abandoned": self._abandoned,
        }

    def _do_stats(self) -> dict[str, Any]:
        with self._lock:
            clients = sorted(self._clients.items())
            cameras = len(self._placed) - len(self._pending)
        stats: dict[str, Any] = {"shards": {}, "cameras": cameras}
        for shard_id, client in clients:
            try:
                stats["shards"][str(shard_id)] = client.stats()
            except Exception as exc:
                stats["shards"][str(shard_id)] = {"detail": f"{type(exc).__name__}: {exc}"}
        return stats

    # -- helpers -------------------------------------------------------------------------

    def _build_plan(self) -> ShardPlan:
        """Who gets which GPUs. Pure, and printed before a process exists.

        The plan carries **no cameras**: they arrive over ``AddCamera`` and are placed one at
        a time (arch.md §2). What it decides is the device split and, with it, the sharing
        every shard is told — which is the number that keeps two shards on one GPU from each
        loading a full set of engines.
        """
        gpus = list(self._wanted_gpus or self._settings.devices.visible_gpus or [])
        if not gpus:
            # The driver is deliberately NOT asked here. `runners` imports no `runtime`
            # (check_layers.py), and the reason survives this runner: the parent process is
            # the one CPU-only process in the deployment, and `device_count()` initialises
            # CUDA in whatever process calls it. `cli/commands/run.py` resolves "every device
            # the driver reports" and passes the list in, the way `shipinfer fleet` did.
            raise ConfigurationError(
                "a fleet needs to know which GPUs it may use: none were passed and "
                "devices.visible_gpus is empty. Pass gpus=, or set devices.visible_gpus "
                "(`shipinfer run` fills it in from the driver)"
            )
        count = self._wanted_shards or self._settings.runner.shards or len(gpus)
        return plan_shards({}, shards=count, gpus=gpus)

    def _command_for(self, shard: Shard) -> Sequence[str]:
        if self._command is not None:
            return self._command(shard)
        return self.shard_command(shard.index, self.control_port(shard.index))

    def _client_for(self, shard: Shard) -> ShardClient:
        """The channel to one shard, made once. Called from the install pool, so: locked."""
        port = self.control_port(shard.index)
        made = (
            ShardClient(port, shard_id=shard.index)
            if self._client_factory is None
            else self._client_factory(shard, port)
        )
        with self._lock:
            # Registered before it is used, so a failed install still leaves a channel the
            # unwind's `Stop` can reach - and closes. Construction happens outside the lock:
            # a real `ShardClient` builds a gRPC channel, and `health()` may not wait on that.
            return self._clients.setdefault(shard.index, made)

    def _shards(self) -> list[tuple[int, ShardClient]]:
        """Every (id, client) pair, in id order, snapshotted under :attr:`_lock`."""
        with self._lock:
            return sorted(self._clients.items())

    def _by_load(self) -> list[int]:
        """Shard ids, least-loaded first, ties on the id. Call under :attr:`_lock`.

        Reservations count. A camera being placed right now is one this shard is about to
        hold, and ignoring it would send a second concurrent placement to the same "emptiest"
        shard — which is the balance failure ADR-005 exists to prevent, one level up.
        """
        counts = dict.fromkeys(self._clients, 0)
        for shard_id in self._placed.values():
            counts[shard_id] = counts.get(shard_id, 0) + 1
        return sorted(counts, key=lambda shard_id: (counts[shard_id], shard_id))

    def _require_running(self) -> None:
        if not self.is_running or not self._clients:
            raise ServerStateError(
                f"the fleet is not running; call start() before managing cameras "
                f"({len(self._clients)} shard(s) connected)"
            )

    def describe_plan(self) -> str:
        """The plan as an operator reads it, computed if nothing has been spawned yet.

        Planning is pure — who gets which GPUs, decided before a process exists — so a dry run
        is the same computation the start does and not a second description of it. Which
        camera lands on which GPU is stable across restarts, and that is worth looking at
        before fifty of them start reconnecting.
        """
        return (self._plan or self._build_plan()).describe()

    def __repr__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return (
            f"<FleetRunner {len(self._clients)} shard(s) topology="
            f"{self._topology.name or '<unnamed>'} {state}>"
        )


def _placed_on(placed: dict[str, int], pending: set[str], shard_id: int) -> list[str]:
    """The cameras this launcher has *confirmed* on one shard, out of a snapshot.

    A reservation is left out on purpose: ``placed`` under a shard's entry is what the
    launcher believes is running there, and a camera the shard has not accepted yet is not.
    The whole camera map beside it says ``pending`` for those, so nothing is hidden.
    """
    return sorted(c for c, s in placed.items() if s == shard_id and c not in pending)
