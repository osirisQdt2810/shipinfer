"""The shard's half of the control plane: a servicer over a :class:`Runner`.

The parent's half is :mod:`shipinfer.launch.client`; this is what answers it. It lives in
``runners/`` and not in ``launch/`` because of what it holds: a runner. A launcher that
imported the thing it launches would pay for the executor in the parent process and there
would be no reason left for the shard to be a separate process — the dependency goes
``runners`` -> ``launch``, never back (``scripts/hooks/check_layers.py``).

**It is thin on purpose.** Every invariant these RPCs expose already exists and was hardened
somewhere else: the fleet-deadline stop that returns an abandonment count, the insert →
start → re-check that refuses a camera the fleet forgot mid-add, the health snapshot that is
safe against a concurrent removal (``ingest/manager.py``, #33-#41). This class translates
between those and protobuf and does nothing else. A servicer that re-derived any of them
would buy back four review rounds of concurrency fixes.

**No traceback ever reaches the wire.** Every method catches, logs and answers — a refusal in
``accepted``/``reason``, a failure in ``detail`` — because the caller is a supervisor
deciding whether to respawn a process, and "the call failed with UNKNOWN" and "the camera is
a duplicate" are the same status code and completely different decisions.

**Why this is not a subclass of the generated** ``ShardServicer``. Subclassing needs
``shard_pb2_grpc`` at class-definition time, and that module imports ``grpc`` on its first
line — which would make ``import shipinfer.runners`` fail wherever the optional extra is not
installed. ``add_ShardServicer_to_server`` binds handlers by attribute name and does not
check the type, so a duck-typed servicer is exactly as good and costs no import. The method
names are therefore protobuf's CamelCase; the ``N802`` ignore for this file in
``pyproject.toml`` is that, and says so.
"""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from typing import Any

from shipinfer.core.errors import ConfigurationError, ShipInferError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.launch.control import CameraSpec, ShardHealth, ShardIdentity, ShardState
from shipinfer.runners.base import Runner
from shipinfer.topology import ChainSpec

__all__ = ["ShardServer", "ShardService", "serve_shard"]

_LOG = get_logger("runners.service")

#: Enough threads to answer a health probe while a Stop is draining, and few enough that a
#: wedged RPC cannot spawn a pool. Every handler here is either instant or bounded by the
#: deadline its request carries.
_DEFAULT_MAX_WORKERS = 8


class ShardService:
    """The eight RPCs of ``shard.proto``, over one runner.

    Args:
        runner: what actually executes this shard's chain. Handed in rather than built here:
            the process entry point owns construction, and a test drives this class with
            whatever runner makes its property visible.
        identity: who this shard is. ``control_port`` must be the port actually bound, which
            is why :func:`serve_shard` fills it in after binding rather than before.

    The state machine is derived, not stored — apart from the two facts nothing else can
    know, which is whether ``Drain`` or ``Stop`` has been called. Deriving it means a runner
    that dies on its own cannot leave this class asserting ``running``.
    """

    def __init__(self, runner: Runner, identity: ShardIdentity) -> None:
        self._runner = runner
        self._identity = identity
        self._lock = threading.Lock()
        self._draining = False
        self._stopped = False
        #: What the last accepted ``UpdateTopology`` carried. Read by the process entry point
        #: (A2 PR-6's ``launch/shard.py``), which is what builds a runner from it; kept here
        #: because this is the object that received it and the only one that can say a later
        #: ``UpdateTopology`` superseded it.
        self._chain_yaml = ""
        self._shared_by: tuple[int, ...] = ()
        self._share_rank: tuple[int, ...] = ()

    # -- what the shard knows about itself -----------------------------------------------

    @property
    def runner(self) -> Runner:
        return self._runner

    @property
    def identity(self) -> ShardIdentity:
        return self._identity

    @property
    def chain_yaml(self) -> str:
        """The chain text of the last accepted ``UpdateTopology``; empty before the first."""
        return self._chain_yaml

    @property
    def shared_by(self) -> tuple[int, ...]:
        """How many shard processes share each of this shard's devices.

        Not decoration: two shards on one GPU must each load HALF the configured instances,
        or the device holds twice the engines for the same throughput and nothing says so.
        It used to ride in the child's environment; it rides in ``TopologyRequest`` now.
        """
        return self._shared_by

    @property
    def share_rank(self) -> tuple[int, ...]:
        """This shard's rank among the processes sharing each device."""
        return self._share_rank

    def state(self) -> ShardState:
        """The shard's state, derived from the runner rather than remembered."""
        if self._stopped:
            return ShardState.STOPPED
        if self._draining:
            return ShardState.DRAINING
        if not self._runner.is_running:
            return ShardState.STARTING
        return ShardState.READY if not self._cameras() else ShardState.RUNNING

    def _cameras(self) -> dict[str, Any]:
        report = self._runner.health()
        cameras = report.get("cameras", {})
        return cameras if isinstance(cameras, dict) else {}

    # -- the RPCs ------------------------------------------------------------------------

    def Ready(self, request: Any, context: Any = None) -> Any:
        """The reply *arriving* is the readiness signal; ``state`` says what was found.

        There is no ``ready`` boolean because a field that is true whenever it can be read is
        not information. A process that answers has finished importing and bound its port,
        which is what the parent is waiting to learn; whether it can yet take a camera is
        ``state``.
        """
        pb = _pb()
        return pb.ReadyReply(identity=self._identity.to_pb(), state=str(self._safe_state()))

    def UpdateTopology(self, request: Any, context: Any = None) -> Any:
        """Install the chain this shard runs, and start executing it.

        Three refusals, all typed, all ``accepted=False``:

        * the document does not parse — refused here, before any camera, with the loader's
          own message. This is the one that earns the RPC its keep: a mistyped chain fails
          the deploy instead of producing a shard that runs nothing;
        * the shard is already running — a live topology swap would have to close nine
          stateful elements under in-flight frames, and that is not this PR's problem;
        * the chain named is not the one this runner holds. A runner is *given* its topology
          at construction (``runners/base.py``), so accepting a different name would start
          the wrong chain and report the right one.

        On acceptance the sharing lists are recorded (see :attr:`shared_by`) and the runner
        is started — installing a topology is precisely what makes a shard usable, and a
        parent that had to send a separate "now go" RPC would have a two-step handshake with
        a failure state in the middle.
        """
        pb = _pb()
        try:
            spec = ChainSpec.from_yaml(
                request.chain_yaml, source=f"shard {self._identity.shard_id} UpdateTopology"
            )
        except ShipInferError as exc:
            return pb.TopologyReply(accepted=False, reason=str(exc))

        with self._lock:
            if self._runner.is_running:
                return pb.TopologyReply(
                    accepted=False,
                    reason=(
                        f"shard {self._identity.shard_id} is already running topology "
                        f"{self._runner.topology.name!r}; stop it before installing another"
                    ),
                )
            held = self._runner.topology.name
            if spec.name and held and spec.name != held:
                return pb.TopologyReply(
                    accepted=False,
                    reason=(
                        f"shard {self._identity.shard_id} was built for topology {held!r} "
                        f"and was sent {spec.name!r}"
                    ),
                )
            try:
                self._runner.start()
            # A start failure is an answer, not a crash: the launcher has to be told
            # which shard could not open its chain, and told it as data.
            except Exception as exc:
                _LOG.exception("shard %d could not start", self._identity.shard_id)
                return pb.TopologyReply(accepted=False, reason=f"{type(exc).__name__}: {exc}")
            self._chain_yaml = request.chain_yaml
            self._shared_by = tuple(request.shared_by)
            self._share_rank = tuple(request.share_rank)
            self._stopped = False
            self._draining = False

        _LOG.info(
            "shard %d installed topology %r (shared_by=%s share_rank=%s)",
            self._identity.shard_id,
            held or spec.name,
            list(self._shared_by),
            list(self._share_rank),
        )
        return pb.TopologyReply(accepted=True, topology=held or spec.name)

    def AddCamera(self, request: Any, context: Any = None) -> Any:
        """Start one camera. Every refusal is an answer.

        The runner's typed vocabulary maps straight through: ``ConfigurationError`` is a
        duplicate id or an unknown source, ``ServerStateError`` is a shard that is stopping
        or a fleet that forgot the camera between the insert and the start
        (``ingest/manager.py``). Both mean the same thing to a launcher — this camera is not
        running here, place it elsewhere — and both must reach it as data, because a gRPC
        error status would put them in the same bucket as a dead process.
        """
        pb = _pb()
        camera = CameraSpec.from_pb(request.camera)
        try:
            self._runner.add_camera(camera)
        except ShipInferError as exc:
            _LOG.info(
                "shard %d refused camera %s: %s",
                self._identity.shard_id,
                camera.camera_id,
                exc,
                extra=log_context(camera_id=camera.camera_id),
            )
            return pb.AddCameraReply(accepted=False, reason=str(exc))
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception(
                "shard %d failed adding camera %s",
                self._identity.shard_id,
                camera.camera_id,
            )
            return pb.AddCameraReply(accepted=False, reason=f"{type(exc).__name__}: {exc}")
        return pb.AddCameraReply(accepted=True)

    def RemoveCamera(self, request: Any, context: Any = None) -> Any:
        """Stop and forget one camera. ``clean=False`` means its thread was abandoned."""
        pb = _pb()
        try:
            clean = self._runner.remove_camera(request.camera_id, timeout_s=request.timeout_s)
        except ShipInferError as exc:
            return pb.RemoveCameraReply(removed=False, clean=False, reason=str(exc))
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception(
                "shard %d failed removing camera %s", self._identity.shard_id, request.camera_id
            )
            return pb.RemoveCameraReply(
                removed=False, clean=False, reason=f"{type(exc).__name__}: {exc}"
            )
        return pb.RemoveCameraReply(removed=True, clean=bool(clean))

    def Health(self, request: Any, context: Any = None) -> Any:
        """What this shard is doing, from one snapshot.

        One call to ``runner.health()``, split rather than two calls merged: asking twice
        would let a camera be removed between the two halves and produce a report that
        contradicts itself.
        """
        try:
            report = self._runner.health()
            cameras = report.pop("cameras", {})
            health = ShardHealth(
                state=str(self.state()),
                engine=report,
                cameras=cameras if isinstance(cameras, dict) else {},
                vram_budget={},
            )
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception("shard %d could not report health", self._identity.shard_id)
            health = ShardHealth(
                state=str(ShardState.UNKNOWN), detail=f"{type(exc).__name__}: {exc}"
            )
        return health.to_pb()

    def Stats(self, request: Any, context: Any = None) -> Any:
        """Counters an operator would page on, as the runner reports them."""
        pb = _pb()
        reply = pb.StatsReply()
        try:
            reply.stats.update(self._runner.stats())
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception("shard %d could not report stats", self._identity.shard_id)
            reply.detail = f"{type(exc).__name__}: {exc}"
        return reply

    def Drain(self, request: Any, context: Any = None) -> Any:
        """Stop reading cameras; let what is in flight finish."""
        pb = _pb()
        self._draining = True
        abandoned, detail = self._drain(request.timeout_s)
        return pb.DrainReply(abandoned=abandoned, detail=detail)

    def Stop(self, request: Any, context: Any = None) -> Any:
        """Drain, then stop the executor. Idempotent — a second Stop reports 0 abandoned.

        ``timeout_s`` is ONE deadline for the whole shutdown, charged to the camera set and
        the executor together, because everything is signalled at t0 and a worker still
        unfinished at the deadline is genuinely stuck (``ingest/manager.py``,
        ``runners/base.py``). The returned count is a lifetime signal, not a statistic.
        """
        pb = _pb()
        with self._lock:
            self._draining = True
            abandoned, detail = self._drain(request.timeout_s)
            try:
                self._runner.stop(timeout_s=request.timeout_s)
            except Exception as exc:  # see the module docstring: no traceback reaches the wire
                _LOG.exception("shard %d could not stop its runner", self._identity.shard_id)
                detail = f"{detail}; {type(exc).__name__}: {exc}".lstrip("; ")
            self._draining = False
            self._stopped = True
        _LOG.info(
            "shard %d stopped%s",
            self._identity.shard_id,
            f", {abandoned} camera(s) abandoned" if abandoned else "",
        )
        return pb.StopReply(abandoned=abandoned, detail=detail)

    # -- helpers -------------------------------------------------------------------------

    def _drain(self, timeout_s: float) -> tuple[int, str]:
        """Ask the runner to release its cameras, and say how many it could not.

        A runner that manages no cameras refuses with ``ServerStateError`` — that is the
        documented default on :class:`~shipinfer.runners.base.Runner` — and it abandons
        exactly none of them, so ``0`` is the honest count rather than a swallowed error.
        Any *other* failure lands in ``detail``, where an abandonment count of zero cannot
        be mistaken for a clean shutdown.
        """
        try:
            return int(self._runner.drain(timeout_s)), ""
        except ShipInferError as exc:
            if not self._runner.manages_cameras:
                return 0, ""
            return 0, str(exc)
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception("shard %d could not drain", self._identity.shard_id)
            return 0, f"{type(exc).__name__}: {exc}"

    def _safe_state(self) -> ShardState:
        try:
            return self.state()
        except Exception:  # Ready must answer even from a runner that cannot say
            _LOG.exception("shard %d could not determine its state", self._identity.shard_id)
            return ShardState.UNKNOWN

    def __repr__(self) -> str:
        return (
            f"<ShardService shard={self._identity.shard_id} "
            f"port={self._identity.control_port} {self.state()}>"
        )


@dataclass(frozen=True, slots=True)
class ShardServer:
    """A bound, started gRPC server and the identity it is serving.

    Returned rather than just the grpc server because the port is the interesting part: a
    shard told ``--control-port 0`` picks an ephemeral one, and ``identity.control_port`` is
    where the real number is — the same number ``Ready`` reports to the parent.
    """

    server: Any
    identity: ShardIdentity
    service: ShardService

    def stop(self, grace_s: float = 5.0) -> None:
        """Stop serving. Idempotent; does **not** stop the runner — ``Stop`` does that."""
        with contextlib.suppress(Exception):
            self.server.stop(grace_s).wait(grace_s + 1.0)

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        """Block until the server stops. What a shard process's ``main`` sits on."""
        return bool(self.server.wait_for_termination(timeout))


def serve_shard(
    runner: Runner,
    *,
    shard_id: int,
    control_port: int,
    host: str = "127.0.0.1",
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> ShardServer:
    """Bind the control plane for one shard and start serving. Does not start the runner.

    Args:
        runner: what executes this shard's chain.
        shard_id: which shard this is.
        control_port: the port to bind. ``0`` picks an ephemeral one, and the chosen number
            comes back in the returned identity.
        host: loopback by default and on purpose — this control plane is unauthenticated and
            a shard lives on the same node as its launcher (arch.md section 2). Binding
            ``0.0.0.0`` by default would expose ``Stop`` to the network.
        max_workers: handler threads.

    The runner is deliberately **not** started here: ``UpdateTopology`` starts it, so a shard
    that binds its port before it has been told what to run reports ``starting`` and refuses
    cameras, rather than silently running an empty chain.

    Raises:
        ConfigurationError: grpcio is not installed, or the port is already bound. The port
            check is why this function exists rather than four lines at a call site — the
            refusal has to happen **before any camera is opened**, or a shard that cannot be
            controlled is already holding decoder threads and a CUDA context that nothing
            can now tell it to release.
    """
    from concurrent import futures

    try:
        import grpc

        from shipinfer.launch.proto import shard_pb2_grpc
    except ImportError as exc:
        raise ConfigurationError(
            'the gRPC control plane needs grpcio: pip install "shipinfer[grpc]"'
        ) from exc

    # `grpc.so_reuseport` is ON by default in grpc-python, and it is wrong for this server:
    # with it, a second shard handed a port an earlier run still holds binds SUCCESSFULLY,
    # the kernel load-balances connections between the two, and the launcher's `AddCamera`
    # reaches whichever process the kernel felt like. The option exists for a pre-fork server
    # sharing one port on purpose; a shard is one process owning one port, so the refusal
    # below only means anything with reuseport off.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[("grpc.so_reuseport", 0)],
    )

    # Bound BEFORE the service is built, so the identity can carry the port that was actually
    # taken rather than the one that was asked for. A shard whose port is held by an earlier
    # run is a shard the launcher will never reach, and finding that out at the first
    # AddCamera is finding it out after the decoder threads are already open.
    #
    # Both failure shapes are handled because grpcio has used both: older versions return 0
    # from `add_insecure_port`, current ones raise `RuntimeError` with the address in the
    # message. Either way what an operator needs is which shard, which port, and why —
    # not a grpc string that names neither.
    try:
        bound = server.add_insecure_port(f"{host}:{control_port}")
    except RuntimeError as exc:
        bound = 0
        detail: Exception | None = exc
    else:
        detail = None
    if bound == 0:
        with contextlib.suppress(Exception):
            server.stop(0)
        raise ConfigurationError(
            f"shard {shard_id} could not bind its control port {host}:{control_port}; "
            "another shard or an earlier run still holds it"
        ) from detail

    identity = ShardIdentity(shard_id=shard_id, control_port=bound, pid=os.getpid())
    service = ShardService(runner, identity)
    # No annotations on protoc's output, and no `isinstance` inside it either: this is
    # what binds the duck-typed servicer above by attribute name.
    shard_pb2_grpc.add_ShardServicer_to_server(service, server)  # type: ignore[no-untyped-call]
    server.start()
    _LOG.info("shard %d control plane on %s:%d", shard_id, host, bound)
    return ShardServer(server=server, identity=identity, service=service)


def _pb() -> Any:
    """The generated messages. Imported per call, never at module scope.

    ``shard_pb2`` needs ``protobuf`` and ``shard_pb2_grpc`` needs ``grpc``; both are the
    optional ``grpc`` extra, and ``import shipinfer.runners`` must not require either — the
    in-process runner is the one a laptop uses.
    """
    from shipinfer.launch.proto import shard_pb2

    return shard_pb2
