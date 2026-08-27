"""The launcher's half of the shard control plane (arch.md section 2).

One :class:`ShardClient` per shard process, held by the parent for the child's whole life.
Everything the old argv-and-environment mechanism could only say **once, at exec time** —
camera set, topology, device sharing — is a method here, and can therefore be said again to
a shard that is already serving. That is the entire argument for the change (V140): add,
remove and drain a camera on a live shard; read *ready* vs *running* vs *draining* as a typed
answer instead of inferring it from an exit code.

**grpcio is an optional extra and this module keeps it that way.** Nothing here imports
``grpc`` at module scope: the launcher runs in the one CPU-only process of the deployment and
``import shipinfer.launch`` must work on a host that installed neither grpcio nor protobuf —
``tests/launch/test_client_without_grpcio.py`` is what pins that. The refusal, when the first
call is made, is the same shape ``api/app.py`` uses for FastAPI: a
:class:`~shipinfer.core.errors.ConfigurationError` naming the extra to install, rather than an
``ImportError`` from four frames down.

**A refused camera is not a failed call.** ``AddCamera`` answers ``accepted=False`` with a
reason for every condition under which the shard is healthy and the camera is not running —
a duplicate id, a shard that is stopping, a fleet that forgot the camera mid-add. Those are
placement facts the launcher acts on (put it on another shard); a gRPC error status would
flatten them into "the call failed" alongside a dead process, which is a different action.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from shipinfer.core.errors import ConfigurationError, ServerStateError
from shipinfer.core.logging import get_logger
from shipinfer.launch.control import (
    AddCameraResult,
    CameraSpec,
    ShardHealth,
    ShardIdentity,
    StopResult,
)

__all__ = ["ShardClient"]

_LOG = get_logger("launch.client")

#: What to tell an operator who has not installed the extra. One string, so the message in
#: the refusal and the message a test asserts on cannot drift.
_MISSING_GRPCIO = 'the gRPC control plane needs grpcio: pip install "shipinfer[grpc]"'

#: What a lazy import of the generated stubs can fail with. ``ImportError`` is the extra
#: being absent; ``RuntimeError`` is it being present and too old — protoc's output compares
#: ``grpc.__version__`` against its own ``GRPC_GENERATED_VERSION`` at import and raises a
#: bare ``RuntimeError`` below it, and ``shard_pb2`` does the same for the protobuf runtime.
#: The floors in ``pyproject.toml`` are set so a supported install never sees the second
#: (``tests/launch/test_generated_floor.py``); catching it anyway means a floor that drifts
#: reaches an operator as a typed refusal naming the extra rather than as a raw traceback.
_UNUSABLE_GRPCIO = (ImportError, RuntimeError)

#: How long to sleep between the first two `Ready` polls, and the cap it backs off to. A
#: freshly spawned interpreter takes O(1s) to import and bind, so polling every 50 ms wastes
#: little and answers fast; a child that is going to take two minutes should not be polled
#: 2400 times, hence the cap.
_POLL_MIN_S = 0.05
_POLL_MAX_S = 1.0


class ShardClient:
    """A launcher's connection to one shard process.

    Args:
        control_port: the port the shard's gRPC server is bound to.
        host: where it is bound. Loopback by default and on purpose — this control plane is
            unauthenticated, a shard lives on the same node as its launcher (arch.md
            section 2), and a default of ``0.0.0.0`` would expose ``Stop`` to the network.
        shard_id: which shard this is. Carried for log lines and for the identity the parent
            already knows before the child has told it anything.
        timeout_s: the default per-call deadline. Every method takes one so a caller can
            widen it; none of them blocks forever, because a launcher that hangs on a wedged
            child cannot take the rest of the fleet down.

    Not thread-safe for :meth:`close` racing a call — a gRPC channel itself is, so concurrent
    RPCs are fine, but a supervisor thread closing while a health probe is in flight is a
    use-after-close. The launcher owns one client per shard on one thread; keep it that way.
    """

    def __init__(
        self,
        control_port: int,
        *,
        host: str = "127.0.0.1",
        shard_id: int = 0,
        timeout_s: float = 10.0,
    ) -> None:
        self._host = host
        self._control_port = control_port
        self._shard_id = shard_id
        self._timeout_s = timeout_s
        self._channel: Any = None
        self._stub: Any = None
        self._identity: ShardIdentity | None = None

    # -- what the launcher already knows -------------------------------------------------

    @property
    def address(self) -> str:
        return f"{self._host}:{self._control_port}"

    @property
    def shard_id(self) -> int:
        return self._shard_id

    @property
    def identity(self) -> ShardIdentity | None:
        """Who the shard said it is, or ``None`` before it has answered a :meth:`Ready`.

        Worth reading after :meth:`wait_ready`: it carries the child's **pid** and the port
        it actually bound, which the parent cannot know from the spawn alone when the shard
        was told to pick an ephemeral one.
        """
        return self._identity

    # -- the connection ------------------------------------------------------------------

    def _rpc(self) -> Any:
        """The stub, connecting on first use.

        Lazy for two reasons, both load-bearing. ``grpc`` is an optional extra, so importing
        it at module scope would make ``import shipinfer.launch`` fail on a host without it;
        and a client is constructed by the launcher *before* the child it addresses has
        bound its port, so connecting in ``__init__`` would connect to nothing.

        Raises:
            ConfigurationError: grpcio is not installed, or is older than the committed
                stubs were generated against (see :data:`_UNUSABLE_GRPCIO`).
        """
        if self._stub is not None:
            return self._stub
        try:
            import grpc

            from shipinfer.launch.proto import shard_pb2_grpc
        except _UNUSABLE_GRPCIO as exc:
            raise ConfigurationError(
                f"{_MISSING_GRPCIO} ({type(exc).__name__}: {exc})"
            ) from exc

        self._channel = grpc.insecure_channel(self.address)
        # protoc emits no annotations for the grpc stub, and mypy is strict here.
        self._stub = shard_pb2_grpc.ShardStub(self._channel)  # type: ignore[no-untyped-call]
        return self._stub

    @staticmethod
    def _pb() -> Any:
        """The generated messages, imported here for the same reason :meth:`_rpc` is lazy."""
        try:
            from shipinfer.launch.proto import shard_pb2
        except _UNUSABLE_GRPCIO as exc:  # protobuf rides with grpcio, and has its own floor
            raise ConfigurationError(
                f"{_MISSING_GRPCIO} ({type(exc).__name__}: {exc})"
            ) from exc
        return shard_pb2

    def _call(self, name: str, request: Any, timeout_s: float | None) -> Any:
        """One RPC, with the shard named in whatever goes wrong.

        Raises:
            ServerStateError: the shard did not answer — it is still starting, it died, or it
                is wedged. Typed rather than a bare ``grpc.RpcError`` because every caller
                above this is launcher code that must decide whether to respawn, and a
                library exception carrying a channel state does not help it decide.
        """
        stub = self._rpc()
        import grpc

        try:
            return getattr(stub, name)(request, timeout=self._deadline(timeout_s))
        except grpc.RpcError as exc:
            # `grpc.RpcError` is not required to be a `Call`, so `code`/`details` may be
            # absent; the string form is the fallback rather than an AttributeError on the
            # error path, which is the worst place to raise a second exception.
            code = exc.code() if hasattr(exc, "code") else None
            details = exc.details() if hasattr(exc, "details") else str(exc)
            raise ServerStateError(
                f"shard {self._shard_id} at {self.address} did not answer {name}: "
                f"{code} {details}"
            ) from exc

    def _deadline(self, timeout_s: float | None) -> float:
        return self._timeout_s if timeout_s is None else timeout_s

    def close(self) -> None:
        """Release the channel. Idempotent, and safe before the first call."""
        channel, self._channel, self._stub = self._channel, None, None
        if channel is not None:
            channel.close()

    def __enter__(self) -> ShardClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the surface ---------------------------------------------------------------------

    def wait_ready(self, timeout_s: float = 120.0, *, poll_s: float = _POLL_MIN_S) -> bool:
        """Poll ``Ready`` until the shard answers, or the deadline passes.

        **The parent polls; the child does not call back.** A child that announced itself
        would need to know the parent's address, which is one more thing in argv, and a
        crash between ``exec`` and the announcement would hang the parent forever with no
        deadline to hit.

        Returns:
            Whether the shard answered in time. ``False`` rather than an exception because
            the caller's action is to kill the child and either respawn or fail the fleet —
            it is not an error condition to be propagated but a decision to be made, and
            :attr:`identity` is populated on success for the caller that needs the pid.

        The deadline is generous by default: a shard imports torch and deserialises engines
        before it binds, which is tens of seconds on a cold page cache.

        ``timeout_s`` is a **budget for the whole call**, and each poll's RPC deadline is
        clipped to what is left of it. Without the clip a caller that asked for 0.3 s could
        wait several seconds: the per-poll deadline grows to 5 s as the backoff does, and one
        poll that hangs (a bound-but-not-answering port, which is what a wedged child looks
        like) burns its own deadline in full before the budget is even consulted.
        """
        deadline = time.monotonic() + timeout_s
        sleep_s = max(poll_s, 0.001)
        while True:
            try:
                poll_deadline = min(sleep_s * 4, 5.0, max(0.05, deadline - time.monotonic()))
                reply = self._call("Ready", self._pb().ReadyRequest(), poll_deadline)
            except ServerStateError:
                if time.monotonic() >= deadline:
                    _LOG.warning(
                        "shard %d at %s was not ready within %.1fs",
                        self._shard_id,
                        self.address,
                        timeout_s,
                    )
                    return False
                time.sleep(min(sleep_s, max(0.0, deadline - time.monotonic())))
                sleep_s = min(sleep_s * 2, _POLL_MAX_S)
                continue
            self._identity = ShardIdentity.from_pb(reply.identity)
            _LOG.info(
                "shard %d ready at %s (pid %d, state %s)",
                self._shard_id,
                self.address,
                self._identity.pid,
                reply.state,
            )
            return True

    def update_topology(
        self,
        chain_yaml: str,
        *,
        shared_by: Sequence[int] = (),
        share_rank: Sequence[int] = (),
        timeout_s: float | None = None,
    ) -> str:
        """Install the chain this shard runs, and the device sharing it must honour.

        Args:
            chain_yaml: the chain file's **text**, not a parsed object. The loader is the
                single door through which a chain becomes trustworthy (ADR-017) and it lives
                on the shard; a pre-parsed graph on the wire would be a second parser. It is
                also why this parameter is a ``str`` and not a ``ChainSpec``: ``launch`` sits
                on ``core`` and ``scheduling`` only, so that the launcher pays for neither the
                topology package nor anything it might one day import.
            shared_by: how many shard processes share each of this shard's devices, aligned
                with its logical ordinals.
            share_rank: this shard's rank among them, same alignment.

        The two sharing lists are not optional decoration. Two shards on one GPU must each
        load **half** the configured instances; a shard that never hears about its co-tenant
        loads the full count and the device silently holds twice the engines for the same
        throughput.

        Returns:
            The name of the chain the shard says it installed — echoed so a log names what
            the shard understood rather than what the launcher believes it sent.

        Raises:
            ConfigurationError: the shard refused the chain, with its reason. A refusal here
                is fatal to this shard (it has nothing to run), which is why this one *is* an
                exception where ``AddCamera``'s refusal is a field.
            ServerStateError: the shard did not answer.
        """
        request = self._pb().TopologyRequest(
            chain_yaml=chain_yaml, shared_by=list(shared_by), share_rank=list(share_rank)
        )
        reply = self._call("UpdateTopology", request, timeout_s)
        if not reply.accepted:
            raise ConfigurationError(
                f"shard {self._shard_id} refused the topology: {reply.reason}"
            )
        return str(reply.topology)

    def add_camera(
        self, camera: CameraSpec, *, timeout_s: float | None = None
    ) -> AddCameraResult:
        """Ask this shard to start one camera. A refusal is the result, not an exception."""
        request = self._pb().AddCameraRequest(camera=camera.to_pb())
        return AddCameraResult.from_pb(self._call("AddCamera", request, timeout_s))

    def remove_camera(
        self, camera_id: str, *, timeout_s: float = 5.0, deadline_s: float | None = None
    ) -> bool:
        """Stop and forget one camera.

        Args:
            timeout_s: the shard's deadline for stopping this camera's thread.
            deadline_s: the RPC deadline. Defaults to ``timeout_s`` plus a margin, because an
                RPC that gives up before the shard has finished stopping would report a
                failure for a removal that then succeeds.

        Returns:
            Whether the camera's thread stopped cleanly. ``False`` means it was abandoned —
            the caller's to know, not the log's to bury (``ingest/manager.py``).

        Raises:
            ConfigurationError: no such camera on this shard. A typo in an operator's call
                deserves an answer rather than a silent no-op.
        """
        request = self._pb().RemoveCameraRequest(camera_id=camera_id, timeout_s=timeout_s)
        reply = self._call(
            "RemoveCamera", request, timeout_s + 5.0 if deadline_s is None else deadline_s
        )
        if not reply.removed:
            raise ConfigurationError(
                f"shard {self._shard_id} does not run camera {camera_id!r}: {reply.reason}"
            )
        return bool(reply.clean)

    def health(self, *, timeout_s: float | None = None) -> ShardHealth:
        """What state this shard is in, from a snapshot safe against a concurrent removal."""
        return ShardHealth.from_pb(self._call("Health", self._pb().HealthRequest(), timeout_s))

    def stats(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Counters an operator would page on."""
        from google.protobuf import json_format

        reply = self._call("Stats", self._pb().StatsRequest(), timeout_s)
        stats: dict[str, Any] = json_format.MessageToDict(reply.stats)
        if reply.detail:
            stats["detail"] = reply.detail
        return stats

    def drain(self, timeout_s: float = 20.0, *, deadline_s: float | None = None) -> int:
        """Stop reading cameras; let what is in flight finish.

        Returns:
            How many camera threads had to be abandoned. ``0`` is the clean drain.
        """
        request = self._pb().DrainRequest(timeout_s=timeout_s)
        reply = self._call(
            "Drain", request, timeout_s + 5.0 if deadline_s is None else deadline_s
        )
        return int(reply.abandoned)

    def stop(self, timeout_s: float = 20.0, *, deadline_s: float | None = None) -> StopResult:
        """Drain, then stop the shard's executor. Idempotent: a second call reports 0.

        ``timeout_s`` is ONE deadline for the whole camera set, not one per camera — the rule
        the ingest manager already follows, restated on the wire. The returned
        :attr:`~shipinfer.launch.control.StopResult.abandoned` is a lifetime signal: while it
        is non-zero a detached thread still references the shard's buffers.
        """
        request = self._pb().StopRequest(timeout_s=timeout_s)
        reply = self._call(
            "Stop", request, timeout_s + 5.0 if deadline_s is None else deadline_s
        )
        return StopResult.from_pb(reply)

    def __repr__(self) -> str:
        connected = "connected" if self._stub is not None else "idle"
        return f"<ShardClient shard={self._shard_id} {self.address} {connected}>"
