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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from shipinfer.core.errors import ConfigurationError, ShipInferError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.launch.control import CameraSpec, ShardHealth, ShardIdentity, ShardState
from shipinfer.launch.proto import load_grpc, load_pb
from shipinfer.runners.base import Runner
from shipinfer.topology import ChainSpec

__all__ = ["RunnerFactory", "ShardServer", "ShardService", "serve_shard"]

#: How a shard builds its runner once ``UpdateTopology`` has told it what to run.
#:
#: ``(chain, shared_by, share_rank) -> Runner``, unstarted. It exists because a shard is
#: spawned with **two flags** and nothing else (arch.md section 2): it has no chain when it
#: binds its port, and ``Topology.from_spec`` refuses an empty one, so there is nothing to
#: construct a runner over until the first RPC arrives. The two sharing lists are arguments
#: rather than something the servicer applies, because what they configure is the *engine* a
#: shard's runner is handed — a factory can put them in the settings tree before it builds
#: one; this class cannot, and must not know that an engine exists.
RunnerFactory = Callable[[ChainSpec, Sequence[int], Sequence[int]], Runner]

_LOG = get_logger("runners.service")

#: Enough threads to answer a health probe while a Stop is draining, and few enough that a
#: wedged RPC cannot spawn a pool. Every handler here is either instant or bounded by the
#: deadline its request carries.
_DEFAULT_MAX_WORKERS = 8

#: What a ``timeout_s`` of ``0.0`` on the wire means.
#:
#: proto3 has no field presence for scalars: an **unset** ``double`` and a deliberate ``0.0``
#: arrive as the same bytes, and the servicer cannot tell them apart. Read literally, a client
#: that simply did not set the field would ask this shard to give every camera thread zero
#: seconds - which is to say, detach all of them and report a fleet-wide lifetime signal for a
#: shutdown that was perfectly ordinary. So zero reads as the default, and the numbers match
#: the ones :class:`~shipinfer.launch.client.ShardClient` and
#: :class:`~shipinfer.runners.base.Runner` already use, so a caller that omits the field and
#: one that lets its client fill it in get the same shutdown.
#:
#: The same sentence is in ``shard.proto``, because the ``.proto`` ships in the wheel and an
#: other-language client reads that and never this file. A caller that really does want no
#: grace has ``Stop`` with a tiny positive number.
DEFAULT_DRAIN_TIMEOUT_S = 20.0
DEFAULT_STOP_TIMEOUT_S = 20.0
DEFAULT_REMOVE_TIMEOUT_S = 5.0


class ShardService:
    """The eight RPCs of ``shard.proto``, over one runner.

    Args:
        runner: what executes this shard's chain, when the caller already has one — a test,
            or an embedded shard whose chain is known up front. ``None`` is the fleet's case:
            a spawned shard has only its identity, so the runner arrives with the chain, and
            ``build`` is what makes it.
        identity: who this shard is. ``control_port`` must be the port actually bound, which
            is why :func:`serve_shard` fills it in after binding rather than before.
        build: how to make the runner when ``UpdateTopology`` brings the chain. Required
            when ``runner`` is ``None`` and ignored when it is not: a shard runs one chain
            for its life, and a second ``UpdateTopology`` is refused rather than swapping a
            live one under in-flight frames.

    Raises:
        ConfigurationError: neither a runner nor a factory. A servicer with nothing to run
            would bind a port, answer ``Ready``, and refuse every RPC after it — a shard the
            launcher counts as alive and that can never take a camera.

    The state machine is derived, not stored — apart from the three facts nothing else can
    know, which is whether ``Drain`` is in flight, whether one completed, and whether ``Stop``
    has been called. Deriving the rest means a runner that dies on its own cannot leave this
    class asserting ``running``.
    """

    def __init__(
        self,
        runner: Runner | None,
        identity: ShardIdentity,
        *,
        build: RunnerFactory | None = None,
    ) -> None:
        if runner is None and build is None:
            raise ConfigurationError(
                f"shard {identity.shard_id} was given neither a runner nor a way to build "
                "one; it would answer Ready and then refuse every camera"
            )
        self._runner = runner
        self._build = build
        self._identity = identity
        self._lock = threading.Lock()
        #: A ``Drain`` is in flight *right now*. Set and cleared under :attr:`_lock`, and read
        #: without it by the lock-free probes, which is the point: a ``Health`` arriving
        #: during a slow drain must answer ``draining`` rather than block behind it.
        self._draining = False
        #: A ``Drain`` completed. Distinct from :attr:`_draining` because "still releasing
        #: cameras" and "released them, still serving nothing" are different answers to a
        #: launcher deciding whether to wait or to place the cameras elsewhere. Cleared only
        #: by a fresh ``UpdateTopology``, which is what makes the shard usable again.
        self._drained = False
        #: Why the last ``Drain`` did not release its cameras, or ``""``. Kept beside the flag
        #: rather than folded into it: "released" and "tried and could not" are different
        #: answers to a launcher deciding whether to place those cameras elsewhere.
        self._drain_detail = ""
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
    def runner(self) -> Runner | None:
        """What executes the chain, or ``None`` before the first ``UpdateTopology``."""
        return self._runner

    @property
    def identity(self) -> ShardIdentity:
        return self._identity

    @property
    def chain_yaml(self) -> str:
        """The chain text of the last accepted ``UpdateTopology``; empty before the first."""
        return self._chain_yaml

    @property
    def drain_detail(self) -> str:
        """Why the last ``Drain`` could not release the cameras, or ``""`` when it did."""
        return self._drain_detail

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

    @property
    def stopped(self) -> bool:
        """Whether ``Stop`` has been accepted. The cheap read a shutdown poll needs.

        Identical in *meaning* to ``state() == ShardState.STOPPED`` - :meth:`_lifecycle`
        checks the same flag first and nothing overrides it - and not in *cost*: ``state()``
        with no snapshot in hand falls through to :meth:`Runner.health`, which walks every
        element of the chain and snapshots the ingest manager. The process entry point
        (``cli/shard.py``) asks once a second, only to learn whether it may exit and give its
        CUDA context back, and a full health report per second is not what that answer is
        worth.
        """
        return self._stopped

    def state(self, report: dict[str, Any] | None = None) -> ShardState:
        """The shard's state, derived from the runner rather than remembered.

        Args:
            report: a snapshot from :meth:`Runner.health` the caller *already holds*. Passed
                in rather than fetched again because a caller that has one and lets this
                method take a second (``Health`` did) can report a camera count from one
                snapshot beside a state derived from another — the self-contradicting report
                the one-snapshot rule exists to prevent (arch.md section 2). ``None`` means
                "nobody has asked yet", and only then is the runner asked.
        """
        lifecycle = self._lifecycle()
        if lifecycle is not None:
            return lifecycle
        if self._runner is None or not self._runner.is_running:
            return ShardState.STARTING
        return ShardState.READY if not self._cameras(report) else ShardState.RUNNING

    def _lifecycle(self) -> ShardState | None:
        """The states only this class can know, or ``None`` when the runner decides."""
        if self._stopped:
            return ShardState.STOPPED
        if self._draining:
            return ShardState.DRAINING
        if self._drained:
            return ShardState.DRAINED
        return None

    def _refusal(self) -> ShardState | None:
        """The state that makes this shard take no cameras, or ``None`` when it takes them.

        ``starting`` joins :meth:`_lifecycle`'s three: a shard that has not been told what to
        run has no runner to hand a camera to, and answering ``accepted=True`` for one would
        have the launcher mark it placed and stop looking for a home for it.
        """
        return self._lifecycle() or (ShardState.STARTING if self._runner is None else None)

    def _refused(self, pb: Any, state: ShardState) -> Any:
        """One refusal sentence, so the fast path and the locked path cannot word it apart."""
        return pb.AddCameraReply(
            accepted=False,
            reason=(
                f"shard {self._identity.shard_id} is {state} and takes no cameras; "
                "place this one on another shard"
            ),
        )

    def _cameras(self, report: dict[str, Any] | None = None) -> dict[str, Any]:
        """The camera map out of a snapshot, taking one only if the caller has none."""
        if report is None:
            report = self._report()
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
        pb = load_pb()
        return pb.ReadyReply(identity=self._identity.to_pb(), state=str(self._safe_state()))

    def UpdateTopology(self, request: Any, context: Any = None) -> Any:
        """Install the chain this shard runs, and start executing it.

        This is the RPC that replaced the argv, and it carries what the child's environment
        used to (arch.md section 2, V140): the chain *and* the device sharing. A spawned
        shard has neither when it binds, so on the first call it **builds** its runner
        through the factory it was given — that is why a shard can be started before anybody
        has decided what it runs.

        Four refusals, all typed, all ``accepted=False``:

        * the document does not parse — refused here, before any camera, with the loader's
          own message. This is the one that earns the RPC its keep: a mistyped chain fails
          the deploy instead of producing a shard that runs nothing;
        * the runner could not be built — an unloadable model, a chain whose elements this
          host cannot open. The launcher is told which shard and why, as data;
        * the shard is already running — a live topology swap would have to close nine
          stateful elements under in-flight frames, and that is not this PR's problem;
        * the chain named is not the one this runner holds, for a shard that was *given* one
          at construction. Accepting a different name would start the wrong chain and report
          the right one.

        On acceptance the sharing lists are recorded (see :attr:`shared_by`) and the runner
        is started — installing a topology is precisely what makes a shard usable, and a
        parent that had to send a separate "now go" RPC would have a two-step handshake with
        a failure state in the middle. On a *refused* start the runner this call built is
        dropped again (:meth:`_discard_runner`), so a retry goes back through the factory with
        the sharing that retry carries rather than recording new numbers over an old engine.

        **This call is slow, and the caller must budget for it.** Building the runner is
        where a shard loads its models and deserialises its engines: tens of seconds on a
        cold page cache, which is why ``FleetRunner`` sends it with a deadline in minutes
        rather than the client's default ten seconds.
        """
        pb = load_pb()
        try:
            spec = ChainSpec.from_yaml(
                request.chain_yaml, source=f"shard {self._identity.shard_id} UpdateTopology"
            )
        except ShipInferError as exc:
            return pb.TopologyReply(accepted=False, reason=str(exc))

        with self._lock:
            if self._runner is not None and self._runner.is_running:
                return pb.TopologyReply(
                    accepted=False,
                    reason=(
                        f"shard {self._identity.shard_id} is already running topology "
                        f"{self._runner.topology.name!r}; stop it before installing another"
                    ),
                )
            built_here = False
            if self._runner is None:
                assert self._build is not None  # __init__ refuses without one
                try:
                    self._runner = self._build(
                        spec, tuple(request.shared_by), tuple(request.share_rank)
                    )
                    built_here = True
                # Same rule as the start below: the launcher has to be told which shard
                # could not be built and why, as data rather than as an UNKNOWN status.
                except Exception as exc:
                    _LOG.exception(
                        "shard %d could not build its runner", self._identity.shard_id
                    )
                    return pb.TopologyReply(
                        accepted=False, reason=f"{type(exc).__name__}: {exc}"
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
                if built_here:
                    self._discard_runner()
                return pb.TopologyReply(accepted=False, reason=f"{type(exc).__name__}: {exc}")
            self._chain_yaml = request.chain_yaml
            self._shared_by = tuple(request.shared_by)
            self._share_rank = tuple(request.share_rank)
            self._stopped = False
            self._draining = False
            self._drained = False
            self._drain_detail = ""

        _LOG.info(
            "shard %d installed topology %r (shared_by=%s share_rank=%s)",
            self._identity.shard_id,
            held or spec.name,
            list(self._shared_by),
            list(self._share_rank),
        )
        return pb.TopologyReply(accepted=True, topology=held or spec.name)

    def _discard_runner(self) -> None:
        """Forget a runner this ``UpdateTopology`` built and could not start. Under the lock.

        Not tidiness. Leaving it assigned makes a **retried** ``UpdateTopology`` take the
        "already built" path, skip the factory entirely, and then record the *new*
        ``shared_by``/``share_rank`` over an engine the first call built with the old ones.
        That is the number deciding whether two shards on one GPU load two instances each or
        four (``cli/shard.py::apply_sharing``), so the shard would report a sharing it is not
        running - the silent, expensive kind of wrong.

        Only a runner *this call* made is dropped: one handed in at construction is the only
        one this servicer will ever have, and forgetting it would leave a shard permanently
        answering ``starting`` with no factory to rebuild from.

        :meth:`Runner.start` has already run its own single unwind by the time we get here
        (``runners/base.py`` - a subclass that released twice is how an abandonment count came
        to be zeroed), so this only asks for a stop, best effort, in case the object holds
        something ``start`` never acquired.
        """
        runner, self._runner = self._runner, None
        if runner is not None:
            with contextlib.suppress(Exception):
                runner.stop(timeout_s=0.0)

    def AddCamera(self, request: Any, context: Any = None) -> Any:
        """Start one camera. Every refusal is an answer.

        The runner's typed vocabulary maps straight through: ``ConfigurationError`` is a
        duplicate id or an unknown source, ``ServerStateError`` is a shard that is stopping
        or a fleet that forgot the camera between the insert and the start
        (``ingest/manager.py``). Both mean the same thing to a launcher — this camera is not
        running here, place it elsewhere — and both must reach it as data, because a gRPC
        error status would put them in the same bucket as a dead process.

        The first refusal is this class's own: a shard that is draining, drained or stopped
        takes no cameras. Without it a ``Stop`` racing an ``AddCamera`` answers
        ``accepted=True`` for a camera nothing will ever read - the launcher marks it placed
        and stops looking for a home for it, and the camera is dark until an operator reads a
        dashboard. The runner cannot make that refusal for us: by then its own camera set is
        already gone, so what reaches it looks like an ordinary add.

        **Holding the lock across the add is deliberate, and it is cheap.**
        ``IngestManager.add_camera`` starts the camera's actor *thread* and returns - it does
        not wait on the RTSP open, and the actor's own re-check is what makes a camera
        forgotten mid-start safe (``ingest/manager.py``). So this handler is microseconds and
        serialising it against ``Stop`` costs nothing. The launcher above still never holds
        *its* lock across the RPC (``FleetRunner.add_camera``), for a different reason: a
        network call takes its whole deadline when the peer is wedged or unreachable, however
        quick the handler would have been.

        The guard is read **twice**, the same shape and for the same reason as ``Stop``'s.
        Once outside :attr:`_lock` as a fast path, so a camera offered to a shard that is
        already draining is refused *now* and placed on a sibling rather than queueing behind
        a twenty-second drain for the same answer. Then again inside the lock, together with
        the add, and that is the check that decides: the servicer runs on a thread pool, so
        without it an ``AddCamera`` passes a still-false ``_stopped`` and reaches the runner
        while a ``Stop`` on another thread is releasing that runner's camera set - and the
        launcher is told ``accepted=True`` for a camera on a shard that is going down, marks
        it placed, and stops looking for a home for it. The same lock ``Stop``, ``Drain`` and
        ``UpdateTopology`` take, so the four lifecycle-changing calls are serialised with each
        other and with nothing else: the probes an operator needs during a shutdown
        (``Ready``, ``Health``, ``Stats``) still take no lock at all.

        **The decode is refused as data too, and that is not decoration.**
        :meth:`~shipinfer.launch.control.CameraSpec.from_pb` raises a
        ``ConfigurationError`` for a band this build has no name for -- proto3 enums are open,
        so a newer launcher's lane arrives here as an unknown integer -- and it runs *before*
        the guard below, so an uncaught one would leave the servicer through gRPC's generic
        handler as ``UNKNOWN``. ``ShardClient`` reads a status that is not ``OK`` as a shard
        that failed, and ``FleetRunner.add_camera`` treats that as a dead peer and **aborts
        the whole placement** instead of offering the camera to the next shard. A refusal in
        the reply body says the one true thing -- this shard cannot take this camera -- and
        leaves the launcher free to try a sibling that can.

        Raises:
            Nothing. Every failure this handler can produce is an ``AddCameraReply`` with
            ``accepted=False`` and a reason, which is the module docstring's rule.
        """
        pb = load_pb()
        try:
            camera = CameraSpec.from_pb(request.camera)
        except ShipInferError as exc:
            _LOG.info(
                "shard %d could not read the camera spec it was offered: %s",
                self._identity.shard_id,
                exc,
            )
            return pb.AddCameraReply(accepted=False, reason=str(exc))
        except Exception as exc:  # module docstring: no traceback reaches the wire
            _LOG.exception("shard %d failed reading a camera spec", self._identity.shard_id)
            return pb.AddCameraReply(accepted=False, reason=f"{type(exc).__name__}: {exc}")
        refuse = self._refusal()  # fast path only; the check that decides is under the lock
        if refuse is not None:
            return self._refused(pb, refuse)
        with self._lock:
            refuse = self._refusal()
            if refuse is not None:
                return self._refused(pb, refuse)
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
            except Exception as exc:  # module docstring: no traceback reaches the wire
                _LOG.exception(
                    "shard %d failed adding camera %s",
                    self._identity.shard_id,
                    camera.camera_id,
                )
                return pb.AddCameraReply(accepted=False, reason=f"{type(exc).__name__}: {exc}")
        return pb.AddCameraReply(accepted=True)

    def RemoveCamera(self, request: Any, context: Any = None) -> Any:
        """Stop and forget one camera. ``clean=False`` means its thread was abandoned.

        ``removed`` answers exactly the question ``shard.proto`` says it does - **whether the
        shard knew this camera at all** - and two things mean it did not: the runner's
        ``ConfigurationError`` ("no such camera", ``runners/base.py``), and a runner that
        manages no cameras in the first place, which refuses everything with a
        ``ServerStateError`` and has therefore never heard of this one. That second case is
        the same reading :meth:`_drain` already takes of the same refusal.

        Anything else went wrong *while removing a camera this shard has*, and answers
        ``removed=True, clean=False`` with the reason. The distinction is the operator's:
        "there is no cam-7 here" sends them to look for a typo or at another shard, while
        "cam-7 raised on the way out" is this shard's problem and the camera may still be
        holding a thread. Collapsing the two into ``removed=False`` sent them to the wrong
        place - the client turns ``removed=False`` into "this shard does not run that
        camera", in precisely those words.
        """
        pb = load_pb()
        if self._runner is None:
            return pb.RemoveCameraReply(
                removed=False,
                clean=False,
                reason=f"shard {self._identity.shard_id} runs no topology yet",
            )
        # `or DEFAULT`: proto3 cannot tell an unset double from a deliberate 0.0, and zero
        # here would abandon this camera's thread instantly (see DEFAULT_REMOVE_TIMEOUT_S).
        timeout_s = request.timeout_s or DEFAULT_REMOVE_TIMEOUT_S
        try:
            clean = self._runner.remove_camera(request.camera_id, timeout_s=timeout_s)
        except ConfigurationError as exc:
            return pb.RemoveCameraReply(removed=False, clean=False, reason=str(exc))
        except ShipInferError as exc:
            if not self._runner.manages_cameras:
                return pb.RemoveCameraReply(removed=False, clean=False, reason=str(exc))
            _LOG.warning(
                "shard %d could not remove camera %s: %s",
                self._identity.shard_id,
                request.camera_id,
                exc,
                extra=log_context(camera_id=request.camera_id),
            )
            return pb.RemoveCameraReply(removed=True, clean=False, reason=str(exc))
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception(
                "shard %d failed removing camera %s", self._identity.shard_id, request.camera_id
            )
            return pb.RemoveCameraReply(
                removed=True, clean=False, reason=f"{type(exc).__name__}: {exc}"
            )
        return pb.RemoveCameraReply(removed=True, clean=bool(clean))

    def Health(self, request: Any, context: Any = None) -> Any:
        """What this shard is doing, from **one** snapshot.

        Exactly one call to ``runner.health()`` per RPC, split into state, engine and cameras
        rather than two calls merged: :meth:`state` is handed the report this method already
        holds, so a camera removed between two would-be halves makes the whole reply smaller
        instead of making ``state`` disagree with ``cameras``.

        The encoding is inside the guard with the fetch. ``Struct`` accepts string keys and
        JSON-shaped values only, so a report carrying an int key or a set raises out of
        ``to_pb`` — and a runner's health dict is the *least* controlled data this class
        handles. Encoded outside the guard, that reaches the wire as UNKNOWN plus a
        traceback, which is the one thing this servicer promises never to do.
        """
        try:
            report = self._report()
            state = str(self.state(report))
            cameras = report.pop("cameras", {})
            health = ShardHealth(
                state=state,
                engine=report,
                cameras=cameras if isinstance(cameras, dict) else {},
                vram_budget={},
            )
            return health.to_pb()
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception("shard %d could not report health", self._identity.shard_id)
            return ShardHealth(
                state=str(ShardState.UNKNOWN), detail=f"{type(exc).__name__}: {exc}"
            ).to_pb()

    def Stats(self, request: Any, context: Any = None) -> Any:
        """Counters an operator would page on, as the runner reports them."""
        pb = load_pb()
        reply = pb.StatsReply()
        if self._runner is None:
            return reply
        try:
            reply.stats.update(self._runner.stats())
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception("shard %d could not report stats", self._identity.shard_id)
            reply.detail = f"{type(exc).__name__}: {exc}"
        return reply

    def Drain(self, request: Any, context: Any = None) -> Any:
        """Stop reading cameras; let what is in flight finish.

        Under :attr:`_lock`, exactly as ``Stop`` is. Two drains are then serialised instead of
        both being inside the runner's camera set at once, and a ``Drain`` racing a ``Stop``
        cannot have the executor pulled out from under its in-flight work. The lock-free
        probes (``Ready``, ``Health``, ``Stats``) still answer throughout, which is what makes
        holding a lock across a blocking wait acceptable here.

        A completed drain leaves the shard ``drained``, not ``draining``: the flag used to be
        set and never cleared, so a shard that had finished releasing its cameras reported
        "still finishing" forever and a launcher waiting for the drain to end waited out its
        whole deadline. Only a fresh ``UpdateTopology`` clears it - a drained shard is done,
        and refuses cameras (``AddCamera``) until it is given something to run again.

        **A drain that FAILED is not a drain.** ``drained`` means *released*, and it is set
        only when the runner actually released: a drain that raised left the cameras where
        they were, and reporting ``drained`` for it would have a launcher stop waiting, place
        those cameras on another shard, and end up with two shards reading one camera. When it
        fails, the reason is in :attr:`drain_detail` and in ``DrainReply.detail``, and
        ``state`` goes back to what the runner says it is - which is the truth: this shard is
        still serving. Abandoning threads is *not* failing: a drain that released its cameras
        and could not join some of them answers ``abandoned>0`` with no detail, and that is a
        completed drain with a lifetime signal attached.
        """
        pb = load_pb()
        with self._lock:
            self._draining = True
            try:
                # `or DEFAULT`: proto3 cannot tell an unset double from a deliberate 0.0,
                # and zero here would detach every camera thread (DEFAULT_DRAIN_TIMEOUT_S).
                abandoned, detail = self._drain(request.timeout_s or DEFAULT_DRAIN_TIMEOUT_S)
            finally:
                self._draining = False
            self._drain_detail = detail
            self._drained = not detail
        return pb.DrainReply(abandoned=abandoned, detail=detail)

    def Stop(self, request: Any, context: Any = None) -> Any:
        """Drain, then stop the executor. Idempotent — a second Stop reports 0 abandoned.

        ``timeout_s`` is ONE deadline for the whole shutdown, charged to the camera set and
        the executor together, because everything is signalled at t0 and a worker still
        unfinished at the deadline is genuinely stuck (``ingest/manager.py``,
        ``runners/base.py``). The returned count is a lifetime signal, not a statistic.

        Idempotence is **answered here**, not delegated to the runner. A second Stop returns
        before touching it: the count belongs to the shutdown that actually happened, and
        asking a stopped runner to drain and stop again would re-enter two idempotent-but-not-
        free paths only to report a zero this class already knows.

        The flag is read **twice**: once outside the lock as a fast path, and once inside it
        as the check that is actually load-bearing. The servicer runs on a thread pool, so two
        Stops - a supervisor's and a signal handler's - are an ordinary wire event, and with
        the outer check alone both pass it while the flag is still false. The second would
        then re-drain a runner with nothing left to release and answer ``abandoned=0``, which
        ``StopResult.clean`` and ``shard.proto`` both define as "the clean shutdown, safe to
        unwind" - for a shard that abandoned threads still referencing its buffers. It would
        also burn a second full ``timeout_s`` doing it.

        :attr:`_lock` is held across both blocking waits - the drain and the executor stop -
        for the whole of ``timeout_s``. That is deliberate: it is what serialises a Stop
        against a concurrent Drain or UpdateTopology. The probes an operator needs *during* a
        shutdown (``Ready``, ``Health``, ``Stats``) take no lock at all, so a shard that is
        stopping still says so instead of timing out the supervisor watching it.
        """
        pb = load_pb()
        if self._stopped:  # fast path only; the check that decides is the one below
            return pb.StopReply(abandoned=0, detail="already stopped")
        with self._lock:
            if self._stopped:
                return pb.StopReply(abandoned=0, detail="already stopped")
            self._draining = True
            # `or DEFAULT`: proto3 cannot tell an unset double from a deliberate 0.0, and
            # zero here would abandon every camera thread on this shard and report a
            # fleet-wide lifetime signal for an ordinary shutdown (DEFAULT_STOP_TIMEOUT_S).
            timeout_s = request.timeout_s or DEFAULT_STOP_TIMEOUT_S
            abandoned, detail = self._drain(timeout_s)
            try:
                if self._runner is not None:
                    self._runner.stop(timeout_s=timeout_s)
            except Exception as exc:  # see the module docstring: no traceback reaches the wire
                _LOG.exception("shard %d could not stop its runner", self._identity.shard_id)
                detail = f"{detail}; {type(exc).__name__}: {exc}".lstrip("; ")
            self._draining = False
            self._drained = False
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
        if self._runner is None:
            return 0, ""
        try:
            return int(self._runner.drain(timeout_s)), ""
        except ShipInferError as exc:
            if not self._runner.manages_cameras:
                return 0, ""
            return 0, str(exc)
        except Exception as exc:  # see the module docstring: no traceback reaches the wire
            _LOG.exception("shard %d could not drain", self._identity.shard_id)
            return 0, f"{type(exc).__name__}: {exc}"

    def _report(self) -> dict[str, Any]:
        """One health snapshot, or an empty one from a shard with no runner yet."""
        return {} if self._runner is None else self._runner.health()

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
        """Whether the server has stopped. What a shard process's ``main`` polls.

        ``True`` means terminated, ``False`` means ``timeout`` elapsed with it still serving.
        grpc's own method answers the *opposite* question — it returns True when the wait
        times out — so the inversion belongs here, once, rather than at the call site. Read
        the wrong way round, a healthy server looks stopped and the shard exits one tick
        after it started: the fleet came up, installed its chain, and every shard was gone
        before the first camera was placed.
        """
        return not self.server.wait_for_termination(timeout)


def serve_shard(
    runner: Runner | None,
    *,
    shard_id: int,
    control_port: int,
    host: str = "127.0.0.1",
    max_workers: int = _DEFAULT_MAX_WORKERS,
    build: RunnerFactory | None = None,
) -> ShardServer:
    """Bind the control plane for one shard and start serving. Does not start the runner.

    Args:
        runner: what executes this shard's chain, or ``None`` for a spawned shard that will
            be told over ``UpdateTopology`` — see ``build``.
        build: how to make the runner from the chain the first ``UpdateTopology`` carries.
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
        ConfigurationError: grpcio is unusable, or the port is already bound. The port
            check is why this function exists rather than four lines at a call site — the
            refusal has to happen **before any camera is opened**, or a shard that cannot be
            controlled is already holding decoder threads and a CUDA context that nothing
            can now tell it to release.
    """
    from concurrent import futures

    grpc, shard_pb2_grpc = load_grpc()

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
        _release(server)
        raise ConfigurationError(
            f"shard {shard_id} could not bind its control port {host}:{control_port}; "
            "another shard or an earlier run still holds it"
        ) from detail

    identity = ShardIdentity(shard_id=shard_id, control_port=bound, pid=os.getpid())
    # Everything after a successful bind is unwound on failure. The port is held from the
    # `add_insecure_port` above, so a servicer that cannot be attached or a `start()` that
    # raises would otherwise leave a listening socket and a thread pool owned by nothing -
    # and the caller's retry, or the next shard handed this port, would be refused by a
    # server no object references any more.
    try:
        # Constructed inside the unwind: it refuses a shard given neither a runner nor a
        # factory, and that refusal must not strand the port bound two lines above.
        service = ShardService(runner, identity, build=build)
        # No annotations on protoc's output, and no `isinstance` inside it either: this is
        # what binds the duck-typed servicer above by attribute name.
        shard_pb2_grpc.add_ShardServicer_to_server(service, server)  # type: ignore[no-untyped-call]
        server.start()
    except BaseException:
        _release(server)
        raise
    _LOG.info("shard %d control plane on %s:%d", shard_id, host, bound)
    return ShardServer(server=server, identity=identity, service=service)


def _release(server: Any) -> None:
    """Give back whatever a half-built server took, and never raise doing it.

    ``stop()`` releases a listening socket only on a server that was **started**: a grpc
    server that bound a port and then failed before ``start()`` keeps that socket until the
    process exits, even once the last reference to it is gone (measured against grpcio
    1.83 - ``stop(0).wait()``, ``del``, ``gc.collect()``, all still EADDRINUSE). So the
    unwind starts it first. A server with no servicer attached answers UNIMPLEMENTED for the
    microseconds before it is stopped, which is the price of not stranding a port that the
    operator's retry - or the next shard given this number - would then be refused.

    Both calls are best-effort: this runs on a failure path, and an exception here would
    replace the diagnosis the caller is in the middle of raising.
    """
    with contextlib.suppress(Exception):
        server.start()
    with contextlib.suppress(Exception):
        server.stop(0).wait(1.0)
