"""The control plane's vocabulary, with no transport in it.

Five values move between a launcher and a shard (arch.md section 2), and this module is all
five as ordinary frozen dataclasses. The protobuf messages next door are how they *travel*;
these are what the rest of the codebase passes around, compares, logs and asserts on.

**Why both, rather than passing the generated messages about.** A protobuf message is a
mutable object with a global descriptor registration behind it, no ``__eq__`` a test wants to
read in a failure report, and — the one that decides it — an import of ``google.protobuf``.
Keeping the vocabulary here means a caller in ``runners/`` or a test can name a
:class:`CameraSpec` without the optional extra installed, and the two ``to_pb`` /
``from_pb`` helpers are the only code in the project that has to know a wire format exists.
Both helpers reach the generated module through
:mod:`shipinfer.launch.proto`'s loaders - lazily, and behind the guard that turns a missing
extra into a typed refusal - for exactly that reason.

**The dicts stay dicts.** ``ShardHealth`` carries three of them —
``engine`` (``engine/health.py::HealthReport.as_dict``), ``cameras`` (the ingest manager's
per-camera snapshot) and ``vram_budget`` (ADR-016 section 3.4, phase D) — and they travel as
``google.protobuf.Struct``. Freezing three independently evolving report shapes into wire
messages would make every field a schema migration, and a health report is read by an
operator, not dispatched on. One consequence is worth knowing: ``Struct`` numbers are
doubles, so an integer count comes back as ``2.0``. It compares equal to ``2`` in Python,
which is why the round trip is still an equality assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from shipinfer.launch.proto import load_json_format, load_pb

if TYPE_CHECKING:  # pragma: no cover - typing only; never imported at runtime
    from shipinfer.launch.proto import shard_pb2

__all__ = [
    "AddCameraResult",
    "CameraSpec",
    "ShardHealth",
    "ShardIdentity",
    "ShardState",
    "StopResult",
]


class ShardState(str, Enum):
    """What a shard says it is doing, in the launcher's vocabulary.

    The wire field is a plain string and this enum is the *vocabulary*, not a parser: a
    launcher must be able to read a state a newer shard invented instead of failing to
    decode the reply. Compare with ``==`` against either form — it subclasses ``str``, as
    :class:`shipinfer.engine.health.HealthStatus` does.
    """

    #: Spawned and answering, but no topology installed yet — it can take no cameras.
    STARTING = "starting"
    #: Topology installed, executor running, no cameras. The state a launcher waits for
    #: before its first ``AddCamera``.
    READY = "ready"
    #: At least one camera is being served.
    RUNNING = "running"
    #: Told to stop reading; still finishing what is in flight.
    DRAINING = "draining"
    #: A drain **completed**: the cameras are released and the executor is still up. Distinct
    #: from :attr:`DRAINING` because a launcher waiting for a drain to end needs to be able
    #: to tell "still finishing" from "finished", and distinct from :attr:`STOPPED` because
    #: the process is alive and can be given a new topology.
    DRAINED = "drained"
    #: Executor stopped. Terminal for this process.
    STOPPED = "stopped"
    #: The shard could not ask its executor what state it is in.
    #: :attr:`ShardHealth.detail` says why. Not a healthy answer, and deliberately not
    #: spelled ``stopped``: "I do not know" and "it is down" call for different actions.
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """The wire spelling, not ``ShardState.READY``.

        ``str.Enum.__str__`` is ``Enum``'s, so ``f"{state}"`` yields the *member* name — and
        that string would go straight onto the wire and into an operator's dashboard. One
        override, rather than remembering ``.value`` at seven call sites.
        """
        return self.value


@dataclass(frozen=True, slots=True)
class ShardIdentity:
    """Who a shard is. Assigned by the launcher, except for the pid.

    ``control_port`` is the port the shard is **actually** bound to, which is not necessarily
    the one it was told: a shard asked for port 0 picks an ephemeral one, and the parent
    learns the real number from :meth:`ShardClient.wait_ready` rather than assuming it.
    """

    shard_id: int
    control_port: int
    pid: int = 0

    def to_pb(self) -> shard_pb2.ShardIdentity:
        shard_pb2 = load_pb()

        return shard_pb2.ShardIdentity(
            shard_id=self.shard_id, control_port=self.control_port, pid=self.pid
        )

    @classmethod
    def from_pb(cls, message: shard_pb2.ShardIdentity) -> ShardIdentity:
        return cls(
            shard_id=message.shard_id, control_port=message.control_port, pid=message.pid
        )


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """One camera, as a launcher hands it to a shard.

    Deliberately three fields and not
    :class:`shipinfer.core.settings.ingest.CameraConfig`'s twenty. The rest of a camera's
    configuration — codec, transport, decode size, priority — is *deployment* settings, and a
    shard resolves them from the settings tree it loaded (CONVENTIONS 2.6). What only the
    launcher knows is which camera goes where, and that is this.
    """

    camera_id: str
    #: ``rtsp://...`` for a camera, or a file path for a replayed video.
    url: str
    #: Target frame rate; ``0.0`` means "whatever the source delivers".
    fps: float = 0.0

    def to_pb(self) -> shard_pb2.CameraSpec:
        shard_pb2 = load_pb()

        return shard_pb2.CameraSpec(camera_id=self.camera_id, url=self.url, fps=self.fps)

    @classmethod
    def from_pb(cls, message: shard_pb2.CameraSpec) -> CameraSpec:
        return cls(camera_id=message.camera_id, url=message.url, fps=message.fps)


@dataclass(frozen=True, slots=True)
class AddCameraResult:
    """Whether a shard took a camera, and why not.

    A refusal is an **ordinary answer**, not an exception, and that is the whole reason this
    type exists. The shard-side invariant it carries is the ingest manager's insert →
    start → re-check (``ingest/manager.py``): a duplicate id, a manager that is stopping, or
    a fleet that forgot the camera while it was starting are all conditions under which the
    camera is *not* running and the launcher must place it elsewhere — a distinction a gRPC
    error status flattens into "the call failed".
    """

    accepted: bool
    #: Why not. Empty when ``accepted``.
    reason: str = ""

    def to_pb(self) -> shard_pb2.AddCameraReply:
        shard_pb2 = load_pb()

        return shard_pb2.AddCameraReply(accepted=self.accepted, reason=self.reason)

    @classmethod
    def from_pb(cls, message: shard_pb2.AddCameraReply) -> AddCameraResult:
        return cls(accepted=message.accepted, reason=message.reason)


@dataclass(frozen=True, slots=True)
class StopResult:
    """What a shard's shutdown cost.

    ``abandoned`` is a **lifetime signal**, not a statistic (arch.md section 2, mirroring
    ``ingest/manager.py::IngestManager.stop``): one deadline is charged to the whole camera
    set, and a thread still unfinished at it is genuinely stuck rather than merely slow.
    While the count is non-zero, a detached thread still references this shard's buffers and
    the caller must not unwind them. ``0`` is the clean shutdown.
    """

    abandoned: int
    #: The shard's own failure text when the shutdown itself went wrong. Empty on a normal
    #: stop. Without it ``abandoned=0`` would read identically whether the shard stopped
    #: cleanly or could not be asked at all.
    detail: str = ""

    @property
    def clean(self) -> bool:
        return self.abandoned == 0 and not self.detail

    def to_pb(self) -> shard_pb2.StopReply:
        shard_pb2 = load_pb()

        return shard_pb2.StopReply(abandoned=self.abandoned, detail=self.detail)

    @classmethod
    def from_pb(cls, message: shard_pb2.StopReply) -> StopResult:
        return cls(abandoned=message.abandoned, detail=message.detail)


@dataclass(frozen=True, slots=True)
class ShardHealth:
    """One shard's answer to "what are you doing".

    Safe against a concurrent removal by construction: every dict here is built from a
    snapshot taken on the shard, so a camera removed between two RPCs makes one report
    smaller rather than making either report inconsistent with itself.
    """

    state: str
    #: What the shard's executor reports about the models it holds
    #: (``engine/health.py::HealthReport.as_dict``).
    engine: dict[str, Any] = field(default_factory=dict)
    #: One entry per camera, keyed by camera id.
    cameras: dict[str, Any] = field(default_factory=dict)
    #: ``slabs + own_ctx + K * C_ctx + engines <= device memory - reserve``
    #: (ADR-016 section 3.4). Empty until the DataPool lands in phase D.
    vram_budget: dict[str, Any] = field(default_factory=dict)
    #: Empty on a healthy answer; the shard's own failure text when it could not ask its
    #: executor. Paired with ``state == ShardState.UNKNOWN``.
    detail: str = ""

    def to_pb(self) -> shard_pb2.HealthReply:
        shard_pb2 = load_pb()

        reply = shard_pb2.HealthReply(state=str(self.state), detail=self.detail)
        reply.engine.update(self.engine)
        reply.cameras.update(self.cameras)
        reply.vram_budget.update(self.vram_budget)
        return reply

    @classmethod
    def from_pb(cls, message: shard_pb2.HealthReply) -> ShardHealth:
        json_format = load_json_format()

        return cls(
            state=message.state,
            engine=json_format.MessageToDict(message.engine),
            cameras=json_format.MessageToDict(message.cameras),
            vram_budget=json_format.MessageToDict(message.vram_budget),
            detail=message.detail,
        )
