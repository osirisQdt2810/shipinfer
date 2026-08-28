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

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import Priority
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
    "mint_camera_id",
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

    Deliberately five fields and not
    :class:`shipinfer.core.settings.ingest.CameraConfig`'s twenty. The rest of a camera's
    configuration — codec, transport, decode size — is *deployment* settings, and a shard
    resolves them from the settings tree it loaded (CONVENTIONS 2.6). What only the launcher
    knows is which camera goes where, and that is this.

    :attr:`loop` is the field that had to join them, and the argument for it is the one the
    rest of the class makes in reverse: it is not a deployment default, because it decides
    whether *this* camera ever ends. ``shipinfer run --inputs clip.mp4`` mints its cameras in
    ``cli/commands/run.py`` and they appear in no ``ingest.cameras`` list, so before this
    field there was no configuration anywhere that could make a file be processed once —
    and a configured camera placed across a fleet would have lost the ``loop: false`` its
    operator wrote, because a shard is told its cameras and not the file they came from.

    :attr:`priority` joined them for the same reason one layer further out, and it is the
    field that had to *stop* being deployment configuration. A fleet shard is an
    ``InprocessRunner`` built from an env-only settings tree whose ``ingest.cameras`` is
    **cleared** on purpose (``runners/inprocess.py::_ingest``, #71), so the shard has no
    configured camera table to resolve a band from: ``priority: tracking_critical`` written
    against a camera reached the launcher, was dropped at the wire, and every camera placed by
    RPC was admitted at ``normal`` — the one customisation ADR-005 exists for, configured and
    then silently discarded. ``None`` still means "the shard decides", which is the right
    answer for a single-process deployment that *can* read its own config.
    """

    camera_id: str
    #: ``rtsp://...`` for a camera, or a file path for a replayed video.
    url: str
    #: Target frame rate; ``0.0`` means "whatever the source delivers".
    fps: float = 0.0
    #: ``replay`` only: restart the file at EOF. ``True`` keeps a stress test running;
    #: ``False`` makes a finite input finish, which is what ``--no-loop`` asks for.
    loop: bool = True
    #: The scheduler lane this camera's frames are admitted into. ``None`` is **not** a band:
    #: it means "whatever the shard's own config says, and ``NORMAL`` if it says nothing",
    #: which is what a camera nobody gave an opinion about should get. A value here overrides
    #: the shard's table, because a launcher that read the fleet config is better informed
    #: than a shard whose copy of it was stripped.
    priority: Priority | None = None

    def to_pb(self) -> shard_pb2.CameraSpec:
        shard_pb2 = load_pb()

        return shard_pb2.CameraSpec(
            camera_id=self.camera_id,
            url=self.url,
            fps=self.fps,
            loop=self.loop,
            priority=_priority_to_pb(self.priority),
        )

    @classmethod
    def from_pb(cls, message: shard_pb2.CameraSpec) -> CameraSpec:
        # `HasField`, because the wire default for a `bool` is false and this field's default
        # is TRUE: read literally, a client that simply did not set it would ask every shard
        # to stop each camera at EOF. `optional` in the .proto buys that presence back — the
        # same trap `RemoveCameraRequest.timeout_s` documents, in the other direction.
        return cls(
            camera_id=message.camera_id,
            url=message.url,
            fps=message.fps,
            loop=message.loop if message.HasField("loop") else True,
            priority=_priority_from_pb(message.priority),
        )


#: What ``CameraPriority``'s value names add to :class:`~shipinfer.core.request.Priority`'s.
#: protobuf enum values share their enclosing scope, so a bare ``NORMAL`` would be a name
#: this package owns globally; the prefix is the convention that keeps it ours.
_BAND_PREFIX = "CAMERA_PRIORITY_"


def _priority_to_pb(priority: Priority | None) -> int:
    """One :class:`~shipinfer.core.request.Priority`, or ``None``, as a wire band.

    Mapped **by name**, never by number. The two vocabularies are numbered differently on
    purpose — the wire reserves 0 for "unspecified" so that
    :attr:`~shipinfer.core.request.Priority.TRACKING_CRITICAL`, which *is* 0, cannot be
    confused with silence — and a hand-written offset between them would be one edit away
    from admitting every critical camera into ``high``. A name that does not exist on the
    wire is a loud failure instead.

    Raises:
        ConfigurationError: a band exists in ``Priority`` that ``shard.proto`` has no name
            for. That is a missed edit to the ``.proto``, not a client's mistake, and it
            fails here rather than travelling as a neighbouring lane.
    """
    shard_pb2 = load_pb()
    if priority is None:
        return int(shard_pb2.CAMERA_PRIORITY_UNSPECIFIED)
    try:
        return int(shard_pb2.CameraPriority.Value(_BAND_PREFIX + priority.name))
    except ValueError:
        raise ConfigurationError(
            f"priority {priority.name} has no name in shard.proto's CameraPriority; "
            f"add {_BAND_PREFIX + priority.name} to it and regenerate the stubs "
            "(python scripts/gen_proto.py)"
        ) from None


def _priority_from_pb(value: int) -> Priority | None:
    """A wire band as a :class:`~shipinfer.core.request.Priority`, or ``None`` when unset.

    ``None`` for the unspecified zero, and that is the whole point of spending a value on it:
    a reader here cannot mistake "the launcher said nothing" for the critical band, and does
    not have to remember to ask ``HasField`` to avoid doing so.

    Raises:
        ConfigurationError: a band this build has no name for. proto3 enums are open, so a
            newer launcher's lane arrives as an unknown integer; refusing it names the value,
            where mapping it to ``NORMAL`` would place a camera in a lane nobody asked for
            and say nothing.
    """
    shard_pb2 = load_pb()
    if value == shard_pb2.CAMERA_PRIORITY_UNSPECIFIED:
        return None
    try:
        name = shard_pb2.CameraPriority.Name(value)
    except ValueError:
        raise ConfigurationError(
            f"camera priority {value} is not a band this build knows; "
            "it comes from a newer shard.proto than this process was built against"
        ) from None
    try:
        return Priority[name.removeprefix(_BAND_PREFIX)]
    except KeyError:
        raise ConfigurationError(
            f"wire priority {name} has no counterpart in core.request.Priority; "
            "the two vocabularies are mapped by name and this one has drifted"
        ) from None


def mint_camera_id(index: int) -> str:
    """The name a camera gets when nobody supplied one: ``cam-000``, ``cam-001``, ...

    One function rather than two format strings, because the two callers must agree or they
    collide. ``shipinfer run --inputs`` names its cameras by position
    (``cli/commands/run.py::cameras_from_inputs``) and ``POST /streams`` with no ``camera_id``
    names the next free one (``api/streams.py``) — on the same running deployment, where a
    second ``cam-1`` and a ``cam-001`` that are the same camera under two names is a tracker
    keyed on nothing.

    Zero-padded to three digits so ``cam-010`` sorts after ``cam-009`` in every log line,
    dashboard and metric label that sorts strings, and left un-truncated past 999 rather than
    wrapping: a fiftieth camera is the design point, a thousandth is somebody's bug and it
    should read as one.
    """
    return f"cam-{index:03d}"


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
