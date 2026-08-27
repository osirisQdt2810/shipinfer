"""The shard's servicer, driven directly — no socket, no channel, no port.

Everything worth asserting about :class:`~shipinfer.runners.service.ShardService` is a
mapping: a runner's typed refusal becomes ``accepted=False`` with a reason, an abandonment
count becomes an integer on a reply, an unexpected exception becomes a ``detail`` string
instead of a traceback on the wire. None of that needs a transport, and a test that stood one
up would be testing grpcio. One socket test exists — ``tests/launch/test_shard_rpc.py`` — and
it is deliberately the only one.

The runner here is a real :class:`~shipinfer.runners.base.Runner` subclass rather than a mock
object, because the *default* behaviour of the three camera methods is half of what is under
test: a runner that does not manage cameras must produce a refusal a launcher can act on, and
a ``Mock`` would happily return a truthy answer for a method the ABC refuses.
"""

from __future__ import annotations

import textwrap
from typing import Any, ClassVar

import pytest

pytest.importorskip("google.protobuf", reason="the grpc extra is not installed")

from shipinfer.core.errors import ConfigurationError, ServerStateError
from shipinfer.core.request import ResponseFuture
from shipinfer.launch.control import (
    CameraSpec,
    ShardHealth,
    ShardIdentity,
    ShardState,
)
from shipinfer.launch.proto import shard_pb2 as pb
from shipinfer.runners.base import Runner
from shipinfer.runners.service import ShardService
from shipinfer.topology import ChainItem, ChainSpec, Topology

CHAIN = """
name: linear
elements:
  decode: {impl: mock}
  detect: {impl: mock, model: ship_detector}
  output: {impl: mock}
"""


def chain(name: str = "linear") -> Topology:
    text = textwrap.dedent(CHAIN).replace("name: linear", f"name: {name}", 1)
    return Topology.from_spec(ChainSpec.from_yaml(text))


class FakeRunner(Runner):
    """A runner that starts, stops and answers, and does whatever a test tells it to.

    Deliberately a subclass: the three camera methods are left at the ABC's defaults unless a
    test replaces them, so "the servicer maps the ABC's refusal" is asserted against the real
    refusal rather than against a stand-in for it.
    """

    name: ClassVar[str] = "fake"

    def __init__(self, topology: Topology, **kwargs: Any) -> None:
        super().__init__(topology, **kwargs)
        self.started = 0
        self.stopped = 0
        self.start_error: Exception | None = None
        self.health_error: Exception | None = None
        self.extra_health: dict[str, Any] = {}

    def _do_start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started += 1

    def _do_stop(self, timeout_s: float) -> None:
        self.stopped += 1

    def _do_submit(self, item: ChainItem) -> ResponseFuture:  # pragma: no cover - unused
        raise NotImplementedError

    def _do_health(self) -> dict[str, Any]:
        if self.health_error is not None:
            raise self.health_error
        return dict(self.extra_health)


class CameraRunner(FakeRunner):
    """A runner that *does* manage cameras — the shape the fleet shard will have."""

    manages_cameras: ClassVar[bool] = True

    def __init__(self, topology: Topology, **kwargs: Any) -> None:
        super().__init__(topology, **kwargs)
        self.cameras: dict[str, CameraSpec] = {}
        self.abandoned = 0
        self.drain_error: Exception | None = None

    def add_camera(self, camera: CameraSpec) -> None:
        if camera.camera_id in self.cameras:
            raise ConfigurationError(
                f"camera {camera.camera_id!r} is already running; "
                "remove it before adding it again"
            )
        self.cameras[camera.camera_id] = camera

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool:
        if camera_id not in self.cameras:
            raise ConfigurationError(f"camera {camera_id!r} is not running")
        del self.cameras[camera_id]
        return self.abandoned == 0

    def drain(self, timeout_s: float = 20.0) -> int:
        if self.drain_error is not None:
            raise self.drain_error
        self.cameras.clear()
        return self.abandoned

    def _do_health(self) -> dict[str, Any]:
        health = super()._do_health()
        health["cameras"] = {
            cid: {"url": c.url, "fps": c.fps} for cid, c in self.cameras.items()
        }
        return health


def service(runner: Runner, *, shard_id: int = 3, port: int = 50103) -> ShardService:
    return ShardService(runner, ShardIdentity(shard_id=shard_id, control_port=port, pid=99))


def install(svc: ShardService, *, yaml: str | None = None, **kw: Any) -> Any:
    text = textwrap.dedent(CHAIN if yaml is None else yaml)
    return svc.UpdateTopology(pb.TopologyRequest(chain_yaml=text, **kw))


# -- state --------------------------------------------------------------------------------


class TestTheStateIsDerivedNotRemembered:
    def test_health_on_a_runner_that_never_started_says_starting(self) -> None:
        """The first question asked of a shard that will not serve is what it thinks it is.

        `starting` rather than `stopped`: nothing has gone wrong, the shard simply has not
        been told what to run, and a launcher that read `stopped` would respawn it.
        """
        svc = service(FakeRunner(chain()))

        reply = svc.Health(pb.HealthRequest())

        assert reply.state == ShardState.STARTING
        assert reply.detail == ""

    def test_a_running_shard_with_no_cameras_is_ready_and_with_one_is_running(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)

        assert svc.Health(pb.HealthRequest()).state == ShardState.READY

        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        assert svc.Health(pb.HealthRequest()).state == ShardState.RUNNING

    def test_after_stop_the_state_is_stopped(self) -> None:
        svc = service(FakeRunner(chain()))
        install(svc)

        svc.Stop(pb.StopRequest(timeout_s=1.0))

        assert svc.Health(pb.HealthRequest()).state == ShardState.STOPPED

    def test_ready_answers_with_the_identity_and_the_state(self) -> None:
        """The reply *arriving* is the readiness signal; the fields say who and what."""
        svc = service(FakeRunner(chain()), shard_id=5, port=50105)

        reply = svc.Ready(pb.ReadyRequest())

        assert ShardIdentity.from_pb(reply.identity) == ShardIdentity(5, 50105, 99)
        assert reply.state == ShardState.STARTING


# -- the topology -------------------------------------------------------------------------


class TestInstallingATopology:
    def test_the_sharing_fields_reach_the_shard(self) -> None:
        """Two shards on one GPU must each load HALF the instances.

        This used to ride in the child's environment (`SHIPINFER_DEVICES__SHARED_BY`). A
        shard that never hears it loads the full instance count and the device silently
        holds twice the engines for the same throughput — which is why the fields are on
        `TopologyRequest` and not deferred to the runner that will read them.
        """
        svc = service(FakeRunner(chain()))

        reply = install(svc, shared_by=[2, 2], share_rank=[0, 1])

        assert reply.accepted
        assert reply.topology == "linear"
        assert svc.shared_by == (2, 2)
        assert svc.share_rank == (0, 1)
        assert svc.chain_yaml.strip().startswith("name: linear")

    def test_installing_starts_the_runner(self) -> None:
        runner = FakeRunner(chain())
        svc = service(runner)

        install(svc)

        assert runner.started == 1
        assert runner.is_running

    def test_a_document_that_does_not_parse_is_refused_before_any_camera(self) -> None:
        runner = FakeRunner(chain())
        svc = service(runner)

        reply = install(svc, yaml="elements: [this is a list, not a mapping]")

        assert not reply.accepted
        assert "UpdateTopology" in reply.reason
        assert runner.started == 0

    def test_a_second_install_on_a_running_shard_is_refused(self) -> None:
        svc = service(FakeRunner(chain()))
        install(svc)

        reply = install(svc)

        assert not reply.accepted
        assert "already running" in reply.reason

    def test_a_chain_this_runner_was_not_built_for_is_refused_by_name(self) -> None:
        """Accepting it would start the wrong chain and report the right one."""
        runner = FakeRunner(chain("linear"))
        svc = service(runner)

        reply = install(svc, yaml=CHAIN.replace("name: linear", "name: something_else"))

        assert not reply.accepted
        assert "linear" in reply.reason and "something_else" in reply.reason
        assert runner.started == 0

    def test_a_runner_that_cannot_start_answers_rather_than_raising(self) -> None:
        runner = FakeRunner(chain())
        runner.start_error = ServerStateError("this element cannot open")
        svc = service(runner)

        reply = install(svc)

        assert not reply.accepted
        assert "this element cannot open" in reply.reason
        assert not runner.is_running


# -- cameras ------------------------------------------------------------------------------


class TestAddingACamera:
    def test_a_runner_that_manages_no_cameras_refuses_with_its_reason(self) -> None:
        """A refusal is an ordinary answer, and the reason is the payload.

        `accepted=False` with an empty reason would tell a launcher only that something went
        wrong; the reason is what distinguishes "place it on another shard" from "this shard
        is broken".
        """
        svc = service(FakeRunner(chain()))
        install(svc)

        reply = svc.AddCamera(
            pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x", 20.0).to_pb())
        )

        assert not reply.accepted
        assert "this runner does not manage cameras" in reply.reason

    def test_a_duplicate_is_refused_typed_and_not_silently_replaced(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        request = pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x", 20.0).to_pb())

        assert svc.AddCamera(request).accepted

        reply = svc.AddCamera(request)

        assert not reply.accepted
        assert "already running" in reply.reason
        assert list(runner.cameras) == ["cam-1"]

    def test_the_spec_arrives_intact(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)

        svc.AddCamera(
            pb.AddCameraRequest(
                camera=CameraSpec("cam-9", "rtsp://10.0.0.9/live", 19.5).to_pb()
            )
        )

        assert runner.cameras["cam-9"] == CameraSpec("cam-9", "rtsp://10.0.0.9/live", 19.5)


class TestRemovingACamera:
    def test_an_unknown_camera_is_a_typed_answer_not_a_silent_no_op(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)

        reply = svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-x", timeout_s=1.0))

        assert not reply.removed
        assert "cam-x" in reply.reason

    def test_an_abandoned_thread_is_reported_as_not_clean(self) -> None:
        runner = CameraRunner(chain())
        runner.abandoned = 1
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        reply = svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-1", timeout_s=1.0))

        assert reply.removed
        assert not reply.clean


# -- shutdown -----------------------------------------------------------------------------


class TestStopReportsTheAbandonmentCount:
    def test_the_runners_count_is_what_the_reply_carries(self) -> None:
        """A lifetime signal, not a statistic: while it is non-zero, buffers are still live."""
        runner = CameraRunner(chain())
        runner.abandoned = 3
        svc = service(runner)
        install(svc)

        reply = svc.Stop(pb.StopRequest(timeout_s=1.0))

        assert reply.abandoned == 3
        assert reply.detail == ""
        assert runner.stopped == 1

    def test_a_runner_that_manages_no_cameras_abandons_none(self) -> None:
        """Its `drain` refuses, and 0 is the honest count — not a swallowed error."""
        runner = FakeRunner(chain())
        svc = service(runner)
        install(svc)

        reply = svc.Stop(pb.StopRequest(timeout_s=1.0))

        assert reply.abandoned == 0
        assert reply.detail == ""

    def test_a_second_stop_is_idempotent(self) -> None:
        runner = CameraRunner(chain())
        runner.abandoned = 2
        svc = service(runner)
        install(svc)

        first = svc.Stop(pb.StopRequest(timeout_s=1.0))
        runner.abandoned = 0
        second = svc.Stop(pb.StopRequest(timeout_s=1.0))

        assert first.abandoned == 2
        assert second.abandoned == 0
        assert svc.Health(pb.HealthRequest()).state == ShardState.STOPPED

    def test_drain_reports_the_count_without_stopping_the_runner(self) -> None:
        runner = CameraRunner(chain())
        runner.abandoned = 1
        svc = service(runner)
        install(svc)

        reply = svc.Drain(pb.DrainRequest(timeout_s=1.0))

        assert reply.abandoned == 1
        assert runner.stopped == 0
        assert svc.Health(pb.HealthRequest()).state == ShardState.DRAINING


# -- nothing reaches the wire as a traceback ----------------------------------------------


class TestAnUnexpectedFailureBecomesAReply:
    """A supervisor decides whether to respawn; it cannot decide from a stack trace.

    Each of these drives an exception the servicer has no vocabulary for — a bare
    ``RuntimeError`` out of a runner, which is what a bug looks like — and asserts the RPC
    still answers, with the failure where a caller will read it.
    """

    def test_health_answers_unknown_with_the_reason(self) -> None:
        runner = FakeRunner(chain())
        runner.health_error = RuntimeError("the executor is wedged")
        svc = service(runner)

        reply = svc.Health(pb.HealthRequest())

        assert reply.state == ShardState.UNKNOWN
        assert "the executor is wedged" in reply.detail
        assert ShardHealth.from_pb(reply).engine == {}

    def test_ready_still_answers_when_the_state_cannot_be_determined(self) -> None:
        runner = FakeRunner(chain())
        svc = service(runner)
        install(svc)
        # Set only after the install: `state()` reaches the runner's health to count cameras,
        # and it only gets that far once the runner is running.
        runner.health_error = RuntimeError("the executor is wedged")

        reply = svc.Ready(pb.ReadyRequest())

        assert reply.state == ShardState.UNKNOWN
        assert reply.identity.shard_id == 3

    def test_add_camera_answers_rather_than_raising(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)

        def boom(camera: CameraSpec) -> None:
            raise RuntimeError("the source factory blew up")

        runner.add_camera = boom  # type: ignore[assignment]

        reply = svc.AddCamera(
            pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb())
        )

        assert not reply.accepted
        assert "RuntimeError: the source factory blew up" in reply.reason

    def test_stop_reports_a_broken_drain_in_detail_rather_than_as_a_clean_zero(self) -> None:
        """0 abandoned and a failed drain must not read the same."""
        runner = CameraRunner(chain())
        runner.drain_error = RuntimeError("the ingest manager is gone")
        svc = service(runner)
        install(svc)

        reply = svc.Stop(pb.StopRequest(timeout_s=1.0))

        assert reply.abandoned == 0
        assert "the ingest manager is gone" in reply.detail
        assert runner.stopped == 1

    def test_stats_reports_a_broken_runner_in_detail(self) -> None:
        runner = FakeRunner(chain())
        svc = service(runner)

        def boom() -> dict[str, Any]:
            raise RuntimeError("no counters")

        runner.stats = boom  # type: ignore[assignment]

        reply = svc.Stats(pb.StatsRequest())

        assert "no counters" in reply.detail
        assert not reply.stats.fields


class TestTheServicerNeedsNoChannel:
    def test_every_rpc_accepts_a_none_context(self) -> None:
        """The whole file's premise, stated once: these are plain methods.

        gRPC hands a servicer a context object; nothing here uses one, which is what lets the
        rest of this file exist and what keeps a socket out of the offline tier.
        """
        svc = service(CameraRunner(chain()))
        install(svc)

        assert svc.Ready(pb.ReadyRequest(), None).identity.shard_id == 3
        assert svc.Health(pb.HealthRequest(), None).state
        assert svc.Stats(pb.StatsRequest(), None) is not None
        assert svc.Drain(pb.DrainRequest(timeout_s=0.1), None).abandoned == 0
        assert svc.Stop(pb.StopRequest(timeout_s=0.1), None).abandoned == 0
