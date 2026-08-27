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
import threading
import time
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
        self.health_calls = 0
        self.start_error: Exception | None = None
        self.health_error: Exception | None = None
        self.extra_health: dict[Any, Any] = {}

    def _do_start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started += 1

    def _do_stop(self, timeout_s: float) -> None:
        self.stopped += 1

    def _do_submit(self, item: ChainItem) -> ResponseFuture:  # pragma: no cover - unused
        raise NotImplementedError

    def _do_health(self) -> dict[str, Any]:
        # Counted here rather than by overriding `health`, because `health` is the template
        # method every caller goes through and `_do_health` is called exactly once by it.
        self.health_calls += 1
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
        self.drains = 0
        self.removed_with: list[tuple[str, float]] = []
        #: Raised by `remove_camera` for a camera this runner *does* have, which is how the
        #: servicer's "the shard knew it, the removal failed" branch is reached.
        self.remove_error: Exception | None = None
        self.drain_delay_s = 0.0
        #: Every deadline `drain` was actually given. The servicer clamps a proto3 zero to
        #: its default before it gets here, and this is what shows which number arrived.
        self.drain_timeouts: list[float] = []
        #: The highest number of threads ever inside `drain` at once. 1 is the assertion:
        #: the servicer's lock is what keeps two Drains out of one camera set.
        self.drain_concurrency = 0
        self._inside = 0
        self._counter = threading.Lock()

    def add_camera(self, camera: CameraSpec) -> None:
        if camera.camera_id in self.cameras:
            raise ConfigurationError(
                f"camera {camera.camera_id!r} is already running; "
                "remove it before adding it again"
            )
        self.cameras[camera.camera_id] = camera

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool:
        self.removed_with.append((camera_id, timeout_s))
        if camera_id not in self.cameras:
            raise ConfigurationError(f"camera {camera_id!r} is not running")
        if self.remove_error is not None:
            raise self.remove_error
        del self.cameras[camera_id]
        return self.abandoned == 0

    def drain(self, timeout_s: float = 20.0) -> int:
        with self._counter:
            self.drains += 1
            self.drain_timeouts.append(timeout_s)
            self._inside += 1
            self.drain_concurrency = max(self.drain_concurrency, self._inside)
        try:
            if self.drain_delay_s:
                time.sleep(self.drain_delay_s)
            if self.drain_error is not None:
                raise self.drain_error
            self.cameras.clear()
            return self.abandoned
        finally:
            with self._counter:
                self._inside -= 1

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


class TestTheShutdownPollReadsAFlag:
    """``cli/shard.py`` polls once a second, only to learn whether it may exit and give its
    CUDA context back. ``state()`` answers the same question and takes a full
    ``runner.health()`` to do it - every element walked, the ingest manager snapshotted - for
    every second the shard is alive."""

    def test_stopped_costs_no_snapshot_and_state_costs_one(self) -> None:
        runner = FakeRunner(chain())
        svc = service(runner)
        install(svc)
        before = runner.health_calls

        assert svc.stopped is False
        assert runner.health_calls == before, "the poll took a health snapshot"

        assert svc.state() == ShardState.READY
        assert runner.health_calls == before + 1, "the expensive path stopped being expensive"

    def test_it_answers_what_state_answers(self) -> None:
        """Same meaning, so the cheaper read is a substitution and not a second opinion."""
        runner = FakeRunner(chain())
        svc = service(runner)
        install(svc)

        assert svc.stopped is (svc.state() == ShardState.STOPPED) is False

        svc.Stop(pb.StopRequest(timeout_s=0.1))

        assert svc.stopped is (svc.state() == ShardState.STOPPED) is True


class TestHealthAsksTheRunnerExactlyOnce:
    """Two snapshots are two different moments, and a reply must be one moment.

    ``Health`` used to call ``runner.health()`` for the engine report and then again, through
    ``state()``, to count the cameras. A removal landing between them produces a reply whose
    ``state`` says ``running`` and whose ``cameras`` map is empty - self-contradicting, and
    the contradiction is invisible in any test that only reads one field.
    """

    def test_one_snapshot_per_rpc(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))
        runner.health_calls = 0

        svc.Health(pb.HealthRequest())

        assert runner.health_calls == 1

    def test_ready_costs_one_snapshot_too(self) -> None:
        """`Ready` derives the same state, and is the RPC a launcher polls in a loop."""
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        runner.health_calls = 0

        svc.Ready(pb.ReadyRequest())

        assert runner.health_calls == 1

    def test_the_cameras_on_the_reply_are_the_ones_the_state_was_derived_from(self) -> None:
        """A non-empty camera map, consistent with `running` - the half never asserted."""
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        for index in (1, 2):
            svc.AddCamera(
                pb.AddCameraRequest(
                    camera=CameraSpec(
                        f"cam-{index}", f"rtsp://10.0.0.{index}/live", 20.0
                    ).to_pb()
                )
            )

        health = ShardHealth.from_pb(svc.Health(pb.HealthRequest()))

        assert health.state == ShardState.RUNNING
        assert set(health.cameras) == {"cam-1", "cam-2"}
        assert health.cameras["cam-2"]["url"] == "rtsp://10.0.0.2/live"
        # Struct numbers are doubles; 20.0 is what a double round trip gives back.
        assert health.cameras["cam-1"]["fps"] == 20.0
        # The camera map is not left in the engine report as well as beside it.
        assert "cameras" not in health.engine


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


class TestAShardThatIsGoingDownTakesNoCameras:
    """`accepted=True` from a stopped shard is a camera nobody will ever read.

    The runner cannot make this refusal: by the time `Stop` has run, its camera set is empty
    and an add looks entirely ordinary to it. So the servicer makes it, and names the state
    in the reason - the launcher's cue is "place it elsewhere", and it needs to know why.
    """

    def test_a_stopped_shard_refuses_and_says_so(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.Stop(pb.StopRequest(timeout_s=1.0))

        reply = svc.AddCamera(
            pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb())
        )

        assert not reply.accepted
        assert ShardState.STOPPED in reply.reason
        assert runner.cameras == {}

    def test_a_drained_shard_refuses_and_says_so(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.Drain(pb.DrainRequest(timeout_s=1.0))

        reply = svc.AddCamera(
            pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb())
        )

        assert not reply.accepted
        assert ShardState.DRAINED in reply.reason
        assert runner.cameras == {}

    def test_a_drain_in_flight_refuses_with_draining(self) -> None:
        """The window that matters: the camera arrives while the drain is still running."""
        runner = CameraRunner(chain())
        runner.drain_delay_s = 0.2
        svc = service(runner)
        install(svc)
        draining = threading.Thread(target=svc.Drain, args=(pb.DrainRequest(timeout_s=1.0),))
        draining.start()
        try:
            time.sleep(0.05)

            reply = svc.AddCamera(
                pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb())
            )
        finally:
            draining.join(timeout=5.0)

        assert not reply.accepted
        assert ShardState.DRAINING in reply.reason
        assert runner.cameras == {}

    def test_a_fresh_topology_makes_the_shard_usable_again(self) -> None:
        """`drained` is not terminal: it is cleared by being told what to run next."""
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.Drain(pb.DrainRequest(timeout_s=1.0))
        runner.stop(timeout_s=1.0)

        install(svc)

        reply = svc.AddCamera(
            pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb())
        )

        assert reply.accepted
        assert svc.Health(pb.HealthRequest()).state == ShardState.RUNNING


class TestRemovingACamera:
    def test_an_unknown_camera_is_a_typed_answer_not_a_silent_no_op(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)

        reply = svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-x", timeout_s=1.0))

        assert not reply.removed
        assert "cam-x" in reply.reason

    def test_the_requested_deadline_is_the_one_the_runner_is_given(self) -> None:
        """`timeout_s` is the whole point of the field, and it crossed no boundary before.

        A launcher that asks for a 0.5 s stop and gets the runner's 5 s default waits ten
        times as long per camera, and nothing anywhere says so.
        """
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-1", timeout_s=0.25))

        assert runner.removed_with == [("cam-1", 0.25)]

    def test_a_runner_that_owns_no_cameras_also_answers_not_removed(self) -> None:
        """ "I manage no cameras" is still "I never knew this one", not a failed removal.

        The same reading `_drain` takes of the same refusal: a runner whose
        `manages_cameras` is False has no camera set for `cam-1` to be missing from.
        """
        svc = service(FakeRunner(chain()))
        install(svc)

        reply = svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-1", timeout_s=1.0))

        assert not reply.removed
        assert "does not manage cameras" in reply.reason

    def test_a_removal_that_raises_is_not_an_unknown_camera(self) -> None:
        """`removed` means "the shard knew this camera", and it did - the removal failed.

        Answering `removed=False` here made the client say "shard 3 does not run camera
        cam-1", which sends an operator looking for a typo while cam-1 sits on this shard
        possibly still holding a thread.
        """
        runner = CameraRunner(chain())
        runner.remove_error = ServerStateError("the manager is stopping")
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        reply = svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-1", timeout_s=1.0))

        assert reply.removed
        assert not reply.clean
        assert reply.reason == "the manager is stopping"

    def test_an_unexpected_failure_is_reported_the_same_way_with_its_type(self) -> None:
        """A bug in a decoder is still a camera this shard has; no traceback on the wire."""
        runner = CameraRunner(chain())
        runner.remove_error = RuntimeError("the decoder segfaulted its way out")
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        reply = svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-1", timeout_s=1.0))

        assert reply.removed
        assert not reply.clean
        assert reply.reason == "RuntimeError: the decoder segfaulted its way out"

    def test_an_abandoned_thread_is_reported_as_not_clean(self) -> None:
        runner = CameraRunner(chain())
        runner.abandoned = 1
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        reply = svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-1", timeout_s=1.0))

        assert reply.removed
        assert not reply.clean
        # Empty, and load-bearing: a non-empty reason on a removed camera is how the client
        # tells "the thread was abandoned" from "the removal raised".
        assert reply.reason == ""


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
        # `drained`, not `draining`: the call has returned, so "still finishing" would be a
        # lie a launcher would wait out its whole deadline on.
        assert svc.Health(pb.HealthRequest()).state == ShardState.DRAINED


class TestDrainAndStopAreSerialisedAndIdempotent:
    """Both mutate the camera set, so both take the lock; neither may be entered twice.

    `Drain` used to take no lock at all while `Stop` did, which meant two drains could be
    inside the runner's camera set at once and a drain could have the executor stopped out
    from under its in-flight work.
    """

    def test_two_concurrent_drains_never_overlap(self) -> None:
        runner = CameraRunner(chain())
        runner.drain_delay_s = 0.1
        svc = service(runner)
        install(svc)
        replies: list[Any] = []

        def drain() -> None:
            replies.append(svc.Drain(pb.DrainRequest(timeout_s=1.0)))

        threads = [threading.Thread(target=drain) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        assert [thread.is_alive() for thread in threads] == [False, False]
        assert len(replies) == 2
        assert runner.drains == 2
        assert runner.drain_concurrency == 1
        assert svc.Health(pb.HealthRequest()).state == ShardState.DRAINED

    def test_a_second_drain_answers_the_same_way(self) -> None:
        """Idempotent as a launcher sees it: a repeat is an answer, not a second shutdown."""
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        first = svc.Drain(pb.DrainRequest(timeout_s=1.0))
        second = svc.Drain(pb.DrainRequest(timeout_s=1.0))

        assert (first.abandoned, first.detail) == (0, "")
        assert (second.abandoned, second.detail) == (0, "")
        assert runner.cameras == {}

    def test_a_second_stop_answers_without_touching_the_runner(self) -> None:
        """The count belongs to the shutdown that happened; a repeat re-runs nothing.

        Delegating idempotence to the runner meant a second `Stop` re-entered the drain and
        the executor stop, and reported whatever the runner happened to say the second time.
        """
        runner = CameraRunner(chain())
        runner.abandoned = 2
        svc = service(runner)
        install(svc)
        first = svc.Stop(pb.StopRequest(timeout_s=1.0))
        drains_after_the_first_stop = runner.drains

        second = svc.Stop(pb.StopRequest(timeout_s=1.0))

        assert first.abandoned == 2
        assert (second.abandoned, second.detail) == (0, "already stopped")
        assert runner.drains == drains_after_the_first_stop
        assert runner.stopped == 1

    def test_two_concurrent_stops_produce_exactly_one_shutdown(self) -> None:
        """The idempotence check has to be re-read *under* the lock, not only outside it.

        The servicer is bound to a thread pool, so two Stops - a supervisor's drain-then-stop
        and a signal handler's - are an ordinary wire event. With the guard read only outside
        the lock, both threads passed it while `_stopped` was still false: the second then
        re-drained a runner with nothing left to release and answered `abandoned=0`, which
        `StopResult.clean` and `shard.proto` both define as "the clean shutdown, safe to
        unwind" - for a shard that had just abandoned two threads still holding its buffers.

        The drain is slow on purpose: 0.3 s is long enough that the second thread is reliably
        inside the window the first one owns.
        """
        runner = CameraRunner(chain())
        runner.abandoned = 2
        runner.drain_delay_s = 0.3
        svc = service(runner)
        install(svc)
        replies: list[Any] = []
        lock = threading.Lock()

        def stop() -> None:
            reply = svc.Stop(pb.StopRequest(timeout_s=2.0))
            with lock:
                replies.append(reply)

        threads = [threading.Thread(target=stop) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        assert [thread.is_alive() for thread in threads] == [False, False]
        assert len(replies) == 2
        answers = sorted((reply.abandoned, reply.detail) for reply in replies)
        assert answers == [(0, "already stopped"), (2, "")]
        # The count is a lifetime signal, so the shutdown that produced it must have happened
        # exactly once - and the second caller must not be told a clean zero.
        assert runner.drains == 1
        assert runner.stopped == 1

    def test_a_health_probe_answers_while_a_stop_holds_the_lock(self) -> None:
        """The reason holding the lock across two blocking waits is acceptable.

        A supervisor watching a shard shut down must keep getting answers, or it concludes
        the process is wedged and kills it mid-drain - which is precisely the abandonment
        the deadline exists to avoid.
        """
        runner = CameraRunner(chain())
        runner.drain_delay_s = 0.3
        svc = service(runner)
        install(svc)
        stopping = threading.Thread(target=svc.Stop, args=(pb.StopRequest(timeout_s=2.0),))
        stopping.start()
        try:
            time.sleep(0.05)
            started = time.monotonic()

            state = svc.Health(pb.HealthRequest()).state
            elapsed = time.monotonic() - started
        finally:
            stopping.join(timeout=5.0)

        assert state == ShardState.DRAINING
        assert elapsed < 0.2, "Health blocked behind the stop instead of answering"


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

    def test_a_report_that_cannot_be_encoded_is_a_detail_and_not_a_traceback(self) -> None:
        """The failure is in `to_pb`, not in `health()`, and it used to escape the guard.

        A `Struct` takes string keys and JSON-shaped values only, and a runner's health dict
        is the least controlled data this class handles - an int key or a set is a plain bug
        one layer down. Encoding outside the try turned that into UNKNOWN plus a traceback on
        the wire, which is the one thing this servicer promises never to do.
        """
        runner = FakeRunner(chain())
        runner.extra_health = {1: "an int key a Struct cannot take"}
        svc = service(runner)

        reply = svc.Health(pb.HealthRequest())

        assert reply.state == ShardState.UNKNOWN
        assert "TypeError" in reply.detail
        assert ShardHealth.from_pb(reply).engine == {}

    def test_a_value_a_struct_cannot_take_is_the_same_answer(self) -> None:
        """The other half of the same door: a set raises ValueError rather than TypeError."""
        runner = FakeRunner(chain())
        runner.extra_health = {"queues": {1, 2, 3}}
        svc = service(runner)

        reply = svc.Health(pb.HealthRequest())

        assert reply.state == ShardState.UNKNOWN
        assert "ValueError" in reply.detail

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


class TestAShardThatIsToldWhatToRun:
    """A spawned shard has two flags and no chain (arch.md section 2), so it has no runner.

    ``Topology.from_spec`` refuses an empty chain, so there is genuinely nothing to construct
    one over until the first ``UpdateTopology`` arrives — which is why the servicer takes a
    factory instead of a runner. Everything before that call has to answer *something*: a
    shard that could not say "starting" would be a shard a launcher either respawns or hands
    a camera to, and both are wrong.
    """

    @staticmethod
    def _service(build: Any, **kw: Any) -> ShardService:
        return ShardService(
            None, ShardIdentity(shard_id=3, control_port=50103, pid=99), build=build, **kw
        )

    def test_neither_a_runner_nor_a_factory_is_refused_at_construction(self) -> None:
        """It would bind a port, answer Ready, and refuse every RPC after it."""
        with pytest.raises(ConfigurationError, match="neither a runner nor"):
            ShardService(None, ShardIdentity(shard_id=3, control_port=50103))

    def test_before_the_first_topology_it_says_starting(self) -> None:
        svc = self._service(lambda spec, shared, rank: FakeRunner(chain()))

        assert svc.Ready(pb.ReadyRequest()).state == ShardState.STARTING
        assert svc.Health(pb.HealthRequest()).state == ShardState.STARTING
        assert svc.runner is None

    def test_it_takes_no_camera_before_it_has_a_topology(self) -> None:
        """`accepted=True` here would have the launcher mark the camera placed and stop
        looking for a home for it — dark until somebody reads a dashboard."""
        svc = self._service(lambda spec, shared, rank: FakeRunner(chain()))

        reply = svc.AddCamera(
            pb.AddCameraRequest(camera=CameraSpec("quay-1", "rtsp://host").to_pb())
        )

        assert reply.accepted is False and "starting" in reply.reason

    def test_stopping_one_costs_nothing_and_says_so(self) -> None:
        svc = self._service(lambda spec, shared, rank: FakeRunner(chain()))

        reply = svc.Stop(pb.StopRequest(timeout_s=0.1))

        assert reply.abandoned == 0 and reply.detail == ""

    def test_the_topology_builds_the_runner_and_starts_it(self) -> None:
        built: list[Any] = []

        def build(spec: Any, shared: Any, rank: Any) -> Runner:
            built.append((spec.name, tuple(shared), tuple(rank)))
            return FakeRunner(chain())

        svc = self._service(build)
        reply = install(svc, shared_by=[2], share_rank=[1])

        assert reply.accepted and reply.topology == "linear"
        assert built == [("linear", (2,), (1,))]
        assert svc.runner is not None and svc.runner.is_running
        assert svc.Health(pb.HealthRequest()).state == ShardState.READY

    def test_the_sharing_reaches_the_factory_and_not_only_the_property(self) -> None:
        """`shared_by` used to arrive in the child's environment, where the settings tree read
        it before anything was built. It arrives here now, and the factory is what puts it
        back in front of the engine — so a servicer that only recorded it would be a shard
        loading twice the engines with the right number in its logs."""
        seen: list[tuple[int, ...]] = []
        svc = self._service(
            lambda spec, shared, rank: (seen.append(tuple(shared)), FakeRunner(chain()))[1]
        )

        install(svc, shared_by=[4], share_rank=[3])

        assert seen == [(4,)]
        assert svc.shared_by == (4,) and svc.share_rank == (3,)

    def test_a_factory_that_fails_is_an_answer_not_a_crash(self) -> None:
        def build(spec: Any, shared: Any, rank: Any) -> Runner:
            raise ConfigurationError("model 'ship_detector' is not in this repository")

        svc = self._service(build)
        reply = install(svc)

        assert reply.accepted is False
        assert "ship_detector" in reply.reason
        assert svc.runner is None, "a failed build must not leave half a runner behind"

    def test_a_start_that_failed_sends_the_retry_back_through_the_factory(self) -> None:
        """Otherwise the retry records the NEW sharing over an engine built with the OLD one.

        `shared_by` is the number that decides whether two shards on one GPU load two
        instances each or four, so a shard that skipped the factory would report a sharing it
        is not running - and `UpdateTopology` is the one RPC a launcher would retry.
        """
        built: list[tuple[int, ...]] = []

        def build(spec: Any, shared: Any, rank: Any) -> Runner:
            built.append(tuple(shared))
            runner = FakeRunner(chain())
            if len(built) == 1:
                runner.start_error = RuntimeError("the decode element could not open")
            return runner

        svc = self._service(build)

        first = install(svc, shared_by=[4], share_rank=[3])

        assert first.accepted is False and "could not open" in first.reason
        assert svc.runner is None, "a runner that could not start was left assigned"

        second = install(svc, shared_by=[2], share_rank=[1])

        assert second.accepted is True
        assert built == [(4,), (2,)], "the retry skipped the factory"
        assert svc.shared_by == (2,) and svc.share_rank == (1,)
        assert svc.runner is not None and svc.runner.is_running

    def test_a_runner_handed_in_at_construction_is_kept_across_a_failed_start(self) -> None:
        """The other side of the same coin: this servicer has no factory, so dropping its one
        runner would leave a shard answering `starting` forever with nothing to rebuild."""
        runner = FakeRunner(chain())
        runner.start_error = RuntimeError("the decode element could not open")
        svc = service(runner)

        assert install(svc).accepted is False
        assert svc.runner is runner

        runner.start_error = None

        assert install(svc).accepted is True

    def test_a_chain_that_does_not_parse_never_reaches_the_factory(self) -> None:
        """The refusal that earns the RPC its keep: a mistyped chain fails the deploy."""
        calls: list[Any] = []
        svc = self._service(lambda *a: calls.append(a) or FakeRunner(chain()))

        reply = install(svc, yaml="elements: [this, is, not, a, mapping]")

        assert reply.accepted is False and not calls


class TestAZeroTimeoutOnTheWireIsTheDefaultNotNoGrace:
    """proto3 has no field presence for scalars.

    An unset ``double timeout_s`` and a deliberate ``0.0`` are the same bytes, and no servicer
    can tell them apart. Read literally, a client that simply omits the field asks this shard
    to detach every camera thread at once and report ``abandoned>0`` — which ``StopReply`` and
    ``StopResult.clean`` both define as "a detached thread still references this shard's
    buffers, do not unwind them". That is a fleet-wide lifetime signal raised by an ordinary
    shutdown, and the ``.proto`` ships in the wheel for clients that never see this package's
    defaults. So zero reads as the default, and the numbers are the ones ``ShardClient`` and
    ``Runner`` already use.
    """

    def test_stop_with_an_unset_timeout_gets_the_default(self) -> None:
        from shipinfer.runners.service import DEFAULT_STOP_TIMEOUT_S

        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        svc.Stop(pb.StopRequest())  # no timeout_s: 0.0 on the wire

        assert runner.drain_timeouts == [DEFAULT_STOP_TIMEOUT_S]

    def test_drain_with_an_unset_timeout_gets_the_default(self) -> None:
        from shipinfer.runners.service import DEFAULT_DRAIN_TIMEOUT_S

        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)

        svc.Drain(pb.DrainRequest())

        assert runner.drain_timeouts == [DEFAULT_DRAIN_TIMEOUT_S]

    def test_remove_camera_with_an_unset_timeout_gets_the_default(self) -> None:
        from shipinfer.runners.service import DEFAULT_REMOVE_TIMEOUT_S

        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))

        svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-1"))

        assert runner.removed_with == [("cam-1", DEFAULT_REMOVE_TIMEOUT_S)]

    def test_a_number_the_caller_did_set_is_still_the_one_used(self) -> None:
        """The clamp is for the unset field, not a floor on what a caller may ask for."""
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)

        svc.Drain(pb.DrainRequest(timeout_s=0.25))

        assert runner.drain_timeouts == [0.25]


class TestADrainThatFailedDoesNotReadAsDrained:
    """``drained`` means RELEASED, not "a Drain was attempted".

    The flag used to be set in a ``finally``, so a drain whose runner raised left the shard
    reporting ``drained`` with its cameras still running. A launcher reads that as "this shard
    is finished, place its cameras elsewhere" — and two shards end up reading one camera.
    """

    def test_a_failed_drain_leaves_the_shard_where_it_was(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        svc.AddCamera(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()))
        runner.drain_error = ServerStateError("the manager would not release cam-1")

        reply = svc.Drain(pb.DrainRequest(timeout_s=1.0))

        assert reply.abandoned == 0
        assert "would not release" in reply.detail
        assert svc.state() != ShardState.DRAINED
        assert svc.state() == ShardState.RUNNING, "the shard is still serving cam-1"
        assert svc.drain_detail == "the manager would not release cam-1"

    def test_a_failed_drain_still_takes_cameras(self) -> None:
        """Which is the operational consequence: the shard did not release anything, so it is
        not a shard that must refuse the next camera."""
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        runner.drain_error = ServerStateError("nope")
        svc.Drain(pb.DrainRequest(timeout_s=1.0))

        reply = svc.AddCamera(
            pb.AddCameraRequest(camera=CameraSpec("cam-2", "rtsp://x").to_pb())
        )

        assert reply.accepted

    def test_a_drain_that_abandoned_threads_is_still_a_completed_drain(self) -> None:
        """Abandoning is not failing: the cameras WERE released, and `abandoned` is the
        lifetime signal that goes with it."""
        runner = CameraRunner(chain())
        runner.abandoned = 2
        svc = service(runner)
        install(svc)

        reply = svc.Drain(pb.DrainRequest(timeout_s=1.0))

        assert reply.abandoned == 2 and reply.detail == ""
        assert svc.state() == ShardState.DRAINED
        assert svc.drain_detail == ""

    def test_a_fresh_topology_clears_the_failure_with_the_flag(self) -> None:
        runner = CameraRunner(chain())
        svc = service(runner)
        install(svc)
        runner.drain_error = ServerStateError("nope")
        svc.Drain(pb.DrainRequest(timeout_s=1.0))
        runner.drain_error = None
        runner.stop()

        install(svc)

        assert svc.drain_detail == ""


class TestAddCameraIsSerialisedAgainstTheLifecycle:
    """The guard is read twice — outside the lock as a fast path, inside it as the decision.

    The servicer runs on a thread pool, so an ``AddCamera`` and a ``Stop`` on two threads is
    an ordinary wire event. With the guard read only outside the lock, the add passes a
    still-false ``_stopped`` and reaches the runner *while* the Stop is releasing that
    runner's camera set: the launcher is told ``accepted=True``, marks the camera placed, and
    stops looking for a home for it. The camera is then dark until somebody reads a dashboard
    — ADR-005's failure, one layer up.
    """

    def test_a_stop_waits_for_an_add_in_flight_instead_of_racing_it(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class SlowAdd(CameraRunner):
            def add_camera(self, camera: CameraSpec) -> None:
                entered.set()
                assert release.wait(10.0), "the test never released the add"
                super().add_camera(camera)

        runner = SlowAdd(chain())
        svc = service(runner)
        install(svc)
        adding = threading.Thread(
            target=svc.AddCamera,
            args=(pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb()),),
        )
        adding.start()
        try:
            assert entered.wait(5.0)
            stopping = threading.Thread(target=svc.Stop, args=(pb.StopRequest(timeout_s=1.0),))
            stopping.start()
            time.sleep(0.1)

            assert stopping.is_alive(), "the Stop ran while the add was inside the runner"
        finally:
            release.set()
            adding.join(timeout=10.0)
            stopping.join(timeout=10.0)

        # The camera was inserted first and drained by the Stop that waited for it - never
        # left running on a shard the launcher believes is down.
        assert runner.cameras == {}
        assert svc.state() == ShardState.STOPPED

    def test_the_fast_path_still_refuses_a_draining_shard_without_waiting(self) -> None:
        """Read twice, and the outer read is why: a camera offered to a shard that is already
        draining is refused NOW and placed on a sibling, rather than queueing behind a
        twenty-second drain for the same answer."""
        runner = CameraRunner(chain())
        runner.drain_delay_s = 0.5
        svc = service(runner)
        install(svc)
        draining = threading.Thread(target=svc.Drain, args=(pb.DrainRequest(timeout_s=2.0),))
        draining.start()
        try:
            time.sleep(0.05)
            started = time.monotonic()

            reply = svc.AddCamera(
                pb.AddCameraRequest(camera=CameraSpec("cam-1", "rtsp://x").to_pb())
            )

            assert time.monotonic() - started < 0.3, "the refusal waited out the drain"
        finally:
            draining.join(timeout=5.0)

        assert not reply.accepted and ShardState.DRAINING in reply.reason


class TestOverARealInprocessRunner:
    """The one class here that does not use a double, because a double cannot show this.

    Every other test in this file asserts a *mapping* — a typed refusal into a reason string,
    an abandonment count onto a reply — and :class:`CameraRunner` is the right shape for that
    because the servicer's job is the mapping. What it cannot show is whether the runner the
    fleet actually ships emits the key the mapping reads: ``state()`` derives ``running`` from
    ``health()["cameras"]``, so an ``InprocessRunner`` that reported no such key would leave
    every shard answering ``ready`` forever while it read fifty cameras, and no fake would
    ever notice. That is a contract between two files, so it is tested across both.
    """

    CHAIN_YAML = textwrap.dedent("""
        name: replayed
        elements:
          decode: {impl: replay}
          detect: {impl: mock, model: ship_detector}
          output: {impl: mock}
        """)

    def _runner(self) -> Any:
        """A real runner over a source that opens, delivers nothing, and never blocks.

        The camera's *state* is what is under test, not its frames, so the source is the
        cheapest thing that keeps an actor alive without a decoder anywhere near it.
        """
        from shipinfer.core.settings import ServerSettings
        from shipinfer.ingest.base import FrameSource
        from shipinfer.runners.inprocess import InprocessRunner
        from shipinfer.topology import ChainSpec as Spec
        from shipinfer.topology import Topology as Chain

        class SilentSource(FrameSource):
            name = "silent"

            def _do_open(self) -> None:
                self._set_format(4, 4, 20.0)

            def _do_read(self) -> None:
                return None

            def _do_close(self) -> None:
                return None

        return InprocessRunner(
            Chain.from_spec(Spec.from_yaml(self.CHAIN_YAML)),
            ServerSettings(
                pipeline={"workers": 1},
                ingest={"read_timeout_ms": 20, "empty_read_sleep_ms": 5},
            ),
            source_factory=lambda config, counter: SilentSource(config, counter),
        )

    def test_a_camera_makes_the_shard_running_and_appears_in_its_health(self) -> None:
        runner = self._runner()
        svc = service(runner)
        install(svc, yaml=self.CHAIN_YAML)
        try:
            assert svc.state() == ShardState.READY

            reply = svc.AddCamera(
                pb.AddCameraRequest(camera=CameraSpec("cam-1", "injected://one", 20.0).to_pb())
            )

            assert reply.accepted, reply.reason
            health = ShardHealth.from_pb(svc.Health(pb.HealthRequest()))
            assert health.state == ShardState.RUNNING
            assert set(health.cameras) == {"cam-1"}
            assert health.cameras["cam-1"]["camera_id"] == "cam-1"
        finally:
            svc.Stop(pb.StopRequest(timeout_s=5.0))

    def test_removing_the_last_camera_takes_it_back_to_ready(self) -> None:
        """Derived, not remembered: the shard has nothing to read, and says so."""
        runner = self._runner()
        svc = service(runner)
        install(svc, yaml=self.CHAIN_YAML)
        try:
            svc.AddCamera(
                pb.AddCameraRequest(camera=CameraSpec("cam-1", "injected://one").to_pb())
            )
            assert svc.state() == ShardState.RUNNING

            reply = svc.RemoveCamera(pb.RemoveCameraRequest(camera_id="cam-1", timeout_s=5.0))

            assert reply.removed and reply.clean
            assert svc.state() == ShardState.READY
        finally:
            svc.Stop(pb.StopRequest(timeout_s=5.0))
