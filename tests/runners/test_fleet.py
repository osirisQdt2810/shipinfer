"""The fleet runner: what a shard is told, where a camera lands, and what a stop costs.

Two doubles and no GPU, deliberately. The ``command=`` a real :class:`Fleet` spawns is a
``sleep``, so the supervision under test is the real one — processes really start, really get
SIGTERM and really are reaped — while the *conversation* runs against a fake
:class:`~shipinfer.launch.client.ShardClient`. That split is the point: everything this class
decides (which argv, which shard, which deadline, what to do when one shard refuses) is a
launcher-side decision, and testing it against a real shard would be testing CUDA.

The one thing asserted against strings rather than behaviour is the argv, and that is
deliberate too: it is a *contract* between :meth:`FleetRunner.shard_command` and
``cli/shard.py``'s parser, nothing else holds the two ends together, and the previous
mechanism's first version emitted a flag the command it launched did not define.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any, ClassVar

import pytest

from shipinfer.core.errors import (
    ConfigurationError,
    NoShardAvailableError,
    ServerStateError,
)
from shipinfer.core.request import RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.launch.control import AddCameraResult, CameraSpec, ShardHealth, StopResult
from shipinfer.runners import build_runner
from shipinfer.runners.fleet import FleetRunner
from shipinfer.topology import Caps, ChainItem, ChainSpec, Topology
from shipinfer.topology.base import (
    CameraGroup,
    Element,
    ElementContext,
    ElementKind,
)
from shipinfer.topology.registry import registry_for

CHAIN = textwrap.dedent("""
    name: linear
    elements:
      decode: {impl: mock}
      detect: {impl: mock, model: ship_detector}
      output: {impl: mock}
    """)


def topology() -> Topology:
    return Topology.from_spec(ChainSpec.from_yaml(CHAIN))


def sleeps(seconds: float = 30.0):
    """A child that is not a shard. Everything under test is the parent's half."""
    return lambda shard: [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def kill_shard(runner, shard_id: int) -> None:
    """Kill one shard's child the way a segfault would, and wait until the parent can see it.

    ``supervise()`` is deliberately not called: what is under test is the report a live
    launcher gives between polls, and the supervisor's answer to a dead shard is to stop the
    whole fleet. The ``wait()`` is what makes the test deterministic rather than timing-
    dependent — it reaps the child, so ``poll()`` has an exit code from the next line on.
    """
    for running in runner._fleet.running:
        if running.shard.index == shard_id:
            running.process.kill()
            running.process.wait(timeout=5.0)
            return
    raise AssertionError(f"no shard {shard_id} in this fleet")


class FakeClient:
    """A shard that answers, and remembers what it was asked.

    Duck-typed rather than a ``Mock``: every method here is one the runner actually calls,
    so a signature that drifts is an ``AttributeError`` in a test rather than a mock that
    cheerfully returns another mock.
    """

    def __init__(self, shard_id: int, port: int) -> None:
        self.shard_id = shard_id
        self.address = f"127.0.0.1:{port}"
        self.ready_calls = 0
        self.topologies: list[tuple[str, tuple[int, ...], tuple[int, ...], float]] = []
        self.cameras: list[str] = []
        self.stops: list[float] = []
        self.drains: list[float] = []
        self.closed = 0
        #: What a test tells this shard to do.
        self.ready = True
        self.refuse_topology: str | None = None
        self.refuse_cameras: str | None = None
        self.abandoned = 0
        self.remove_clean = True

    def wait_ready(self, timeout_s: float = 120.0, **_: Any) -> bool:
        self.ready_calls += 1
        return self.ready

    def update_topology(
        self,
        chain_yaml: str,
        *,
        shared_by=(),
        share_rank=(),
        timeout_s: float | None = None,
    ) -> str:
        if self.refuse_topology is not None:
            raise ConfigurationError(self.refuse_topology)
        self.topologies.append(
            (chain_yaml, tuple(shared_by), tuple(share_rank), timeout_s or 0.0)
        )
        return "linear"

    def add_camera(self, camera: CameraSpec, **_: Any) -> AddCameraResult:
        if self.refuse_cameras is not None:
            return AddCameraResult(accepted=False, reason=self.refuse_cameras)
        self.cameras.append(camera.camera_id)
        return AddCameraResult(accepted=True)

    def remove_camera(self, camera_id: str, **_: Any) -> bool:
        if camera_id in self.cameras:
            self.cameras.remove(camera_id)
        return self.remove_clean

    def health(self, **_: Any) -> ShardHealth:
        return ShardHealth(
            state="running" if self.cameras else "ready",
            cameras={name: {"state": "running"} for name in self.cameras},
            engine={"status": "healthy"},
        )

    def stats(self, **_: Any) -> dict[str, Any]:
        return {"cameras": len(self.cameras)}

    def drain(self, timeout_s: float = 20.0, **_: Any) -> int:
        self.drains.append(timeout_s)
        self.cameras.clear()
        return self.abandoned

    def stop(self, timeout_s: float = 20.0, **_: Any) -> StopResult:
        self.stops.append(timeout_s)
        return StopResult(abandoned=self.abandoned)

    def close(self) -> None:
        self.closed += 1


@pytest.fixture()
def clients() -> dict[int, FakeClient]:
    return {}


@pytest.fixture()
def runner(clients):
    """A two-shard fleet over GPUs 2 and 3, spawning sleeps and talking to fakes."""

    def factory(shard, port):
        clients[shard.index] = FakeClient(shard.index, port)
        return clients[shard.index]

    built = FleetRunner(
        topology(),
        ServerSettings(),
        chain_yaml=CHAIN,
        shards=2,
        gpus=[2, 3],
        command=sleeps(),
        client=factory,
    )
    yield built
    built.stop(timeout_s=5.0)


class TestTheArgvIsTheIdentityAndNothingElse:
    """arch.md section 2: the child receives ``--shard-id N --control-port P`` and no more."""

    def test_it_is_exactly_the_two_flags(self) -> None:
        assert FleetRunner.shard_command(3, 50103) == [
            sys.executable,
            "-m",
            "shipinfer.cli.shard",
            "--shard-id",
            "3",
            "--control-port",
            "50103",
        ]

    def test_it_runs_this_interpreter_not_whatever_is_on_path(self) -> None:
        """A shard under a different interpreter is a debugging session nobody wants."""
        assert FleetRunner.shard_command(0, 1)[0] == sys.executable

    def test_the_child_accepts_every_flag_it_emits(self) -> None:
        """The two ends of the contract, checked against each other.

        The mechanism this replaces emitted ``--cameras``, which the command it launched did
        not define: the argv read plausibly and the child would have refused it. So this
        parses the real argv with the real parser instead of asserting a shape.
        """
        from shipinfer.cli.shard import build_parser

        argv = FleetRunner.shard_command(7, 50107)
        args = build_parser().parse_args(argv[3:])

        assert (args.shard_id, args.control_port) == (7, 50107)

    def test_the_child_refuses_a_third_flag(self) -> None:
        """Refused rather than ignored: the argv is the contract, so a launcher that grew a
        flag must fail on the first shard rather than have it silently dropped by all of them.
        """
        from shipinfer.cli.shard import build_parser

        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(
                ["--shard-id", "1", "--control-port", "2", "--repository", "/models"]
            )

        assert exit_info.value.code == 2

    def test_ports_are_one_per_shard(self) -> None:
        """One port for the whole fleet is not an option: the second bind is refused."""
        runner = FleetRunner(topology(), chain_yaml=CHAIN, gpus=[0], control_port_base=51000)

        assert [runner.control_port(i) for i in range(3)] == [51000, 51001, 51002]


class TestStartingTheFleet:
    def test_every_shard_is_waited_for_and_told_the_chain(self, runner, clients) -> None:
        runner.start()

        assert sorted(clients) == [0, 1]
        assert all(c.ready_calls == 1 for c in clients.values())
        assert [c.topologies[0][0] for c in clients.values()] == [CHAIN, CHAIN]

    def test_the_sharing_rides_with_the_chain(self, clients) -> None:
        """The assertion ``tests/server/test_shard_settings.py`` used to make of the
        environment: two shards on one GPU must each be told they share it, or each loads the
        full instance count and the device silently holds twice the engines."""

        def factory(shard, port):
            clients[shard.index] = FakeClient(shard.index, port)
            return clients[shard.index]

        runner = FleetRunner(
            topology(),
            chain_yaml=CHAIN,
            shards=2,
            gpus=[4],  # one GPU, two shards
            command=sleeps(),
            client=factory,
        )
        try:
            runner.start()
        finally:
            runner.stop(timeout_s=5.0)

        assert [c.topologies[0][1] for c in clients.values()] == [(2,), (2,)]
        assert sorted(c.topologies[0][2] for c in clients.values()) == [(0,), (1,)]

    def test_the_topology_deadline_is_minutes_not_the_client_default(
        self, runner, clients
    ) -> None:
        """`UpdateTopology` is where a shard deserialises its engines; ten seconds would kill
        a fleet that was starting perfectly well."""
        runner.start()

        assert all(c.topologies[0][3] >= 300.0 for c in clients.values())

    def test_a_shard_that_never_answers_takes_the_fleet_down(self, runner, clients) -> None:
        def factory(shard, port):
            client = FakeClient(shard.index, port)
            client.ready = shard.index == 0
            clients[shard.index] = client
            return client

        runner._client_factory = factory  # the seam the fixture injects

        with pytest.raises(ServerStateError, match="did not answer"):
            runner.start()

        assert not runner.is_running
        assert runner.plan is not None and len(runner.plan) == 2

    def test_the_installs_overlap_instead_of_queueing_behind_each_other(self, clients) -> None:
        """Each install is a `wait_ready` poll plus an engine load — both waits on another
        process. Sequentially, sixteen shards would take sixteen times one shard's start with
        every GPU but one idle. The barrier is the proof: it only clears if all four threads
        reach it, so a sequential loop deadlocks on the first shard and times out."""
        import threading

        barrier = threading.Barrier(4, timeout=10.0)

        class Overlapping(FakeClient):
            def wait_ready(self, timeout_s: float = 120.0, **kwargs: Any) -> bool:
                barrier.wait()  # BrokenBarrierError if the installs are one after another
                return super().wait_ready(timeout_s, **kwargs)

        def factory(shard, port):
            clients[shard.index] = Overlapping(shard.index, port)
            return clients[shard.index]

        runner = FleetRunner(
            topology(),
            chain_yaml=CHAIN,
            shards=4,
            gpus=[0, 1, 2, 3],
            command=sleeps(),
            client=factory,
        )
        try:
            runner.start()  # raises BrokenBarrierError if they do not overlap
        finally:
            runner.stop(timeout_s=5.0)

        assert sorted(clients) == [0, 1, 2, 3]
        assert all(c.topologies for c in clients.values())

    def test_the_failure_reported_is_the_first_in_plan_order_not_the_first_to_lose(
        self, clients
    ) -> None:
        """A fleet must fail the same way twice, or the operator debugs the scheduler instead
        of the deployment. Shard 3 refuses instantly; shard 1 refuses after a pause, so it is
        the *last* thread to finish and still the one named."""
        import time as clock

        def factory(shard, port):
            client = FakeClient(shard.index, port)
            if shard.index == 1:
                client.refuse_topology = "shard 1: no such model 'ship_detector'"
            if shard.index == 3:
                client.refuse_topology = "shard 3: no such element 'mock'"
            clients[shard.index] = client
            return client

        slow = factory

        def factory_with_delay(shard, port):
            client = slow(shard, port)
            if shard.index == 1:
                original = client.update_topology

                def delayed(*args: Any, **kwargs: Any):
                    clock.sleep(0.2)
                    return original(*args, **kwargs)

                client.update_topology = delayed  # type: ignore[method-assign]
            return client

        runner = FleetRunner(
            topology(),
            chain_yaml=CHAIN,
            shards=4,
            gpus=[0, 1, 2, 3],
            command=sleeps(),
            client=factory_with_delay,
        )
        try:
            with pytest.raises(ConfigurationError, match="shard 1"):
                runner.start()
        finally:
            runner.stop(timeout_s=5.0)

    def test_a_refused_topology_stops_the_shards_that_did_start(self, runner, clients) -> None:
        """Half a fleet is the state hardest to notice: every process that *is* up says it is
        healthy. So shard 0 is stopped over RPC, not left running a chain nobody asked for."""

        def factory(shard, port):
            client = FakeClient(shard.index, port)
            if shard.index == 1:
                client.refuse_topology = "no such model 'ship_detector'"
            clients[shard.index] = client
            return client

        runner._client_factory = factory

        with pytest.raises(ConfigurationError, match="no such model"):
            runner.start()

        assert clients[0].stops, "shard 0 was left running after shard 1 refused"
        assert clients[0].closed == 1 and clients[1].closed == 1
        assert not runner.is_running


class TestWhatAFailedStartStillOwesTheOperator:
    """The unwind, and the one number it must not round to zero.

    ``abandoned`` is a lifetime signal, not a statistic (arch.md section 2): while it is
    non-zero, a detached decoder thread on some shard still references buffers nobody may
    unwind. A failed start abandons threads exactly like a shutdown does — the shards that
    *did* come up have frames in flight — and reporting 0 for it is the single lie the signal
    exists to prevent, because 0 is what a caller reads as "safe, take the buffers back".
    """

    @staticmethod
    def _fleet(clients, *, shards: int, refuses: int, abandoned: int = 3):
        def factory(shard, port):
            client = FakeClient(shard.index, port)
            client.abandoned = abandoned
            if shard.index == refuses:
                client.refuse_topology = f"shard {shard.index}: no such model"
            clients[shard.index] = client
            return client

        return FleetRunner(
            topology(),
            ServerSettings(),
            chain_yaml=CHAIN,
            shards=shards,
            gpus=list(range(shards)),
            command=sleeps(),
            client=factory,
        )

    def test_a_failed_start_reports_every_thread_it_abandoned(self, clients) -> None:
        """Two shards, three abandoned each, one refused chain: six, not zero.

        Zero was what it reported, because the ABC's unwind ran a second `_do_stop` over an
        already-emptied client map and *assigned* its result over the six.
        """
        runner = self._fleet(clients, shards=2, refuses=1, abandoned=3)

        with pytest.raises(ConfigurationError, match="no such model"):
            runner.start()

        assert runner.abandoned == 6, "the unwind's abandonment count was overwritten"

    def test_a_drain_and_the_stop_after_it_both_count(self, runner, clients) -> None:
        """Both sets of threads are detached at once; the second writer must not zero the
        first. Accumulated, never assigned (`FleetRunner._abandoned`)."""
        runner.start()
        for client in clients.values():
            client.abandoned = 2

        runner.drain(timeout_s=2.0)
        runner.stop(timeout_s=5.0)

        assert runner.abandoned == 8  # 2 shards x 2, twice

    def test_every_shard_is_stopped_and_the_processes_taken_down(self, clients) -> None:
        """The mutation guard for the unwind itself: delete `Runner.start`'s
        `except BaseException: _do_stop(...)` and this is the test that goes red.

        Half a fleet is the state hardest to notice — every process that *is* up reports
        healthy — so a chain refused on shard 2 stops shards 0 and 1 over RPC, closes every
        channel, and takes the processes down. All three of those, because dropping any one
        leaves either a live chain nobody asked for, a leaked channel, or three children
        holding CUDA contexts on a shared box.
        """
        seen: dict[str, Any] = {}
        runner = self._fleet(clients, shards=3, refuses=2)
        original = runner._client_for

        def capture(shard):
            seen.setdefault("fleet", runner._fleet)
            return original(shard)

        runner._client_for = capture  # type: ignore[method-assign]

        with pytest.raises(ConfigurationError, match="no such model"):
            runner.start()

        assert sorted(clients) == [0, 1, 2]
        assert all(c.stops for c in clients.values()), "a shard was left running its chain"
        assert all(c.closed == 1 for c in clients.values()), "a channel was leaked"
        assert seen["fleet"].running == (), "the shard processes outlived the failed start"
        assert not runner.is_running

    def test_the_unwind_gets_the_shutdown_budget_and_not_zero(self, clients) -> None:
        """`Runner.start` unwinds with `_unwind_timeout_s()`, which a fleet overrides: those
        shards have frames in flight whether or not their sibling ever answered, and a zero
        budget would abandon work an ordinary shutdown finishes — and then report it."""
        settings = ServerSettings(runner={"drain_s": 7.0})

        def factory(shard, port):
            client = FakeClient(shard.index, port)
            if shard.index == 1:
                client.refuse_topology = "no"
            clients[shard.index] = client
            return client

        runner = FleetRunner(
            topology(),
            settings,
            chain_yaml=CHAIN,
            shards=2,
            gpus=[0, 1],
            command=sleeps(),
            client=factory,
        )
        with pytest.raises(ConfigurationError):
            runner.start()

        assert clients[0].stops[0] == pytest.approx(7.0, abs=0.5)


class TestPlacingCameras:
    def test_the_least_loaded_shard_takes_the_next_camera(self, runner, clients) -> None:
        runner.start()
        for index in range(5):
            runner.add_camera(CameraSpec(camera_id=f"quay-{index}", url="rtsp://host"))

        assert len(clients[0].cameras) == 3 and len(clients[1].cameras) == 2
        assert clients[0].cameras == ["quay-0", "quay-2", "quay-4"]

    def test_a_tie_goes_to_the_lower_shard_so_a_fleet_fills_the_same_way_twice(
        self, runner, clients
    ) -> None:
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert clients[0].cameras == ["quay-1"] and clients[1].cameras == []

    def test_a_refusal_places_the_camera_on_another_shard(self, runner, clients) -> None:
        """A refusal is a placement fact, not a failure (``launch/control.py``)."""
        runner.start()
        clients[0].refuse_cameras = "shard 0 is draining and takes no cameras"

        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert clients[1].cameras == ["quay-1"]

    def test_every_shard_refusing_is_an_error_that_names_what_each_said(
        self, runner, clients
    ) -> None:
        """And it is a *capacity* error, not a configuration one.

        `NoShardAvailableError` is a `ServerStateError`, so an HTTP caller gets 503 and backs
        off; the duplicate below stays a `ConfigurationError` and gets 400. Nothing about this
        request is malformed — the shards are all draining, and the one that refused now will
        take the camera in a minute (`api/errors.py`).
        """
        runner.start()
        for client in clients.values():
            client.refuse_cameras = "draining"

        with pytest.raises(NoShardAvailableError, match="no shard would take") as caught:
            runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert caught.value.camera_id == "quay-1"
        assert caught.value.refusals == ("shard 0: draining", "shard 1: draining")
        assert not isinstance(caught.value, ConfigurationError)

    def test_a_duplicate_is_refused_by_the_launcher_before_any_rpc(
        self, runner, clients
    ) -> None:
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        with pytest.raises(ConfigurationError, match="already on shard 0"):
            runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert clients[0].cameras == ["quay-1"], "the second add reached the shard"

    def test_removing_frees_the_slot_for_the_next_camera(self, runner, clients) -> None:
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))
        runner.add_camera(CameraSpec(camera_id="quay-2", url="rtsp://host"))

        assert runner.remove_camera("quay-1") is True
        runner.add_camera(CameraSpec(camera_id="quay-3", url="rtsp://host"))

        assert clients[0].cameras == ["quay-3"]

    def test_an_abandoned_thread_still_frees_the_placement(self, runner, clients) -> None:
        """`clean=False` means the decoder was abandoned, not that the camera is still served.
        Keeping the placement would make that camera unplaceable for the fleet's whole life."""
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))
        clients[0].remove_clean = False

        assert runner.remove_camera("quay-1") is False

        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))  # no refusal

    def test_removing_a_camera_nobody_holds_is_an_error(self, runner) -> None:
        runner.start()

        with pytest.raises(ConfigurationError, match="no shard holds"):
            runner.remove_camera("quay-9")

    def test_cameras_before_start_are_refused_rather_than_queued(self, runner) -> None:
        with pytest.raises(ServerStateError, match="not running"):
            runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))


@registry_for(ElementKind.TRACK).register("grouped-track")
class GroupedTrack(Element):
    """A ``track`` element that declares a camera group. Not a real thing — the point of it.

    ``Element.camera_group()`` is on the ABC and not on ``ShipvisionMtmc``, so a launcher that
    honours it honours *any* element that needs its cameras co-located. This double is how
    that is asserted rather than assumed: it is not an ``mtmc`` element, and the fleet must
    still pin its group.
    """

    kind: ClassVar[ElementKind] = ElementKind.TRACK
    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "meta@cpu")
    produces: ClassVar[tuple[str, ...]] = ("meta@cpu",)

    def camera_group(self) -> CameraGroup:
        return CameraGroup("berth", ("b-0", "b-1", "b-2"))

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem:
        return item


class TestACameraGroupIsAnAtomicUnitOfPlacement:
    """``docs/arch.md`` §4, enforced here because nothing else can enforce it.

    An ``mtmc`` element associates one camera group against one identity space, and that
    tracker lives in one shard process. Split the group across two shards and each half runs
    its own tracker: one object is given two global ids, both of them plausible, and nothing
    in the metrics disagrees. The element cannot see the split — it is told
    :attr:`ElementContext.shard_id` and nothing about where any *camera* went, and it opens
    before a single camera is placed. This class owns ``{camera_id: shard_id}``, so this is
    where the invariant is decidable.
    """

    @pytest.fixture()
    def grouped(self, clients):
        """A two-shard fleet whose chain declares one group of four cameras."""
        chain_yaml = textwrap.dedent("""
            name: grouped
            elements:
              decode: {impl: mock}
              detect: {impl: mock, model: ship_detector}
              track:  {impl: mock}
              mtmc:   {impl: shipvision, params: {group: quay, cameras: [q-0, q-1, q-2, q-3]}}
              output: {impl: mock}
            """)

        def factory(shard, port):
            clients[shard.index] = FakeClient(shard.index, port)
            return clients[shard.index]

        built = FleetRunner(
            Topology.from_spec(ChainSpec.from_yaml(chain_yaml)),
            ServerSettings(),
            chain_yaml=chain_yaml,
            shards=2,
            gpus=[2, 3],
            command=sleeps(),
            client=factory,
        )
        yield built
        built.stop(timeout_s=5.0)

    def test_the_whole_group_follows_the_first_camera_of_it(self, grouped, clients) -> None:
        """Least-loaded placement would have alternated them; the group pins them together."""
        grouped.start()
        for index in range(4):
            grouped.add_camera(CameraSpec(camera_id=f"q-{index}", url="rtsp://host"))

        assert clients[0].cameras == ["q-0", "q-1", "q-2", "q-3"]
        assert clients[1].cameras == []

    def test_a_camera_outside_the_group_is_still_placed_by_load(self, grouped, clients) -> None:
        """The pin is the group's, not a global override: everything else balances as before."""
        grouped.start()
        grouped.add_camera(CameraSpec(camera_id="q-0", url="rtsp://host"))
        grouped.add_camera(CameraSpec(camera_id="visitor", url="rtsp://host"))

        assert clients[0].cameras == ["q-0"]
        assert clients[1].cameras == ["visitor"]

    def test_a_group_whose_shard_died_is_refused_naming_the_group_and_both_shards(
        self, grouped, clients
    ) -> None:
        """Re-placing the survivors would be the tracker reset dressed as failover ADR-018
        refuses, one tier up: every global id under the group would change silently."""
        grouped.start()
        grouped.add_camera(CameraSpec(camera_id="q-0", url="rtsp://host"))
        assert clients[0].cameras == ["q-0"]
        kill_shard(grouped, 0)

        with pytest.raises(NoShardAvailableError) as caught:
            grouped.add_camera(CameraSpec(camera_id="q-1", url="rtsp://host"))

        message = str(caught.value)
        assert "quay" in message
        assert "shard 0" in message and "shard 1" in message
        assert "atomic unit of placement" in message
        assert clients[1].cameras == [], "half the group was placed on the survivor"

    def test_the_refusal_is_a_capacity_answer_and_not_a_configuration_one(
        self, grouped
    ) -> None:
        grouped.start()
        grouped.add_camera(CameraSpec(camera_id="q-0", url="rtsp://host"))
        kill_shard(grouped, 0)

        with pytest.raises(NoShardAvailableError) as caught:
            grouped.add_camera(CameraSpec(camera_id="q-1", url="rtsp://host"))

        assert not isinstance(caught.value, ConfigurationError)

    def test_removing_the_last_camera_frees_the_group_to_move(self, grouped, clients) -> None:
        """The pin is on where the group *is*, not on where it once was."""
        grouped.start()
        grouped.add_camera(CameraSpec(camera_id="q-0", url="rtsp://host"))
        grouped.add_camera(CameraSpec(camera_id="visitor", url="rtsp://host"))
        assert grouped.remove_camera("q-0") is True

        grouped.add_camera(CameraSpec(camera_id="q-1", url="rtsp://host"))

        # Nothing of the group is placed any more, so q-1 is placed by load like any other
        # camera -- onto shard 0, which the removal emptied.
        assert clients[0].cameras == ["q-1"]
        assert clients[1].cameras == ["visitor"]

    def test_a_chain_with_no_roster_places_by_load_as_before(self, runner, clients) -> None:
        """Declaring the roster is what buys the invariant; a chain that did not say gets the
        honest answer, which is that the group is whatever ended up together."""
        runner.start()
        for index in range(4):
            runner.add_camera(CameraSpec(camera_id=f"q-{index}", url="rtsp://host"))

        assert clients[0].cameras == ["q-0", "q-2"]
        assert clients[1].cameras == ["q-1", "q-3"]

    def test_the_launcher_asks_every_element_and_never_what_kind_it_is(self, clients) -> None:
        """The seam, asserted directly. ``_camera_groups`` walks ``Element.camera_group()``
        with no ``ElementKind`` test and no import of an element implementation module, so a
        *second* kind that needs co-located cameras is a method override rather than an
        ``elif`` in ``runners/`` (ADR-017 §2). ``grouped-track`` is that second kind, invented
        here, and it is a ``track`` element rather than an ``mtmc`` one on purpose."""
        chain_yaml = textwrap.dedent("""
            name: grouped-by-another-kind
            elements:
              decode: {impl: mock}
              track:  {impl: grouped-track}
              output: {impl: mock}
            """)

        def factory(shard, port):
            clients[shard.index] = FakeClient(shard.index, port)
            return clients[shard.index]

        built = FleetRunner(
            Topology.from_spec(ChainSpec.from_yaml(chain_yaml)),
            ServerSettings(),
            chain_yaml=chain_yaml,
            shards=2,
            gpus=[2, 3],
            command=sleeps(),
            client=factory,
        )
        try:
            built.start()
            for index in range(3):
                built.add_camera(CameraSpec(camera_id=f"b-{index}", url="rtsp://host"))

            assert clients[0].cameras == ["b-0", "b-1", "b-2"]
            assert clients[1].cameras == []
        finally:
            built.stop(timeout_s=5.0)

    def test_one_camera_in_two_groups_is_refused_when_the_fleet_is_built(self) -> None:
        """A camera that would have to be on two shards at once, refused before a start."""
        chain_yaml = textwrap.dedent("""
            name: overlapping
            elements:
              decode: {impl: mock}
              track:  {impl: mock}
              mtmc:   {impl: shipvision, params: {group: quay, cameras: [q-0]}}
              mtmc_2: {impl: shipvision, kind: mtmc, params: {group: gate, cameras: [q-0]}}
              output: {impl: mock}
            """)

        with pytest.raises(ConfigurationError, match="claimed by camera groups"):
            FleetRunner(
                Topology.from_spec(ChainSpec.from_yaml(chain_yaml)),
                ServerSettings(),
                chain_yaml=chain_yaml,
                shards=2,
                gpus=[2, 3],
                command=sleeps(),
            )


class TestTheLockIsNeverHeldAcrossAnRpc:
    """`health()` during a slow `AddCamera` is the call an operator makes while wondering why
    a camera is dark. Holding `_lock` across the RPC made it the one call that hangs."""

    def test_health_answers_while_an_add_camera_is_blocked(self, runner, clients) -> None:
        import threading

        released = threading.Event()
        entered = threading.Event()

        def blocking(camera: CameraSpec, **_: Any) -> AddCameraResult:
            entered.set()
            assert released.wait(10.0), "the test never released the fake shard"
            clients[0].cameras.append(camera.camera_id)
            return AddCameraResult(accepted=True)

        runner.start()
        clients[0].add_camera = blocking  # type: ignore[method-assign]
        adding = threading.Thread(
            target=runner.add_camera,
            args=(CameraSpec(camera_id="quay-1", url="rtsp://host"),),
        )
        adding.start()
        try:
            assert entered.wait(5.0), "the add never reached the shard"

            report = runner.health()  # would block here for as long as the RPC took
            stats = runner.stats()

            assert sorted(report["shards"]) == ["0", "1"]
            assert report["cameras"] == {"quay-1": {"shard": 0, "pending": True}}
            assert report["shards"]["0"]["placed"] == [], "an unconfirmed camera read as placed"
            assert stats["cameras"] == 0
        finally:
            released.set()
            adding.join(timeout=10.0)

        assert runner.health()["cameras"] == {"quay-1": {"shard": 0}}
        assert runner.health()["shards"]["0"]["placed"] == ["quay-1"]

    def test_a_reservation_counts_against_the_shard_it_is_offered_to(
        self, runner, clients
    ) -> None:
        """Otherwise a second concurrent placement picks the same "emptiest" shard, which is
        the balance failure ADR-005 exists to prevent, one level up."""
        import threading

        released = threading.Event()

        def blocking(camera: CameraSpec, **_: Any) -> AddCameraResult:
            assert released.wait(10.0)
            clients[0].cameras.append(camera.camera_id)
            return AddCameraResult(accepted=True)

        runner.start()
        clients[0].add_camera = blocking  # type: ignore[method-assign]
        first = threading.Thread(
            target=runner.add_camera,
            args=(CameraSpec(camera_id="quay-1", url="rtsp://host"),),
        )
        first.start()
        try:
            while runner.health()["cameras"].get("quay-1") is None:
                pass

            runner.add_camera(CameraSpec(camera_id="quay-2", url="rtsp://host"))
        finally:
            released.set()
            first.join(timeout=10.0)

        assert clients[1].cameras == ["quay-2"], "the second camera piled onto shard 0"

    def test_a_camera_no_shard_took_is_placeable_again(self, runner, clients) -> None:
        """The reservation is released on every way out but a commit."""
        runner.start()
        for client in clients.values():
            client.refuse_cameras = "draining"

        with pytest.raises(NoShardAvailableError, match="no shard would take"):
            runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert runner.health()["cameras"] == {}
        for client in clients.values():
            client.refuse_cameras = None

        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))  # not a duplicate

    def test_an_rpc_that_raises_releases_the_reservation_too(self, runner, clients) -> None:
        runner.start()

        def explode(*_a: Any, **_k: Any):
            raise ServerStateError("the channel died mid-add")

        for client in clients.values():
            client.add_camera = explode  # type: ignore[method-assign]

        with pytest.raises(ServerStateError, match="channel died"):
            runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert runner.health()["cameras"] == {}


class _HookedLock:
    """A lock that runs a callback the first time it is taken.

    The window N2 is about - between a guard read outside the lock and the lock that follows
    it - is microseconds wide, so it is *provoked* rather than waited for. The callback (a
    concurrent ``stop()``) runs before the lock is acquired, which is exactly where a real
    stop would have landed, and every later acquisition is the plain lock again.
    """

    def __init__(self, inner: Any, once: Any) -> None:
        self._inner = inner
        self._once = once

    def __enter__(self) -> Any:
        once, self._once = self._once, None
        if once is not None:
            once()
        return self._inner.__enter__()

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc)


class TestAStopRacingACameraCallIsATypedRefusal:
    """`stop()` empties the client map. A guard that ran beside the lock rather than inside it
    let the next line index that empty map, so the caller got an `IndexError`/`KeyError` where
    the docstring promised a `ServerStateError` - the difference between "the fleet is down,
    place this elsewhere" and a traceback a supervisor cannot classify."""

    def test_a_stop_during_an_add_camera_rpc_refuses_the_next_shard_typed(
        self, runner, clients
    ) -> None:
        """The real race: the lock is down for the whole of shard 0's RPC, and the stop lands
        in that window. Shard 0 refuses, so the loop comes back for the lock - and the map it
        used to index is empty."""
        import threading

        entered = threading.Event()
        released = threading.Event()
        raised: list[BaseException] = []

        def blocking(camera: CameraSpec, **_: Any) -> AddCameraResult:
            entered.set()
            assert released.wait(10.0), "the test never released the fake shard"
            return AddCameraResult(accepted=False, reason="draining")

        def add() -> None:
            try:
                runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))
            # Caught broadly on purpose: which type comes out is the whole assertion.
            except BaseException as exc:
                raised.append(exc)

        runner.start()
        clients[0].add_camera = blocking  # type: ignore[method-assign]
        adding = threading.Thread(target=add)
        adding.start()
        try:
            assert entered.wait(5.0), "the add never reached the shard"
            runner.stop(timeout_s=5.0)
        finally:
            released.set()
            adding.join(timeout=10.0)

        assert raised, "the add returned as if the camera had been placed"
        assert isinstance(raised[0], ServerStateError), f"got {raised[0]!r}"
        assert "not running" in str(raised[0])

    def test_a_stop_between_the_guard_and_the_lock_is_still_typed_on_remove(
        self, runner, clients
    ) -> None:
        """`_release()` empties the client map without emptying the placements, so this is the
        one where the old guard handed the next line a `KeyError`."""
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))
        runner._lock = _HookedLock(runner._lock, lambda: runner.stop(timeout_s=5.0))

        with pytest.raises(ServerStateError, match="not running"):
            runner.remove_camera("quay-1")

    def test_a_stop_between_the_guard_and_the_lock_is_still_typed_on_add(
        self, runner, clients
    ) -> None:
        """The same window on the placement side: `_by_load()` over an empty map returns an
        empty order, and `order[0]` was an `IndexError`."""
        runner.start()
        runner._lock = _HookedLock(runner._lock, lambda: runner.stop(timeout_s=5.0))

        with pytest.raises(ServerStateError, match="not running"):
            runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))


class TestStoppingCostsWhatItCosts:
    def test_every_shard_is_stopped_and_its_channel_closed(self, runner, clients) -> None:
        runner.start()
        runner.stop(timeout_s=5.0)

        assert all(c.stops for c in clients.values())
        assert all(c.closed == 1 for c in clients.values())
        assert not runner.is_running

    def test_the_deadline_is_one_budget_for_the_whole_fleet(self, runner, clients) -> None:
        """Not one per shard: each is given what is *left* of the budget, so a shard that
        takes four seconds of a six-second stop leaves two for its sibling rather than
        starting a fresh six. Charging per shard would make N of them N waits."""
        import time

        def slow(timeout_s: float = 20.0, **_: Any) -> StopResult:
            clients[0].stops.append(timeout_s)
            time.sleep(1.0)
            return StopResult(abandoned=0)

        runner.start()
        clients[0].stop = slow  # type: ignore[method-assign]
        runner.stop(timeout_s=3.0)

        assert clients[0].stops[0] == pytest.approx(3.0, abs=0.2)
        assert clients[1].stops[0] < 2.5, "the second shard was given a fresh budget"

    def test_the_abandonment_counts_are_summed(self, runner, clients) -> None:
        """A lifetime signal, not a statistic: while it is non-zero a detached thread on some
        shard still references buffers nobody may unwind (arch.md section 2)."""
        runner.start()
        clients[0].abandoned = 2
        clients[1].abandoned = 3

        runner.stop(timeout_s=5.0)

        assert runner.abandoned == 5

    def test_draining_sums_them_too_and_forgets_the_placements(self, runner, clients) -> None:
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))
        clients[0].abandoned = 1

        assert runner.drain(timeout_s=4.0) == 1
        assert runner.health()["cameras"] == {}

    def test_a_shard_that_cannot_be_drained_does_not_skip_the_others(
        self, runner, clients
    ) -> None:
        """The one unreachable shard is precisely the one whose neighbours must still be
        emptied — and what came back from them is still added, because those threads are
        detached whatever the unreachable shard is doing."""
        runner.start()

        def explode(*_a: Any, **_k: Any) -> int:
            raise ServerStateError("shard 0 did not answer Drain")

        clients[0].drain = explode  # type: ignore[method-assign]
        clients[1].abandoned = 2

        assert runner.drain(timeout_s=4.0) == 2
        assert clients[1].drains, "shard 1 was skipped because shard 0 could not be asked"
        assert runner.abandoned == 2

    def test_a_drain_and_the_stop_after_it_accumulate(self, runner, clients) -> None:
        """Both sets of threads are still detached. A lifetime signal the next writer can
        overwrite is worse than none: the last writer is the one with nothing left to
        release, and assigning made its zero the only truth."""
        runner.start()
        clients[0].abandoned = 1
        clients[1].abandoned = 2

        runner.drain(timeout_s=4.0)
        runner.stop(timeout_s=5.0)

        assert runner.abandoned == 6, "3 abandoned by the drain plus 3 by the stop"

    def test_a_drain_does_not_forget_a_camera_whose_add_is_still_in_flight(
        self, runner, clients
    ) -> None:
        """The carried bug: clearing the reservations too made an accepted camera vanish.

        `drain()` forgets everything the launcher *knows* is placed, but a camera whose
        `AddCamera` is in flight has an outcome nobody has heard yet. Clearing `_pending` as
        well meant the add committed into an emptied map a moment later: the shard was
        reading a camera the authoritative placement map did not contain, so it appeared in
        no listing and `remove_camera` refused it for the fleet's whole life.
        """
        import threading

        entered, released = threading.Event(), threading.Event()

        def blocking(camera: CameraSpec, **_: Any) -> AddCameraResult:
            entered.set()
            assert released.wait(10.0), "the test never released the fake shard"
            clients[0].cameras.append(camera.camera_id)
            return AddCameraResult(accepted=True)

        runner.start()
        clients[0].add_camera = blocking  # type: ignore[method-assign]
        adding = threading.Thread(
            target=runner.add_camera,
            args=(CameraSpec(camera_id="quay-1", url="rtsp://host"),),
        )
        adding.start()
        try:
            assert entered.wait(5.0), "the add never reached the shard"

            assert runner.drain(timeout_s=4.0) == 0
            assert runner.health()["cameras"] == {"quay-1": {"shard": 0, "pending": True}}
        finally:
            released.set()
            adding.join(10.0)

        assert runner.health()["cameras"] == {"quay-1": {"shard": 0}}
        assert runner.remove_camera("quay-1") is True, "the accepted camera was untracked"

    def test_a_drain_still_forgets_a_placement_whose_add_already_returned(
        self, runner, clients
    ) -> None:
        """The other half of the same rule: a confirmed placement was drained, so it goes.
        Keeping it would make every camera unplaceable after the first drain."""
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        runner.drain(timeout_s=4.0)
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert runner.health()["cameras"] == {"quay-1": {"shard": 0}}

    def test_a_refused_add_across_a_drain_leaves_nothing_behind(self, runner, clients) -> None:
        """The reservation survives the drain, and `add_camera`'s own rollback resolves it:
        refused by every shard, and the map is empty rather than holding a ghost."""
        import threading

        entered, released = threading.Event(), threading.Event()

        def blocking(camera: CameraSpec, **_: Any) -> AddCameraResult:
            entered.set()
            assert released.wait(10.0), "the test never released the fake shard"
            return AddCameraResult(accepted=False, reason="draining")

        runner.start()
        clients[0].add_camera = blocking  # type: ignore[method-assign]
        clients[1].refuse_cameras = "draining"
        failed: list[BaseException] = []

        def place() -> None:
            try:
                runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))
            except BaseException as exc:
                failed.append(exc)

        adding = threading.Thread(target=place)
        adding.start()
        try:
            assert entered.wait(5.0), "the add never reached the shard"
            runner.drain(timeout_s=4.0)
        finally:
            released.set()
            adding.join(10.0)

        assert isinstance(failed[0], NoShardAvailableError)
        assert runner.health()["cameras"] == {}

    def test_a_shard_that_cannot_be_asked_does_not_skip_the_others(
        self, runner, clients
    ) -> None:
        runner.start()

        def explode(*_a: Any, **_k: Any):
            raise ServerStateError("shard 0 did not answer Stop")

        clients[0].stop = explode  # type: ignore[method-assign]
        runner.stop(timeout_s=5.0)

        assert clients[1].stops, "shard 1 was skipped because shard 0 could not be asked"

    def test_stopping_twice_is_idempotent(self, runner, clients) -> None:
        runner.start()
        runner.stop(timeout_s=5.0)
        runner.stop(timeout_s=5.0)

        assert [len(c.stops) for c in clients.values()] == [1, 1]


class TestWhatTheFleetReports:
    def test_health_has_one_entry_per_shard(self, runner, clients) -> None:
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        report = runner.health()

        assert report["runner"] == "fleet" and report["state"] == "running"
        assert sorted(report["shards"]) == ["0", "1"]
        assert report["shards"]["0"]["placed"] == ["quay-1"]
        assert report["shards"]["0"]["cameras"] == {"quay-1": {"state": "running"}}
        assert report["cameras"] == {"quay-1": {"shard": 0}}

    def test_an_unreachable_shard_is_named_rather_than_omitted(self, runner, clients) -> None:
        """An absent entry reads as "no such shard", and naming the one that stopped answering
        is the whole point of a fleet health report."""
        runner.start()

        def explode(**_k: Any):
            raise ServerStateError("shard 1 did not answer Health")

        clients[1].health = explode  # type: ignore[method-assign]

        assert runner.health()["shards"]["1"]["state"] == "unreachable"

    def test_stats_survive_a_shard_that_cannot_answer(self, runner, clients) -> None:
        runner.start()

        def explode(**_k: Any):
            raise ServerStateError("no")

        clients[0].stats = explode  # type: ignore[method-assign]
        stats = runner.stats()

        assert "detail" in stats["shards"]["0"] and stats["shards"]["1"] == {"cameras": 0}


class TestWhenAShardDies:
    """Its cameras are reported **lost**, and nothing re-places them (ADR-018).

    The three reasons, restated so a reader of the tests has them: a camera's tracker is
    stateful and lives on its home shard (arch.md section 4), so moving the camera is a
    tracker reset dressed as high availability; nothing respawns the process, so a survivor
    would hold those cameras for the rest of the deployment; and the survivors' GPUs were
    given a ``shared_by`` when the plan was made, so piling a dead shard's cameras onto them
    oversubscribes devices that were never told.

    The processes here are real ``sleeps()`` children and one of them is really killed —
    that is the only part of a shard's death this parent can observe anyway.
    """

    @staticmethod
    def _four_cameras(runner) -> None:
        for index in range(4):
            runner.add_camera(CameraSpec(camera_id=f"quay-{index}", url="rtsp://host"))

    def test_health_names_exactly_the_dead_shards_cameras(self, runner, clients) -> None:
        runner.start()
        self._four_cameras(runner)  # quay-0/2 on shard 0, quay-1/3 on shard 1

        kill_shard(runner, 0)
        report = runner.health()

        assert report["lost"] == {"quay-0": 0, "quay-2": 0}

    def test_no_camera_goes_missing_from_the_report(self, runner, clients) -> None:
        """Reported, never deleted: an entry that vanished would read as "never placed", and
        "I have never heard of it" is the wrong answer to "where did my camera go"."""
        runner.start()
        self._four_cameras(runner)

        kill_shard(runner, 0)
        report = runner.health()

        assert sorted(report["cameras"]) == ["quay-0", "quay-1", "quay-2", "quay-3"]
        assert report["cameras"]["quay-0"] == {"shard": 0}

    def test_one_report_does_not_contradict_itself(self, runner, clients) -> None:
        """The dead shard's `placed` list and the `lost` map are derived from ONE snapshot.

        Asking twice is how a report comes to list a camera as running on shard 0 three lines
        above saying it is lost.
        """
        runner.start()
        self._four_cameras(runner)

        kill_shard(runner, 0)
        report = runner.health()

        assert report["shards"]["0"]["placed"] == []
        assert set(report["lost"]) == {"quay-0", "quay-2"}

    def test_the_survivor_is_unaffected(self, runner, clients) -> None:
        runner.start()
        self._four_cameras(runner)

        kill_shard(runner, 0)
        report = runner.health()

        assert report["shards"]["1"]["placed"] == ["quay-1", "quay-3"]
        assert report["shards"]["1"]["state"] == "running"
        assert not {"quay-1", "quay-3"} & set(report["lost"])

    def test_a_live_fleet_reports_nothing_lost(self, runner, clients) -> None:
        """Through the report, which is the only way anything reads loss: the launcher has no
        loss method of its own, and a test that called one would be pinning a private helper
        rather than the answer an operator gets."""
        runner.start()
        self._four_cameras(runner)
        report = runner.health()

        assert report["lost"] == {}
        assert [entry["exited"] for entry in report["shards"].values()] == [False, False]
        assert runner.stats()["lost"] == 0

    def test_an_unreachable_shard_is_not_a_lost_one(self, runner, clients) -> None:
        """A shard that is wedged, paging in an engine or slow is ALIVE and may answer again.

        Conflating the two would report a camera lost — which is terminal until an operator
        removes it — because its shard took two seconds over a health probe.
        """
        runner.start()
        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        def explode(**_k: Any):
            raise ServerStateError("shard 0 did not answer Health")

        clients[0].health = explode  # type: ignore[method-assign]
        report = runner.health()

        assert report["shards"]["0"]["state"] == "unreachable"
        assert report["lost"] == {}
        assert report["shards"]["0"]["placed"] == ["quay-1"]

    def test_a_dead_shard_with_no_cameras_still_says_it_exited(self, runner, clients) -> None:
        """`unreachable` with an empty `lost` is exactly what a wedged shard reads, so on its
        own it cannot answer the one question an operator has: wait, or remove-and-re-add.
        `exited` is the only field that separates them.

        Both shards are made to raise from `health()` here because that is what a real
        channel does on either side of the distinction - a dead process refuses the
        connection, a wedged one runs out the deadline - and the fake would otherwise keep
        answering cheerfully for a child that has been killed.
        """
        runner.start()

        def explode(**_k: Any):
            raise ServerStateError("no answer")

        clients[0].health = explode  # type: ignore[method-assign]
        clients[1].health = explode  # type: ignore[method-assign]
        kill_shard(runner, 0)
        report = runner.health()

        assert report["lost"] == {}, "neither shard held a camera"
        assert report["shards"]["0"]["state"] == report["shards"]["1"]["state"] == "unreachable"
        assert report["shards"]["0"]["exited"] is True
        assert report["shards"]["1"]["exited"] is False

    def test_a_camera_being_placed_when_its_shard_dies_is_pending_not_lost(
        self, runner, clients
    ) -> None:
        """`lost` is a subset of the placed cameras, and `cameras` has always subtracted the
        reservations - so counting a camera whose `AddCamera` is still in flight would report
        `cameras: 0, lost: 1`, one dark camera out of none placed.

        The add is held inside the shard's RPC, which is where the launcher's lock is down by
        design, and the shard is killed in that window. `add_camera`'s own commit resolves it
        afterwards: this shard accepted, so the camera becomes a placement and *then* a
        lost one.
        """
        import threading

        runner.start()
        started, release = threading.Event(), threading.Event()
        accept = clients[0].add_camera

        def block_then_accept(camera, **kwargs):
            started.set()
            release.wait(timeout=5.0)
            return accept(camera, **kwargs)

        clients[0].add_camera = block_then_accept  # type: ignore[method-assign]
        placing = threading.Thread(
            target=runner.add_camera, args=(CameraSpec(camera_id="quay-0", url="rtsp://h"),)
        )
        placing.start()
        try:
            assert started.wait(5.0), "the placement never reached the shard"
            kill_shard(runner, 0)
            mid_flight, stats = runner.health(), runner.stats()
        finally:
            release.set()
            placing.join(timeout=5.0)

        assert mid_flight["cameras"] == {"quay-0": {"shard": 0, "pending": True}}
        assert mid_flight["lost"] == {}, "pending and lost are exclusive"
        assert (stats["cameras"], stats["lost"]) == (0, 0)
        assert runner.health()["lost"] == {"quay-0": 0}, "the commit resolved it into a loss"

    def test_stats_count_the_lost_as_a_subset_of_the_placed(self, runner, clients) -> None:
        runner.start()
        self._four_cameras(runner)

        kill_shard(runner, 0)
        stats = runner.stats()

        assert stats["cameras"] == 4 and stats["lost"] == 2

    def test_removing_a_lost_camera_is_not_clean_and_asks_nobody(self, runner, clients) -> None:
        """`clean=True` would say the camera was shut down in an orderly way. Its process was.

        Nothing ran the decoder's release, so whatever that thread held was abandoned rather
        than finished — and there is no one left to ask, so no RPC is attempted.
        """
        runner.start()
        self._four_cameras(runner)
        kill_shard(runner, 0)

        assert runner.remove_camera("quay-0") is False
        assert clients[0].cameras == ["quay-0", "quay-2"], "a corpse was sent RemoveCamera"

    def test_removing_a_lost_camera_frees_the_id_for_a_survivor(self, runner, clients) -> None:
        """The one recovery a fleet that does not respawn can offer, and it is the operator's
        to ask for: remove, then add, and the camera lands somewhere alive."""
        runner.start()
        self._four_cameras(runner)
        kill_shard(runner, 0)
        runner.remove_camera("quay-0")

        runner.add_camera(CameraSpec(camera_id="quay-0", url="rtsp://host"))

        assert "quay-0" in clients[1].cameras
        assert runner.health()["lost"] == {"quay-2": 0}

    def test_a_lost_camera_is_still_a_duplicate_until_it_is_removed(
        self, runner, clients
    ) -> None:
        """Because the placement is reported, not deleted. Re-adding the same id without
        removing it would leave two records of one camera and no way to tell them apart."""
        runner.start()
        self._four_cameras(runner)
        kill_shard(runner, 0)

        with pytest.raises(ConfigurationError, match="already on shard 0"):
            runner.add_camera(CameraSpec(camera_id="quay-0", url="rtsp://host"))

    def test_a_new_camera_lands_on_a_survivor(self, runner, clients) -> None:
        """Shard 0 is the one `_by_load` would choose and is not offered the camera at all:
        `AddCamera` against a corpse spends the whole deadline discovering what the process
        table already said, and the placement loop does not catch an RPC that *raises*.

        Nothing is placed first on purpose. The counts tie at `{0: 0, 1: 0}`, so the tie-break
        on the shard index puts the DEAD shard 0 at the head of the order, and only the dead
        filter moves the camera along: remove that filter and this test fails, which an
        earlier version - which placed a camera on shard 1 before killing shard 0, and so had
        already ordered shard 1 first - did not.
        """
        runner.start()
        kill_shard(runner, 0)

        runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert clients[1].cameras == ["quay-1"]
        assert clients[0].cameras == [], "a corpse was sent AddCamera"

    def test_every_shard_dead_is_a_capacity_refusal_that_names_them(
        self, runner, clients
    ) -> None:
        """503 rather than 400 (`api/errors.py`): nothing about the request is wrong, and the
        fleet a supervisor is about to restart will take the camera."""
        runner.start()
        kill_shard(runner, 0)
        kill_shard(runner, 1)

        with pytest.raises(NoShardAvailableError) as caught:
            runner.add_camera(CameraSpec(camera_id="quay-1", url="rtsp://host"))

        assert caught.value.refusals == (
            "shard 0: its process has exited",
            "shard 1: its process has exited",
        )
        assert not isinstance(caught.value, ConfigurationError)
        assert runner.health()["cameras"] == {}, "a refused placement kept its reservation"


class TestSupervisingTheShards:
    def test_request_stop_reaches_the_fleet_underneath(self, runner, clients) -> None:
        """`Fleet.supervise` watches the fleet's own flag, so a handler that set only the
        runner's would be noticed a poll later at best — and not at all while the loop is
        inside its sleep."""
        runner.start()
        assert runner._fleet is not None

        runner.request_stop()

        assert runner.stop_requested and runner._fleet._stopped.is_set()

    def test_supervise_returns_when_asked_and_takes_the_processes_down(
        self, runner, clients
    ) -> None:
        import threading
        import time as clock

        runner.start()
        threading.Timer(0.1, runner.request_stop).start()
        started = clock.monotonic()

        runner.supervise(poll_s=0.02)

        assert clock.monotonic() - started < 5.0
        assert runner._fleet is None or runner._fleet.running == ()

    def test_supervising_a_fleet_that_is_not_running_is_a_typed_refusal(self, runner) -> None:
        with pytest.raises(ServerStateError, match="not running"):
            runner.supervise()

    def test_a_fleet_released_under_the_guard_refuses_typed_and_not_by_assertion(
        self, runner, clients
    ) -> None:
        """This used to be `assert self._fleet is not None`, which vanishes under `python -O`
        and leaves an `AttributeError` on `None` in its place - on the one call whose job is
        to tell an operator what state the deployment is in."""
        runner.start()
        runner._fleet = None  # what a stop that landed under the guard leaves behind

        with pytest.raises(ServerStateError, match="nothing left to supervise"):
            runner.supervise()


class TestWhatTheFleetRefuses:
    def test_it_executes_no_items_of_its_own(self, runner) -> None:
        """Funnelling every camera's frames through the parent would rebuild the single
        shared buffer this project exists to delete, one process higher up."""
        runner.start()
        item = ChainItem(
            context=RequestContext(camera_id="quay-1", frame_id=1), caps=Caps.parse("nv12@gpu")
        )

        with pytest.raises(ServerStateError, match="does not execute items"):
            runner.submit(item)

    def test_a_fleet_with_no_chain_text_is_refused_at_construction(self) -> None:
        """The loader lives on the shard (ADR-017), so the *text* is what travels."""
        with pytest.raises(ConfigurationError, match="chain's \\*text\\*"):
            FleetRunner(topology(), gpus=[0])

    def test_a_fleet_with_no_gpus_is_refused_at_start(self) -> None:
        """The driver is deliberately not asked here — `runners` imports no `runtime`, and
        the parent is the one process in the deployment that needs no device at all."""
        runner = FleetRunner(topology(), chain_yaml=CHAIN, gpus=[])

        with pytest.raises(ConfigurationError, match="which GPUs"):
            runner.start()


class TestItIsReachedThroughTheRegistry:
    def test_build_runner_makes_one(self) -> None:
        runner = build_runner("fleet", topology(), None, chain_yaml=CHAIN, gpus=[0])

        assert isinstance(runner, FleetRunner) and runner.name == "fleet"

    def test_the_plan_is_readable_before_anything_is_spawned(self, runner) -> None:
        """`--dry-run` is the same computation the start does, not a second description of
        it: planning is pure, so asking before spawning gives the plan that would be run."""
        planned = runner.describe_plan()

        assert "2 shard(s)" in planned and "gpu(s) [2]" in planned

        runner.start()

        assert runner.describe_plan() == planned
