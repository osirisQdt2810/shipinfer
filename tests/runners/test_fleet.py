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
from typing import Any

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
