"""One test over a real gRPC server, because everything else here fakes the transport.

``tests/runners/test_shard_service.py`` drives the servicer as plain methods and
``test_client_without_grpcio.py`` drives the client with no transport at all. Both are the
right shape for what they assert — and neither would notice if the servicer were never bound
to the server, if a message name in the ``.proto`` did not match the handler's, or if the
generated stub's package-relative import were wrong. Those are wiring failures, and wiring is
what a socket proves.

So: **one** file, one loopback server on an ephemeral port, and the four calls a launcher
actually makes in order. It is still the offline tier — no GPU, no container, no driver: the
chain is mock elements end to end, which ``runners/`` already relies on.

Every timeout here is bounded and the server is stopped in a ``finally``. A test that leaves a
gRPC server alive leaves a thread pool and a listening socket behind, and the next test to ask
for "an ephemeral port" is the one that fails.
"""

from __future__ import annotations

import socket
import textwrap
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

pytest.importorskip("grpc", reason="the grpc extra is not installed")

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import Priority
from shipinfer.core.settings.ingest import CameraConfig
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.frame import FrameCounter
from shipinfer.ingest.registry import create_source
from shipinfer.launch import CameraSpec, ShardClient, ShardState
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.runners.service import ShardServer, serve_shard
from shipinfer.topology import ChainSpec, Topology

#: A straight line of mock elements behind a real decode element: the smallest chain a real
#: runner will start, and enough to prove the shard is executing something rather than
#: answering from an empty object. The head is ``replay`` rather than ``mock`` so that
#: ``AddCamera`` reaches a real ingest source — a path that does not exist, which the replay
#: source refuses before it imports a codec, so the actor retries on its own thread and this
#: file needs no OpenCV and no network.
CHAIN = """
name: rpc_linear
elements:
  decode: {impl: replay}
  detect: {impl: mock, model: ship_detector}
  output: {impl: mock}
"""


class RecordingRunner(InprocessRunner):
    """The in-process runner, keeping the :class:`CameraConfig` it resolved per camera.

    One added attribute and no changed behaviour: the factory notes the record and then builds
    the source the registry would have built anyway, so the shard under test is still driving
    a real replay source at a path that does not exist.

    It exists so this file can assert what a launcher's band *became* without reading the
    runner's private tables. That record is the runner's own resolution of the band
    (``runners/inprocess.py::_camera_config``), which is the thing that has to be right after
    the value has crossed a socket.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.records: dict[str, CameraConfig] = {}
        super().__init__(*args, source_factory=self._record, **kwargs)

    def _record(self, config: CameraConfig, counter: FrameCounter) -> FrameSource:
        self.records[config.camera_id] = config
        return create_source(config, counter)


def _eventually(predicate: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    """Poll a predicate to a deadline. The camera's record is written on the actor's thread."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _free_port() -> int:
    """A port nothing is listening on. The usual trick: bind 0, read it back, release it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def shard() -> Iterator[tuple[ShardServer, RecordingRunner]]:
    """A shard serving on a loopback ephemeral port, torn down whatever the test does."""
    chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
    runner = RecordingRunner(chain, workers=1)
    server = serve_shard(runner, shard_id=4, control_port=0)
    try:
        yield server, runner
    finally:
        server.stop(grace_s=1.0)
        runner.stop(timeout_s=2.0)


@pytest.fixture()
def client(shard: tuple[ShardServer, RecordingRunner]) -> Iterator[ShardClient]:
    server, _ = shard
    with ShardClient(server.identity.control_port, shard_id=4, timeout_s=5.0) as connected:
        yield connected


class TestALauncherTalksToAShard:
    def test_the_whole_sequence_a_launcher_performs(
        self, shard: tuple[ShardServer, RecordingRunner], client: ShardClient
    ) -> None:
        """Spawn -> wait_ready -> UpdateTopology -> AddCamera -> Health -> Stop, in order.

        One test rather than six, deliberately: these calls are a *sequence* — the state the
        shard reports after each one is the assertion — and splitting them would need the
        earlier calls repeated as setup in every later one anyway.
        """
        server, runner = shard

        assert client.wait_ready(timeout_s=15.0)
        assert client.identity is not None
        assert client.identity.shard_id == 4
        # The port the shard actually bound, which is not the 0 it was asked for. A launcher
        # that assumed its request was honoured would address nothing.
        assert client.identity.control_port == server.identity.control_port != 0
        assert client.health().state == ShardState.STARTING

        assert client.update_topology(
            textwrap.dedent(CHAIN), shared_by=[2], share_rank=[1]
        ) == ("rpc_linear")
        assert server.service.shared_by == (2,)
        assert server.service.share_rank == (1,)
        assert runner.is_running

        health = client.health()
        assert health.state == ShardState.READY
        assert health.engine["runner"] == "inprocess"
        assert health.engine["topology"] == "rpc_linear"
        assert health.cameras == {}
        assert health.detail == ""

        # The in-process runner owns an ingest manager (phase B1), so the camera is taken and
        # the shard's state follows from the camera map rather than from a flag.
        accepted = client.add_camera(
            CameraSpec(
                "cam-1", "/nonexistent/clip.mp4", 20.0, priority=Priority.TRACKING_CRITICAL
            )
        )
        assert accepted.accepted, accepted.reason
        assert client.health().state == ShardState.RUNNING
        assert set(client.health().cameras) == {"cam-1"}
        # The band crossed a real socket into a shard with an EMPTY camera config -- which is
        # the shape a fleet shard actually has (`runners/inprocess.py::_ingest`), and the one
        # in which `tracking_critical` used to arrive as `normal`. `TRACKING_CRITICAL` is 0,
        # so this also fails if the wire ever carries the band as a bare int. Read off the
        # camera record the runner resolved rather than out of its band tables: the record is
        # what the ingest plane was told, and it is written on the actor's thread.
        assert not runner.settings.ingest.cameras
        assert _eventually(lambda: "cam-1" in runner.records)
        assert runner.records["cam-1"].priority is Priority.TRACKING_CRITICAL

        # A camera that is not running is a typed refusal reaching the launcher as data, and
        # the id it names is the one the launcher asked about.
        duplicate = client.add_camera(CameraSpec("cam-1", "/nonexistent/clip.mp4", 20.0))
        assert not duplicate.accepted
        assert "already running" in duplicate.reason

        result = client.stop(timeout_s=2.0)
        assert result.abandoned == 0
        assert result.clean
        assert not runner.is_running
        assert client.health().state == ShardState.STOPPED

    def test_stats_cross_the_wire_as_the_runners_own_counters(
        self, client: ShardClient
    ) -> None:
        client.wait_ready(timeout_s=15.0)
        client.update_topology(textwrap.dedent(CHAIN))

        stats = client.stats()

        assert stats["runner"] == "inprocess"
        assert stats["topology"] == "rpc_linear"
        # Struct numbers are doubles, which is why this compares against 0 rather than `is 0`.
        assert stats["items"]["walked"] == 0

    def test_removing_a_camera_this_shard_never_had_is_a_typed_answer(
        self, client: ShardClient
    ) -> None:
        client.wait_ready(timeout_s=15.0)
        client.update_topology(textwrap.dedent(CHAIN))

        with pytest.raises(ConfigurationError, match="cam-nope"):
            client.remove_camera("cam-nope", timeout_s=0.5)


class TestABoundPortIsRefusedBeforeAnyCamera:
    def test_a_second_shard_on_the_same_port_refuses_typed(
        self, shard: tuple[ShardServer, RecordingRunner]
    ) -> None:
        """grpc reports a failed bind by returning 0, which is the shape that gets ignored.

        A shard whose port is held by an earlier run is a shard the launcher can never reach.
        Finding that out at the first ``AddCamera`` would be finding it out after the decoder
        threads and the CUDA context are already open — hence the check inside
        :func:`serve_shard` rather than at its call sites.
        """
        server, _ = shard
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        intruder = InprocessRunner(chain, workers=1)

        with pytest.raises(ConfigurationError, match="control port"):
            serve_shard(intruder, shard_id=5, control_port=server.identity.control_port)

        assert not intruder.is_running


class TestTheBoundPortIsReleasedWhenTheWiringFails:
    def test_a_servicer_that_cannot_be_attached_leaves_no_listening_socket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Everything after a successful bind has to be unwound, or the port is held forever.

        ``add_insecure_port`` takes the port; ``add_ShardServicer_to_server`` and ``start``
        come after it. If either raises, the ``grpc.Server`` goes out of scope holding a
        listening socket and a thread pool that nothing references and nothing can stop - and
        the operator's retry, or the next shard handed this port, is refused by a server that
        no longer exists as far as the program is concerned.

        Asserted the only way that means anything: bind the port afterwards, from this test.
        """
        from shipinfer.launch.proto import shard_pb2_grpc

        def boom(servicer: object, server: object) -> None:
            raise RuntimeError("the generated stub and the servicer disagree")

        monkeypatch.setattr(shard_pb2_grpc, "add_ShardServicer_to_server", boom)
        chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
        runner = InprocessRunner(chain, workers=1)
        port = _free_port()

        with pytest.raises(RuntimeError, match="disagree"):
            serve_shard(runner, shard_id=6, control_port=port)

        assert not runner.is_running
        with socket.socket() as probe:
            # No SO_REUSEADDR: a leaked *listening* socket must make this raise.
            probe.bind(("127.0.0.1", port))
            probe.listen(1)


class TestAnUnreachableShard:
    def test_wait_ready_stops_polling_when_its_budget_is_spent(self) -> None:
        """``timeout_s`` is the budget for the whole call, including the poll in flight.

        A port that is **bound but never answers** is what a wedged child looks like, and it
        is the case the clip exists for: the per-poll RPC deadline backs off towards 5 s, so
        one unanswered poll used to run past the caller's entire budget before the loop got
        to check it. A launcher supervising fifty shards with a 0.3 s probe cannot spend
        seconds on each.
        """
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)  # accepts the TCP connection, then says nothing at all
            port = listener.getsockname()[1]

            with ShardClient(port, shard_id=9, timeout_s=0.2) as client:
                started = time.monotonic()

                assert not client.wait_ready(timeout_s=0.3)

                elapsed = time.monotonic() - started

        assert elapsed < 0.5, f"wait_ready overran its 0.3s budget by far: {elapsed:.2f}s"
        assert elapsed >= 0.2, "it cannot have polled at all in that time"
        assert client.identity is None

    def test_wait_ready_gives_up_rather_than_blocking_forever(self) -> None:
        """A never-ready child is a decision for the caller (kill it), not an exception.

        Bounded hard: a launcher that hangs on one wedged child cannot take the rest of the
        fleet down, which is the whole reason this returns a bool.
        """
        # Port 1 is privileged and unbound here, so the connection is refused immediately.
        with ShardClient(1, shard_id=9, timeout_s=0.2) as client:
            assert not client.wait_ready(timeout_s=0.5)
            assert client.identity is None
