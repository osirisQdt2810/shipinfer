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

import textwrap
from collections.abc import Iterator

import pytest

pytest.importorskip("grpc", reason="the grpc extra is not installed")

from shipinfer.core.errors import ConfigurationError
from shipinfer.launch import CameraSpec, ShardClient, ShardState
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.runners.service import ShardServer, serve_shard
from shipinfer.topology import ChainSpec, Topology

#: A straight line of mock elements: the smallest chain a real runner will start, and enough
#: to prove the shard is executing something rather than answering from an empty object.
CHAIN = """
name: rpc_linear
elements:
  decode: {impl: mock}
  detect: {impl: mock, model: ship_detector}
  output: {impl: mock}
"""


@pytest.fixture()
def shard() -> Iterator[tuple[ShardServer, InprocessRunner]]:
    """A shard serving on a loopback ephemeral port, torn down whatever the test does."""
    chain = Topology.from_spec(ChainSpec.from_yaml(textwrap.dedent(CHAIN)))
    runner = InprocessRunner(chain, workers=1)
    server = serve_shard(runner, shard_id=4, control_port=0)
    try:
        yield server, runner
    finally:
        server.stop(grace_s=1.0)
        runner.stop(timeout_s=2.0)


@pytest.fixture()
def client(shard: tuple[ShardServer, InprocessRunner]) -> Iterator[ShardClient]:
    server, _ = shard
    with ShardClient(server.identity.control_port, shard_id=4, timeout_s=5.0) as connected:
        yield connected


class TestALauncherTalksToAShard:
    def test_the_whole_sequence_a_launcher_performs(
        self, shard: tuple[ShardServer, InprocessRunner], client: ShardClient
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

        # The in-process runner manages no cameras, and the refusal is the launcher's cue to
        # place this one elsewhere. It arrives as data, not as an RPC error status.
        refusal = client.add_camera(CameraSpec("cam-1", "rtsp://10.0.0.1/live", 20.0))
        assert not refusal.accepted
        assert "this runner does not manage cameras" in refusal.reason

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
        self, shard: tuple[ShardServer, InprocessRunner]
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


class TestAnUnreachableShard:
    def test_wait_ready_gives_up_rather_than_blocking_forever(self) -> None:
        """A never-ready child is a decision for the caller (kill it), not an exception.

        Bounded hard: a launcher that hangs on one wedged child cannot take the rest of the
        fleet down, which is the whole reason this returns a bool.
        """
        # Port 1 is privileged and unbound here, so the connection is refused immediately.
        with ShardClient(1, shard_id=9, timeout_s=0.2) as client:
            assert not client.wait_ready(timeout_s=0.5)
            assert client.identity is None
