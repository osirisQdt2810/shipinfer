"""How the launcher reads a reply, with the transport replaced by a two-line double.

``tests/launch/test_shard_rpc.py`` proves the wiring over a real socket and is deliberately
the only file that does; ``tests/runners/test_shard_service.py`` proves the servicer's side of
the mapping. What is left is the *client's* half - the reply fields it dispatches on and the
exception each one becomes - and that needs no transport at all: a stub with one method is
exactly as good, and it is the only way to hand the client a reply combination a real runner
would have to be broken to produce.

The seam used here is :meth:`ShardClient._rpc`'s cache: a client with ``_stub`` already set
never connects, which is the same laziness that keeps grpcio optional.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.protobuf", reason="the grpc extra is not installed")

from shipinfer.core.errors import ConfigurationError, ServerStateError
from shipinfer.launch import ShardClient
from shipinfer.launch.proto import shard_pb2 as pb


class FakeGrpc:
    """Just enough of the module for ``except self._grpc.RpcError`` to be a valid clause."""

    class RpcError(Exception):
        pass


class FakeStub:
    """One reply, and a record of what was asked for."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[tuple[Any, float]] = []

    # protobuf's CamelCase, because that is the attribute the client calls by name.
    def RemoveCamera(self, request: Any, timeout: float) -> Any:
        self.calls.append((request, timeout))
        return self.reply


def client_answering(reply: Any) -> tuple[ShardClient, FakeStub]:
    client = ShardClient(50100, shard_id=3)
    stub = FakeStub(reply)
    client._stub = stub
    client._grpc = FakeGrpc
    return client, stub


class TestRemoveCameraTellsTheTwoFailuresApart:
    def test_a_camera_the_shard_never_had_is_a_configuration_error(self) -> None:
        client, _ = client_answering(
            pb.RemoveCameraReply(
                removed=False, clean=False, reason="camera 'cam-7' is not running"
            )
        )

        with pytest.raises(ConfigurationError) as excinfo:
            client.remove_camera("cam-7")

        assert "does not run camera 'cam-7'" in str(excinfo.value)

    def test_a_removal_that_failed_is_a_server_state_error_with_the_reason(self) -> None:
        """Not "this shard does not run cam-7": it does, and the removal is what went wrong.

        The exception type is the load-bearing part. A launcher placing cameras treats a
        ``ConfigurationError`` as "wrong shard, try another"; this camera is on *this* shard
        and may still be holding a thread, which is a different decision entirely.
        """
        client, _ = client_answering(
            pb.RemoveCameraReply(removed=True, clean=False, reason="the manager is stopping")
        )

        with pytest.raises(ServerStateError) as excinfo:
            client.remove_camera("cam-7")

        message = str(excinfo.value)
        assert "could not remove camera 'cam-7'" in message
        assert "the manager is stopping" in message

    def test_an_abandoned_thread_is_a_false_return_not_an_exception(self) -> None:
        """``removed`` with no reason: the removal worked, the thread outlived its deadline.

        The caller's to know rather than the log's to bury, so it is a return value.
        """
        client, _ = client_answering(pb.RemoveCameraReply(removed=True, clean=False))

        assert client.remove_camera("cam-7") is False

    def test_a_clean_removal_returns_true(self) -> None:
        client, stub = client_answering(pb.RemoveCameraReply(removed=True, clean=True))

        assert client.remove_camera("cam-7", timeout_s=0.25) is True
        request, timeout = stub.calls[0]
        assert (request.camera_id, request.timeout_s) == ("cam-7", 0.25)
        # The RPC deadline outlives the stop deadline it carries, or a removal that succeeds
        # would be reported as a failure.
        assert timeout > 0.25
