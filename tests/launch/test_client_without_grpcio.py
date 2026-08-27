"""``import shipinfer.launch`` works on a host with no grpcio, and says so when asked to talk.

grpcio and protobuf are the optional ``grpc`` extra, and the launcher is the one CPU-only
process in the deployment: it must import on a host that installed neither, because the
alternative is that ``shipinfer serve`` on a laptop fails at import time over a dependency it
was never going to use. That is the same promise ``api/`` keeps for FastAPI, and the same
shape of refusal — a :class:`~shipinfer.core.errors.ConfigurationError` naming the extra to
install, raised at the first *use*, rather than an ``ImportError`` from four frames down that
names ``grpc._channel``.

The masking below is ``sys.modules[name] = None``, which is CPython's own "this module is not
importable" marker: an ``import grpc`` against it raises ``ImportError`` exactly as an absent
package would, without needing a host that actually lacks one.
"""

from __future__ import annotations

import sys

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.launch import ShardClient
from shipinfer.launch.control import CameraSpec

#: Everything that would let a call through: `grpc` itself, and the generated stub module
#: that imports it. Masking only `grpc` would leave an already-imported `shard_pb2_grpc`
#: usable in a suite where another test imported it first.
MASKED = ("grpc", "shipinfer.launch.proto.shard_pb2_grpc")


@pytest.fixture()
def no_grpcio(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MASKED:
        monkeypatch.setitem(sys.modules, name, None)


class TestTheLauncherImportsWithoutTheExtra:
    def test_the_package_and_the_client_class_are_importable(self) -> None:
        """Asserted by the imports at the top of this file, and stated here so it is visible."""
        assert callable(ShardClient)

    def test_constructing_a_client_costs_no_import(self, no_grpcio: None) -> None:
        """The connection is made on first use, not in ``__init__``.

        Two reasons, and the optional extra is only the first: a launcher constructs a client
        for a child it has just spawned, *before* that child has bound its port, so connecting
        in the constructor would connect to nothing.
        """
        client = ShardClient(50100, shard_id=2)

        assert client.address == "127.0.0.1:50100"
        assert client.shard_id == 2
        assert client.identity is None


class TestTheFirstCallRefusesTyped:
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("health", ()),
            ("stats", ()),
            ("update_topology", ("name: x\nelements: {}\n",)),
            ("add_camera", (CameraSpec("cam-1", "rtsp://x"),)),
            ("remove_camera", ("cam-1",)),
            ("drain", (0.1,)),
            ("stop", (0.1,)),
        ],
    )
    def test_every_call_names_the_extra(
        self, no_grpcio: None, method: str, args: tuple
    ) -> None:
        client = ShardClient(50100)

        with pytest.raises(ConfigurationError) as excinfo:
            getattr(client, method)(*args)

        message = str(excinfo.value)
        assert "grpcio" in message
        assert 'pip install "shipinfer[grpc]"' in message

    def test_wait_ready_refuses_rather_than_polling_for_two_minutes(
        self, no_grpcio: None
    ) -> None:
        """A missing dependency is not a child that is slow to start.

        ``wait_ready`` swallows "the shard did not answer" on purpose — that is what polling
        is — so it would have burned its whole deadline retrying an import that can never
        succeed. The refusal has to come out past the retry loop.
        """
        client = ShardClient(50100)

        with pytest.raises(ConfigurationError):
            client.wait_ready(timeout_s=60.0)

    def test_close_is_safe_before_the_first_call(self, no_grpcio: None) -> None:
        client = ShardClient(50100)

        client.close()
        client.close()
