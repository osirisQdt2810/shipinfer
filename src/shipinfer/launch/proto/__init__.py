"""The generated gRPC stubs for the shard control plane (``shard.proto``).

**This package deliberately exports nothing at module scope.** ``shard_pb2_grpc`` imports
``grpc`` at its first line, and ``grpc`` is an optional extra (``pip install
"shipinfer[grpc]"``); re-exporting either module from here would make ``import
shipinfer.launch`` — which every launcher does, on the one CPU-only process in the
deployment — fail on a host that never installed it. Import them where they are used, inside
the function that uses them:

    from shipinfer.launch.proto import shard_pb2

``shard_pb2`` itself needs only ``protobuf``, but it is left out of this file for the same
reason: one rule for the package is easier to keep than two.

The stubs are generated and committed; ``python scripts/gen_proto.py`` puts them back and
``tests/launch/test_proto_is_current.py`` fails if they drift from the ``.proto``.
"""

from __future__ import annotations

__all__: list[str] = []
