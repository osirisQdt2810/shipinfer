"""The launcher: spawn the shards, supervise them, stop them together (arch.md §2).

A node runs one launcher process and *n* shard processes, one per GPU, and this package is
the parent's half of that arrangement. It owns three things:

* :class:`~shipinfer.launch.supervisor.Fleet` — spawn one process per shard, notice when one
  dies, and take the rest down rather than serve three quarters of the cameras behind a green
  dashboard;
* :func:`~shipinfer.launch.signals.forward_signals` — make the operator's Ctrl-C mean that;
* :class:`~shipinfer.launch.client.ShardClient` — the parent's half of the gRPC control
  plane (arch.md §2), with :mod:`~shipinfer.launch.control`'s transport-free vocabulary and
  the generated stubs under :mod:`shipinfer.launch.proto`. A shard receives nothing in argv
  beyond its own identity; its camera set, topology and configuration arrive as RPCs after it
  reports ready, and this is what sends them.

The shard's *other* half — the servicer that answers these calls — is
:mod:`shipinfer.runners.service`, and the direction is deliberate: a launcher that imported
the thing it launches would pay for the executor in the parent process. The runner that
drives this client across a whole fleet is :class:`shipinfer.runners.fleet.FleetRunner`, and
it lives above this package for the same reason.

**grpcio and protobuf are an optional extra** (``pip install "shipinfer[grpc]"``) and nothing
here imports either at module scope, so ``import shipinfer.launch`` works on a host that has
neither. The first call on a :class:`ShardClient` raises a typed
:class:`~shipinfer.core.errors.ConfigurationError` naming the extra — the same shape
``api/app.py`` uses for FastAPI.

**Nothing here may import torch, and that is not tidiness.** The whole reason a shard is a
subprocess rather than a thread is that ``CUDA_VISIBLE_DEVICES`` has to be in the child's
environment *before* its interpreter imports torch (see the supervisor's module docstring). A
launcher that had imported torch itself would have a CUDA context of its own on every device
it can see, on the one process in the deployment that needs no device at all —
``FORBIDDEN_EXTERNAL["launch"]`` in ``scripts/hooks/check_layers.py`` is what keeps that true.
"""

from shipinfer.launch.client import ShardClient
from shipinfer.launch.control import (
    AddCameraResult,
    CameraSpec,
    ShardHealth,
    ShardIdentity,
    ShardState,
    StopResult,
)
from shipinfer.launch.signals import Stoppable, forward_signals
from shipinfer.launch.supervisor import DEFAULT_DRAIN_S, Fleet, ShardProcess

__all__ = [
    "DEFAULT_DRAIN_S",
    "AddCameraResult",
    "CameraSpec",
    "Fleet",
    "ShardClient",
    "ShardHealth",
    "ShardIdentity",
    "ShardProcess",
    "ShardState",
    "StopResult",
    "Stoppable",
    "forward_signals",
]
