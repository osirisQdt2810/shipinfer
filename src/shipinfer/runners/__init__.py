"""Runners — *how* a topology executes (arch.md §1), one implementation per module.

The third of the three concepts. A :class:`~shipinfer.runners.base.Runner` is handed a
validated :class:`~shipinfer.topology.chain.Topology` and executes it: ``inprocess`` here,
``fleet`` (shard processes over gRPC) and ``deepstream`` (the chain compiled into a GStreamer
graph) in later phases. One chain definition, three executions.

Typical use::

    from shipinfer.runners import build_runner
    from shipinfer.topology import load_topology

    chain = load_topology("topology/ship_person.yaml")
    with build_runner("inprocess", topology=chain, models=server) as runner:
        runner.submit(item)

This package imports ``core``, ``topology`` and ``scheduling`` — all pure — and reaches the
model pool only through the :class:`~shipinfer.topology.base.ModelResolver` it is *given*.
So ``import shipinfer.runners`` costs no torch, no TensorRT and no engine, and a chain can be
started with mock elements on a host with no driver. A runner that genuinely needs a heavy
import (the DeepStream compiler) registers lazily, as the element registries do.
"""

from __future__ import annotations

from shipinfer.runners.base import Runner

# Imported for the side effect: this is what puts `inprocess` in the registry. Not
# re-exported as a class -- a runner is reached through `build_runner`, by the name a
# settings tree uses, which is what keeps the registry the seam.
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.runners.registry import RUNNERS, build_runner

# The shard's half of the gRPC control plane (arch.md §2). It lives here rather than in
# `launch/` because it holds a runner, and a launcher that imported the thing it launches
# would pay for the executor in the parent process. Nothing it imports needs grpcio at module
# scope, so this re-export costs a laptop nothing.
from shipinfer.runners.service import ShardService, serve_shard

__all__ = [
    "RUNNERS",
    "InprocessRunner",
    "Runner",
    "ShardService",
    "build_runner",
    "serve_shard",
]
