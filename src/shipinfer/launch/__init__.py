"""The launcher: spawn the shards, supervise them, stop them together (arch.md §2).

A node runs one launcher process and *n* shard processes, one per GPU, and this package is
the parent's half of that arrangement. It owns exactly two things today:

* :class:`~shipinfer.launch.supervisor.Fleet` — spawn one process per shard, notice when one
  dies, and take the rest down rather than serve three quarters of the cameras behind a green
  dashboard;
* :func:`~shipinfer.launch.signals.forward_signals` — make the operator's Ctrl-C mean that.

**What it deliberately does not own yet is the conversation with the children.** arch.md §2
puts the control plane on gRPC: a shard receives nothing in argv beyond its own identity, and
its camera set, topology and configuration arrive as RPCs after it reports ready. That client
lands here in A2 PR-5, the runner that drives it in PR-6, and the argv-rendering
``server/topology/`` classes are deleted in the same change. Until then the supervisor is
what it was under ``server/``: unchanged code, in the package it belongs to.

**Nothing here may import torch, and that is not tidiness.** The whole reason a shard is a
subprocess rather than a thread is that ``CUDA_VISIBLE_DEVICES`` has to be in the child's
environment *before* its interpreter imports torch (see the supervisor's module docstring). A
launcher that had imported torch itself would have a CUDA context of its own on every device
it can see, on the one process in the deployment that needs no device at all —
``FORBIDDEN_EXTERNAL["launch"]`` in ``scripts/hooks/check_layers.py`` is what keeps that true.
"""

from shipinfer.launch.signals import forward_signals
from shipinfer.launch.supervisor import DEFAULT_DRAIN_S, Fleet, ShardProcess

__all__ = ["DEFAULT_DRAIN_S", "Fleet", "ShardProcess", "forward_signals"]
